from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch

from thrml.block_management import block_state_to_global

from ..discrete_ebm_packing import PackedSpinGammaBatch, pack_spin_gamma_batches
from ..interaction_lowering import lower_block_interactions
from ...runtime_config import SPIN_PARAMETER_FAMILY
from ..spin_ops import SpinGammaInputs
from .runtime import (
    TTMLIRConfig,
    compile_stablehlo_to_flatbuffer,
    get_or_compile_cached_artifact,
    make_ttmlir_config,
    run_flatbuffer,
    stablehlo_text,
    supports_direct_ttnn_inputs,
)
from .tensor_bridge import (
    abstract_jax_input,
    default_scale_tensor_like,
    default_float32_scale,
    normalized_dtype_name,
    execute_single_output_flatbuffer,
    numpy_to_torch,
    torch_to_jax,
    tensor_shape,
    optional_ttnn_input_as_torch,
    ttnn_input_as_torch,
)
from ..ttnn_kernels import spin_gamma_dense_expected


@dataclass(frozen=True)
class TTMLIRSpinGammaOpSignature:
    flat_weights_shape: tuple[int, ...]
    flat_weights_dtype: str
    flat_index_shape: tuple[int, ...] | None
    flat_index_dtype: str | None
    interaction_scale_shape: tuple[int, ...]
    interaction_scale_dtype: str
    n_nodes: int
    n_interactions: int

    def stable_cache_key(self) -> str:
        payload = {
            "flat_weights_shape": list(self.flat_weights_shape),
            "flat_weights_dtype": self.flat_weights_dtype,
            "flat_index_shape": None
            if self.flat_index_shape is None
            else list(self.flat_index_shape),
            "flat_index_dtype": self.flat_index_dtype,
            "interaction_scale_shape": list(self.interaction_scale_shape),
            "interaction_scale_dtype": self.interaction_scale_dtype,
            "n_nodes": self.n_nodes,
            "n_interactions": self.n_interactions,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


@dataclass(frozen=True)
class TTMLIRSpinGammaOpArtifact:
    signature: TTMLIRSpinGammaOpSignature
    artifact_dir: Path
    stablehlo_path: Path
    ttir_path: Path
    ttnn_path: Path
    flatbuffer_path: Path
    stablehlo_to_ttir_command: tuple[str, ...]
    ttir_to_ttnn_command: tuple[str, ...]
    ttnn_to_flatbuffer_command: tuple[str, ...]


class TTMLIRSpinGammaOp:
    def __init__(
        self,
        *,
        config: TTMLIRConfig | None = None,
        system_desc_path: Path | str | None = None,
        artifact_root: Path | str | None = None,
        build_dir: Path | str | None = None,
        ttmlir_opt: Path | str | None = None,
        ttmlir_translate: Path | str | None = None,
        base_name_prefix: str = "spin_gamma_op",
    ):
        self.config = make_ttmlir_config(
            config=config,
            system_desc_path=system_desc_path,
            artifact_root=artifact_root,
            build_dir=build_dir,
            ttmlir_opt=ttmlir_opt,
            ttmlir_translate=ttmlir_translate,
        )
        self.base_name_prefix = base_name_prefix

    def _compile_artifact(
        self,
        signature: TTMLIRSpinGammaOpSignature,
    ) -> TTMLIRSpinGammaOpArtifact:
        compiled = get_or_compile_cached_artifact(
            self.config,
            family=SPIN_PARAMETER_FAMILY.value,
            signature=signature,
            base_name_prefix=self.base_name_prefix,
            stablehlo_module_text_factory=lambda: lower_spin_gamma_inputs_to_stablehlo(
                signature=signature,
            ),
            compile_fn=compile_stablehlo_to_flatbuffer,
        )
        return TTMLIRSpinGammaOpArtifact(
            signature=signature,
            artifact_dir=compiled.artifact_dir,
            stablehlo_path=compiled.stablehlo_path,
            ttir_path=compiled.ttir_path,
            ttnn_path=compiled.ttnn_path,
            flatbuffer_path=compiled.flatbuffer_path,
            stablehlo_to_ttir_command=compiled.stablehlo_to_ttir_command,
            ttir_to_ttnn_command=compiled.ttir_to_ttnn_command,
            ttnn_to_flatbuffer_command=compiled.ttnn_to_flatbuffer_command,
        )

    def __call__(
        self,
        *,
        ttnn,
        device,
        inputs: SpinGammaInputs,
    ):
        signature = spin_gamma_op_signature(
            flat_weights_shape=tensor_shape(inputs.flat_weights),
            flat_weights_dtype=torch.float32,
            flat_index_shape=None
            if inputs.flat_index is None
            else tensor_shape(inputs.flat_index),
            flat_index_dtype=None
            if inputs.flat_index is None
            else getattr(inputs.flat_index, "dtype", torch.uint32),
            interaction_scale_shape=_default_scale_shape(
                flat_weights_shape=tensor_shape(inputs.flat_weights),
                flat_index_shape=None
                if inputs.flat_index is None
                else tensor_shape(inputs.flat_index),
                interaction_scale=inputs.interaction_scale,
            ),
            interaction_scale_dtype=torch.float32,
            n_nodes=inputs.n_nodes,
            n_interactions=inputs.n_interactions,
        )
        artifact = self._compile_artifact(signature)
        direct_runtime_inputs = [inputs.flat_weights]
        if inputs.flat_index is not None:
            direct_runtime_inputs.append(inputs.flat_index)
        direct_runtime_inputs.append(
            inputs.interaction_scale
            if inputs.interaction_scale is not None
            else default_scale_tensor_like(
                ttnn,
                device=device,
                flat_weights=inputs.flat_weights,
                flat_index=inputs.flat_index,
            )
        )
        return execute_single_output_flatbuffer(
            config=self.config,
            ttnn=ttnn,
            device=device,
            flatbuffer_path=artifact.flatbuffer_path,
            input_tensors_factory=lambda: _runtime_host_inputs(ttnn, inputs),
            direct_input_tensors=direct_runtime_inputs,
            output_reference=inputs.flat_weights,
            op_name="spin-gamma op",
            run_flatbuffer_fn=run_flatbuffer,
            supports_direct_ttnn_inputs_fn=supports_direct_ttnn_inputs,
        )


def extract_program_spin_gamma_batches(
    *,
    program,
    block_index: int,
    state_free,
    state_clamp,
    global_state=None,
    max_instances_per_batch: int | None = None,
) -> list[PackedSpinGammaBatch]:
    if global_state is None:
        global_state = block_state_to_global(
            list(state_free) + list(state_clamp), program.gibbs_spec
        )

    lowered_interactions = lower_block_interactions(
        program,
        block_index,
        parameter_family=SPIN_PARAMETER_FAMILY,
    )
    interaction_states = []
    for lowered_interaction in lowered_interactions:
        this_interaction_states = []
        for global_index, source_slice in zip(
            lowered_interaction.source_global_inds,
            lowered_interaction.source_global_slices,
        ):
            this_interaction_states.append(
                jax.tree.map(
                    lambda value: jnp.take(value, source_slice, axis=0),
                    global_state[global_index],
                )
            )
        interaction_states.append(this_interaction_states)

    return pack_spin_gamma_batches(
        [entry.contribution for entry in lowered_interactions],
        [entry.active_mask for entry in lowered_interactions],
        interaction_states,
        max_instances_per_batch=max_instances_per_batch,
    )


def spin_gamma_expected_from_batches(
    batches: Sequence[PackedSpinGammaBatch],
) -> torch.Tensor:
    gamma = None
    for batch in batches:
        partial = (
            spin_gamma_dense_expected(
                batch.weights,
                batch.active_mask,
                batch.spin_conditions,
                batch.categorical_conditions,
            )
            .float()
            .squeeze(0)
            .squeeze(-1)
        )
        gamma = partial if gamma is None else gamma + partial

    if gamma is None:
        raise ValueError("At least one packed spin-gamma batch is required.")

    return gamma


def _to_jax_float(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.float32)


def _to_jax_int32(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.int32)


def _to_jax_uint32(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.uint32)


def _make_spin_gamma_kernel(batch: PackedSpinGammaBatch):
    tail_shape = tuple(int(size) for size in batch.tail_shape)

    def gamma_kernel(weights_in, active_mask_in, *conditions):
        spin_inputs = conditions[: batch.n_spin]
        categorical_inputs = conditions[batch.n_spin :]
        gathered = weights_in

        if tail_shape:
            if len(categorical_inputs) != len(tail_shape):
                raise ValueError(
                    "Categorical condition count must match the trailing "
                    "weight dimensions."
                )
            flat_index = jnp.zeros_like(categorical_inputs[0], dtype=jnp.uint32)
            stride = 1
            for condition, size in zip(reversed(categorical_inputs), reversed(tail_shape)):
                flat_index = flat_index + condition.astype(jnp.uint32) * jnp.asarray(
                    int(stride),
                    dtype=jnp.uint32,
                )
                stride *= size
            flat_weights = weights_in.reshape(
                *active_mask_in.shape,
                math.prod(tail_shape),
            )
            selector = jax.nn.one_hot(
                flat_index.astype(jnp.int32),
                flat_weights.shape[-1],
                dtype=flat_weights.dtype,
            )
            gathered = jnp.sum(flat_weights * selector, axis=-1)

        spin_product = jnp.ones_like(gathered)
        for condition in spin_inputs:
            signed = jnp.where(
                condition > 0,
                jnp.ones_like(gathered),
                -jnp.ones_like(gathered),
            )
            spin_product = spin_product * signed

        return jnp.sum(
            gathered * active_mask_in.astype(gathered.dtype) * spin_product,
            axis=-1,
            keepdims=True,
        )

    return gamma_kernel


def packed_spin_gamma_batch_inputs(
    batch: PackedSpinGammaBatch,
) -> list[torch.Tensor]:
    return [
        batch.weights.to(torch.float32).contiguous(),
        batch.active_mask.to(torch.float32).contiguous(),
        *[
            condition.to(torch.int32).contiguous()
            for condition in batch.spin_conditions
        ],
        *[
            condition.to(torch.uint32).contiguous()
            for condition in batch.categorical_conditions
        ],
    ]


def compute_packed_spin_gamma_jax(batch: PackedSpinGammaBatch) -> torch.Tensor:
    kernel = _make_spin_gamma_kernel(batch)
    inputs = [
        _to_jax_float(batch.weights),
        _to_jax_float(batch.active_mask),
        *[_to_jax_int32(condition) for condition in batch.spin_conditions],
        *[_to_jax_uint32(condition) for condition in batch.categorical_conditions],
    ]
    result = jax.jit(kernel)(*inputs)
    return numpy_to_torch(result, dtype=np.float32, shape=(-1,))


def lower_packed_spin_gamma_to_stablehlo(batch: PackedSpinGammaBatch) -> str:
    kernel = _make_spin_gamma_kernel(batch)
    inputs = [
        _to_jax_float(batch.weights),
        _to_jax_float(batch.active_mask),
        *[_to_jax_int32(condition) for condition in batch.spin_conditions],
        *[_to_jax_uint32(condition) for condition in batch.categorical_conditions],
    ]
    lowered = jax.jit(kernel).lower(*inputs)
    return stablehlo_text(lowered)


def _make_spin_gamma_inputs_kernel(
    *,
    flat_index: torch.Tensor | None,
    flat_tail_size: int,
    n_nodes: int,
    n_interactions: int,
):
    def _normalize_selected_interaction_scale(
        interaction_scale_in,
        *,
        target_ndim: int,
    ):
        if interaction_scale_in.ndim == target_ndim:
            return interaction_scale_in
        if (
            interaction_scale_in.ndim == target_ndim + 1
            and int(interaction_scale_in.shape[-1]) == 1
        ):
            return jnp.squeeze(interaction_scale_in, axis=-1)
        return interaction_scale_in

    def gamma_kernel(flat_weights_in, *dynamic_inputs):
        if flat_index is None:
            interaction_scale_in = dynamic_inputs[0]
            gathered = flat_weights_in
        else:
            flat_index_in, interaction_scale_in = dynamic_inputs
            selector = jax.nn.one_hot(
                jnp.squeeze(flat_index_in, axis=-1).astype(jnp.int32),
                flat_tail_size,
                dtype=flat_weights_in.dtype,
            )
            gathered = jnp.sum(
                flat_weights_in * selector,
                axis=-1,
            )
            interaction_scale_in = _normalize_selected_interaction_scale(
                interaction_scale_in,
                target_ndim=gathered.ndim,
            )

        weighted = gathered * interaction_scale_in.astype(gathered.dtype)
        if weighted.ndim != 4:
            weighted = jnp.reshape(
                weighted,
                (weighted.shape[0], 1, n_nodes, n_interactions),
            )
        return jnp.sum(weighted, axis=3, keepdims=True)

    return gamma_kernel


def spin_gamma_op_signature(
    *,
    flat_weights: torch.Tensor | None = None,
    flat_index: torch.Tensor | None = None,
    interaction_scale: torch.Tensor | None = None,
    flat_weights_shape: tuple[int, ...] | None = None,
    flat_weights_dtype=None,
    flat_index_shape: tuple[int, ...] | None = None,
    flat_index_dtype=None,
    interaction_scale_shape: tuple[int, ...] | None = None,
    interaction_scale_dtype=None,
    n_nodes: int,
    n_interactions: int,
) -> TTMLIRSpinGammaOpSignature:
    if flat_weights_shape is None:
        if flat_weights is None:
            raise TypeError("spin_gamma_op_signature requires flat_weights or flat_weights_shape.")
        flat_weights_shape = tensor_shape(flat_weights)
    if flat_weights_dtype is None:
        flat_weights_dtype = torch.float32 if flat_weights is None else flat_weights.dtype
    if flat_index is not None and flat_index_shape is None:
        flat_index_shape = tensor_shape(flat_index)
    if flat_index_dtype is None and flat_index is not None:
        flat_index_dtype = flat_index.dtype
    if interaction_scale_shape is None:
        if interaction_scale is not None:
            interaction_scale_shape = tensor_shape(interaction_scale)
        else:
            interaction_scale_shape = _default_scale_shape(
                flat_weights_shape=flat_weights_shape,
                flat_index_shape=flat_index_shape,
                interaction_scale=None,
            )
    if interaction_scale_dtype is None:
        interaction_scale_dtype = (
            torch.float32 if interaction_scale is None else interaction_scale.dtype
        )
    return TTMLIRSpinGammaOpSignature(
        flat_weights_shape=tuple(int(dim) for dim in flat_weights_shape),
        flat_weights_dtype=normalized_dtype_name(flat_weights_dtype),
        flat_index_shape=None if flat_index_shape is None else tuple(int(dim) for dim in flat_index_shape),
        flat_index_dtype=None if flat_index_dtype is None else normalized_dtype_name(flat_index_dtype),
        interaction_scale_shape=tuple(int(dim) for dim in interaction_scale_shape),
        interaction_scale_dtype=normalized_dtype_name(interaction_scale_dtype),
        n_nodes=int(n_nodes),
        n_interactions=int(n_interactions),
    )


def lower_spin_gamma_inputs_to_stablehlo(
    *,
    signature: TTMLIRSpinGammaOpSignature | None = None,
    flat_weights: torch.Tensor | None = None,
    flat_index: torch.Tensor | None = None,
    interaction_scale: torch.Tensor | None = None,
    n_nodes: int | None = None,
    n_interactions: int | None = None,
) -> str:
    signature = signature or spin_gamma_op_signature(
        flat_weights=flat_weights,
        flat_index=flat_index,
        interaction_scale=interaction_scale,
        n_nodes=n_nodes,
        n_interactions=n_interactions,
    )
    kernel = _make_spin_gamma_inputs_kernel(
        flat_index=None if signature.flat_index_shape is None else object(),
        flat_tail_size=int(signature.flat_weights_shape[-1]),
        n_nodes=signature.n_nodes,
        n_interactions=signature.n_interactions,
    )
    lowered_inputs = [
        abstract_jax_input(signature.flat_weights_shape, dtype=signature.flat_weights_dtype),
    ]
    if signature.flat_index_shape is not None:
        lowered_inputs.append(
            abstract_jax_input(signature.flat_index_shape, dtype=signature.flat_index_dtype)
        )
    lowered_inputs.append(
        abstract_jax_input(
            signature.interaction_scale_shape,
            dtype=signature.interaction_scale_dtype,
        )
    )
    lowered = jax.jit(kernel).lower(*lowered_inputs)
    return stablehlo_text(lowered)


def _runtime_host_inputs(ttnn, inputs: SpinGammaInputs) -> list[torch.Tensor]:
    flat_weights = ttnn_input_as_torch(ttnn, inputs.flat_weights, dtype=torch.float32)
    flat_index = optional_ttnn_input_as_torch(
        ttnn,
        inputs.flat_index,
        dtype=torch.uint32,
    )
    interaction_scale = optional_ttnn_input_as_torch(
        ttnn,
        inputs.interaction_scale,
        dtype=torch.float32,
    )
    if interaction_scale is None:
        interaction_scale = default_float32_scale(flat_weights, flat_index)
    runtime_inputs = [flat_weights]
    if flat_index is not None:
        runtime_inputs.append(flat_index)
    runtime_inputs.append(interaction_scale)
    return runtime_inputs


def _default_scale_shape(
    *,
    flat_weights_shape: tuple[int, ...],
    flat_index_shape: tuple[int, ...] | None,
    interaction_scale,
) -> tuple[int, ...]:
    if interaction_scale is not None:
        return tensor_shape(interaction_scale)
    if flat_index_shape is not None:
        return flat_index_shape
    return flat_weights_shape


def make_ttmlir_spin_gamma_op(
    *,
    config: TTMLIRConfig | None = None,
    system_desc_path: Path | str | None = None,
    artifact_root: Path | str | None = None,
    build_dir: Path | str | None = None,
    ttmlir_opt: Path | str | None = None,
    ttmlir_translate: Path | str | None = None,
    base_name_prefix: str = "spin_gamma_op",
) -> TTMLIRSpinGammaOp:
    return TTMLIRSpinGammaOp(
        config=config,
        system_desc_path=system_desc_path,
        artifact_root=artifact_root,
        build_dir=build_dir,
        ttmlir_opt=ttmlir_opt,
        ttmlir_translate=ttmlir_translate,
        base_name_prefix=base_name_prefix,
    )
