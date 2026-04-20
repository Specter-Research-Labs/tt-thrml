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

from ..categorical_ops import CategoricalThetaInputs, tail_strides
from ..discrete_ebm_packing import (
    PackedCategoricalThetaBatch,
    pack_categorical_theta_batches,
)
from ..interaction_lowering import lower_block_interactions
from ...runtime_config import CATEGORICAL_PARAMETER_FAMILY
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
    execute_single_output_flatbuffer,
    numpy_to_torch,
    optional_ttnn_input_as_torch,
    torch_to_jax,
    ttnn_input_as_torch,
)
from ..ttnn_kernels import categorical_theta_dense_expected

@dataclass(frozen=True)
class TTMLIRCategoricalThetaOpSignature:
    flat_weights_shape: tuple[int, ...]
    flat_weights_dtype: str
    flat_index_shape: tuple[int, ...] | None
    flat_index_dtype: str | None
    interaction_scale_shape: tuple[int, ...]
    interaction_scale_dtype: str
    n_nodes: int
    n_interactions: int
    n_categories: int

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
            "n_categories": self.n_categories,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


@dataclass(frozen=True)
class TTMLIRCategoricalThetaOpArtifact:
    signature: TTMLIRCategoricalThetaOpSignature
    artifact_dir: Path
    stablehlo_path: Path
    ttir_path: Path
    ttnn_path: Path
    flatbuffer_path: Path
    stablehlo_to_ttir_command: tuple[str, ...]
    ttir_to_ttnn_command: tuple[str, ...]
    ttnn_to_flatbuffer_command: tuple[str, ...]


class TTMLIRCategoricalThetaOp:
    def __init__(
        self,
        *,
        config: TTMLIRConfig | None = None,
        system_desc_path: Path | str | None = None,
        artifact_root: Path | str | None = None,
        build_dir: Path | str | None = None,
        ttmlir_opt: Path | str | None = None,
        ttmlir_translate: Path | str | None = None,
        base_name_prefix: str = "categorical_theta_op",
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
        signature: TTMLIRCategoricalThetaOpSignature,
        *,
        flat_weights: torch.Tensor,
        flat_index: torch.Tensor | None,
        interaction_scale: torch.Tensor,
    ) -> TTMLIRCategoricalThetaOpArtifact:
        compiled = get_or_compile_cached_artifact(
            self.config,
            family=CATEGORICAL_PARAMETER_FAMILY.value,
            signature=signature,
            base_name_prefix=self.base_name_prefix,
            stablehlo_module_text_factory=lambda: lower_categorical_theta_inputs_to_stablehlo(
                flat_weights=flat_weights,
                flat_index=flat_index,
                interaction_scale=interaction_scale,
                n_nodes=signature.n_nodes,
                n_interactions=signature.n_interactions,
                n_categories=signature.n_categories,
            ),
            compile_fn=compile_stablehlo_to_flatbuffer,
        )
        return TTMLIRCategoricalThetaOpArtifact(
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
        inputs: CategoricalThetaInputs,
    ):
        flat_weights = ttnn_input_as_torch(ttnn, inputs.flat_weights, dtype=torch.float32)
        interaction_scale = ttnn_input_as_torch(
            ttnn,
            inputs.interaction_scale,
            dtype=torch.float32,
        )
        flat_index = optional_ttnn_input_as_torch(
            ttnn,
            inputs.flat_index,
            dtype=torch.uint32,
        )

        signature = categorical_theta_op_signature(
            flat_weights=flat_weights,
            flat_index=flat_index,
            interaction_scale=interaction_scale,
            n_nodes=inputs.n_nodes,
            n_interactions=inputs.n_interactions,
            n_categories=inputs.n_categories,
        )
        artifact = self._compile_artifact(
            signature,
            flat_weights=flat_weights,
            flat_index=flat_index,
            interaction_scale=interaction_scale,
        )
        runtime_inputs = [flat_weights]
        if flat_index is not None:
            runtime_inputs.append(flat_index)
        runtime_inputs.append(interaction_scale)
        direct_runtime_inputs = [inputs.flat_weights]
        if inputs.flat_index is not None:
            direct_runtime_inputs.append(inputs.flat_index)
        direct_runtime_inputs.append(inputs.interaction_scale)
        return execute_single_output_flatbuffer(
            config=self.config,
            ttnn=ttnn,
            device=device,
            flatbuffer_path=artifact.flatbuffer_path,
            input_tensors=runtime_inputs,
            direct_input_tensors=direct_runtime_inputs,
            output_reference=inputs.flat_weights,
            op_name="categorical-theta op",
            run_flatbuffer_fn=run_flatbuffer,
            supports_direct_ttnn_inputs_fn=supports_direct_ttnn_inputs,
        )


def extract_program_categorical_theta_batches(
    *,
    program,
    block_index: int,
    state_free,
    state_clamp,
    global_state=None,
    max_instances_per_batch: int | None = None,
) -> list[PackedCategoricalThetaBatch]:
    if global_state is None:
        global_state = block_state_to_global(
            list(state_free) + list(state_clamp), program.gibbs_spec
        )

    lowered_interactions = lower_block_interactions(
        program,
        block_index,
        parameter_family=CATEGORICAL_PARAMETER_FAMILY,
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

    return pack_categorical_theta_batches(
        [entry.contribution for entry in lowered_interactions],
        [entry.active_mask for entry in lowered_interactions],
        interaction_states,
        max_instances_per_batch=max_instances_per_batch,
    )


def categorical_theta_expected_from_batches(
    batches: Sequence[PackedCategoricalThetaBatch],
) -> torch.Tensor:
    theta = None
    for batch in batches:
        partial = (
            categorical_theta_dense_expected(
                batch.weights,
                batch.active_mask,
                batch.spin_conditions,
                batch.categorical_conditions,
            )
            .float()
            .squeeze(0)
        )
        theta = partial if theta is None else theta + partial

    if theta is None:
        raise ValueError("At least one packed categorical-theta batch is required.")

    return theta


def _to_jax_float(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.float32)


def _to_jax_int32(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.int32)


def _to_jax_uint32(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.uint32)


def categorical_theta_op_signature(
    *,
    flat_weights: torch.Tensor,
    flat_index: torch.Tensor | None,
    interaction_scale: torch.Tensor,
    n_nodes: int,
    n_interactions: int,
    n_categories: int,
) -> TTMLIRCategoricalThetaOpSignature:
    return TTMLIRCategoricalThetaOpSignature(
        flat_weights_shape=tuple(int(size) for size in flat_weights.shape),
        flat_weights_dtype=str(flat_weights.dtype),
        flat_index_shape=None
        if flat_index is None
        else tuple(int(size) for size in flat_index.shape),
        flat_index_dtype=None if flat_index is None else str(flat_index.dtype),
        interaction_scale_shape=tuple(int(size) for size in interaction_scale.shape),
        interaction_scale_dtype=str(interaction_scale.dtype),
        n_nodes=int(n_nodes),
        n_interactions=int(n_interactions),
        n_categories=int(n_categories),
    )


def _make_categorical_theta_kernel(batch: PackedCategoricalThetaBatch):
    tail_shape = tuple(int(size) for size in batch.tail_shape)
    tail_flat_size = math.prod(tail_shape) if tail_shape else 1
    tail_index_strides = tail_strides(tail_shape)

    def theta_kernel(weights_in, active_mask_in, *conditions):
        spin_inputs = conditions[: batch.n_spin]
        categorical_inputs = conditions[batch.n_spin :]
        selected = weights_in

        if tail_shape:
            if len(categorical_inputs) != len(tail_shape):
                raise ValueError(
                    "Categorical condition count must match the trailing "
                    "weight dimensions."
                )
            flat_index = jnp.zeros_like(categorical_inputs[0], dtype=jnp.uint32)
            for condition, stride in zip(categorical_inputs, tail_index_strides):
                flat_index = flat_index + condition.astype(jnp.uint32) * jnp.asarray(
                    int(stride), dtype=jnp.uint32
                )
            flat_weights = weights_in.reshape(
                *active_mask_in.shape,
                batch.n_categories,
                tail_flat_size,
            )
            selector = jax.nn.one_hot(
                flat_index.astype(jnp.int32),
                tail_flat_size,
                dtype=flat_weights.dtype,
            )
            selected = jnp.sum(
                flat_weights * selector[..., None, :],
                axis=-1,
            )

        interaction_scale = active_mask_in.astype(selected.dtype)
        for condition in spin_inputs:
            interaction_scale = interaction_scale * jnp.where(
                condition > 0,
                jnp.ones_like(interaction_scale),
                -jnp.ones_like(interaction_scale),
            )

        theta = jnp.sum(selected * interaction_scale[..., None], axis=2)
        return theta.reshape(1, 1, active_mask_in.shape[1], batch.n_categories)

    return theta_kernel


def _make_categorical_theta_inputs_kernel(
    *,
    flat_weights_shape: Sequence[int],
    has_flat_index: bool,
    n_nodes: int,
    n_interactions: int,
    n_categories: int,
):
    flat_tail_size = int(flat_weights_shape[-1]) if flat_weights_shape else 1
    flat_weight_rows = int(flat_weights_shape[1]) if len(flat_weights_shape) > 1 else 1

    def theta_kernel(flat_weights_in, *args):
        if has_flat_index:
            flat_index_in, interaction_scale_in = args
            selector = jax.nn.one_hot(
                jnp.reshape(
                    jnp.squeeze(flat_index_in, axis=-1).astype(jnp.int32),
                    (1, flat_weight_rows),
                ),
                flat_tail_size,
                dtype=flat_weights_in.dtype,
            )
            selector = jnp.reshape(
                selector,
                (1, flat_weight_rows, 1, flat_tail_size),
            )
            selected = jnp.sum(
                flat_weights_in * selector,
                axis=-1,
                keepdims=True,
            )
        else:
            (interaction_scale_in,) = args
            selected = flat_weights_in

        scaled = selected * interaction_scale_in
        return jnp.reshape(
            jnp.sum(
                jnp.reshape(
                    scaled,
                    (1, n_nodes, n_interactions, n_categories),
                ),
                axis=2,
                keepdims=False,
            ),
            (1, 1, n_nodes, n_categories),
        )

    return theta_kernel


def packed_categorical_theta_batch_inputs(
    batch: PackedCategoricalThetaBatch,
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


def compute_packed_categorical_theta_jax(
    batch: PackedCategoricalThetaBatch,
) -> torch.Tensor:
    kernel = _make_categorical_theta_kernel(batch)
    inputs = [
        _to_jax_float(batch.weights),
        _to_jax_float(batch.active_mask),
        *[_to_jax_int32(condition) for condition in batch.spin_conditions],
        *[_to_jax_uint32(condition) for condition in batch.categorical_conditions],
    ]
    result = jax.jit(kernel).lower(*inputs).compile()(*inputs)
    return numpy_to_torch(
        result,
        dtype=np.float32,
        shape=(batch.weights.shape[1], batch.n_categories),
    )


def lower_packed_categorical_theta_to_stablehlo(
    batch: PackedCategoricalThetaBatch,
) -> str:
    kernel = _make_categorical_theta_kernel(batch)
    inputs = [
        _to_jax_float(batch.weights),
        _to_jax_float(batch.active_mask),
        *[_to_jax_int32(condition) for condition in batch.spin_conditions],
        *[_to_jax_uint32(condition) for condition in batch.categorical_conditions],
    ]
    lowered = jax.jit(kernel).lower(*inputs)
    return stablehlo_text(lowered)


def lower_categorical_theta_inputs_to_stablehlo(
    *,
    flat_weights: torch.Tensor,
    flat_index: torch.Tensor | None,
    interaction_scale: torch.Tensor,
    n_nodes: int,
    n_interactions: int,
    n_categories: int,
) -> str:
    kernel = _make_categorical_theta_inputs_kernel(
        flat_weights_shape=tuple(flat_weights.shape),
        has_flat_index=flat_index is not None,
        n_nodes=n_nodes,
        n_interactions=n_interactions,
        n_categories=n_categories,
    )
    lowered_inputs = [_to_jax_float(flat_weights)]
    if flat_index is not None:
        lowered_inputs.append(_to_jax_uint32(flat_index))
    lowered_inputs.append(_to_jax_float(interaction_scale))
    lowered = jax.jit(kernel).lower(*lowered_inputs)
    return stablehlo_text(lowered)


def make_ttmlir_categorical_theta_op(
    *,
    config: TTMLIRConfig | None = None,
    system_desc_path: Path | str | None = None,
    artifact_root: Path | str | None = None,
    build_dir: Path | str | None = None,
    ttmlir_opt: Path | str | None = None,
    ttmlir_translate: Path | str | None = None,
    base_name_prefix: str = "categorical_theta_op",
) -> TTMLIRCategoricalThetaOp:
    return TTMLIRCategoricalThetaOp(
        config=config,
        system_desc_path=system_desc_path,
        artifact_root=artifact_root,
        build_dir=build_dir,
        ttmlir_opt=ttmlir_opt,
        ttmlir_translate=ttmlir_translate,
        base_name_prefix=base_name_prefix,
    )
