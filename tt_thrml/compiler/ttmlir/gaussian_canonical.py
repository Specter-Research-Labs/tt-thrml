from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import torch

from ..gaussian_ops import GaussianCanonicalInputs
from ...runtime_config import GAUSSIAN_PARAMETER_FAMILY
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
    default_float32_scale,
    execute_single_output_flatbuffer,
    optional_ttnn_input_as_torch,
    torch_to_jax,
    ttnn_input_as_torch,
)


@dataclass(frozen=True)
class TTMLIRGaussianCanonicalOpSignature:
    flat_weights_shape: tuple[int, ...]
    flat_weights_dtype: str
    flat_index_shape: tuple[int, ...] | None
    flat_index_dtype: str | None
    interaction_scale_shape: tuple[int, ...]
    interaction_scale_dtype: str
    n_nodes: int
    n_interactions: int
    contribution_kind: str

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
            "contribution_kind": self.contribution_kind,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]


@dataclass(frozen=True)
class TTMLIRGaussianCanonicalOpArtifact:
    signature: TTMLIRGaussianCanonicalOpSignature
    artifact_dir: Path
    stablehlo_path: Path
    ttir_path: Path
    ttnn_path: Path
    flatbuffer_path: Path
    stablehlo_to_ttir_command: tuple[str, ...]
    ttir_to_ttnn_command: tuple[str, ...]
    ttnn_to_flatbuffer_command: tuple[str, ...]


class TTMLIRGaussianCanonicalOp:
    def __init__(
        self,
        *,
        config: TTMLIRConfig | None = None,
        system_desc_path: Path | str | None = None,
        artifact_root: Path | str | None = None,
        build_dir: Path | str | None = None,
        ttmlir_opt: Path | str | None = None,
        ttmlir_translate: Path | str | None = None,
        base_name_prefix: str = "gaussian_canonical_op",
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
        signature: TTMLIRGaussianCanonicalOpSignature,
        *,
        flat_weights: torch.Tensor,
        flat_index: torch.Tensor | None,
        interaction_scale: torch.Tensor,
    ) -> TTMLIRGaussianCanonicalOpArtifact:
        compiled = get_or_compile_cached_artifact(
            self.config,
            family=GAUSSIAN_PARAMETER_FAMILY.value,
            signature=signature,
            base_name_prefix=self.base_name_prefix,
            stablehlo_module_text_factory=lambda: lower_gaussian_canonical_inputs_to_stablehlo(
                flat_weights=flat_weights,
                flat_index=flat_index,
                interaction_scale=interaction_scale,
                n_nodes=signature.n_nodes,
                n_interactions=signature.n_interactions,
                contribution_kind=signature.contribution_kind,
            ),
            compile_fn=compile_stablehlo_to_flatbuffer,
        )
        return TTMLIRGaussianCanonicalOpArtifact(
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
        inputs: GaussianCanonicalInputs,
    ):
        flat_weights = ttnn_input_as_torch(ttnn, inputs.flat_weights, dtype=torch.float32)
        flat_index = optional_ttnn_input_as_torch(
            ttnn,
            inputs.flat_index,
            dtype=torch.uint32,
        )
        if inputs.interaction_scale is None:
            interaction_scale = default_float32_scale(flat_weights, flat_index)
        else:
            interaction_scale = ttnn_input_as_torch(
                ttnn,
                inputs.interaction_scale,
                dtype=torch.float32,
            )

        signature = gaussian_canonical_op_signature(
            flat_weights=flat_weights,
            flat_index=flat_index,
            interaction_scale=interaction_scale,
            n_nodes=inputs.n_nodes,
            n_interactions=inputs.n_interactions,
            contribution_kind=inputs.contribution_kind,
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
        direct_runtime_inputs.append(
            inputs.interaction_scale
            if inputs.interaction_scale is not None
            else interaction_scale
        )
        return execute_single_output_flatbuffer(
            config=self.config,
            ttnn=ttnn,
            device=device,
            flatbuffer_path=artifact.flatbuffer_path,
            input_tensors=runtime_inputs,
            direct_input_tensors=direct_runtime_inputs,
            output_reference=inputs.flat_weights,
            op_name="gaussian-canonical op",
            run_flatbuffer_fn=run_flatbuffer,
            supports_direct_ttnn_inputs_fn=supports_direct_ttnn_inputs,
        )


def _to_jax_float(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.float32)


def _to_jax_int32(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.int32)


def _to_jax_uint32(value: torch.Tensor) -> jax.Array:
    return torch_to_jax(value, dtype=jnp.uint32)


def gaussian_canonical_op_signature(
    *,
    flat_weights: torch.Tensor,
    flat_index: torch.Tensor | None,
    interaction_scale: torch.Tensor,
    n_nodes: int,
    n_interactions: int,
    contribution_kind: str,
) -> TTMLIRGaussianCanonicalOpSignature:
    return TTMLIRGaussianCanonicalOpSignature(
        flat_weights_shape=tuple(int(dim) for dim in flat_weights.shape),
        flat_weights_dtype=str(flat_weights.dtype),
        flat_index_shape=None
        if flat_index is None
        else tuple(int(dim) for dim in flat_index.shape),
        flat_index_dtype=None if flat_index is None else str(flat_index.dtype),
        interaction_scale_shape=tuple(int(dim) for dim in interaction_scale.shape),
        interaction_scale_dtype=str(interaction_scale.dtype),
        n_nodes=int(n_nodes),
        n_interactions=int(n_interactions),
        contribution_kind=str(contribution_kind),
    )


def _make_gaussian_canonical_kernel(
    *,
    flat_index: torch.Tensor | None,
    flat_tail_size: int,
    n_nodes: int,
    n_interactions: int,
    contribution_kind: str,
):
    if contribution_kind not in {"linear", "precision"}:
        raise TypeError(
            "Gaussian canonical tt-mlir lowering only supports 'linear' and "
            f"'precision' contributions, got {contribution_kind!r}."
        )

    def gaussian_kernel(flat_weights_in, *dynamic_inputs):
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
                keepdims=True,
            )

        weighted = gathered * interaction_scale_in.astype(gathered.dtype)
        if weighted.ndim != 4:
            weighted = jnp.reshape(weighted, (weighted.shape[0], 1, n_nodes, n_interactions))
        partial = jnp.sum(weighted, axis=3, keepdims=True)
        zeros = jnp.zeros_like(partial)
        if contribution_kind == "linear":
            return jnp.concatenate([partial, zeros], axis=-1)
        return jnp.concatenate([zeros, partial], axis=-1)

    return gaussian_kernel


def lower_gaussian_canonical_inputs_to_stablehlo(
    *,
    flat_weights: torch.Tensor,
    flat_index: torch.Tensor | None,
    interaction_scale: torch.Tensor,
    n_nodes: int,
    n_interactions: int,
    contribution_kind: str,
) -> str:
    kernel = _make_gaussian_canonical_kernel(
        flat_index=flat_index,
        flat_tail_size=int(flat_weights.shape[-1]),
        n_nodes=n_nodes,
        n_interactions=n_interactions,
        contribution_kind=contribution_kind,
    )
    lowered = jax.jit(kernel).lower(
        _to_jax_float(flat_weights),
        *(
            [_to_jax_uint32(flat_index)]
            if flat_index is not None
            else []
        ),
        _to_jax_float(interaction_scale),
    )
    return stablehlo_text(lowered)


def make_ttmlir_gaussian_canonical_op(
    *,
    config: TTMLIRConfig | None = None,
    system_desc_path: Path | str | None = None,
    artifact_root: Path | str | None = None,
    build_dir: Path | str | None = None,
    ttmlir_opt: Path | str | None = None,
    ttmlir_translate: Path | str | None = None,
    base_name_prefix: str = "gaussian_canonical_op",
) -> TTMLIRGaussianCanonicalOp:
    return TTMLIRGaussianCanonicalOp(
        config=config,
        system_desc_path=system_desc_path,
        artifact_root=artifact_root,
        build_dir=build_dir,
        ttmlir_opt=ttmlir_opt,
        ttmlir_translate=ttmlir_translate,
        base_name_prefix=base_name_prefix,
    )
