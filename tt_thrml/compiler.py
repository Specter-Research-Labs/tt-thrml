"""Fused kernel compilation for THRML programs.

Each block compiles to a single flatbuffer kernel with signature
    (global_state, rng_slice) -> new_global_state
All interaction weights, masks, gather indices, and the block-local write-back
offset are baked into the StableHLO as constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import subprocess
import threading
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from thrml.block_sampling import BlockSamplingProgram

from .core import (
    Family,
    TTMLIRConfig,
    FusedInteractionSpec,
    FusedBlockSpec,
    CompiledFusedBlock,
    CompiledProgram,
)


@dataclass(frozen=True)
class _CompileContext:
    ttnn: object
    device: object
    config: TTMLIRConfig
    state_layout: object
    state_dtype: object
    index_dtype: object


_ARTIFACT_CACHE: dict[str, Path] = {}
_ARTIFACT_CACHE_LOCK = threading.Lock()


def _infer_family(block, gibbs_spec) -> Family:
    node_type = type(block.nodes[0]).__name__.lower()
    if "spin" in node_type or "bool" in node_type:
        return Family.SPIN
    if "categorical" in node_type or "discrete" in node_type:
        return Family.CATEGORICAL
    if "gaussian" in node_type or "continuous" in node_type:
        return Family.GAUSSIAN
    sd = gibbs_spec.node_shape_dtypes.get(type(block.nodes[0]))
    if sd is not None:
        dtype_kind = np.dtype(sd[0][0].dtype).kind
        if dtype_kind == "b":
            return Family.SPIN
        if dtype_kind in ("i", "u"):
            return Family.CATEGORICAL
        if dtype_kind == "f":
            return Family.GAUSSIAN
    return Family.SPIN


def _get_n_categories(block, family: Family) -> int | None:
    if family != Family.CATEGORICAL:
        return None
    n_categories = getattr(block.nodes[0], "n_categories", None)
    if n_categories is not None:
        return int(n_categories)
    return 2


def _lower_interaction(interaction, family: Family) -> dict:
    hook = getattr(interaction, "tt_interaction_contribution", None)
    if callable(hook):
        contrib = hook(parameter_family=family.value)
        return {
            "weights": np.asarray(contrib.weights),
            "n_spin": int(contrib.n_spin),
            "contribution_kind": str(getattr(contrib, "contribution_kind", "default")),
        }
    if hasattr(interaction, "n_spin") and hasattr(interaction, "weights"):
        return {
            "weights": np.asarray(interaction.weights),
            "n_spin": int(interaction.n_spin),
            "contribution_kind": "default",
        }
    if family == Family.GAUSSIAN:
        if hasattr(interaction, "weights"):
            return {
                "weights": np.asarray(interaction.weights),
                "n_spin": 0,
                "contribution_kind": "linear",
            }
        if hasattr(interaction, "inverse_weights"):
            return {
                "weights": np.reciprocal(np.asarray(interaction.inverse_weights)),
                "n_spin": 0,
                "contribution_kind": "precision",
            }
    raise TypeError(f"Cannot lower interaction of type {type(interaction)}")


def _build_block_global_starts(gibbs_spec) -> tuple[tuple[int, ...], int]:
    starts = []
    offset = 0
    for block in gibbs_spec.blocks:
        starts.append(offset)
        offset += len(block.nodes)
    return tuple(starts), offset


def _build_interaction_spec(
    interaction,
    active_mask,
    global_inds,
    global_slices,
    family: Family,
    n_nodes: int,
) -> FusedInteractionSpec:
    lowered = _lower_interaction(interaction, family)
    weights = np.asarray(lowered["weights"], dtype=np.float32)
    mask = np.asarray(active_mask, dtype=np.float32)
    if weights.ndim == 1:
        weights = weights.reshape(n_nodes, -1)
    if mask.ndim == 1:
        mask = mask.reshape(n_nodes, -1)

    # Categorical weights: (n_nodes, n_terms, n_categories[, n_cat_source, ...])
    # Spin/gaussian bias: (n_nodes, n_terms)
    if weights.ndim >= 3 and weights.shape[:2] == mask.shape:
        n_terms = int(mask.shape[1])
        # Broadcast mask over all trailing category/source dims
        mask_expanded = mask.reshape(mask.shape + (1,) * (weights.ndim - 2))
        weighted_mask = (weights * mask_expanded).astype(np.float32)
        if weights.ndim == 3:
            # Bias: flatten (n_nodes, n_terms, n_categories) -> (n_nodes, n_terms*n_categories)
            weighted_mask = weighted_mask.reshape(n_nodes, -1)
        # For ndim >= 4 (Potts): keep shape; kernel handles dynamic source indexing
    elif weights.shape == mask.shape:
        weighted_mask = (weights * mask).astype(np.float32)
        n_terms = int(weighted_mask.shape[-1])
    else:
        raise ValueError(
            f"weights shape {weights.shape} does not match mask shape {mask.shape}"
        )

    n_spin = int(lowered["n_spin"])
    # Trailing dims beyond (n_nodes, n_terms, n_categories) each require one categorical source
    n_categorical_sources = max(0, weights.ndim - 3)
    expected_slices = n_spin + n_categorical_sources

    gather_arrays: list[np.ndarray] = []
    for gslice in global_slices:
        arr = np.asarray(gslice, dtype=np.int32)
        if arr.shape != (n_nodes, n_terms):
            raise ValueError(
                f"gather slice shape {arr.shape} does not match ({n_nodes}, {n_terms})"
            )
        gather_arrays.append(arr)

    # Extra slices beyond n_spin + n_categorical_sources are continuous source conditioning
    # (e.g. gaussian coupling where n_spin=0 but there are source node gathers)
    if len(gather_arrays) < n_spin + n_categorical_sources:
        raise ValueError(
            f"too few gather slices: need at least {n_spin + n_categorical_sources} "
            f"(n_spin={n_spin} + n_categorical_sources={n_categorical_sources}), "
            f"got {len(gather_arrays)}"
        )

    return FusedInteractionSpec(
        weighted_mask=weighted_mask,
        gather_indices=tuple(gather_arrays),
        n_spin=n_spin,
        n_terms=n_terms,
        contribution_kind=lowered["contribution_kind"],
    )


def _build_fused_block_spec(
    program: BlockSamplingProgram,
    block_index: int,
    block,
    family: Family,
    gibbs_spec,
    block_global_start: int,
    total_nodes: int,
) -> FusedBlockSpec:
    n_nodes = len(block.nodes)
    n_categories = _get_n_categories(block, family)

    interactions = []
    for interaction, active_mask, global_inds, global_slices in zip(
        program.per_block_interactions[block_index],
        program.per_block_interaction_active[block_index],
        program.per_block_interaction_global_inds[block_index],
        program.per_block_interaction_global_slices[block_index],
    ):
        interactions.append(
            _build_interaction_spec(
                interaction, active_mask, global_inds, global_slices, family, n_nodes
            )
        )

    # CategoricalNode doesn't store n_categories; derive from weighted_mask shape
    if family == Family.CATEGORICAL and interactions:
        for ispec in interactions:
            wm = ispec.weighted_mask
            if wm.ndim >= 3:
                # Potts or other higher-dim: axis 2 is always n_categories
                n_categories = int(wm.shape[2])
                break
            elif ispec.n_terms > 0:
                # Flat bias: (n_nodes, n_terms * n_categories)
                n_categories = wm.shape[-1] // ispec.n_terms

    return FusedBlockSpec(
        block_index=block_index,
        family=family,
        n_nodes=n_nodes,
        n_categories=n_categories,
        block_global_start=block_global_start,
        total_nodes=total_nodes,
        interactions=tuple(interactions),
    )


def _gather_source_scale(global_state, interaction: FusedInteractionSpec):
    """Elementwise product of all non-categorical source states.

    Handles spin (±1.0) and continuous gaussian sources.
    Categorical source gathers (trailing gather_indices) are handled separately.
    """
    n_cat_gathers = max(0, interaction.weighted_mask.ndim - 3)
    n_source_gathers = len(interaction.gather_indices) - n_cat_gathers
    if n_source_gathers == 0:
        return None
    gathered = global_state[jnp.asarray(interaction.gather_indices[0])]
    for extra in interaction.gather_indices[1:n_source_gathers]:
        gathered = gathered * global_state[jnp.asarray(extra)]
    return gathered


def _accumulate_gamma(global_state, interactions) -> jnp.ndarray:
    """Sum weighted_mask * source_scale across interactions for spin/gaussian-linear."""
    gamma = None
    for interaction in interactions:
        wm = jnp.asarray(interaction.weighted_mask)
        source_scale = _gather_source_scale(global_state, interaction)
        if source_scale is None:
            contrib = jnp.sum(wm, axis=-1, keepdims=True)
        else:
            contrib = jnp.sum(wm * source_scale, axis=-1, keepdims=True)
        gamma = contrib if gamma is None else gamma + contrib
    if gamma is None:
        raise ValueError("Block has no interactions; cannot build kernel")
    return gamma


def _make_fused_spin_kernel(spec: FusedBlockSpec) -> Callable:
    block_start = spec.block_global_start
    interactions = spec.interactions

    def kernel(global_state, rng_slice):
        gamma = _accumulate_gamma(global_state, interactions)
        # THRML BernoulliConditional uses sigmoid(2*gamma): p(+1) = sigmoid(2*gamma)
        # logistic noise comparison: gamma > logistic/2 ↔ 2*gamma > logistic
        new_block_state = jnp.where(2.0 * gamma > rng_slice, 1.0, -1.0).reshape(spec.n_nodes)
        return jax.lax.dynamic_update_slice(
            global_state, new_block_state, (block_start,)
        )

    return kernel


def _make_fused_categorical_kernel(spec: FusedBlockSpec) -> Callable:
    n_categories = spec.n_categories or 2
    n_nodes = spec.n_nodes
    block_start = spec.block_global_start
    interactions = spec.interactions

    def kernel(global_state, rng_slice):
        theta = None
        for interaction in interactions:
            wm = jnp.asarray(interaction.weighted_mask)
            n_cat_sources = len(interaction.gather_indices) - interaction.n_spin
            if n_cat_sources > 0:
                # Potts: wm is (n_nodes, n_terms, n_categories, n_cat_source)
                # Gather source categorical state and index into last weight axis
                cat_gathers = interaction.gather_indices[interaction.n_spin:]
                if len(cat_gathers) != 1:
                    raise NotImplementedError(
                        f"Only single categorical source supported, got {len(cat_gathers)}"
                    )
                # Categorical states are exact integer-valued floats (0.0, 1.0, 2.0, ...)
                # Use one-hot selection to avoid gather ops (unsupported index dtypes)
                cat_src_float = global_state[jnp.asarray(cat_gathers[0])]
                # wm: (n_nodes, n_terms, n_categories, n_cat_source)
                n_cat_source = wm.shape[-1]
                arange_vals = jnp.arange(n_cat_source, dtype=jnp.float32)
                # one_hot: (n_nodes, n_terms, n_cat_source)
                one_hot = (cat_src_float[:, :, None] == arange_vals).astype(jnp.float32)
                # selected: (n_nodes, n_terms, n_categories)
                selected = jnp.sum(wm * one_hot[:, :, None, :], axis=-1)
                if interaction.n_spin > 0:
                    spin_scale = _gather_source_scale(global_state, interaction)
                    selected = selected * spin_scale[:, :, None]
                contrib = jnp.sum(selected, axis=1)
            else:
                # Bias or spin-conditional: wm is (n_nodes, n_terms * n_categories)
                wm_reshaped = jnp.reshape(wm, (n_nodes, interaction.n_terms, n_categories))
                source_scale = _gather_source_scale(global_state, interaction)
                if source_scale is None:
                    contrib = jnp.sum(wm_reshaped, axis=1)
                else:
                    contrib = jnp.sum(wm_reshaped * source_scale[..., None], axis=1)
            theta = contrib if theta is None else theta + contrib

        perturbed = theta + rng_slice
        new_block_state = jnp.argmax(perturbed, axis=-1).astype(jnp.float32)
        return jax.lax.dynamic_update_slice(
            global_state, new_block_state, (block_start,)
        )

    return kernel


def _make_fused_gaussian_kernel(spec: FusedBlockSpec) -> Callable:
    block_start = spec.block_global_start
    interactions = spec.interactions
    linear_interactions = tuple(i for i in interactions if i.contribution_kind != "precision")
    precision_interactions = tuple(i for i in interactions if i.contribution_kind == "precision")

    def kernel(global_state, rng_slice):
        linear = _accumulate_gamma(global_state, linear_interactions) if linear_interactions else jnp.zeros((spec.n_nodes, 1), dtype=jnp.float32)
        if precision_interactions:
            precision = _accumulate_gamma(global_state, precision_interactions)
        else:
            precision = jnp.ones((spec.n_nodes, 1), dtype=jnp.float32)

        inv_precision = 1.0 / (precision + 1e-6)
        # mean/std have shape (n_nodes, 1); rng_slice is (n_nodes,) — reshape to avoid broadcast to (n_nodes, n_nodes)
        mean_vec = (linear * inv_precision).reshape(spec.n_nodes)
        std_vec = jnp.sqrt(inv_precision).reshape(spec.n_nodes)
        new_block_state = mean_vec + std_vec * rng_slice
        return jax.lax.dynamic_update_slice(
            global_state, new_block_state, (block_start,)
        )

    return kernel


def _make_fused_kernel(spec: FusedBlockSpec) -> Callable:
    if spec.family == Family.SPIN:
        return _make_fused_spin_kernel(spec)
    if spec.family == Family.CATEGORICAL:
        return _make_fused_categorical_kernel(spec)
    if spec.family == Family.GAUSSIAN:
        return _make_fused_gaussian_kernel(spec)
    raise ValueError(f"Unknown family: {spec.family}")


def _rng_slice_shape(spec: FusedBlockSpec) -> tuple[int, ...]:
    if spec.family == Family.SPIN:
        return (spec.n_nodes, 1)
    if spec.family == Family.CATEGORICAL:
        return (spec.n_nodes, spec.n_categories or 2)
    if spec.family == Family.GAUSSIAN:
        return (spec.n_nodes,)
    raise ValueError(f"Unknown family: {spec.family}")


def _kernel_signature_key(spec: FusedBlockSpec) -> str:
    """Cache key covering all baked-in constants; any change re-compiles."""
    interactions_payload = []
    for i in spec.interactions:
        interactions_payload.append({
            "weighted_mask": hashlib.sha256(np.ascontiguousarray(i.weighted_mask).tobytes()).hexdigest()[:16],
            "gather_indices": [
                hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()[:16]
                for g in i.gather_indices
            ],
            "n_spin": i.n_spin,
            "n_terms": i.n_terms,
            "contribution_kind": i.contribution_kind,
        })

    payload = {
        "family": spec.family.value,
        "n_nodes": spec.n_nodes,
        "n_categories": spec.n_categories,
        "total_nodes": spec.total_nodes,
        "block_global_start": spec.block_global_start,
        "interactions": interactions_payload,
        "kernel_version": 6,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _compile_fused_kernel(
    config: TTMLIRConfig,
    spec: FusedBlockSpec,
    kernel: Callable,
) -> Path:
    sig_key = _kernel_signature_key(spec)
    cache_key = f"{spec.family.value}_{config.cache_key()}_{sig_key}"

    with _ARTIFACT_CACHE_LOCK:
        if cache_key in _ARTIFACT_CACHE:
            return _ARTIFACT_CACHE[cache_key]

    rng_shape = _rng_slice_shape(spec)
    inputs = [
        jax.ShapeDtypeStruct((spec.total_nodes,), jnp.float32),
        jax.ShapeDtypeStruct(rng_shape, jnp.float32),
    ]

    lowered = jax.jit(kernel).lower(*inputs)
    stablehlo_text = lowered.as_text(dialect="stablehlo")

    artifact_dir = config.artifact_root / "fused" / spec.family.value / sig_key
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stablehlo_path = artifact_dir / "kernel.stablehlo.mlir"
    ttir_path = artifact_dir / "kernel.ttir.mlir"
    ttnn_path = artifact_dir / "kernel.ttnn.mlir"
    flatbuffer_path = artifact_dir / "kernel.ttnn"

    stablehlo_path.write_text(stablehlo_text)

    subprocess.run(
        [
            config.ttmlir_opt,
            "--stablehlo-to-ttir-pipeline",
            str(stablehlo_path),
            "-o", str(ttir_path),
        ],
        check=True,
        text=True,
    )

    subprocess.run(
        [
            config.ttmlir_opt,
            f"--ttir-to-ttnn-backend-pipeline=enable-cpu-hoisted-const-eval=false system-desc-path={config.system_desc_path}",
            str(ttir_path),
            "-o", str(ttnn_path),
        ],
        check=True,
        text=True,
    )

    subprocess.run(
        [
            config.ttmlir_translate,
            "--ttnn-to-flatbuffer",
            str(ttnn_path),
            "-o", str(flatbuffer_path),
        ],
        check=True,
        text=True,
    )

    with _ARTIFACT_CACHE_LOCK:
        _ARTIFACT_CACHE[cache_key] = flatbuffer_path

    return flatbuffer_path


def _compile_block(
    program: BlockSamplingProgram,
    block_index: int,
    block,
    gibbs_spec,
    block_global_start: int,
    total_nodes: int,
    context: _CompileContext,
) -> CompiledFusedBlock:
    family = _infer_family(block, gibbs_spec)
    spec = _build_fused_block_spec(
        program, block_index, block, family, gibbs_spec,
        block_global_start, total_nodes,
    )
    kernel = _make_fused_kernel(spec)
    artifact_path = _compile_fused_kernel(context.config, spec, kernel)

    return CompiledFusedBlock(spec=spec, kernel_artifact=artifact_path)


def compile_program(
    ttnn,
    device,
    program: BlockSamplingProgram,
    config: TTMLIRConfig,
    *,
    n_sweeps: int = 100,
) -> CompiledProgram:
    state_layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None) or getattr(ttnn, "TILE_LAYOUT", None)
    state_dtype = getattr(ttnn, "float32")
    index_dtype = getattr(ttnn, "uint32", None) or getattr(ttnn, "int32")

    context = _CompileContext(
        ttnn=ttnn,
        device=device,
        config=config,
        state_layout=state_layout,
        state_dtype=state_dtype,
        index_dtype=index_dtype,
    )

    gibbs_spec = program.gibbs_spec
    block_global_starts, total_nodes = _build_block_global_starts(gibbs_spec)

    n_free = len(gibbs_spec.free_blocks)
    compiled_blocks = []
    for block_index, block in enumerate(gibbs_spec.blocks):
        if block_index < n_free:
            compiled_blocks.append(
                _compile_block(
                    program, block_index, block, gibbs_spec,
                    block_global_starts[block_index], total_nodes, context,
                )
            )
        else:
            # Clamped block: no kernel (state is fixed, never resampled)
            family = _infer_family(block, gibbs_spec)
            placeholder_spec = FusedBlockSpec(
                block_index=block_index,
                family=family,
                n_nodes=len(block.nodes),
                n_categories=None,
                block_global_start=block_global_starts[block_index],
                total_nodes=total_nodes,
                interactions=(),
            )
            compiled_blocks.append(CompiledFusedBlock(spec=placeholder_spec, kernel_artifact=None))

    sampling_order = tuple(tuple(int(i) for i in group) for group in gibbs_spec.sampling_order)

    from .rng import make_rng_spec
    rng_spec = make_rng_spec(tuple(compiled_blocks), n_sweeps)

    return CompiledProgram(
        blocks=tuple(compiled_blocks),
        n_free_blocks=len(gibbs_spec.free_blocks),
        total_nodes=total_nodes,
        block_global_starts=block_global_starts,
        sampling_order=sampling_order,
        state_dtype=state_dtype,
        index_dtype=index_dtype,
        layout=state_layout,
        rng_spec=rng_spec,
    )
