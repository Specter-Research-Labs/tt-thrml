"""Fused kernel compilation for THRML programs.

Each THRML sampling group compiles to a single flatbuffer kernel with signature
    (global_state, *rng_slices) -> new_global_state
All interaction weights, masks, gather indices, and the block-local write-back
offset are baked into the StableHLO as constants.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from thrml.block_sampling import BlockSamplingProgram

from .core import (
    CompiledFusedBlock,
    CompiledProgram,
    CompiledSamplingGroup,
    Family,
    FusedBlockSpec,
    FusedInteractionSpec,
    TTMLIRConfig,
)
from .rng import make_rng_spec

_ARTIFACT_CACHE: dict[str, Path] = {}
_ARTIFACT_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _GlobalStateLayout:
    block_starts: tuple[int, ...]
    global_to_flat: tuple[tuple[int, ...], ...]
    total_nodes: int

    def flatten_slices(self, global_inds, global_slices) -> tuple[np.ndarray, ...]:
        flat_slices = []
        for global_ind, global_slice in zip(global_inds, global_slices, strict=True):
            index_map = np.asarray(self.global_to_flat[int(global_ind)], dtype=np.int32)
            flat_slices.append(index_map[np.asarray(global_slice, dtype=np.int32)])
        return tuple(flat_slices)


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
            "weights": np.asarray(getattr(contrib, "weights")),
            "n_spin": int(getattr(contrib, "n_spin")),
            "n_categorical": getattr(contrib, "n_categorical", None),
            "contribution_kind": str(getattr(contrib, "contribution_kind", "default")),
        }
    if hasattr(interaction, "n_spin") and hasattr(interaction, "weights"):
        return {
            "weights": np.asarray(getattr(interaction, "weights")),
            "n_spin": int(getattr(interaction, "n_spin")),
            "n_categorical": None,
            "contribution_kind": "default",
        }
    if family == Family.GAUSSIAN:
        if hasattr(interaction, "weights"):
            return {
                "weights": np.asarray(interaction.weights),
                "n_spin": 0,
                "n_categorical": 0,
                "contribution_kind": "linear",
            }
        if hasattr(interaction, "inverse_weights"):
            return {
                "weights": np.reciprocal(np.asarray(interaction.inverse_weights)),
                "n_spin": 0,
                "n_categorical": 0,
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


def _build_global_state_layout(gibbs_spec) -> _GlobalStateLayout:
    block_starts, total_nodes = _build_block_global_starts(gibbs_spec)
    max_global_ind = max(
        (location[0] for location in gibbs_spec.node_global_location_map.values()),
        default=-1,
    )
    global_to_flat: list[list[int | None]] = [[] for _ in range(max_global_ind + 1)]

    for block_index, block in enumerate(gibbs_spec.blocks):
        block_start = block_starts[block_index]
        for node_offset, node in enumerate(block.nodes):
            global_ind, global_offset = gibbs_spec.node_global_location_map[node]
            mapping = global_to_flat[global_ind]
            if len(mapping) <= global_offset:
                mapping.extend([None] * (global_offset + 1 - len(mapping)))
            mapping[global_offset] = block_start + node_offset

    compact_maps = []
    for mapping in global_to_flat:
        if any(value is None for value in mapping):
            raise ValueError("THRML global state map is not dense.")
        compact_maps.append(tuple(value for value in mapping if value is not None))

    return _GlobalStateLayout(
        block_starts=block_starts,
        global_to_flat=tuple(compact_maps),
        total_nodes=total_nodes,
    )


def _build_interaction_spec(
    interaction,
    active_mask,
    global_inds,
    global_slices,
    family: Family,
    n_nodes: int,
    layout: _GlobalStateLayout,
) -> FusedInteractionSpec:
    lowered = _lower_interaction(interaction, family)
    weights = np.asarray(lowered["weights"], dtype=np.float32)
    mask = np.asarray(active_mask, dtype=np.float32)
    if weights.ndim == 1:
        weights = weights.reshape(n_nodes, -1)
    if mask.ndim == 1:
        mask = mask.reshape(n_nodes, -1)

    if weights.shape[:2] == mask.shape:
        n_terms = int(mask.shape[1])
        mask_expanded = mask.reshape(mask.shape + (1,) * (weights.ndim - 2))
        weighted_mask = (weights * mask_expanded).astype(np.float32)
    elif weights.shape == mask.shape:
        weighted_mask = (weights * mask).astype(np.float32)
        n_terms = int(weighted_mask.shape[-1])
    else:
        raise ValueError(f"weights shape {weights.shape} does not match mask shape {mask.shape}")

    n_spin = int(lowered["n_spin"])
    explicit_n_categorical = lowered.get("n_categorical")
    if explicit_n_categorical is not None:
        n_categorical_sources = int(explicit_n_categorical)
    elif family == Family.CATEGORICAL:
        n_categorical_sources = max(0, weights.ndim - 3)
    elif lowered["contribution_kind"] == "default":
        n_categorical_sources = max(0, weights.ndim - 2)
    else:
        n_categorical_sources = 0

    gather_arrays: list[np.ndarray] = []
    flat_slices = layout.flatten_slices(global_inds, global_slices)
    for gslice in flat_slices:
        arr = np.asarray(gslice, dtype=np.int32)
        if arr.shape != (n_nodes, n_terms):
            raise ValueError(f"gather slice shape {arr.shape} does not match ({n_nodes}, {n_terms})")
        gather_arrays.append(arr)

    min_sources = n_spin + n_categorical_sources
    if len(gather_arrays) < min_sources:
        raise ValueError(
            f"too few gather slices: need at least {min_sources} "
            f"(n_spin={n_spin} + n_categorical_sources={n_categorical_sources}), "
            f"got {len(gather_arrays)}"
        )

    return FusedInteractionSpec(
        weighted_mask=weighted_mask,
        gather_indices=tuple(gather_arrays),
        n_spin=n_spin,
        n_categorical=n_categorical_sources,
        n_terms=n_terms,
        contribution_kind=lowered["contribution_kind"],
    )


def _build_fused_block_spec(
    program: BlockSamplingProgram,
    block_index: int,
    block,
    family: Family,
    block_global_start: int,
    total_nodes: int,
    layout: _GlobalStateLayout,
) -> FusedBlockSpec:
    n_nodes = len(block.nodes)
    n_categories = _get_n_categories(block, family)

    interactions = []
    for interaction, active_mask, global_inds, global_slices in zip(
        program.per_block_interactions[block_index],
        program.per_block_interaction_active[block_index],
        program.per_block_interaction_global_inds[block_index],
        program.per_block_interaction_global_slices[block_index],
        strict=True,
    ):
        interactions.append(
            _build_interaction_spec(
                interaction,
                active_mask,
                global_inds,
                global_slices,
                family,
                n_nodes,
                layout,
            )
        )

    # CategoricalNode doesn't store n_categories; derive from weighted_mask shape
    if family == Family.CATEGORICAL and interactions:
        for ispec in interactions:
            wm = ispec.weighted_mask
            if wm.ndim >= 3:
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
    n_source_gathers = len(interaction.gather_indices) - interaction.n_categorical
    if n_source_gathers == 0:
        return None
    gathered = global_state[jnp.asarray(interaction.gather_indices[0])]
    for extra in interaction.gather_indices[1:n_source_gathers]:
        gathered = gathered * global_state[jnp.asarray(extra)]
    return gathered


def _select_categorical_sources(weights, global_state, interaction: FusedInteractionSpec, *, source_axis: int):
    selected = weights
    cat_gathers = interaction.gather_indices[len(interaction.gather_indices) - interaction.n_categorical :]
    for gather_indices in cat_gathers:
        cat_src_float = global_state[jnp.asarray(gather_indices)]
        n_categories = selected.shape[source_axis]
        one_hot = (cat_src_float[:, :, None] == jnp.arange(n_categories, dtype=jnp.float32)).astype(jnp.float32)
        prefix = (1,) * (source_axis - 2)
        suffix = (1,) * (selected.ndim - source_axis - 1)
        selector = one_hot.reshape(one_hot.shape[:2] + prefix + (n_categories,) + suffix)
        selected = jnp.sum(selected * selector, axis=source_axis)
    return selected


def _accumulate_gamma(global_state, interactions) -> jnp.ndarray:
    """Sum weighted_mask * source_scale across interactions for spin/gaussian-linear."""
    gamma = None
    for interaction in interactions:
        wm = jnp.asarray(interaction.weighted_mask)
        if interaction.n_categorical:
            wm = _select_categorical_sources(wm, global_state, interaction, source_axis=2)
        source_scale = _gather_source_scale(global_state, interaction)
        if source_scale is None:
            contrib = jnp.sum(wm, axis=-1, keepdims=True)
        else:
            contrib = jnp.sum(wm * source_scale, axis=-1, keepdims=True)
        gamma = contrib if gamma is None else gamma + contrib
    if gamma is None:
        raise ValueError("Block has no interactions; cannot build kernel")
    return gamma


def _sample_spin_block(global_state, rng_slice, spec: FusedBlockSpec):
    gamma = _accumulate_gamma(global_state, spec.interactions)
    # THRML BernoulliConditional uses sigmoid(2*gamma): p(+1) = sigmoid(2*gamma).
    return jnp.where(2.0 * gamma > rng_slice, 1.0, -1.0).reshape(spec.n_nodes)


def _sample_categorical_block(global_state, rng_slice, spec: FusedBlockSpec):
    n_categories = spec.n_categories or 2
    n_nodes = spec.n_nodes
    theta = None
    for interaction in spec.interactions:
        wm = jnp.asarray(interaction.weighted_mask)
        if interaction.n_categorical:
            wm = _select_categorical_sources(wm, global_state, interaction, source_axis=3)
        wm = jnp.reshape(wm, (n_nodes, interaction.n_terms, n_categories))
        source_scale = _gather_source_scale(global_state, interaction)
        if source_scale is None:
            contrib = jnp.sum(wm, axis=1)
        else:
            contrib = jnp.sum(wm * source_scale[..., None], axis=1)
        theta = contrib if theta is None else theta + contrib

    perturbed = theta + rng_slice
    return jnp.argmax(perturbed, axis=-1).astype(jnp.float32)


def _sample_gaussian_block(global_state, rng_slice, spec: FusedBlockSpec):
    interactions = spec.interactions
    linear_interactions = tuple(i for i in interactions if i.contribution_kind != "precision")
    precision_interactions = tuple(i for i in interactions if i.contribution_kind == "precision")

    linear = (
        _accumulate_gamma(global_state, linear_interactions)
        if linear_interactions
        else jnp.zeros((spec.n_nodes, 1), dtype=jnp.float32)
    )
    if precision_interactions:
        precision = _accumulate_gamma(global_state, precision_interactions)
    else:
        precision = jnp.ones((spec.n_nodes, 1), dtype=jnp.float32)

    inv_precision = 1.0 / (precision + 1e-6)
    mean_vec = (linear * inv_precision).reshape(spec.n_nodes)
    std_vec = jnp.sqrt(inv_precision).reshape(spec.n_nodes)
    return mean_vec + std_vec * rng_slice


def _sample_block(global_state, rng_slice, spec: FusedBlockSpec):
    if spec.family == Family.SPIN:
        return _sample_spin_block(global_state, rng_slice, spec)
    if spec.family == Family.CATEGORICAL:
        return _sample_categorical_block(global_state, rng_slice, spec)
    if spec.family == Family.GAUSSIAN:
        return _sample_gaussian_block(global_state, rng_slice, spec)
    raise ValueError(f"Unknown family: {spec.family}")


def _make_fused_kernel(spec: FusedBlockSpec) -> Callable:
    block_start = spec.block_global_start

    def kernel(global_state, rng_slice):
        new_block_state = _sample_block(global_state, rng_slice, spec)
        return jax.lax.dynamic_update_slice(global_state, new_block_state, (block_start,))

    return kernel


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
        interactions_payload.append(
            {
                "weighted_mask": hashlib.sha256(np.ascontiguousarray(i.weighted_mask).tobytes()).hexdigest()[:16],
                "gather_indices": [
                    hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()[:16] for g in i.gather_indices
                ],
                "n_spin": i.n_spin,
                "n_categorical": i.n_categorical,
                "n_terms": i.n_terms,
                "contribution_kind": i.contribution_kind,
            }
        )

    payload = {
        "family": spec.family.value,
        "n_nodes": spec.n_nodes,
        "n_categories": spec.n_categories,
        "total_nodes": spec.total_nodes,
        "block_global_start": spec.block_global_start,
        "interactions": interactions_payload,
        "kernel_version": 7,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _group_signature_key(specs: tuple[FusedBlockSpec, ...]) -> str:
    payload = {
        "blocks": [_kernel_signature_key(spec) for spec in specs],
        "block_indices": [spec.block_index for spec in specs],
        "kernel_version": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _make_sampling_group_kernel(specs: tuple[FusedBlockSpec, ...]) -> Callable:
    def kernel(global_state, *rng_slices):
        block_states = [
            _sample_block(global_state, rng_slice, spec) for spec, rng_slice in zip(specs, rng_slices, strict=True)
        ]
        updated_state = global_state
        for spec, block_state in zip(specs, block_states, strict=True):
            updated_state = jax.lax.dynamic_update_slice(
                updated_state,
                block_state,
                (spec.block_global_start,),
            )
        return updated_state

    return kernel


def _compile_sampling_group(
    config: TTMLIRConfig,
    specs: tuple[FusedBlockSpec, ...],
) -> Path:
    sig_key = _group_signature_key(specs)
    group_label = "_".join(str(spec.block_index) for spec in specs)
    cache_key = f"group_{group_label}_{config.cache_key()}_{sig_key}"

    with _ARTIFACT_CACHE_LOCK:
        if cache_key in _ARTIFACT_CACHE:
            return _ARTIFACT_CACHE[cache_key]

    kernel = _make_sampling_group_kernel(specs)
    inputs = [jax.ShapeDtypeStruct((specs[0].total_nodes,), jnp.float32)]
    inputs.extend(jax.ShapeDtypeStruct(_rng_slice_shape(spec), jnp.float32) for spec in specs)

    lowered = jax.jit(kernel).lower(*inputs)
    stablehlo_text = lowered.as_text(dialect="stablehlo")

    artifact_dir = config.artifact_root / "fused" / "group" / group_label / sig_key
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
            "-o",
            str(ttir_path),
        ],
        check=True,
        text=True,
    )

    subprocess.run(
        [
            config.ttmlir_opt,
            f"--ttir-to-ttnn-backend-pipeline=enable-cpu-hoisted-const-eval=false system-desc-path={config.system_desc_path}",
            str(ttir_path),
            "-o",
            str(ttnn_path),
        ],
        check=True,
        text=True,
    )

    subprocess.run(
        [
            config.ttmlir_translate,
            "--ttnn-to-flatbuffer",
            str(ttnn_path),
            "-o",
            str(flatbuffer_path),
        ],
        check=True,
        text=True,
    )

    with _ARTIFACT_CACHE_LOCK:
        _ARTIFACT_CACHE[cache_key] = flatbuffer_path

    return flatbuffer_path


def compile_program(
    ttnn,
    device,
    program: BlockSamplingProgram,
    config: TTMLIRConfig,
    *,
    n_sweeps: int = 100,
) -> CompiledProgram:
    state_layout = ttnn.ROW_MAJOR_LAYOUT
    state_dtype = ttnn.float32
    index_dtype = ttnn.uint32

    gibbs_spec = program.gibbs_spec
    global_layout = _build_global_state_layout(gibbs_spec)
    block_global_starts = global_layout.block_starts
    total_nodes = global_layout.total_nodes

    n_free = len(gibbs_spec.free_blocks)
    compiled_blocks = []
    for block_index, block in enumerate(gibbs_spec.blocks):
        family = _infer_family(block, gibbs_spec)
        if block_index < n_free:
            spec = _build_fused_block_spec(
                program,
                block_index,
                block,
                family,
                block_global_starts[block_index],
                total_nodes,
                global_layout,
            )
            compiled_blocks.append(CompiledFusedBlock(spec=spec, kernel_artifact=None))
        else:
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
    sampling_groups = []
    for group in sampling_order:
        specs = tuple(compiled_blocks[block_index].spec for block_index in group)
        artifact_path = _compile_sampling_group(config, specs)
        sampling_groups.append(
            CompiledSamplingGroup(
                block_indices=group,
                kernel_artifact=artifact_path,
            )
        )
    rng_spec = make_rng_spec(tuple(compiled_blocks[:n_free]), n_sweeps)

    return CompiledProgram(
        blocks=tuple(compiled_blocks),
        sampling_groups=tuple(sampling_groups),
        n_free_blocks=len(gibbs_spec.free_blocks),
        total_nodes=total_nodes,
        block_global_starts=block_global_starts,
        sampling_order=sampling_order,
        state_dtype=state_dtype,
        index_dtype=index_dtype,
        layout=state_layout,
        rng_spec=rng_spec,
    )
