from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
import torch
from jax import numpy as jnp
import jax

from ..compiler.categorical_ops import (
    CategoricalThetaInputs,
    categorical_gumbel_noise_tensor,
    categorical_flat_index_tensor_device,
    categorical_interaction_scale_tensor_device,
    categorical_tail_index_tensor_device,
)
from ..compiler.gaussian_ops import (
    GaussianCanonicalInputs,
    gaussian_noise_batch_tensor,
    gaussian_noise_tensor,
)
from ..compiler.spin_ops import SpinGammaInputs
from ..compiler.spin_ops import SPIN_PARAMETER_TO_GAMMA_SCALE
from ..runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    ParameterFamily,
    SPIN_PARAMETER_FAMILY,
)
from .compiled_program import (
    CompiledBlock,
    CompiledInteractionGroup,
    CompiledCategoricalFamilyRuntime,
    CompiledGaussianFamilyRuntime,
    CompiledInteraction,
    CompiledSpinFamilyRuntime,
)
from . import state_runtime as _state_runtime
from .runtime_utils import (
    spin_threshold_logits_batch_tensor,
    spin_threshold_logits_tensor,
)


@dataclass(frozen=True)
class SpinPreparedRandom:
    threshold_logits: object


@dataclass(frozen=True)
class CategoricalPreparedRandom:
    gumbel_noise: object


@dataclass(frozen=True)
class GaussianPreparedRandom:
    gaussian_noise: object


PreparedFamilyRandom = (
    SpinPreparedRandom | CategoricalPreparedRandom | GaussianPreparedRandom | None
)


@dataclass(frozen=True)
class ParameterFamilyHandler:
    family: ParameterFamily
    compute_stage: str
    sample_stage: str
    compute_interaction_partial: Callable[..., object]
    compute_interaction_group_partial: Callable[..., object]
    initialize_parameters: Callable[..., object]
    sample: Callable[..., object]
    prepare_batch_sample_inputs: Callable[..., object | None]
    select_prepared_random: Callable[..., PreparedFamilyRandom]
    parameters_to_host: Callable[..., object]
    supports_batch_sampling: bool = False


def _spin_runtime(block: CompiledBlock) -> CompiledSpinFamilyRuntime:
    runtime = block.family_runtime
    if not isinstance(runtime, CompiledSpinFamilyRuntime):
        raise TypeError(f"Block {block.block_index} is not a spin sampler block.")
    return runtime


def _categorical_runtime(block: CompiledBlock) -> CompiledCategoricalFamilyRuntime:
    runtime = block.family_runtime
    if not isinstance(runtime, CompiledCategoricalFamilyRuntime):
        raise TypeError(f"Block {block.block_index} is not a categorical sampler block.")
    return runtime


def _gaussian_runtime(block: CompiledBlock) -> CompiledGaussianFamilyRuntime:
    runtime = block.family_runtime
    if not isinstance(runtime, CompiledGaussianFamilyRuntime):
        raise TypeError(f"Block {block.block_index} is not a continuous Gaussian block.")
    return runtime


def _spin_threshold_tensor(executor, block: CompiledBlock, key):
    return executor.ttnn.from_torch(
        spin_threshold_logits_tensor(key, n_nodes=block.n_nodes),
        dtype=executor.compiled.spin_state_dtype,
        layout=executor.compiled.state_layout,
        device=executor.device,
    )


def _categorical_gumbel_tensor(executor, block: CompiledBlock, key):
    assert block.n_categories is not None
    return executor.ttnn.from_torch(
        torch.from_numpy(
            np.asarray(
                jax.random.gumbel(
                    key,
                    shape=(block.n_nodes, int(block.n_categories)),
                    dtype=jnp.float32,
                ),
                dtype=np.float32,
            ).copy()
        ).reshape(1, 1, block.n_nodes, int(block.n_categories)),
        dtype=executor.compiled.spin_state_dtype,
        layout=executor.compiled.categorical_layout,
        device=executor.device,
    )


def _gaussian_noise_device_tensor(executor, block: CompiledBlock, key):
    return executor.ttnn.from_torch(
        gaussian_noise_tensor(key, n_nodes=block.n_nodes),
        dtype=executor.compiled.spin_state_dtype,
        layout=executor.compiled.state_layout,
        device=executor.device,
    )


def _prepare_spin_batch_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> SpinPreparedRandom:
    return SpinPreparedRandom(
        threshold_logits=executor.ttnn.from_torch(
            spin_threshold_logits_batch_tensor(sample_keys, n_nodes=block.n_nodes),
            dtype=executor.compiled.spin_state_dtype,
            layout=executor.compiled.state_layout,
            device=executor.device,
        )
    )

def _prepare_categorical_batch_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> CategoricalPreparedRandom:
    assert block.n_categories is not None
    return CategoricalPreparedRandom(
        gumbel_noise=executor.ttnn.from_torch(
            categorical_gumbel_noise_tensor(
                sample_keys,
                n_users=block.n_nodes,
                n_categories=int(block.n_categories),
            ),
            dtype=executor.compiled.spin_state_dtype,
            layout=executor.compiled.categorical_layout,
            device=executor.device,
        )
    )


def _prepare_gaussian_batch_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> GaussianPreparedRandom:
    return GaussianPreparedRandom(
        gaussian_noise=executor.ttnn.from_torch(
            gaussian_noise_batch_tensor(sample_keys, n_nodes=block.n_nodes),
            dtype=executor.compiled.spin_state_dtype,
            layout=executor.compiled.state_layout,
            device=executor.device,
        )
    )


def _unsupported_batch_sample_inputs(
    executor,
    block: CompiledBlock,
    sample_keys: Sequence[object],
) -> PreparedFamilyRandom:
    del executor, sample_keys
    raise TypeError(
        f"Batched sample inputs are not implemented for block {block.block_index} "
        f"({block.sampler_lowering.parameter_family.value})."
    )


def _slice_prepared_random_tensor(executor, tensor, iter_index: int):
    if tensor is None:
        return None
    shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
    if shape[0] == 1:
        return tensor
    slicer = getattr(executor.ttnn, "slice", None)
    if not callable(slicer):
        raise TypeError(
            "TT backend must expose slice() to consume prepared batched random buffers."
        )
    return slicer(
        tensor,
        starts=(iter_index, 0, 0, 0),
        ends=(iter_index + 1, *shape[1:]),
        steps=(1, 1, 1, 1),
    )


def _select_spin_prepared_random(executor, block, prepared_random, iter_index: int):
    del block
    if not isinstance(prepared_random, SpinPreparedRandom):
        return None
    return SpinPreparedRandom(
        threshold_logits=_slice_prepared_random_tensor(
            executor, prepared_random.threshold_logits, iter_index
        )
    )


def _select_categorical_prepared_random(executor, block, prepared_random, iter_index: int):
    del block
    if not isinstance(prepared_random, CategoricalPreparedRandom):
        return None
    return CategoricalPreparedRandom(
        gumbel_noise=_slice_prepared_random_tensor(
            executor, prepared_random.gumbel_noise, iter_index
        )
    )


def _select_gaussian_prepared_random(executor, block, prepared_random, iter_index: int):
    del block
    if not isinstance(prepared_random, GaussianPreparedRandom):
        return None
    return GaussianPreparedRandom(
        gaussian_noise=_slice_prepared_random_tensor(
            executor, prepared_random.gaussian_noise, iter_index
        )
    )


def _concat_tensors(executor, tensors, *, dim: int):
    if not tensors:
        raise ValueError("Expected at least one tensor to concatenate.")
    if len(tensors) == 1:
        return tensors[0]
    return executor.ttnn.concat(list(tensors), dim=dim)


def _concat_categorical_group_flattened_tensors(
    executor,
    block: CompiledBlock,
    tensors,
    *,
    interactions: Sequence[CompiledInteraction],
):
    if not tensors:
        raise ValueError("Expected at least one categorical group tensor.")
    if len(tensors) == 1:
        return tensors[0]

    reshaped = []
    for tensor, interaction in zip(tensors, interactions):
        tensor_shape = tuple(int(dim) for dim in tensor.shape)
        trailing_shape = tensor_shape[2:]
        reshaped.append(
            executor.ttnn.reshape(
                tensor,
                (
                    tensor_shape[0],
                    block.n_nodes,
                    interaction.n_interactions,
                    *trailing_shape,
                ),
            )
        )

    grouped = _concat_tensors(executor, reshaped, dim=2)
    grouped_shape = tuple(int(dim) for dim in grouped.shape)
    trailing_shape = grouped_shape[3:]
    return executor.ttnn.reshape(
        grouped,
        (
            grouped_shape[0],
            block.n_nodes * grouped_shape[2],
            *trailing_shape,
        ),
    )


def _group_interactions(
    interaction_group: CompiledInteractionGroup | Sequence[CompiledInteraction],
):
    if isinstance(interaction_group, CompiledInteractionGroup):
        return interaction_group, interaction_group.interactions
    interactions = tuple(interaction_group)
    return None, interactions


def _identity_scale_tensor(
    executor,
    *,
    shape,
    layout,
):
    return executor.ttnn.full(
        list(shape),
        fill_value=1.0,
        dtype=executor.compiled.spin_state_dtype,
        layout=layout,
        device=executor.device,
    )


def _group_tail_size(
    group: CompiledInteractionGroup | None,
    interactions: Sequence[CompiledInteraction],
) -> int:
    if group is not None and group.flat_weights_spec is not None:
        if any(interaction.tail_shape for interaction in interactions):
            shape_tail = tuple(int(dim) for dim in group.flat_weights_spec.shape_tail)
            if shape_tail:
                return int(shape_tail[-1])
        return 1
    return max((math.prod(interaction.tail_shape) or 1) for interaction in interactions)


def _zero_index_tensor(
    executor,
    *,
    shape: tuple[int, ...],
    layout,
):
    return executor.ttnn.full(
        list(shape),
        fill_value=0,
        dtype=executor.compiled.index_dtype,
        layout=layout,
        device=executor.device,
    )


def _promote_spin_or_gaussian_group_scale(
    executor,
    tensor,
    *,
    batch_size: int,
    block: CompiledBlock,
    interaction: CompiledInteraction,
):
    if len(tuple(tensor.shape)) == 5:
        return tensor
    return executor.ttnn.reshape(
        tensor,
        (
            batch_size,
            1,
            block.n_nodes,
            interaction.n_interactions,
            1,
        ),
    )


def _build_categorical_theta_inputs(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    if continuous_sources:
        raise TypeError(
            "categorical_logits interactions do not support continuous tails yet."
        )
    categorical_spin_sources = tuple(spin_sources)
    categorical_sources_row_major = tuple(categorical_sources)
    if interaction.fused_static_theta_bias:
        categorical_spin_sources = tuple(
            executor.ttnn.concat(
                [interaction.fused_static_theta_prefix, source],
                dim=1 if interaction.use_single_node_fused_theta_scale_fast_path else 2,
            )
            for source in categorical_spin_sources
        )
    flat_index = None
    if interaction.tail_shape:
        flat_index = categorical_flat_index_tensor_device(
            ttnn=executor.ttnn,
            device=executor.device,
            categorical_sources=categorical_sources_row_major,
            categorical_tail_strides=interaction.categorical_tail_strides,
            layout=executor.compiled.categorical_layout,
            index_dtype=executor.compiled.index_dtype,
        )

    interaction_scale = categorical_interaction_scale_tensor_device(
        ttnn=executor.ttnn,
        active_mask=interaction.active_mask,
        spin_sources=categorical_spin_sources,
        n_nodes=block.n_nodes,
        n_interactions=interaction.n_interactions,
        active_mask_is_all_ones=interaction.active_mask_is_all_ones,
        skip_flatten_if_aligned=interaction.use_single_node_fused_theta_scale_fast_path,
    )
    return flat_index, interaction_scale


def _build_spin_gamma_inputs(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    batch_size = int(executor._block_state_slots[block.state_view.block_index].shape[0])
    flat_index = None
    if interaction.tail_shape:
        flat_index = categorical_tail_index_tensor_device(
            ttnn=executor.ttnn,
            device=executor.device,
            categorical_sources=categorical_sources,
            categorical_tail_strides=interaction.categorical_tail_strides,
            n_nodes=block.n_nodes,
            n_interactions=interaction.n_interactions,
            n_categories=None,
            layout=executor.compiled.state_layout,
            index_dtype=executor.compiled.index_dtype,
        )

    interaction_scale = None
    if not interaction.active_mask_is_all_ones:
        interaction_scale = executor._ensure_tensor_batch_size(
            interaction.active_mask,
            batch_size,
        )

    source_scale = None
    for gathered_source in (*spin_sources, *continuous_sources):
        aligned_source = executor.ttnn.reshape(
            gathered_source,
            interaction.parameter_spec.shape(batch_size),
        )
        source_scale = (
            aligned_source
            if source_scale is None
            else executor.ttnn.multiply(source_scale, aligned_source)
        )

    if source_scale is not None:
        interaction_scale = (
            source_scale
            if interaction_scale is None
            else executor.ttnn.multiply(interaction_scale, source_scale)
        )
    return flat_index, interaction_scale


def _build_gaussian_canonical_inputs(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    batch_size = int(executor._block_state_slots[block.state_view.block_index].shape[0])
    flat_index = None
    if interaction.tail_shape:
        flat_index = categorical_tail_index_tensor_device(
            ttnn=executor.ttnn,
            device=executor.device,
            categorical_sources=categorical_sources,
            categorical_tail_strides=interaction.categorical_tail_strides,
            n_nodes=block.n_nodes,
            n_interactions=interaction.n_interactions,
            n_categories=None,
            layout=executor.compiled.state_layout,
            index_dtype=executor.compiled.index_dtype,
        )
    interaction_scale = None
    if not interaction.active_mask_is_all_ones:
        interaction_scale = executor._ensure_tensor_batch_size(
            interaction.active_mask,
            batch_size,
        )

    for source in (*spin_sources, *continuous_sources):
        aligned_source = executor.ttnn.reshape(
            source,
            interaction.parameter_spec.shape(batch_size),
        )
        interaction_scale = (
            aligned_source
            if interaction_scale is None
            else executor.ttnn.multiply(interaction_scale, aligned_source)
        )
    return flat_index, interaction_scale


def _compute_categorical_interaction_partial(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    def _build():
        flat_index, interaction_scale = _build_categorical_theta_inputs(
            executor,
            block,
            interaction,
            spin_sources,
            categorical_sources,
            continuous_sources,
        )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "categorical_theta_inputs.build",
                lambda: CategoricalThetaInputs(
                    flat_weights=interaction.flat_weights,
                    flat_index=flat_index,
                    interaction_scale=interaction_scale,
                    n_nodes=block.n_nodes,
                    n_interactions=interaction.n_interactions,
                    n_categories=int(block.n_categories),
                ),
            ),
        )

    return _build()


def _compute_categorical_interaction_group_partial(
    executor,
    block: CompiledBlock,
    interaction_group: CompiledInteractionGroup | Sequence[CompiledInteraction],
):
    group, interactions = _group_interactions(interaction_group)
    if len(interactions) == 1:
        interaction = interactions[0]
        return _compute_categorical_interaction_partial(
            executor,
            block,
            interaction,
            *executor._gather_interaction_sources(block, interaction),
        )

    def _build():
        batch_size = int(executor._block_state_slots[block.state_view.block_index].shape[0])
        group_tail_size = _group_tail_size(group, interactions)
        flat_indices = []
        interaction_scales = []
        flat_weights = [] if group is None or group.flat_weights is None else None

        for interaction in interactions:
            spin_sources, categorical_sources, continuous_sources = (
                executor._gather_interaction_sources(block, interaction)
            )
            flat_index, interaction_scale = _build_categorical_theta_inputs(
                executor,
                block,
                interaction,
                spin_sources,
                categorical_sources,
                continuous_sources,
            )
            if flat_weights is not None:
                flat_weights.append(interaction.flat_weights)
            interaction_scales.append(interaction_scale)
            if flat_index is not None:
                flat_indices.append(flat_index)
            elif group_tail_size > 1:
                flat_indices.append(
                    _zero_index_tensor(
                        executor,
                        shape=(
                            batch_size,
                            block.n_nodes * interaction.n_interactions,
                            int(block.n_categories),
                            1,
                        ),
                        layout=executor.compiled.categorical_layout,
                    )
                )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "categorical_theta_inputs.build_group",
                lambda: CategoricalThetaInputs(
                    flat_weights=(
                        group.flat_weights
                        if group is not None and group.flat_weights is not None
                        else _concat_categorical_group_flattened_tensors(
                            executor,
                            block,
                            flat_weights,
                            interactions=interactions,
                        )
                    ),
                    flat_index=(
                        group.flat_indices
                        if group is not None and group.flat_indices is not None
                        else (
                            None
                            if not flat_indices
                            else _concat_tensors(executor, flat_indices, dim=1)
                        )
                    ),
                    interaction_scale=_concat_categorical_group_flattened_tensors(
                        executor,
                        block,
                        interaction_scales,
                        interactions=interactions,
                    ),
                    n_nodes=block.n_nodes,
                    n_interactions=(
                        group.n_interactions
                        if group is not None
                        else sum(interaction.n_interactions for interaction in interactions)
                    ),
                    n_categories=int(block.n_categories),
                ),
            ),
        )

    return _build()


def _compute_spin_interaction_partial(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    def _build():
        flat_index, interaction_scale = _build_spin_gamma_inputs(
            executor,
            block,
            interaction,
            spin_sources,
            categorical_sources,
            continuous_sources,
        )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "spin_gamma_inputs.build",
                lambda: SpinGammaInputs(
                    flat_weights=interaction.flat_weights,
                    flat_index=flat_index,
                    interaction_scale=interaction_scale,
                    n_nodes=block.n_nodes,
                    n_interactions=interaction.n_interactions,
                ),
            ),
        )

    return _build()


def _compute_spin_interaction_group_partial(
    executor,
    block: CompiledBlock,
    interaction_group: CompiledInteractionGroup | Sequence[CompiledInteraction],
):
    group, interactions = _group_interactions(interaction_group)
    if len(interactions) == 1:
        interaction = interactions[0]
        return _compute_spin_interaction_partial(
            executor,
            block,
            interaction,
            *executor._gather_interaction_sources(block, interaction),
        )

    def _build():
        batch_size = int(executor._block_state_slots[block.state_view.block_index].shape[0])
        group_tail_size = _group_tail_size(group, interactions)
        flat_weights = [] if group is None or group.flat_weights is None else None
        flat_indices = []
        interaction_scales = []

        for interaction in interactions:
            spin_sources, categorical_sources, continuous_sources = (
                executor._gather_interaction_sources(block, interaction)
            )
            flat_index, interaction_scale = _build_spin_gamma_inputs(
                executor,
                block,
                interaction,
                spin_sources,
                categorical_sources,
                continuous_sources,
            )
            if flat_weights is not None:
                flat_weights.append(interaction.flat_weights)
            if flat_index is not None:
                flat_indices.append(flat_index)
            elif group_tail_size > 1:
                flat_indices.append(
                    _zero_index_tensor(
                        executor,
                        shape=(
                            batch_size,
                            1,
                            block.n_nodes,
                            interaction.n_interactions,
                            1,
                        ),
                        layout=executor.compiled.state_layout,
                    )
                )
            if interaction_scale is not None and group_tail_size > 1:
                interaction_scale = _promote_spin_or_gaussian_group_scale(
                    executor,
                    interaction_scale,
                    batch_size=batch_size,
                    block=block,
                    interaction=interaction,
                )
            interaction_scales.append(
                interaction_scale
                if interaction_scale is not None
                else _identity_scale_tensor(
                    executor,
                    shape=(
                        batch_size,
                        1,
                        block.n_nodes,
                        interaction.n_interactions,
                        1,
                    )
                    if group_tail_size > 1
                    else interaction.parameter_spec.shape(batch_size),
                    layout=interaction.parameter_spec.layout,
                )
            )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "spin_gamma_inputs.build_group",
                lambda: SpinGammaInputs(
                    flat_weights=(
                        group.flat_weights
                        if group is not None and group.flat_weights is not None
                        else _concat_tensors(executor, flat_weights, dim=3)
                    ),
                    flat_index=(
                        None
                        if not flat_indices
                        else _concat_tensors(executor, flat_indices, dim=3)
                    ),
                    interaction_scale=_concat_tensors(executor, interaction_scales, dim=3),
                    n_nodes=block.n_nodes,
                    n_interactions=(
                        group.n_interactions
                        if group is not None
                        else sum(interaction.n_interactions for interaction in interactions)
                    ),
                ),
            ),
        )

    return _build()


def _compute_gaussian_interaction_partial(
    executor,
    block: CompiledBlock,
    interaction: CompiledInteraction,
    spin_sources,
    categorical_sources,
    continuous_sources,
):
    def _build():
        flat_index, interaction_scale = _build_gaussian_canonical_inputs(
            executor,
            block,
            interaction,
            spin_sources,
            categorical_sources,
            continuous_sources,
        )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "gaussian_inputs.build",
                lambda: GaussianCanonicalInputs(
                    flat_weights=interaction.flat_weights,
                    flat_index=flat_index,
                    interaction_scale=interaction_scale,
                    n_nodes=block.n_nodes,
                    n_interactions=interaction.n_interactions,
                    contribution_kind=interaction.contribution_kind,
                ),
            ),
        )

    return _build()


def _compute_gaussian_interaction_group_partial(
    executor,
    block: CompiledBlock,
    interaction_group: CompiledInteractionGroup | Sequence[CompiledInteraction],
):
    group, interactions = _group_interactions(interaction_group)
    if len(interactions) == 1:
        interaction = interactions[0]
        return _compute_gaussian_interaction_partial(
            executor,
            block,
            interaction,
            *executor._gather_interaction_sources(block, interaction),
        )

    def _build():
        batch_size = int(executor._block_state_slots[block.state_view.block_index].shape[0])
        group_tail_size = _group_tail_size(group, interactions)
        flat_weights = [] if group is None or group.flat_weights is None else None
        flat_indices = []
        interaction_scales = []

        for interaction in interactions:
            spin_sources, categorical_sources, continuous_sources = (
                executor._gather_interaction_sources(block, interaction)
            )
            flat_index, interaction_scale = _build_gaussian_canonical_inputs(
                executor,
                block,
                interaction,
                spin_sources,
                categorical_sources,
                continuous_sources,
            )
            if flat_weights is not None:
                flat_weights.append(interaction.flat_weights)
            if flat_index is not None:
                flat_indices.append(flat_index)
            elif group_tail_size > 1:
                flat_indices.append(
                    _zero_index_tensor(
                        executor,
                        shape=(
                            batch_size,
                            1,
                            block.n_nodes,
                            interaction.n_interactions,
                            1,
                        ),
                        layout=executor.compiled.state_layout,
                    )
                )
            if interaction_scale is not None and group_tail_size > 1:
                interaction_scale = _promote_spin_or_gaussian_group_scale(
                    executor,
                    interaction_scale,
                    batch_size=batch_size,
                    block=block,
                    interaction=interaction,
                )
            interaction_scales.append(
                interaction_scale
                if interaction_scale is not None
                else _identity_scale_tensor(
                    executor,
                    shape=(
                        batch_size,
                        1,
                        block.n_nodes,
                        interaction.n_interactions,
                        1,
                    )
                    if group_tail_size > 1
                    else interaction.parameter_spec.shape(batch_size),
                    layout=interaction.parameter_spec.layout,
                )
            )

        return executor._parameter_kernel_op_for_block(block)(
            ttnn=executor.ttnn,
            device=executor.device,
            inputs=executor._profile_call(
                "gaussian_inputs.build_group",
                lambda: GaussianCanonicalInputs(
                    flat_weights=(
                        group.flat_weights
                        if group is not None and group.flat_weights is not None
                        else _concat_tensors(executor, flat_weights, dim=3)
                    ),
                    flat_index=(
                        None
                        if not flat_indices
                        else _concat_tensors(executor, flat_indices, dim=3)
                    ),
                    interaction_scale=_concat_tensors(executor, interaction_scales, dim=3),
                    n_nodes=block.n_nodes,
                    n_interactions=(
                        group.n_interactions
                        if group is not None
                        else sum(interaction.n_interactions for interaction in interactions)
                    ),
                    contribution_kind=interactions[0].contribution_kind,
                ),
            ),
        )

    return _build()


def _initialize_spin_parameters(executor, block: CompiledBlock, batch_size: int):
    runtime = _spin_runtime(block)
    return executor._ensure_tensor_batch_size(runtime.zero_parameters, batch_size)


def _initialize_categorical_parameters(executor, block: CompiledBlock, batch_size: int):
    runtime = _categorical_runtime(block)
    base = runtime.static_bias
    if base is None:
        base = runtime.zero_parameters
    return executor._ensure_tensor_batch_size(base, batch_size)


def _initialize_gaussian_parameters(executor, block: CompiledBlock, batch_size: int):
    runtime = _gaussian_runtime(block)
    return executor._ensure_tensor_batch_size(runtime.zero_parameters, batch_size)


def _sample_spin_parameters(
    executor,
    block: CompiledBlock,
    key,
    parameters,
    prepared_random: PreparedFamilyRandom,
):
    runtime = _spin_runtime(block)
    batch_size = int(parameters.shape[0])
    threshold_logits = (
        prepared_random.threshold_logits
        if isinstance(prepared_random, SpinPreparedRandom)
        else None
    )
    if threshold_logits is None:
        threshold_logits = executor._profile_call(
            f"sample_block.block{block.block_index}.threshold_logits",
            lambda: _spin_threshold_tensor(executor, block, key),
        )
    threshold_logits = executor._ensure_tensor_batch_size(threshold_logits, batch_size)
    signed = executor._profile_call(
        f"sample_block.block{block.block_index}.device_op",
        lambda: executor.spin_sample_op(
            ttnn=executor.ttnn,
            device=executor.device,
            gamma=parameters,
            threshold_logits=threshold_logits,
            positive_ones=executor._ensure_tensor_batch_size(
                runtime.positive_ones, batch_size
            ),
            negative_ones=executor._ensure_tensor_batch_size(
                runtime.negative_ones, batch_size
            ),
        ),
    )
    return executor._profile_call(
        f"sample_block.block{block.block_index}.reshape",
        lambda: executor.ttnn.reshape(
            signed,
            (batch_size, 1, 1, block.n_nodes),
        ),
    )


def _sample_categorical_parameters(
    executor,
    block: CompiledBlock,
    key,
    parameters,
    prepared_random: PreparedFamilyRandom,
):
    runtime = _categorical_runtime(block)
    sampler_lowering = block.sampler_lowering
    categorical_sampler = executor.categorical_sampler
    if sampler_lowering.sample_categorical is not None:
        categorical_sampler = lambda **kwargs: sampler_lowering.sample_categorical(
            **kwargs,
            block=block,
            sampler_lowering=sampler_lowering,
            current_sampler_state=_state_runtime.sampler_state_for_block(
                executor, block.block_index
            ),
        )
    return executor._profile_call(
        f"sample_block.block{block.block_index}.categorical_sampler",
        lambda: categorical_sampler(
            ttnn=executor.ttnn,
            device=executor.device,
            logits=parameters,
            key=key,
            output_dtype=block.output_dtype,
            plan=runtime.sampling_plan,
            gumbel_noise=(
                prepared_random.gumbel_noise
                if isinstance(prepared_random, CategoricalPreparedRandom)
                else None
            ),
        ),
    )


def _sample_gaussian_parameters(
    executor,
    block: CompiledBlock,
    key,
    parameters,
    prepared_random: PreparedFamilyRandom,
):
    runtime = _gaussian_runtime(block)
    batch_size = int(parameters.shape[0])
    noise = (
        prepared_random.gaussian_noise
        if isinstance(prepared_random, GaussianPreparedRandom)
        else None
    )
    if noise is None:
        noise = executor._profile_call(
            f"sample_block.block{block.block_index}.gaussian_noise",
            lambda: _gaussian_noise_device_tensor(executor, block, key),
        )
    noise = executor._ensure_tensor_batch_size(noise, batch_size)
    linear_selector = executor._ensure_tensor_batch_size(
        runtime.linear_selector,
        batch_size,
    )
    precision_selector = executor._ensure_tensor_batch_size(
        runtime.precision_selector,
        batch_size,
    )
    linear = executor.ttnn.sum(
        executor.ttnn.multiply(parameters, linear_selector),
        dim=3,
        keepdim=True,
    )
    precision = executor.ttnn.sum(
        executor.ttnn.multiply(parameters, precision_selector),
        dim=3,
        keepdim=True,
    )
    precision = _clamp_min_tensor(
        executor,
        precision,
        min_value=1.0e-6,
        stage=f"sample_block.block{block.block_index}.gaussian_precision_guard",
    )
    variance = executor.ttnn.reciprocal(precision)
    mean = executor.ttnn.multiply(linear, variance)
    std = executor.ttnn.sqrt(variance)
    sample = executor.ttnn.add(mean, executor.ttnn.multiply(std, noise))
    return executor._profile_call(
        f"sample_block.block{block.block_index}.reshape",
        lambda: executor.ttnn.reshape(
            sample,
            (batch_size, 1, 1, block.n_nodes),
        ),
    )


def _spin_parameters_to_host(executor, block: CompiledBlock, parameters):
    del block
    return (
        (1.0 / SPIN_PARAMETER_TO_GAMMA_SCALE)
        * _state_runtime.device_tensor_to_torch(executor, parameters)
    ).reshape(-1)


def _categorical_parameters_to_host(executor, block: CompiledBlock, parameters):
    return _state_runtime.device_tensor_to_torch(executor, parameters).reshape(
        block.n_nodes,
        int(block.n_categories),
    )


def _gaussian_parameters_to_host(executor, block: CompiledBlock, parameters):
    del block
    host_parameters = _state_runtime.device_tensor_to_torch(executor, parameters).to(
        torch.float32
    )
    return (
        host_parameters[..., 0].reshape(-1),
        host_parameters[..., 1].reshape(-1),
    )


def _clamp_min_tensor(executor, value, *, min_value: float, stage: str):
    gt = getattr(executor.ttnn, "gt", None)
    where = getattr(executor.ttnn, "where", None)
    full = getattr(executor.ttnn, "full", None)
    shape = tuple(getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", None)
    layout = getattr(value, "layout", None)
    if not callable(gt) or not callable(where) or not callable(full):
        raise TypeError(
            "TT backend must expose gt(), where(), and full() to guard Gaussian precision."
        )

    epsilon = full(
        list(shape),
        fill_value=float(min_value),
        dtype=dtype,
        layout=layout,
        device=executor.device,
    )
    return executor._profile_call(
        stage,
        lambda: where(gt(value, epsilon), value, epsilon),
    )


PARAMETER_FAMILY_HANDLERS: dict[ParameterFamily, ParameterFamilyHandler] = {
    SPIN_PARAMETER_FAMILY: ParameterFamilyHandler(
        family=SPIN_PARAMETER_FAMILY,
        compute_stage="compute_block_parameters.spin",
        sample_stage="sample_block.spin",
        compute_interaction_partial=_compute_spin_interaction_partial,
        compute_interaction_group_partial=_compute_spin_interaction_group_partial,
        initialize_parameters=_initialize_spin_parameters,
        sample=_sample_spin_parameters,
        prepare_batch_sample_inputs=_prepare_spin_batch_sample_inputs,
        select_prepared_random=_select_spin_prepared_random,
        supports_batch_sampling=True,
        parameters_to_host=_spin_parameters_to_host,
    ),
    CATEGORICAL_PARAMETER_FAMILY: ParameterFamilyHandler(
        family=CATEGORICAL_PARAMETER_FAMILY,
        compute_stage="compute_block_parameters.categorical",
        sample_stage="sample_block.categorical",
        compute_interaction_partial=_compute_categorical_interaction_partial,
        compute_interaction_group_partial=_compute_categorical_interaction_group_partial,
        initialize_parameters=_initialize_categorical_parameters,
        sample=_sample_categorical_parameters,
        prepare_batch_sample_inputs=_prepare_categorical_batch_sample_inputs,
        select_prepared_random=_select_categorical_prepared_random,
        supports_batch_sampling=True,
        parameters_to_host=_categorical_parameters_to_host,
    ),
    GAUSSIAN_PARAMETER_FAMILY: ParameterFamilyHandler(
        family=GAUSSIAN_PARAMETER_FAMILY,
        compute_stage="compute_block_parameters.gaussian",
        sample_stage="sample_block.gaussian",
        compute_interaction_partial=_compute_gaussian_interaction_partial,
        compute_interaction_group_partial=_compute_gaussian_interaction_group_partial,
        initialize_parameters=_initialize_gaussian_parameters,
        sample=_sample_gaussian_parameters,
        prepare_batch_sample_inputs=_prepare_gaussian_batch_sample_inputs,
        select_prepared_random=_select_gaussian_prepared_random,
        supports_batch_sampling=True,
        parameters_to_host=_gaussian_parameters_to_host,
    ),
}


__all__ = [
    "CategoricalPreparedRandom",
    "GaussianPreparedRandom",
    "PARAMETER_FAMILY_HANDLERS",
    "ParameterFamilyHandler",
    "PreparedFamilyRandom",
    "SpinPreparedRandom",
]
