from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from thrml.block_management import get_node_locations
from thrml.block_sampling import BlockSamplingProgram

from .categorical_ops import compile_ttnn_categorical_sampling_plan, tail_strides
from .sampler_lowering import (
    compile_sampler_lowering,
    resolve_sampler_lowering_config,
    unsupported_sampler_message,
)
from ..runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    ParameterKernelBackend,
    ParameterKernelBackends,
    SPIN_PARAMETER_FAMILY,
    resolve_parameter_kernel_backend,
)
from .interaction_lowering import lower_block_interactions
from ..runtime.compiled_program import (
    CompiledBlock,
    CompiledCategoricalFamilyRuntime,
    CompiledDirectSourcePlan,
    CompiledGatherShard,
    CompiledGatherSourcePlan,
    CompiledGlobalSlot,
    CompiledGaussianFamilyRuntime,
    CompiledInteraction,
    CompiledInteractionExecution,
    CompiledInteractionSource,
    CompiledProgram,
    CompiledStateView,
    CompiledSpinFamilyRuntime,
    node_kind_from_template,
)


@dataclass(frozen=True)
class _CompileContext:
    ttnn: object
    device: object
    state_layout: object
    categorical_layout: object
    spin_state_dtype: object
    categorical_state_dtype: object
    index_dtype: object


def _make_compile_context(*, ttnn, device) -> _CompileContext:
    state_layout = getattr(ttnn, "TILE_LAYOUT", getattr(ttnn, "ROW_MAJOR_LAYOUT"))
    categorical_layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", state_layout)
    spin_state_dtype = getattr(ttnn, "bfloat16")
    index_dtype = getattr(ttnn, "uint32", None)
    if index_dtype is None:
        index_dtype = getattr(ttnn, "int32")
    return _CompileContext(
        ttnn=ttnn,
        device=device,
        state_layout=state_layout,
        categorical_layout=categorical_layout,
        spin_state_dtype=spin_state_dtype,
        categorical_state_dtype=index_dtype,
        index_dtype=index_dtype,
    )


def _compile_global_slots(
    *,
    program: BlockSamplingProgram,
    context: _CompileContext,
) -> list[CompiledGlobalSlot]:
    global_slots: list[CompiledGlobalSlot] = []
    for global_slot_index, block_indices in enumerate(
        program.gibbs_spec.block_to_global_slice_spec
    ):
        slot_block_indices = tuple(int(block_index) for block_index in block_indices)
        if not slot_block_indices:
            global_slots.append(
                CompiledGlobalSlot(
                    global_slot_index=global_slot_index,
                    node_kind=None,
                    output_dtype=None,
                    device_dtype=None,
                    block_indices=slot_block_indices,
                )
            )
            continue

        first_block = program.gibbs_spec.blocks[slot_block_indices[0]]
        template_sd = program.gibbs_spec.node_shape_struct[first_block.node_type]
        node_kind = node_kind_from_template(template_sd)
        device_dtype = (
            context.spin_state_dtype
            if node_kind == "spin"
            else context.categorical_state_dtype
        )
        global_slots.append(
            CompiledGlobalSlot(
                global_slot_index=global_slot_index,
                node_kind=node_kind,
                output_dtype=template_sd.dtype,
                device_dtype=device_dtype,
                block_indices=slot_block_indices,
            )
        )
    return global_slots


def _compile_state_views(
    *,
    program: BlockSamplingProgram,
    context: _CompileContext,
) -> list[CompiledStateView]:
    state_views: list[CompiledStateView] = []
    for block_index, block in enumerate(program.gibbs_spec.blocks):
        template_sd = program.gibbs_spec.node_shape_struct[block.node_type]
        node_kind = node_kind_from_template(template_sd)
        global_slot_index, positions = get_node_locations(block, program.gibbs_spec)
        positions_np = np.asarray(positions, dtype=np.int32).copy()
        host_gather_index = torch.from_numpy(positions_np).reshape(1, 1, 1, -1).to(
            torch.int64
        )
        gather_index = context.ttnn.from_torch(
            host_gather_index,
            dtype=context.index_dtype,
            layout=context.state_layout,
            device=context.device,
        )
        state_views.append(
            CompiledStateView(
                block_index=block_index,
                global_slot_index=int(global_slot_index),
                node_kind=node_kind,
                n_nodes=len(block.nodes),
                output_dtype=template_sd.dtype,
                positions=positions_np,
                gather_index=gather_index,
                host_gather_index=host_gather_index,
            )
        )
    return state_views


def _compile_interaction_sources(
    *,
    lowered_interaction,
    global_slots: list[CompiledGlobalSlot],
    state_views: list[CompiledStateView],
    context: _CompileContext,
) -> list[CompiledInteractionSource]:
    sources: list[CompiledInteractionSource] = []
    for source_global_index, global_slice in zip(
        lowered_interaction.source_global_inds,
        lowered_interaction.source_global_slices,
    ):
        slice_np = np.asarray(global_slice, dtype=np.int32)
        slot_meta = global_slots[int(source_global_index)]
        shards: list[CompiledGatherShard] = []
        offset = 0
        for source_block_index in slot_meta.block_indices:
            source_view = state_views[source_block_index]
            mask_np = (
                (slice_np >= offset) & (slice_np < offset + source_view.n_nodes)
            )
            if np.any(mask_np):
                local_np = np.where(mask_np, slice_np - offset, 0).astype(np.int32)
                repeat_sizes = (1, 1, int(slice_np.shape[0]), 1)
                repeat_is_identity = all(size == 1 for size in repeat_sizes)
                gather_is_identity = (
                    repeat_is_identity
                    and local_np.shape[0] == 1
                    and source_view.n_nodes == int(local_np.size)
                    and np.array_equal(
                        local_np.reshape(-1),
                        np.arange(source_view.n_nodes, dtype=np.int32),
                    )
                )
                shards.append(
                    CompiledGatherShard(
                        block_index=source_block_index,
                        repeat_sizes=repeat_sizes,
                        gather_index=context.ttnn.from_torch(
                            torch.from_numpy(local_np.copy()).reshape(
                                1, 1, *slice_np.shape
                            ),
                            dtype=context.index_dtype,
                            layout=context.state_layout,
                            device=context.device,
                        ),
                        membership_mask=context.ttnn.from_torch(
                            torch.from_numpy(mask_np.astype(np.float32)).reshape(
                                1, 1, *slice_np.shape
                            ),
                            dtype=slot_meta.device_dtype,
                            layout=context.state_layout,
                            device=context.device,
                        ),
                        repeat_is_identity=repeat_is_identity,
                        gather_is_identity=gather_is_identity,
                        membership_mask_is_all_ones=bool(np.all(mask_np)),
                    )
                )
            offset += source_view.n_nodes

        sources.append(
            CompiledInteractionSource(
                node_kind=str(slot_meta.node_kind),
                shards=tuple(shards),
            )
        )
    return sources


def _is_trivial_direct_source(source: CompiledInteractionSource) -> bool:
    return (
        len(source.shards) == 1
        and source.shards[0].repeat_is_identity
        and source.shards[0].gather_is_identity
        and source.shards[0].membership_mask_is_all_ones
    )


def _compile_interaction_execution(
    *,
    sources: list[CompiledInteractionSource],
    n_spin: int,
    n_nodes: int,
    source_n_interactions: int,
    output_layout,
    prefer_row_major: bool,
    row_major_cache_block_indices: set[int],
) -> CompiledInteractionExecution:
    target_shape_tail = (n_nodes, source_n_interactions, 1)
    spin_sources = []
    categorical_sources = []
    continuous_sources = []

    for source_position, source in enumerate(sources):
        if _is_trivial_direct_source(source):
            shard = source.shards[0]
            source_plan = CompiledDirectSourcePlan(
                block_index=shard.block_index,
                target_shape_tail=target_shape_tail,
                output_layout=output_layout,
                use_row_major=prefer_row_major,
            )
            if prefer_row_major:
                row_major_cache_block_indices.add(shard.block_index)
        else:
            source_plan = CompiledGatherSourcePlan(
                shards=source.shards,
                target_shape_tail=target_shape_tail,
                output_layout=output_layout,
            )

        if source_position < n_spin:
            spin_sources.append(source_plan)
        elif source.node_kind == "continuous":
            continuous_sources.append(source_plan)
        else:
            categorical_sources.append(source_plan)

    return CompiledInteractionExecution(
        spin_sources=tuple(spin_sources),
        categorical_sources=tuple(categorical_sources),
        continuous_sources=tuple(continuous_sources),
    )


def _float_tensor(context: _CompileContext, values, *, layout):
    return context.ttnn.from_torch(
        torch.from_numpy(np.asarray(values, dtype=np.float32).copy()),
        dtype=context.spin_state_dtype,
        layout=layout,
        device=context.device,
    )


def _compile_spin_block(
    *,
    program: BlockSamplingProgram,
    block_index: int,
    state_view: CompiledStateView,
    global_slots: list[CompiledGlobalSlot],
    state_views: list[CompiledStateView],
    row_major_cache_block_indices: set[int],
    context: _CompileContext,
):
    interactions: list[CompiledInteraction] = []
    pending_constant_spin_partial = None

    for lowered_interaction in lower_block_interactions(
        program,
        block_index,
        parameter_family=SPIN_PARAMETER_FAMILY,
    ):
        n_spin = lowered_interaction.contribution.n_spin
        active_np = np.asarray(lowered_interaction.active_mask, dtype=np.float32)
        n_nodes, n_interactions = active_np.shape
        tail_shape = lowered_interaction.tail_shape
        sources = _compile_interaction_sources(
            lowered_interaction=lowered_interaction,
            global_slots=global_slots,
            state_views=state_views,
            context=context,
        )

        if tail_shape:
            flat_weights = (2.0 * lowered_interaction.contribution.weights).reshape(
                1,
                1,
                n_nodes,
                n_interactions,
                math.prod(tail_shape) or 1,
            )
            active_mask = active_np.reshape(1, 1, n_nodes, n_interactions, 1)
            parameter_scale_shape_tail = (1, n_nodes, n_interactions, 1)
        else:
            flat_weights = (2.0 * lowered_interaction.contribution.weights).reshape(
                1,
                1,
                n_nodes,
                n_interactions,
            )
            active_mask = active_np.reshape(1, 1, n_nodes, n_interactions)
            parameter_scale_shape_tail = (1, n_nodes, n_interactions)

        if not sources:
            constant_partial = flat_weights * active_mask
            if pending_constant_spin_partial is None:
                pending_constant_spin_partial = constant_partial
            else:
                pending_constant_spin_partial = (
                    pending_constant_spin_partial + constant_partial
                )
            continue

        interactions.append(
            CompiledInteraction(
                contribution_kind=lowered_interaction.contribution.contribution_kind,
                n_interactions=n_interactions,
                tail_shape=tail_shape,
                categorical_tail_strides=tail_strides(tail_shape),
                execution=_compile_interaction_execution(
                    sources=sources,
                    n_spin=n_spin,
                    n_nodes=n_nodes,
                    source_n_interactions=n_interactions,
                    output_layout=None,
                    prefer_row_major=False,
                    row_major_cache_block_indices=row_major_cache_block_indices,
                ),
                flat_weights=_float_tensor(
                    context,
                    flat_weights,
                    layout=context.state_layout,
                ),
                active_mask=_float_tensor(
                    context,
                    active_mask,
                    layout=context.state_layout,
                ),
                active_mask_is_all_ones=bool(np.all(active_np == 1.0)),
                parameter_scale_shape_tail=parameter_scale_shape_tail,
                fused_static_theta_bias=False,
                use_single_node_fused_theta_scale_fast_path=False,
                fused_static_theta_prefix=None,
            )
        )

    if pending_constant_spin_partial is not None:
        interactions.append(
            CompiledInteraction(
                contribution_kind="default",
                n_interactions=int(pending_constant_spin_partial.shape[3]),
                tail_shape=(),
                categorical_tail_strides=(),
                execution=CompiledInteractionExecution(
                    spin_sources=tuple(),
                    categorical_sources=tuple(),
                    continuous_sources=tuple(),
                ),
                flat_weights=_float_tensor(
                    context,
                    pending_constant_spin_partial,
                    layout=context.state_layout,
                ),
                active_mask=_float_tensor(
                    context,
                    np.ones_like(pending_constant_spin_partial, dtype=np.float32),
                    layout=context.state_layout,
                ),
                active_mask_is_all_ones=True,
                parameter_scale_shape_tail=(
                    1,
                    state_view.n_nodes,
                    int(pending_constant_spin_partial.shape[3]),
                ),
                fused_static_theta_bias=False,
                use_single_node_fused_theta_scale_fast_path=False,
                fused_static_theta_prefix=None,
            )
        )

    return (
        tuple(interactions),
        CompiledSpinFamilyRuntime(
            zero_parameters=context.ttnn.full(
                [1, 1, state_view.n_nodes, 1],
                fill_value=0.0,
                dtype=context.spin_state_dtype,
                layout=context.state_layout,
                device=context.device,
            ),
            positive_ones=context.ttnn.full(
                [1, 1, state_view.n_nodes, 1],
                fill_value=1.0,
                dtype=context.spin_state_dtype,
                layout=context.state_layout,
                device=context.device,
            ),
            negative_ones=context.ttnn.full(
                [1, 1, state_view.n_nodes, 1],
                fill_value=-1.0,
                dtype=context.spin_state_dtype,
                layout=context.state_layout,
                device=context.device,
            ),
        ),
        None,
    )


def _compile_categorical_block(
    *,
    program: BlockSamplingProgram,
    block_index: int,
    state_view: CompiledStateView,
    sampler_lowering,
    global_slots: list[CompiledGlobalSlot],
    state_views: list[CompiledStateView],
    row_major_cache_block_indices: set[int],
    context: _CompileContext,
):
    interactions: list[CompiledInteraction] = []
    pending_constant_theta_bias = None

    for lowered_interaction in lower_block_interactions(
        program,
        block_index,
        parameter_family=CATEGORICAL_PARAMETER_FAMILY,
    ):
        n_spin = lowered_interaction.contribution.n_spin
        active_np = np.asarray(lowered_interaction.active_mask, dtype=np.float32)
        n_nodes, n_interactions = active_np.shape
        tail_shape = lowered_interaction.tail_shape
        n_categories = int(lowered_interaction.n_categories)
        sources = _compile_interaction_sources(
            lowered_interaction=lowered_interaction,
            global_slots=global_slots,
            state_views=state_views,
            context=context,
        )

        flat_weights = lowered_interaction.contribution.weights.reshape(
            1,
            n_nodes * n_interactions,
            n_categories,
            math.prod(tail_shape) or 1,
        )
        active_mask = active_np.reshape(1, n_nodes * n_interactions, 1, 1)

        if not sources:
            constant_partial = (flat_weights * active_mask).reshape(
                1,
                n_nodes,
                n_interactions,
                n_categories,
            )
            constant_partial = constant_partial.sum(axis=2).reshape(
                1,
                1,
                n_nodes,
                n_categories,
            )
            if pending_constant_theta_bias is None:
                pending_constant_theta_bias = constant_partial
            else:
                pending_constant_theta_bias = (
                    pending_constant_theta_bias + constant_partial
                )
            continue

        fused_static_theta_bias = False
        fused_static_theta_prefix = None
        if pending_constant_theta_bias is not None and not tail_shape:
            flat_weights = np.concatenate(
                [
                    pending_constant_theta_bias.reshape(1, n_nodes, 1, n_categories, 1),
                    flat_weights.reshape(1, n_nodes, n_interactions, n_categories, 1),
                ],
                axis=2,
            ).reshape(
                1,
                n_nodes * (n_interactions + 1),
                n_categories,
                1,
            )
            active_mask = np.concatenate(
                [
                    np.ones((1, n_nodes, 1, 1, 1), dtype=np.float32),
                    active_mask.reshape(1, n_nodes, n_interactions, 1, 1),
                ],
                axis=2,
            ).reshape(1, n_nodes * (n_interactions + 1), 1, 1)
            n_interactions += 1
            pending_constant_theta_bias = None
            fused_static_theta_bias = True
            fused_static_theta_prefix = context.ttnn.full(
                [1, n_nodes, 1, 1],
                fill_value=1.0,
                dtype=context.spin_state_dtype,
                layout=context.categorical_layout,
                device=context.device,
            )

        interactions.append(
            CompiledInteraction(
                contribution_kind=lowered_interaction.contribution.contribution_kind,
                n_interactions=n_interactions,
                tail_shape=tail_shape,
                categorical_tail_strides=tail_strides(tail_shape),
                execution=_compile_interaction_execution(
                    sources=sources,
                    n_spin=n_spin,
                    n_nodes=n_nodes,
                    source_n_interactions=(
                        n_interactions - 1 if fused_static_theta_bias else n_interactions
                    ),
                    output_layout=context.categorical_layout,
                    prefer_row_major=True,
                    row_major_cache_block_indices=row_major_cache_block_indices,
                ),
                flat_weights=_float_tensor(
                    context,
                    flat_weights,
                    layout=context.categorical_layout,
                ),
                active_mask=_float_tensor(
                    context,
                    active_mask,
                    layout=context.categorical_layout,
                ),
                active_mask_is_all_ones=bool(np.all(active_np == 1.0)),
                parameter_scale_shape_tail=(
                    (1, n_nodes, n_interactions, 1)
                    if tail_shape
                    else (1, n_nodes, n_interactions)
                ),
                fused_static_theta_bias=fused_static_theta_bias,
                use_single_node_fused_theta_scale_fast_path=(
                    fused_static_theta_bias and n_nodes == 1
                ),
                fused_static_theta_prefix=fused_static_theta_prefix,
            )
        )

    n_categories = int(sampler_lowering.n_categories)
    return (
        tuple(interactions),
        CompiledCategoricalFamilyRuntime(
            zero_parameters=context.ttnn.full(
                [1, 1, state_view.n_nodes, n_categories],
                fill_value=0.0,
                dtype=context.spin_state_dtype,
                layout=context.categorical_layout,
                device=context.device,
            ),
            static_bias=(
                None
                if pending_constant_theta_bias is None
                else _float_tensor(
                    context,
                    pending_constant_theta_bias,
                    layout=context.categorical_layout,
                )
            ),
            sampling_plan=compile_ttnn_categorical_sampling_plan(
                ttnn=context.ttnn,
                device=context.device,
                n_users=state_view.n_nodes,
                n_categories=n_categories,
            ),
        ),
        n_categories,
    )


def _compile_gaussian_block(
    *,
    program: BlockSamplingProgram,
    block_index: int,
    state_view: CompiledStateView,
    global_slots: list[CompiledGlobalSlot],
    state_views: list[CompiledStateView],
    row_major_cache_block_indices: set[int],
    context: _CompileContext,
):
    interactions: list[CompiledInteraction] = []

    for lowered_interaction in lower_block_interactions(
        program,
        block_index,
        parameter_family=GAUSSIAN_PARAMETER_FAMILY,
    ):
        n_spin = lowered_interaction.contribution.n_spin
        active_np = np.asarray(lowered_interaction.active_mask, dtype=np.float32)
        n_nodes, n_interactions = active_np.shape
        tail_shape = lowered_interaction.tail_shape
        sources = _compile_interaction_sources(
            lowered_interaction=lowered_interaction,
            global_slots=global_slots,
            state_views=state_views,
            context=context,
        )

        if tail_shape:
            flat_weights = lowered_interaction.contribution.weights.reshape(
                1,
                1,
                n_nodes,
                n_interactions,
                math.prod(tail_shape) or 1,
            )
            active_mask = active_np.reshape(1, 1, n_nodes, n_interactions, 1)
            parameter_scale_shape_tail = (1, n_nodes, n_interactions, 1)
        else:
            flat_weights = lowered_interaction.contribution.weights.reshape(
                1,
                1,
                n_nodes,
                n_interactions,
            )
            active_mask = active_np.reshape(1, 1, n_nodes, n_interactions)
            parameter_scale_shape_tail = (1, n_nodes, n_interactions)

        interactions.append(
            CompiledInteraction(
                contribution_kind=lowered_interaction.contribution.contribution_kind,
                n_interactions=n_interactions,
                tail_shape=tail_shape,
                categorical_tail_strides=tail_strides(tail_shape),
                execution=_compile_interaction_execution(
                    sources=sources,
                    n_spin=n_spin,
                    n_nodes=n_nodes,
                    source_n_interactions=n_interactions,
                    output_layout=None,
                    prefer_row_major=False,
                    row_major_cache_block_indices=row_major_cache_block_indices,
                ),
                flat_weights=_float_tensor(
                    context,
                    flat_weights,
                    layout=context.state_layout,
                ),
                active_mask=_float_tensor(
                    context,
                    active_mask,
                    layout=context.state_layout,
                ),
                active_mask_is_all_ones=bool(np.all(active_np == 1.0)),
                parameter_scale_shape_tail=parameter_scale_shape_tail,
                fused_static_theta_bias=False,
                use_single_node_fused_theta_scale_fast_path=False,
                fused_static_theta_prefix=None,
            )
        )

    return (
        tuple(interactions),
        CompiledGaussianFamilyRuntime(
            zero_parameters=context.ttnn.full(
                [1, 1, state_view.n_nodes, 2],
                fill_value=0.0,
                dtype=context.spin_state_dtype,
                layout=context.state_layout,
                device=context.device,
            ),
            linear_selector=_float_tensor(
                context,
                np.broadcast_to(
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    (1, 1, state_view.n_nodes, 2),
                ),
                layout=context.state_layout,
            ),
            precision_selector=_float_tensor(
                context,
                np.broadcast_to(
                    np.asarray([0.0, 1.0], dtype=np.float32),
                    (1, 1, state_view.n_nodes, 2),
                ),
                layout=context.state_layout,
            ),
        ),
        None,
    )


def _compile_block(
    *,
    program: BlockSamplingProgram,
    block_index: int,
    sampler,
    state_view: CompiledStateView,
    global_slots: list[CompiledGlobalSlot],
    state_views: list[CompiledStateView],
    row_major_cache_block_indices: set[int],
    context: _CompileContext,
    parameter_kernel_backends: ParameterKernelBackends | None,
) -> CompiledBlock:
    block = program.gibbs_spec.free_blocks[block_index]
    lowering_config = resolve_sampler_lowering_config(
        sampler=sampler,
        block=block,
        state_view=state_view,
        ttnn=context.ttnn,
        device=context.device,
        state_layout=context.state_layout,
        categorical_layout=context.categorical_layout,
        spin_state_dtype=context.spin_state_dtype,
        categorical_state_dtype=context.categorical_state_dtype,
        index_dtype=context.index_dtype,
    )
    if lowering_config is None:
        raise TypeError(unsupported_sampler_message(block_index, sampler))

    sampler_lowering = compile_sampler_lowering(
        lowering_config,
        sampler=sampler,
        block=block,
        state_view=state_view,
        state_layout=context.state_layout,
        categorical_layout=context.categorical_layout,
        spin_state_dtype=context.spin_state_dtype,
        index_dtype=context.index_dtype,
    )
    parameter_family = sampler_lowering.parameter_family
    parameter_kernel_backend = resolve_parameter_kernel_backend(
        parameter_family,
        parameter_kernel_backends,
    )
    if (
        parameter_family == CATEGORICAL_PARAMETER_FAMILY
        and parameter_kernel_backend is ParameterKernelBackend.NATIVE
    ):
        raise TypeError(
            "Categorical THRML blocks on TT currently require TT-MLIR parameter "
            "kernels. Use tt_thrml.make_ttmlir_backend_binding(...) or set the "
            "categorical parameter-kernel backend to 'ttmlir'."
        )

    if parameter_family == SPIN_PARAMETER_FAMILY:
        interactions, family_runtime, n_categories = _compile_spin_block(
            program=program,
            block_index=block_index,
            state_view=state_view,
            global_slots=global_slots,
            state_views=state_views,
            row_major_cache_block_indices=row_major_cache_block_indices,
            context=context,
        )
    elif parameter_family == CATEGORICAL_PARAMETER_FAMILY:
        interactions, family_runtime, n_categories = _compile_categorical_block(
            program=program,
            block_index=block_index,
            state_view=state_view,
            sampler_lowering=sampler_lowering,
            global_slots=global_slots,
            state_views=state_views,
            row_major_cache_block_indices=row_major_cache_block_indices,
            context=context,
        )
    elif parameter_family == GAUSSIAN_PARAMETER_FAMILY:
        interactions, family_runtime, n_categories = _compile_gaussian_block(
            program=program,
            block_index=block_index,
            state_view=state_view,
            global_slots=global_slots,
            state_views=state_views,
            row_major_cache_block_indices=row_major_cache_block_indices,
            context=context,
        )
    else:
        raise TypeError(
            f"Unsupported parameter family {parameter_family!r} at block {block_index}."
        )

    return CompiledBlock(
        block_index=block_index,
        sampler_lowering=sampler_lowering,
        parameter_kernel_backend=parameter_kernel_backend,
        n_nodes=state_view.n_nodes,
        output_dtype=state_view.output_dtype,
        n_categories=n_categories,
        state_view=state_view,
        interactions=interactions,
        family_runtime=family_runtime,
    )


def compile_program(
    *,
    ttnn,
    device,
    program: BlockSamplingProgram,
    parameter_kernel_backends: ParameterKernelBackends | None = None,
) -> CompiledProgram:
    context = _make_compile_context(ttnn=ttnn, device=device)
    global_slots = _compile_global_slots(program=program, context=context)
    state_views = _compile_state_views(program=program, context=context)
    row_major_cache_block_indices: set[int] = set()
    compiled_blocks = [
        _compile_block(
            program=program,
            block_index=block_index,
            sampler=sampler,
            state_view=state_views[block_index],
            global_slots=global_slots,
            state_views=state_views,
            row_major_cache_block_indices=row_major_cache_block_indices,
            context=context,
            parameter_kernel_backends=parameter_kernel_backends,
        )
        for block_index, sampler in enumerate(program.samplers)
    ]
    return CompiledProgram(
        blocks=tuple(compiled_blocks),
        state_views=tuple(state_views),
        global_slots=tuple(global_slots),
        row_major_cache_block_indices=frozenset(row_major_cache_block_indices),
        state_layout=context.state_layout,
        categorical_layout=context.categorical_layout,
        spin_state_dtype=context.spin_state_dtype,
        categorical_state_dtype=context.categorical_state_dtype,
        index_dtype=context.index_dtype,
    )


__all__ = ["compile_program"]
