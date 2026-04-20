from __future__ import annotations

from typing import Sequence

import jax
from jax import numpy as jnp
import numpy as np

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.observers import AbstractObserver, MomentAccumulatorObserver, StateObserver

from ..runtime_config import SPIN_PARAMETER_FAMILY
from .execution_support import (
    CompiledMomentObserverPlan,
    CompiledObservationPlan,
    LoadedObservationJob,
    LoadedStateJob,
    stack_observer_history,
)
from . import state_runtime
from .runtime_utils import stack_sample_history


def compile_observation_plan(
    executor,
    blocks: Sequence[Block],
) -> CompiledObservationPlan:
    plan_key = tuple(id(block) for block in blocks)
    cached = executor._observation_plan_cache.get(plan_key)
    if cached is not None:
        return cached

    direct_views = []
    gathered_views = []
    block_tuple = tuple(blocks)
    for observe_index, block in enumerate(block_tuple):
        view = state_runtime.view_for_block(executor, block)
        target = direct_views if view.block_index >= 0 else gathered_views
        target.append((observe_index, view))

    grouped_direct_views: dict[str, list[tuple[int, object]]] = {}
    for observe_index, view in direct_views:
        grouped_direct_views.setdefault(view.node_kind, []).append((observe_index, view))

    grouped_gathered_views: dict[int, list[tuple[int, object]]] = {}
    for observe_index, view in gathered_views:
        grouped_gathered_views.setdefault(view.global_slot_index, []).append(
            (observe_index, view)
        )

    plan = CompiledObservationPlan(
        blocks=block_tuple,
        direct_views=tuple(direct_views),
        gathered_views=tuple(gathered_views),
        direct_view_groups=tuple(tuple(grouped) for grouped in grouped_direct_views.values()),
        gathered_view_groups=tuple(
            (global_slot_index, tuple(grouped))
            for global_slot_index, grouped in grouped_gathered_views.items()
        ),
    )
    executor._observation_plan_cache[plan_key] = plan
    return plan


def _observe_direct_view_groups(executor, observed, grouped_views, *, convert_view):
    for grouped in grouped_views:
        direct_slots = [executor._block_state_slots[view.block_index] for _, view in grouped]
        combined = (
            direct_slots[0]
            if len(direct_slots) == 1
            else executor.ttnn.concat(direct_slots, dim=-1)
        )
        combined_torch = state_runtime.device_tensor_to_torch(executor, combined)
        offset = 0
        for observe_index, view in grouped:
            next_offset = offset + view.n_nodes
            observed[observe_index] = convert_view(
                view,
                combined_torch[..., offset:next_offset],
            )
            offset = next_offset


def _observe_gathered_view_groups(
    executor,
    observed,
    grouped_views,
    *,
    gather_view,
    convert_view,
):
    for global_slot_index, grouped in grouped_views:
        source_slot = state_runtime.source_slot_torch_for_global_slot(
            executor,
            global_slot_index,
        )
        for observe_index, view in grouped:
            observed[observe_index] = convert_view(
                view,
                gather_view(view, source_slot),
            )

def _host_view_converter(executor, view, values):
    return state_runtime.state_from_device_torch(
        executor,
        view.node_kind,
        values,
        dtype=view.output_dtype,
    )


def _numpy_view_converter(executor, view, values):
    return state_runtime.state_numpy_from_device_torch(
        executor,
        view.node_kind,
        values,
        dtype=view.output_dtype,
    )


def _numpy_batch_view_converter(executor, view, values):
    return state_runtime.state_batch_numpy_from_device_torch(
        executor,
        view.node_kind,
        values,
        dtype=view.output_dtype,
    )


def _observe_gathered_view_groups_numpy_batch_from_source_slices(
    executor,
    observed,
    grouped_views,
    *,
    clamped_block_numpy_cache: dict[int, np.ndarray] | None = None,
):
    n_free_blocks = len(executor.program.gibbs_spec.free_blocks)
    for global_slot_index, grouped in grouped_views:
        slot_meta = executor.compiled.global_slots[global_slot_index]
        pieces = []
        for block_index in slot_meta.block_indices:
            if (
                clamped_block_numpy_cache is not None
                and block_index >= n_free_blocks
                and block_index in clamped_block_numpy_cache
            ):
                pieces.append(clamped_block_numpy_cache[block_index])
            else:
                pieces.append(state_runtime.source_slot_numpy_batch_piece(executor, block_index))
        source_np = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=-1)

        for observe_index, view in grouped:
            observed_view = np.take(source_np, view.positions, axis=-1)
            observed[observe_index] = np.asarray(observed_view, dtype=view.output_dtype)


def observe_compiled_plan_host(executor, plan: CompiledObservationPlan):
    def _observe():
        state_runtime.require_state(executor)
        observed = [None] * len(plan.blocks)
        _observe_direct_view_groups(
            executor,
            observed,
            plan.direct_view_groups,
            convert_view=lambda view, values: _host_view_converter(executor, view, values),
        )
        _observe_gathered_view_groups(
            executor,
            observed,
            plan.gathered_view_groups,
            gather_view=lambda view, source_slot: state_runtime.gather_block_torch_from_source_slot(
                executor,
                view,
                source_slot=source_slot,
            ),
            convert_view=lambda view, values: _host_view_converter(executor, view, values),
        )
        return observed

    return executor._profile_call("observe_blocks_host", _observe)


def observe_compiled_plan_numpy(executor, plan: CompiledObservationPlan):
    def _observe():
        state_runtime.require_state(executor)
        observed = [None] * len(plan.blocks)
        _observe_direct_view_groups(
            executor,
            observed,
            plan.direct_view_groups,
            convert_view=lambda view, values: _numpy_view_converter(executor, view, values),
        )
        _observe_gathered_view_groups(
            executor,
            observed,
            plan.gathered_view_groups,
            gather_view=lambda view, source_slot: state_runtime.gather_block_torch_from_source_slot(
                executor,
                view,
                source_slot=source_slot,
            ),
            convert_view=lambda view, values: _numpy_view_converter(executor, view, values),
        )
        return observed

    return executor._profile_call("observe_blocks_numpy", _observe)


def observe_compiled_plan_numpy_batch(executor, plan: CompiledObservationPlan):
    def _observe():
        state_runtime.require_state(executor)
        observed = [None] * len(plan.blocks)
        _observe_direct_view_groups(
            executor,
            observed,
            plan.direct_view_groups,
            convert_view=lambda view, values: _numpy_batch_view_converter(
                executor,
                view,
                values,
            ),
        )
        _observe_gathered_view_groups(
            executor,
            observed,
            plan.gathered_view_groups,
            gather_view=lambda view, source_slot: state_runtime.gather_block_torch_from_source_slot_batch(
                executor,
                view,
                source_slot=source_slot,
            ),
            convert_view=lambda view, values: _numpy_batch_view_converter(
                executor,
                view,
                values,
            ),
        )
        return observed

    return executor._profile_call("observe_blocks_numpy_batch", _observe)


def observe_compiled_plan_numpy_batch_moment_helper(
    executor,
    plan: CompiledObservationPlan,
    *,
    clamped_block_numpy_cache: dict[int, np.ndarray] | None = None,
):
    def _observe():
        state_runtime.require_state(executor)
        observed = [None] * len(plan.blocks)
        _observe_direct_view_groups(
            executor,
            observed,
            plan.direct_view_groups,
            convert_view=lambda view, values: _numpy_batch_view_converter(
                executor,
                view,
                values,
            ),
        )
        _observe_gathered_view_groups_numpy_batch_from_source_slices(
            executor,
            observed,
            plan.gathered_view_groups,
            clamped_block_numpy_cache=clamped_block_numpy_cache,
        )
        return observed

    return executor._profile_call("observe_blocks_numpy_batch", _observe)


def observe_blocks_host(executor, blocks: Sequence[Block]):
    return observe_compiled_plan_host(executor, compile_observation_plan(executor, blocks))


def current_state_free_host(executor):
    return observe_blocks_host(executor, executor.program.gibbs_spec.free_blocks)


def current_state_clamp_host(executor):
    return observe_blocks_host(executor, executor.program.gibbs_spec.clamped_blocks)


def compile_moment_observer_plan(
    executor,
    observer: MomentAccumulatorObserver,
) -> CompiledMomentObserverPlan:
    plan_key = id(observer)
    cached = executor._moment_observer_plan_cache.get(plan_key)
    if cached is not None:
        return cached

    unique_blocks = []
    expansion_indices = []
    for block in observer.blocks_to_sample:
        node_to_unique_index = {}
        unique_nodes = []
        block_expansion = []
        for node in block.nodes:
            unique_index = node_to_unique_index.get(id(node))
            if unique_index is None:
                unique_index = len(unique_nodes)
                node_to_unique_index[id(node)] = unique_index
                unique_nodes.append(node)
            block_expansion.append(unique_index)
        unique_blocks.append(Block(unique_nodes))
        expansion_indices.append(np.asarray(block_expansion, dtype=np.int32))

    plan = CompiledMomentObserverPlan(
        observation_plan=compile_observation_plan(executor, tuple(unique_blocks)),
        flat_node_count=len(observer.flat_nodes_list),
        flat_to_type_slices_list=tuple(
            np.asarray(type_slice, dtype=np.int32).copy()
            for type_slice in observer.flat_to_type_slices_list
        ),
        flat_to_full_moment_slices=tuple(
            np.asarray(moment_slice, dtype=np.int32).copy()
            for moment_slice in observer.flat_to_full_moment_slices
        ),
        expansion_indices=tuple(expansion_indices),
    )
    executor._moment_observer_plan_cache[plan_key] = plan
    return plan


def _expand_moment_observed_state(executor, plan: CompiledMomentObserverPlan, observed_state):
    expanded = []
    for state, expansion in zip(observed_state, plan.expansion_indices, strict=True):
        state_np = np.asarray(state)
        if state_np.ndim == 0:
            expanded.append(state_np)
        else:
            expanded.append(state_np[expansion])
    return expanded


def _accumulate_moment_observation(
    executor,
    observer: MomentAccumulatorObserver,
    plan: CompiledMomentObserverPlan,
    mem,
    observed_state,
):
    sampled_state = observer.f_transform(
        _expand_moment_observed_state(executor, plan, observed_state),
        observer.blocks_to_sample,
    )
    sampled_state = list(sampled_state)
    flat_state = None

    for type_slice, state in zip(plan.flat_to_type_slices_list, sampled_state, strict=True):
        state = np.asarray(state)
        if flat_state is None:
            flat_state = np.zeros(
                plan.flat_node_count,
                dtype=np.result_type(
                    *[np.asarray(leaf) for leaf in jax.tree.leaves(sampled_state)]
                ),
            )
        flat_state[type_slice] = state

    if flat_state is None:
        return mem

    def accumulate_moment(mem_entry, sl):
        update = np.prod(flat_state[sl], axis=1)
        return mem_entry.astype(update.dtype) + update

    return [
        np.asarray(accumulate_moment(mem_entry, sl))
        for mem_entry, sl in zip(mem, plan.flat_to_full_moment_slices, strict=True)
    ]


def sample_states(
    executor,
    key,
    schedule: SamplingSchedule,
    *,
    nodes_to_sample: Sequence[Block],
    init_state_free,
    state_clamp,
):
    state_runtime.load_state(executor, init_state_free, state_clamp)
    return sample_observed_blocks(executor, key, schedule, blocks=nodes_to_sample)


def sample_observed_blocks(
    executor,
    key,
    schedule: SamplingSchedule,
    *,
    blocks: Sequence[Block],
):
    observation_plan = compile_observation_plan(executor, blocks)
    return sample_loaded_states(
        executor,
        key,
        schedule,
        observation_plan=observation_plan,
    )


def sample_loaded_observation(
    executor,
    key,
    schedule: SamplingSchedule,
    *,
    state_clamp,
    observation_carry_init,
    f_observe: AbstractObserver,
):
    state_runtime.require_state(executor)

    if isinstance(f_observe, StateObserver) and observation_carry_init is None:
        return None, sample_observed_blocks(
            executor,
            key,
            schedule,
            blocks=f_observe.blocks_to_sample,
        )

    if isinstance(f_observe, MomentAccumulatorObserver):
        return sample_moment_observer(
            executor,
            key,
            schedule,
            observer=f_observe,
            observation_carry_init=observation_carry_init,
        )

    prepared_randoms = executor._prepare_schedule_randoms(key, schedule)
    executor._run_prepared_schedule_warmup(prepared_randoms)

    state_free = current_state_free_host(executor)
    mem, warmup_observation = f_observe(
        executor.program,
        state_free,
        state_clamp,
        observation_carry_init,
        jnp.array(0),
    )

    if schedule.n_samples <= 1:
        return mem, stack_observer_history([warmup_observation])

    observations = [warmup_observation]
    for iteration, _ in enumerate(
        executor._iter_prepared_schedule_sample_intervals(prepared_randoms),
        start=1,
    ):
        state_free = current_state_free_host(executor)
        mem, observation = f_observe(
            executor.program,
            state_free,
            state_clamp,
            mem,
            jnp.array(iteration),
        )
        observations.append(observation)

    return mem, stack_observer_history(observations)


def sample_loaded_observation_jobs(
    executor,
    schedule: SamplingSchedule,
    *,
    jobs: Sequence[LoadedObservationJob],
    f_observe: AbstractObserver,
):
    if isinstance(f_observe, StateObserver) and all(
        job.observation_carry_init is None for job in jobs
    ):
        observation_plan = compile_observation_plan(executor, f_observe.blocks_to_sample)
        return _run_loaded_jobs(
            executor,
            jobs,
            run_loaded_job=lambda job: (
                None,
                sample_loaded_states(
                    executor,
                    job.key,
                    schedule,
                    observation_plan=observation_plan,
                ),
            ),
        )

    return _run_loaded_jobs(
        executor,
        jobs,
        run_loaded_job=lambda job: sample_loaded_observation(
            executor,
            job.key,
            schedule,
            state_clamp=job.state_clamp,
            observation_carry_init=job.observation_carry_init,
            f_observe=f_observe,
        ),
    )


def sample_loaded_state_jobs(
    executor,
    schedule: SamplingSchedule,
    *,
    jobs: Sequence[LoadedStateJob],
    nodes_to_sample: Sequence[Block] | None = None,
    observation_plan: CompiledObservationPlan | None = None,
):
    if not jobs:
        return []

    if observation_plan is None:
        if nodes_to_sample is None:
            raise TypeError("Provide either `nodes_to_sample` or `observation_plan`.")
        observation_plan = compile_observation_plan(executor, nodes_to_sample)

    return _run_loaded_jobs(
        executor,
        jobs,
        run_loaded_job=lambda job: sample_loaded_states(
            executor,
            job.key,
            schedule,
            observation_plan=observation_plan,
        ),
    )


def _run_loaded_jobs(executor, jobs, *, run_loaded_job):
    if not jobs:
        return []

    results = []
    current_group_id = object()
    for job in jobs:
        if job.clamp_group_id is not None and job.clamp_group_id == current_group_id:
            state_runtime.load_free_state(executor, job.state_free)
        elif job.clamp_group_id is not None:
            if job.state_clamp:
                state_runtime.load_clamp_state(executor, job.state_clamp)
            state_runtime.load_free_state(executor, job.state_free)
            current_group_id = job.clamp_group_id
        else:
            state_runtime.load_state(executor, job.state_free, job.state_clamp)
            current_group_id = object()

        results.append(run_loaded_job(job))
    return results


def sample_moment_observer(
    executor,
    key,
    schedule: SamplingSchedule,
    *,
    observer: MomentAccumulatorObserver,
    observation_carry_init,
    plan: CompiledMomentObserverPlan | None = None,
):
    state_runtime.require_state(executor)
    return _sample_moment_observer_impl(
        executor,
        key,
        schedule,
        observer=observer,
        observation_carry_init=observation_carry_init,
        plan=plan,
        to_jax=True,
    )


def sample_moment_observer_numpy(
    executor,
    key,
    schedule: SamplingSchedule,
    *,
    observer: MomentAccumulatorObserver,
    observation_carry_init,
    plan: CompiledMomentObserverPlan | None = None,
):
    state_runtime.require_state(executor)
    return _sample_moment_observer_impl(
        executor,
        key,
        schedule,
        observer=observer,
        observation_carry_init=observation_carry_init,
        plan=plan,
        to_jax=False,
    )


def _sample_moment_observer_impl(
    executor,
    key,
    schedule: SamplingSchedule,
    *,
    observer: MomentAccumulatorObserver,
    observation_carry_init,
    plan: CompiledMomentObserverPlan | None,
    to_jax: bool,
):
    state_runtime.require_state(executor)
    if plan is None:
        plan = compile_moment_observer_plan(executor, observer)

    key, subkey = jax.random.split(key, 2)
    executor.run_blocks(subkey, n_iters=schedule.n_warmup)

    mem = [np.asarray(entry).copy() for entry in observation_carry_init]
    mem = _accumulate_moment_observation(
        executor,
        observer,
        plan,
        mem,
        observe_compiled_plan_numpy(executor, plan.observation_plan),
    )
    if schedule.n_samples <= 1:
        if to_jax:
            return [jnp.asarray(entry) for entry in mem], None
        return mem, None

    for sample_key in jax.random.split(key, schedule.n_samples - 1):
        executor.run_blocks(sample_key, n_iters=schedule.steps_per_sample)
        mem = _accumulate_moment_observation(
            executor,
            observer,
            plan,
            mem,
            observe_compiled_plan_numpy(executor, plan.observation_plan),
        )

    if to_jax:
        return [jnp.asarray(entry) for entry in mem], None
    return mem, None


def _supports_batched_spin_moment_jobs(executor) -> bool:
    return all(
        block.sampler_lowering.parameter_family == SPIN_PARAMETER_FAMILY
        and block.sampler_lowering.sampler_state_spec is None
        for block in executor.compiled.blocks
    )


def _accumulate_moment_observation_batch(
    executor,
    observer: MomentAccumulatorObserver,
    plan: CompiledMomentObserverPlan,
    mem_batch,
    observed_state_batch,
):
    if not mem_batch:
        return []

    batch_size = len(mem_batch)
    observed_state_arrays = [np.asarray(state) for state in observed_state_batch]
    sampled_state_by_type = None

    for batch_index in range(batch_size):
        observed_state = []
        for state_np in observed_state_arrays:
            if state_np.ndim == 0:
                observed_state.append(state_np)
            else:
                observed_state.append(state_np[batch_index])

        sampled_state = list(
            observer.f_transform(
                _expand_moment_observed_state(executor, plan, observed_state),
                observer.blocks_to_sample,
            )
        )
        if sampled_state_by_type is None:
            sampled_state_by_type = [[] for _ in sampled_state]
        for type_index, state in enumerate(sampled_state):
            sampled_state_by_type[type_index].append(np.asarray(state))

    if sampled_state_by_type is None:
        return mem_batch

    sampled_state_batch = [
        np.stack([np.asarray(state).reshape(-1) for state in state_batch], axis=0)
        for state_batch in sampled_state_by_type
    ]
    flat_state_batch = np.zeros(
        (batch_size, plan.flat_node_count),
        dtype=np.result_type(*sampled_state_batch),
    )
    for type_slice, state_batch in zip(plan.flat_to_type_slices_list, sampled_state_batch, strict=True):
        flat_state_batch[:, type_slice] = state_batch

    mem_entry_batch = [
        np.stack([np.asarray(mem[type_index]) for mem in mem_batch], axis=0)
        for type_index in range(len(mem_batch[0]))
    ]
    updated_mem_entry_batch = []
    for mem_entry, sl in zip(mem_entry_batch, plan.flat_to_full_moment_slices, strict=True):
        update = np.prod(flat_state_batch[:, sl], axis=2)
        updated_mem_entry_batch.append(mem_entry.astype(update.dtype) + update)

    return [
        [
            np.asarray(updated_mem_entry_batch[type_index][batch_index])
            for type_index in range(len(updated_mem_entry_batch))
        ]
        for batch_index in range(batch_size)
    ]


def _sample_loaded_batched_spin_moment_job_group_numpy(
    executor,
    schedule: SamplingSchedule,
    *,
    jobs: Sequence[LoadedObservationJob],
    observer: MomentAccumulatorObserver,
    plan: CompiledMomentObserverPlan,
):
    if not jobs:
        return []

    batch_size = len(jobs)
    state_runtime.load_clamp_state_batch(executor, jobs[0].state_clamp, batch_size=batch_size)
    state_runtime.load_free_state_batch(executor, [job.state_free for job in jobs])

    clamped_block_numpy_cache = None
    if executor._enable_grouped_moment_clamped_numpy_cache:
        cached_block_indices = {
            block_index
            for global_slot_index, _ in plan.observation_plan.gathered_view_groups
            for block_index in executor.compiled.global_slots[global_slot_index].block_indices
            if block_index >= len(executor.program.gibbs_spec.free_blocks)
        }
        clamped_block_numpy_cache = {
            block_index: state_runtime.source_slot_numpy_batch_piece(executor, block_index)
            for block_index in cached_block_indices
        }

    sample_roots = []
    warmup_roots = []
    for job in jobs:
        sample_root, warmup_root = jax.random.split(job.key, 2)
        sample_roots.append(sample_root)
        warmup_roots.append(warmup_root)

    executor.run_blocks_batch(warmup_roots, n_iters=schedule.n_warmup)

    mem_batch = [
        [np.asarray(entry).copy() for entry in job.observation_carry_init]
        for job in jobs
    ]
    mem_batch = _accumulate_moment_observation_batch(
        executor,
        observer,
        plan,
        mem_batch,
        observe_compiled_plan_numpy_batch_moment_helper(
            executor,
            plan.observation_plan,
            clamped_block_numpy_cache=clamped_block_numpy_cache,
        ),
    )

    if schedule.n_samples > 1:
        per_job_sample_keys = [
            tuple(jax.random.split(sample_root, schedule.n_samples - 1))
            for sample_root in sample_roots
        ]
        for sample_index in range(schedule.n_samples - 1):
            executor.run_blocks_batch(
                [sample_keys[sample_index] for sample_keys in per_job_sample_keys],
                n_iters=schedule.steps_per_sample,
            )
            mem_batch = _accumulate_moment_observation_batch(
                executor,
                observer,
                plan,
                mem_batch,
                observe_compiled_plan_numpy_batch_moment_helper(
                    executor,
                    plan.observation_plan,
                    clamped_block_numpy_cache=clamped_block_numpy_cache,
                ),
            )

    return [(mem, None) for mem in mem_batch]


def sample_loaded_numpy_moment_observation_jobs(
    executor,
    schedule: SamplingSchedule,
    *,
    jobs: Sequence[LoadedObservationJob],
    observer: MomentAccumulatorObserver,
):
    if not jobs:
        return []

    plan = compile_moment_observer_plan(executor, observer)
    if not _supports_batched_spin_moment_jobs(executor):
        return _run_loaded_jobs(
            executor,
            jobs,
            run_loaded_job=lambda job: sample_moment_observer_numpy(
                executor,
                job.key,
                schedule,
                observer=observer,
                observation_carry_init=job.observation_carry_init,
                plan=plan,
            ),
        )

    results = []
    job_index = 0
    while job_index < len(jobs):
        group_id = jobs[job_index].clamp_group_id
        next_index = job_index + 1
        while (
            group_id is not None
            and next_index < len(jobs)
            and jobs[next_index].clamp_group_id == group_id
        ):
            next_index += 1

        group_jobs = jobs[job_index:next_index]
        if group_id is not None and len(group_jobs) > 1:
            results.extend(
                _sample_loaded_batched_spin_moment_job_group_numpy(
                    executor,
                    schedule,
                    jobs=group_jobs,
                    observer=observer,
                    plan=plan,
                )
            )
        else:
            for job in group_jobs:
                if job.clamp_group_id is not None:
                    state_runtime.load_clamp_state(executor, job.state_clamp)
                    state_runtime.load_free_state(executor, job.state_free)
                else:
                    state_runtime.load_state(executor, job.state_free, job.state_clamp)
                results.append(
                    sample_moment_observer_numpy(
                        executor,
                        job.key,
                        schedule,
                        observer=observer,
                        observation_carry_init=job.observation_carry_init,
                        plan=plan,
                    )
                )

        job_index = next_index

    return results


def sample_loaded_states(
    executor,
    key,
    schedule: SamplingSchedule,
    *,
    nodes_to_sample: Sequence[Block] | None = None,
    observation_plan: CompiledObservationPlan | None = None,
):
    state_runtime.require_state(executor)
    if observation_plan is None:
        if nodes_to_sample is None:
            raise TypeError("Provide either `nodes_to_sample` or `observation_plan`.")
        observation_plan = compile_observation_plan(executor, nodes_to_sample)

    prepared_randoms = executor._prepare_schedule_randoms(key, schedule)
    executor._run_prepared_schedule_warmup(prepared_randoms)

    observations = [observe_compiled_plan_numpy(executor, observation_plan)]
    if schedule.n_samples <= 1:
        return stack_sample_history(observations)

    for _ in executor._iter_prepared_schedule_sample_intervals(prepared_randoms):
        observations.append(observe_compiled_plan_numpy(executor, observation_plan))

    return stack_sample_history(observations)
