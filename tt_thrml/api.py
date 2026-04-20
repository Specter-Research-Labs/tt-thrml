"""Public TT-backed THRML sampling API."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np

from thrml.block_sampling import (
    BlockSamplingProgram,
    SamplingSchedule,
)
from thrml.observers import AbstractObserver, MomentAccumulatorObserver

from .runtime.backend_executor import borrow_executor as _borrow_executor
from .runtime.backend_executor import clear_compiled_program_cache as _clear_compiled_program_cache
from .runtime.backend_executor import make_coordinator
from .runtime.execution_support import LoadedObservationJob
from .runtime.execution_support import LoadedStateJob
from .runtime.program_executor import TTProgramExecutor
from .runtime_config import (
    BackendBinding,
    ExecutionOptions,
    normalize_execution_options,
    require_backend,
)

__all__ = [
    "clear_compiled_program_cache",
    "sample_states",
    "sample_states_many",
    "sample_with_observation",
    "sample_with_observation_many",
]


def _emit_progress(progress, message: str) -> None:
    if progress is None:
        return
    progress(message)


def _resolve_execution(
    *,
    backend: BackendBinding | None = None,
    options: ExecutionOptions | None = None,
) -> tuple[BackendBinding, ExecutionOptions]:
    return require_backend(backend), normalize_execution_options(options)


def clear_compiled_program_cache() -> None:
    _clear_compiled_program_cache()


def _clamp_group_key(state_clamp) -> tuple[tuple[str, tuple[int, ...], bytes], ...]:
    return tuple(
        (
            str(array.dtype),
            tuple(array.shape),
            np.asarray(array).tobytes(),
        )
        for array in [np.asarray(value) for value in state_clamp]
    )


@contextmanager
def _borrow_single_job_executor(
    *,
    program: BlockSamplingProgram,
    progress_name: str,
    backend: BackendBinding | None = None,
    options: ExecutionOptions | None = None,
):
    binding, resolved_options = _resolve_execution(backend=backend, options=options)
    _emit_progress(
        resolved_options.progress,
        f"[tt-thrml.api] {progress_name}.executor_init:start",
    )
    with _borrow_executor(
        program=program,
        backend=binding,
        options=resolved_options,
    ) as executor:
        _emit_progress(
            resolved_options.progress,
            f"[tt-thrml.api] {progress_name}.executor_init:done",
        )
        yield executor


def _run_single_executor_job(
    *,
    program: BlockSamplingProgram,
    progress_name: str,
    backend: BackendBinding | None = None,
    options: ExecutionOptions | None = None,
    emit_run_progress: bool = True,
    run,
):
    _, resolved_options = _resolve_execution(backend=backend, options=options)
    with _borrow_single_job_executor(
        program=program,
        progress_name=progress_name,
        backend=backend,
        options=resolved_options,
    ) as executor:
        if emit_run_progress:
            _emit_progress(
                resolved_options.progress,
                f"[tt-thrml.api] {progress_name}.run:start",
            )
        result = run(executor, resolved_options)
        if emit_run_progress:
            _emit_progress(
                resolved_options.progress,
                f"[tt-thrml.api] {progress_name}.run:done",
            )
        return result


def _run_many_jobs(
    *,
    program: BlockSamplingProgram,
    progress_name: str,
    n_jobs: int,
    backend: BackendBinding | None = None,
    options: ExecutionOptions | None = None,
    run,
):
    binding, resolved_options = _resolve_execution(backend=backend, options=options)
    _emit_progress(
        resolved_options.progress,
        (
            f"[tt-thrml.api] {progress_name}:start "
            f"jobs={n_jobs} devices={len(binding.devices)}"
        ),
    )
    coordinator = make_coordinator(
        program=program,
        backend=binding,
        options=resolved_options,
    )
    result = run(coordinator)
    _emit_progress(
        resolved_options.progress,
        f"[tt-thrml.api] {progress_name}:done",
    )
    return result


def _auto_clamp_group_ids(state_clamps: Sequence) -> list[int]:
    group_index_by_identity: dict[tuple[int, ...], int] = {}
    group_index_by_key: dict[tuple[tuple[str, tuple[int, ...], bytes], ...], int] = {}
    group_ids = []
    for state_clamp in state_clamps:
        identity_key = tuple(id(value) for value in state_clamp)
        if identity_key in group_index_by_identity:
            group_ids.append(group_index_by_identity[identity_key])
            continue
        key = _clamp_group_key(state_clamp)
        group_id = group_index_by_key.setdefault(key, len(group_index_by_key))
        group_index_by_identity[identity_key] = group_id
        group_ids.append(group_id)
    return group_ids


def _sample_with_observation_executor(
    executor: TTProgramExecutor,
    *,
    key,
    schedule: SamplingSchedule,
    init_chain_state,
    state_clamp,
    observation_carry_init,
    f_observe: AbstractObserver,
    progress=None,
):
    _emit_progress(progress, "[tt-thrml.api] sample_with_observation.load_state:start")
    executor.load_state(init_chain_state, state_clamp)
    _emit_progress(progress, "[tt-thrml.api] sample_with_observation.load_state:done")
    _emit_progress(progress, "[tt-thrml.api] sample_with_observation.run:start")
    observed = executor.sample_loaded_observation(
        key,
        schedule,
        state_clamp=state_clamp,
        observation_carry_init=observation_carry_init,
        f_observe=f_observe,
    )
    _emit_progress(progress, "[tt-thrml.api] sample_with_observation.run:done")
    return observed


def sample_with_observation_many(
    keys: Sequence,
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_chain_states: Sequence,
    state_clamps: Sequence,
    observation_carry_inits: Sequence,
    f_observe: AbstractObserver,
    *,
    backend: BackendBinding | None = None,
    options: ExecutionOptions | None = None,
):
    if not (
        len(keys)
        == len(init_chain_states)
        == len(state_clamps)
        == len(observation_carry_inits)
    ):
        raise ValueError(
            "keys, init_chain_states, state_clamps, and observation_carry_inits "
            "must have the same length."
        )

    clamp_group_ids = _auto_clamp_group_ids(state_clamps)
    jobs = [
        LoadedObservationJob(
            key=key,
            state_free=init_chain_state,
            state_clamp=state_clamp,
            observation_carry_init=observation_carry_init,
            clamp_group_id=clamp_group_id,
        )
        for key, init_chain_state, state_clamp, observation_carry_init, clamp_group_id in zip(
            keys,
            init_chain_states,
            state_clamps,
            observation_carry_inits,
            clamp_group_ids,
            strict=True,
        )
    ]
    return _run_many_jobs(
        program=program,
        progress_name="sample_with_observation_many",
        n_jobs=len(jobs),
        backend=backend,
        options=options,
        run=lambda coordinator: (
            coordinator.sample_loaded_numpy_moment_observation_many(
                schedule=schedule,
                jobs=jobs,
                observer=f_observe,
                preserve_clamp_groups=True,
            )
            if isinstance(f_observe, MomentAccumulatorObserver)
            else coordinator.sample_loaded_observation_many(
                schedule=schedule,
                jobs=jobs,
                f_observe=f_observe,
                preserve_clamp_groups=True,
            )
        ),
    )


def sample_states_many(
    keys: Sequence,
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state_frees: Sequence,
    state_clamps: Sequence,
    nodes_to_sample: Sequence,
    *,
    backend: BackendBinding | None = None,
    options: ExecutionOptions | None = None,
):
    if not (len(keys) == len(init_state_frees) == len(state_clamps)):
        raise ValueError(
            "keys, init_state_frees, and state_clamps must have the same length."
        )

    jobs = list(zip(keys, init_state_frees, state_clamps, strict=True))
    loaded_jobs = [
        LoadedStateJob(
            key=key,
            state_free=init_state_free,
            state_clamp=state_clamp,
        )
        for key, init_state_free, state_clamp in jobs
    ]
    return _run_many_jobs(
        program=program,
        progress_name="sample_states_many",
        n_jobs=len(loaded_jobs),
        backend=backend,
        options=options,
        run=lambda coordinator: coordinator.sample_loaded_states_many(
            schedule=schedule,
            jobs=loaded_jobs,
            nodes_to_sample=nodes_to_sample,
            preserve_clamp_groups=False,
        ),
    )


def sample_with_observation(
    key,
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_chain_state,
    state_clamp,
    observation_carry_init,
    f_observe: AbstractObserver,
    *,
    backend: BackendBinding | None = None,
    options: ExecutionOptions | None = None,
):
    return _run_single_executor_job(
        program=program,
        progress_name="sample_with_observation",
        backend=backend,
        options=options,
        emit_run_progress=False,
        run=lambda executor, resolved_options: _sample_with_observation_executor(
            executor,
            key=key,
            schedule=schedule,
            init_chain_state=init_chain_state,
            state_clamp=state_clamp,
            observation_carry_init=observation_carry_init,
            f_observe=f_observe,
            progress=resolved_options.progress,
        ),
    )


def sample_states(
    key,
    program: BlockSamplingProgram,
    schedule: SamplingSchedule,
    init_state_free,
    state_clamp,
    nodes_to_sample: Sequence,
    *,
    backend: BackendBinding | None = None,
    options: ExecutionOptions | None = None,
):
    return _run_single_executor_job(
        program=program,
        progress_name="sample_states",
        backend=backend,
        options=options,
        run=lambda executor, _resolved_options: executor.sample_states(
            key=key,
            schedule=schedule,
            nodes_to_sample=nodes_to_sample,
            init_state_free=init_state_free,
            state_clamp=state_clamp,
        ),
    )
