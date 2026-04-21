from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import weakref
from typing import Sequence

from thrml.block_sampling import BlockSamplingProgram, SamplingSchedule
from thrml.observers import AbstractObserver, MomentAccumulatorObserver

from ..compiler.sampler_lowering import program_supported_by_executor
from ..fingerprint import program_fingerprint
from .execution_support import LoadedObservationJob, LoadedStateJob
from .mesh_executor import TTMeshProgramExecutor, is_multi_device_mesh
from .program_executor import TTProgramExecutor
from ..runtime_config import (
    BackendBinding,
    ExecutionOptions,
    normalize_execution_options,
)


_COMPILED_PROGRAM_CACHE: dict[
    tuple[int, int, int, tuple[tuple[str, str], ...]], object
] = {}
_EXECUTOR_CACHE: dict[
    tuple[
        int,
        int,
        int,
        tuple[tuple[str, str], ...],
        tuple[tuple[str, int], ...],
    ],
    "_ExecutorCacheEntry",
] = {}
_COORDINATOR_CACHE: dict[
    tuple[
        int,
        int,
        tuple[int, ...],
        tuple[tuple[str, str], ...],
        tuple[tuple[str, int], ...],
    ],
    "_MultiDeviceTTCoordinator",
] = {}
_PROGRAM_CACHE_FINALIZERS: dict[int, weakref.finalize] = {}
_PROGRAM_INSTANCE_TOKENS: dict[int, object] = {}
_CACHE_LOCK = threading.RLock()


@dataclass
class _ExecutorCacheEntry:
    executor: TTProgramExecutor
    lock: threading.RLock


def _validate_program(program: BlockSamplingProgram) -> None:
    if not program_supported_by_executor(program):
        raise TypeError(
            "tt_thrml currently supports THRML discrete EBM Gibbs programs, "
            "explicit TT sampler lowerings for spin/categorical/continuous programs, and "
            "explicit parameter-kernel backends attached through the backend binding."
        )


def _program_instance_token(program: BlockSamplingProgram) -> object:
    program_id = id(program)
    token = _PROGRAM_INSTANCE_TOKENS.get(program_id)
    if token is not None:
        return token

    token = object()
    _PROGRAM_INSTANCE_TOKENS[program_id] = token
    if program_id not in _PROGRAM_CACHE_FINALIZERS:
        _PROGRAM_CACHE_FINALIZERS[program_id] = weakref.finalize(
            program,
            _clear_compiled_program_cache_for_program_id,
            program_id,
        )
    return token


def _make_executor(
    *,
    program: BlockSamplingProgram,
    backend: BackendBinding,
    device,
    options: ExecutionOptions,
):
    _validate_program(program)
    parameter_kernel_ops = backend.parameter_kernel_ops
    parameter_kernel_backends = backend.parameter_kernel_backends
    ttnn = backend.ttnn
    semantic_program_key = program_fingerprint(program)
    program_token = _program_instance_token(program)
    cache_key = (semantic_program_key, *backend.device_cache_key(device))
    executor_cache_key = (
        program_token,
        semantic_program_key,
        *backend.executor_cache_key(device),
    )
    cache_executor = options.cacheable
    if cache_executor:
        with _CACHE_LOCK:
            cached_entry = _EXECUTOR_CACHE.get(executor_cache_key)
        if cached_entry is not None:
            return cached_entry.executor

    with _CACHE_LOCK:
        compiled = _COMPILED_PROGRAM_CACHE.get(cache_key)

    executor_cls = TTMeshProgramExecutor if is_multi_device_mesh(device) else TTProgramExecutor

    if compiled is not None:
        executor = executor_cls(
            ttnn=ttnn,
            device=device,
            program=program,
            compiled=compiled,
            parameter_kernel_ops=parameter_kernel_ops,
            parameter_kernel_backends=parameter_kernel_backends,
            profiler=options.profiler,
            profile_sync=options.profile_sync,
            progress=options.progress,
        )
    else:
        executor = executor_cls(
            ttnn=ttnn,
            device=device,
            program=program,
            parameter_kernel_ops=parameter_kernel_ops,
            parameter_kernel_backends=parameter_kernel_backends,
            profiler=options.profiler,
            profile_sync=options.profile_sync,
            progress=options.progress,
        )
        with _CACHE_LOCK:
            _COMPILED_PROGRAM_CACHE[cache_key] = executor.compiled

    if cache_executor:
        with _CACHE_LOCK:
            _EXECUTOR_CACHE[executor_cache_key] = _ExecutorCacheEntry(
                executor=executor,
                lock=threading.RLock(),
            )

    return executor


@contextmanager
def borrow_executor(
    *,
    program: BlockSamplingProgram,
    backend: BackendBinding,
    device=None,
    options: ExecutionOptions | None = None,
):
    options = normalize_execution_options(options)
    device_for_executor = backend.primary_device if device is None else device
    executor = _make_executor(
        program=program,
        backend=backend,
        device=device_for_executor,
        options=options,
    )
    cache_executor = options.cacheable
    if not cache_executor:
        yield executor
        return

    semantic_program_key = program_fingerprint(program)
    executor_cache_key = (
        _program_instance_token(program),
        semantic_program_key,
        *backend.executor_cache_key(device_for_executor),
    )
    with _CACHE_LOCK:
        entry = _EXECUTOR_CACHE[executor_cache_key]
    with entry.lock:
        yield executor


def _clear_compiled_program_cache_for_program_id(program_id: int) -> None:
    program_token = _PROGRAM_INSTANCE_TOKENS.pop(program_id, None)
    if program_token is not None:
        stale_executor_keys = [key for key in _EXECUTOR_CACHE if key[0] is program_token]
        for key in stale_executor_keys:
            _EXECUTOR_CACHE.pop(key, None)
        stale_coordinator_keys = [key for key in _COORDINATOR_CACHE if key[0] is program_token]
        for key in stale_coordinator_keys:
            coordinator = _COORDINATOR_CACHE.pop(key, None)
            if coordinator is not None:
                coordinator.shutdown()
    _PROGRAM_CACHE_FINALIZERS.pop(program_id, None)


def clear_compiled_program_cache() -> None:
    with _CACHE_LOCK:
        _COMPILED_PROGRAM_CACHE.clear()
        _EXECUTOR_CACHE.clear()
        coordinators = tuple(_COORDINATOR_CACHE.values())
        _COORDINATOR_CACHE.clear()
        for finalizer in _PROGRAM_CACHE_FINALIZERS.values():
            finalizer.detach()
        _PROGRAM_CACHE_FINALIZERS.clear()
        _PROGRAM_INSTANCE_TOKENS.clear()
    for coordinator in coordinators:
        coordinator.shutdown()


class _MultiDeviceTTCoordinator:
    def __init__(
        self,
        *,
        program: BlockSamplingProgram,
        backend: BackendBinding,
        options: ExecutionOptions,
    ):
        self.program = program
        self.backend = backend
        self.ttnn = backend.ttnn
        self.devices = backend.devices
        self.parameter_kernel_ops = backend.parameter_kernel_ops
        self.profiler = options.profiler
        self.profile_sync = options.profile_sync
        self.progress = options.progress
        self._thread_pool = (
            None if len(self.devices) <= 1 else ThreadPoolExecutor(max_workers=len(self.devices))
        )

    def sample_loaded_observation_many(
        self,
        schedule: SamplingSchedule,
        *,
        jobs: Sequence[LoadedObservationJob],
        f_observe: AbstractObserver,
        preserve_clamp_groups: bool,
    ):
        return self._run_many_loaded_jobs(
            jobs=jobs,
            preserve_clamp_groups=preserve_clamp_groups,
            run_loaded_jobs=lambda executor, queue_jobs: executor.sample_loaded_observation_jobs(
                schedule,
                jobs=queue_jobs,
                f_observe=f_observe,
            ),
        )

    def sample_loaded_numpy_moment_observation_many(
        self,
        schedule: SamplingSchedule,
        *,
        jobs: Sequence[LoadedObservationJob],
        observer: MomentAccumulatorObserver,
        preserve_clamp_groups: bool,
    ):
        return self._run_many_loaded_jobs(
            jobs=jobs,
            preserve_clamp_groups=preserve_clamp_groups,
            run_loaded_jobs=lambda executor, queue_jobs: executor.sample_loaded_numpy_moment_observation_jobs(
                schedule,
                jobs=queue_jobs,
                observer=observer,
            ),
        )

    def sample_loaded_states_many(
        self,
        schedule: SamplingSchedule,
        *,
        jobs: Sequence[LoadedStateJob],
        nodes_to_sample: Sequence,
        preserve_clamp_groups: bool,
    ):
        return self._run_many_loaded_jobs(
            jobs=jobs,
            preserve_clamp_groups=preserve_clamp_groups,
            run_loaded_jobs=lambda executor, queue_jobs: executor.sample_loaded_state_jobs(
                schedule,
                jobs=queue_jobs,
                nodes_to_sample=nodes_to_sample,
            ),
        )

    def _partition_loaded_jobs(
        self,
        *,
        jobs,
        preserve_clamp_groups: bool,
    ):
        if not jobs:
            return []

        queues: list[list[tuple[int, object]]] = [[] for _ in self.devices]
        if not preserve_clamp_groups:
            for job_index, job in enumerate(jobs):
                queues[job_index % len(self.devices)].append((job_index, job))
            return queues

        grouped_jobs: dict[object, list[tuple[int, object]]] = {}
        for job_index, job in enumerate(jobs):
            grouped_jobs.setdefault(job.clamp_group_id, []).append((job_index, job))

        queue_loads = [0] * len(self.devices)
        target_queue_size = max(1, (len(jobs) + len(self.devices) - 1) // len(self.devices))

        for group_jobs in grouped_jobs.values():
            if len(group_jobs) <= target_queue_size:
                target_queue_index = min(
                    range(len(self.devices)),
                    key=lambda index: (queue_loads[index], index),
                )
                queues[target_queue_index].extend(group_jobs)
                queue_loads[target_queue_index] += len(group_jobs)
                continue

            partition_count = min(
                len(self.devices),
                max(1, (len(group_jobs) + target_queue_size - 1) // target_queue_size),
            )
            target_queue_indices = sorted(
                range(len(self.devices)),
                key=lambda index: (queue_loads[index], index),
            )[:partition_count]
            chunk_size = (len(group_jobs) + partition_count - 1) // partition_count
            for chunk_index, chunk_start in enumerate(range(0, len(group_jobs), chunk_size)):
                target_queue_index = target_queue_indices[chunk_index]
                chunk = group_jobs[chunk_start : chunk_start + chunk_size]
                queues[target_queue_index].extend(chunk)
                queue_loads[target_queue_index] += len(chunk)

        return queues

    def _run_many_loaded_jobs(
        self,
        *,
        jobs,
        preserve_clamp_groups: bool,
        run_loaded_jobs,
    ):
        if not jobs:
            return []

        queues = self._partition_loaded_jobs(
            jobs=jobs,
            preserve_clamp_groups=preserve_clamp_groups,
        )
        results: list[object | None] = [None] * len(jobs)

        def _run_queue(device_for_queue, queue):
            with borrow_executor(
                program=self.program,
                backend=self.backend,
                device=device_for_queue,
                options=ExecutionOptions(
                    profiler=self.profiler,
                    profile_sync=self.profile_sync,
                    progress=self.progress,
                ),
            ) as queue_executor:
                queue_results = run_loaded_jobs(
                    queue_executor,
                    [job for _, job in queue],
                )
                for (job_index, _), result in zip(queue, queue_results, strict=True):
                    results[job_index] = result

        if len(self.devices) == 1:
            _run_queue(self.devices[0], queues[0])
            return list(results)

        non_empty_queues = [
            (device_for_queue, queue)
            for device_for_queue, queue in zip(self.devices, queues, strict=True)
            if queue
        ]
        if len(non_empty_queues) == 1:
            device_for_queue, queue = non_empty_queues[0]
            _run_queue(device_for_queue, queue)
            return list(results)

        assert self._thread_pool is not None
        futures = [
            self._thread_pool.submit(_run_queue, device_for_queue, queue)
            for device_for_queue, queue in non_empty_queues
        ]
        for future in futures:
            future.result()

        return list(results)

    def shutdown(self) -> None:
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None


def make_coordinator(
    *,
    program: BlockSamplingProgram,
    backend: BackendBinding,
    options: ExecutionOptions | None = None,
) -> _MultiDeviceTTCoordinator:
    options = normalize_execution_options(options)
    devices = backend.devices
    cacheable = options.cacheable
    if not cacheable:
        return _MultiDeviceTTCoordinator(
            program=program,
            backend=backend,
            options=options,
        )

    semantic_program_key = program_fingerprint(program)
    cache_key = (_program_instance_token(program), semantic_program_key, *backend.cache_key)
    with _CACHE_LOCK:
        cached = _COORDINATOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    coordinator = _MultiDeviceTTCoordinator(
        program=program,
        backend=backend,
        options=options,
    )
    with _CACHE_LOCK:
        _COORDINATOR_CACHE[cache_key] = coordinator
    return coordinator
