from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    SpinEBMFactor,
    SpinGibbsConditional,
)
from thrml.pgm import AbstractNode, CategoricalNode, SpinNode

try:
    import torch  # type: ignore  # noqa: F401
except ImportError:
    from tests.parity._torch_stub import install_torch_stub

    torch = install_torch_stub()

import tt_thrml
from tt_thrml.compiler.ttmlir import categorical_theta as ttmlir_categorical_theta
from tt_thrml.compiler.ttmlir import gaussian_canonical as ttmlir_gaussian_canonical
from tt_thrml.compiler.ttmlir import runtime as ttmlir_runtime
from tt_thrml.compiler.ttmlir import spin_gamma as ttmlir_spin_gamma
from tt_thrml.conditional_samplers import GaussianConditional
from tt_thrml.runtime.execution_support import RuntimeProfile


pytestmark = [pytest.mark.hardware, pytest.mark.slow]


class ContinuousNode(AbstractNode):
    pass


class LinearInteraction(eqx.Module):
    weights: jax.Array


class QuadraticInteraction(eqx.Module):
    inverse_weights: jax.Array


class LinearFactor(AbstractFactor):
    weights: jax.Array

    def __init__(self, weights: jax.Array, block: Block):
        super().__init__([block])
        self.weights = weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[],
            )
        ]


class QuadraticFactor(AbstractFactor):
    inverse_weights: jax.Array

    def __init__(self, inverse_weights: jax.Array, block: Block):
        super().__init__([block])
        self.inverse_weights = inverse_weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=QuadraticInteraction(self.inverse_weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[],
            )
        ]


class CouplingFactor(AbstractFactor):
    weights: jax.Array

    def __init__(self, weights: jax.Array, blocks: tuple[Block, Block]):
        super().__init__(list(blocks))
        self.weights = weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[self.node_groups[1]],
            ),
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[1],
                tail_nodes=[self.node_groups[0]],
            ),
        ]


@dataclass(frozen=True)
class HardwareCase:
    name: str
    program: FactorSamplingProgram
    init_state_free: list[object]
    state_clamp: list[object]
    nodes_to_sample: list[Block]
    single_sample_schedule: SamplingSchedule
    steady_state_schedule: SamplingSchedule


@dataclass
class TransferSnapshot:
    host_to_device_transfers: int
    device_to_host_transfers: int


class TransferCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._host_to_device_transfers = 0
        self._device_to_host_transfers = 0

    def add_host_to_device(self, count: int = 1) -> None:
        with self._lock:
            self._host_to_device_transfers += count

    def add_device_to_host(self, count: int = 1) -> None:
        with self._lock:
            self._device_to_host_transfers += count

    def snapshot(self) -> TransferSnapshot:
        with self._lock:
            return TransferSnapshot(
                host_to_device_transfers=self._host_to_device_transfers,
                device_to_host_transfers=self._device_to_host_transfers,
            )


@dataclass
class RunMetrics:
    wall_seconds: float
    total_sample_count: int
    dispatch_count: int
    dispatch_count_by_family: dict[str, int]
    host_to_device_transfers: int
    device_to_host_transfers: int
    peak_device_memory_bytes: int | None
    profile: dict[str, dict[str, float | int]]
    official_artifacts: dict[str, str]


class ThreadSafeRuntimeProfile:
    def __init__(self):
        self._inner = RuntimeProfile()
        self._lock = threading.Lock()

    def record(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._inner.record(stage, seconds)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            return self._inner.snapshot()


class ProfilingTTNNProxy:
    def __init__(self, inner, *, transfers: TransferCounter):
        self._inner = inner
        self._transfers = transfers

    @property
    def transfers(self) -> TransferCounter:
        return self._transfers

    def snapshot_transfers(self) -> TransferSnapshot:
        return self._transfers.snapshot()

    def from_torch(self, value, *args, **kwargs):
        self._transfers.add_host_to_device()
        if _using_torch_stub() and isinstance(value, torch.Tensor):
            dtype = kwargs.get("dtype", args[0] if args else None)
            mesh_mapper = kwargs.get("mesh_mapper")
            if hasattr(mesh_mapper, "unwrap"):
                mesh_mapper = mesh_mapper.unwrap()
            return self._inner.Tensor(
                tensor=np.asarray(value),
                data_type=dtype,
                device=kwargs.get("device"),
                layout=kwargs.get("layout"),
                mem_config=kwargs.get("memory_config"),
                tile=kwargs.get("tile"),
                cq_id=kwargs.get("cq_id"),
                pad_value=kwargs.get("pad_value"),
                mesh_mapper=mesh_mapper,
                preserve_nan_values=kwargs.get("preserve_nan_values", False),
                col_tilize=kwargs.get("col_tilize", False),
                fast_approx=kwargs.get("fast_approx", False),
            )
        return self._inner.from_torch(value, *args, **kwargs)

    def to_torch(self, value, *args, **kwargs):
        self._transfers.add_device_to_host()
        if _using_torch_stub():
            is_on_device = getattr(self._inner, "is_tensor_storage_on_device", None)
            host_value = value
            if callable(is_on_device) and is_on_device(value):
                host_value = self._inner.from_device(value, queue_id=kwargs.get("cq_id"))
            numpy_value = host_value.to_numpy(mesh_composer=kwargs.get("mesh_composer"))
            tensor = torch.from_numpy(np.asarray(numpy_value))
            dtype = kwargs.get("dtype", args[0] if args else None)
            if dtype is not None:
                tensor = tensor.to(dtype)
            torch_rank = kwargs.get("torch_rank")
            if torch_rank is not None:
                while tensor.ndim > int(torch_rank):
                    tensor = tensor.squeeze(0)
            return tensor
        return self._inner.to_torch(value, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _using_torch_stub() -> bool:
    return bool(getattr(torch, "_TT_THRML_TORCH_STUB", False))


def _maybe_tracy_signpost(*, header: str, message: str | None = None) -> None:
    try:
        tracy = importlib.import_module("tracy")
    except Exception:
        return
    signpost = getattr(tracy, "signpost", None)
    if not callable(signpost):
        return
    try:
        if message is None:
            signpost(header=header)
        else:
            signpost(header=header, message=message)
    except Exception:
        return


@contextmanager
def _configure_ttnn_reports(*, report_root: Path, report_name: str):
    if _using_torch_stub():
        yield lambda: None
        return

    try:
        ttnn = importlib.import_module("ttnn")
    except Exception:
        yield lambda: None
        return

    config = getattr(ttnn, "CONFIG", None)
    if config is None:
        yield lambda: None
        return

    report_root.mkdir(parents=True, exist_ok=True)
    saved = {
        field_name: getattr(config, field_name)
        for field_name in (
            "enable_fast_runtime_mode",
            "enable_logging",
            "enable_graph_report",
            "enable_detailed_buffer_report",
            "enable_detailed_tensor_report",
            "root_report_path",
            "report_name",
        )
        if hasattr(config, field_name)
    }
    try:
        if hasattr(config, "enable_fast_runtime_mode"):
            config.enable_fast_runtime_mode = False
        if hasattr(config, "enable_logging"):
            config.enable_logging = True
        if hasattr(config, "enable_graph_report"):
            config.enable_graph_report = True
        if hasattr(config, "enable_detailed_buffer_report"):
            config.enable_detailed_buffer_report = True
        if hasattr(config, "enable_detailed_tensor_report"):
            config.enable_detailed_tensor_report = True
        if hasattr(config, "root_report_path"):
            config.root_report_path = str(report_root)
        if hasattr(config, "report_name"):
            config.report_name = report_name
        yield lambda: _resolve_ttnn_report_path(ttnn=ttnn, report_root=report_root)
    finally:
        for field_name, value in saved.items():
            if field_name == "report_name" and value is None:
                continue
            setattr(config, field_name, value)


def _resolve_ttnn_report_path(*, ttnn, report_root: Path) -> str | None:
    configured_path = getattr(getattr(ttnn, "CONFIG", None), "report_path", None)
    if configured_path:
        path = Path(str(configured_path)).resolve()
        if path.exists():
            return str(path)
    if not report_root.exists():
        return None
    candidates = [path for path in report_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def _maybe_read_device_profiler(ttnn_proxy, devices: tuple[object, ...]) -> None:
    if os.environ.get("TT_METAL_DEVICE_PROFILER") not in {"1", "true", "True"}:
        return
    device_namespace = getattr(ttnn_proxy, "device", None)
    read_device_profiler = getattr(device_namespace, "ReadDeviceProfiler", None)
    if not callable(read_device_profiler):
        return
    for device in devices:
        try:
            read_device_profiler(device)
        except Exception:
            continue


def _latest_existing_path(paths: list[Path]) -> str | None:
    if not paths:
        return None
    return str(max(paths, key=lambda path: path.stat().st_mtime))


def _discover_official_artifacts(
    *,
    artifact_root: Path,
    ttnn_report_path: str | None,
) -> dict[str, str]:
    paths_by_key: dict[str, list[Path]] = {
        "ttnn_report_path": [],
        "profiler_report_path": [],
        "device_profile_log_path": [],
        "tracy_profile_log_path": [],
    }
    if ttnn_report_path is not None:
        path = Path(ttnn_report_path)
        if path.exists():
            paths_by_key["ttnn_report_path"].append(path)

    search_roots = [Path.cwd(), artifact_root]
    seen_roots: set[Path] = set()
    for root in search_roots:
        resolved_root = root.resolve()
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)

        profiler_reports_root = resolved_root / "generated" / "profiler" / "reports"
        if profiler_reports_root.exists():
            paths_by_key["profiler_report_path"].extend(
                path for path in profiler_reports_root.iterdir() if path.is_dir()
            )

        device_profile_log = (
            resolved_root / "generated" / "profiler" / ".logs" / "profile_log_device.csv"
        )
        if device_profile_log.exists():
            paths_by_key["device_profile_log_path"].append(device_profile_log)

        tracy_profile_log = (
            resolved_root / "generated" / "profiler" / ".logs" / "tracy_profile_log_host.tracy"
        )
        if tracy_profile_log.exists():
            paths_by_key["tracy_profile_log_path"].append(tracy_profile_log)

    return {
        key: latest_path
        for key, latest_path in (
            (key, _latest_existing_path(paths)) for key, paths in paths_by_key.items()
        )
        if latest_path is not None
    }


@contextmanager
def _count_runtime_transfers(transfers: TransferCounter):
    original_create_runtime_input = ttmlir_runtime._create_runtime_input
    original_runtime_outputs_to_torch = ttmlir_runtime._runtime_outputs_to_torch

    def _wrap_create_runtime_input(*args, _original=original_create_runtime_input, **kwargs):
        transfers.add_host_to_device()
        return _original(*args, **kwargs)

    def _wrap_runtime_outputs_to_torch(
        *args,
        _original=original_runtime_outputs_to_torch,
        **kwargs,
    ):
        outputs, runtime_output_dtypes = _original(*args, **kwargs)
        transfers.add_device_to_host(len(outputs))
        return outputs, runtime_output_dtypes

    ttmlir_runtime._create_runtime_input = _wrap_create_runtime_input
    ttmlir_runtime._runtime_outputs_to_torch = _wrap_runtime_outputs_to_torch
    try:
        yield
    finally:
        ttmlir_runtime._create_runtime_input = original_create_runtime_input
        ttmlir_runtime._runtime_outputs_to_torch = original_runtime_outputs_to_torch


@contextmanager
def _count_ttmlir_dispatches():
    modules = (
        ("spin", ttmlir_spin_gamma),
        ("categorical", ttmlir_categorical_theta),
        ("gaussian", ttmlir_gaussian_canonical),
    )
    counts: dict[str, int] = {family: 0 for family, _ in modules}
    lock = threading.Lock()
    originals = []

    for family, module in modules:
        original = module.run_flatbuffer

        def _wrap(*args, _family=family, _original=original, **kwargs):
            with lock:
                counts[_family] += 1
            return _original(*args, **kwargs)

        originals.append((module, original))
        module.run_flatbuffer = _wrap

    try:
        yield counts
    finally:
        for module, original in originals:
            module.run_flatbuffer = original


def _clear_ttmlir_caches() -> None:
    tt_thrml.clear_compiled_program_cache()
    ttmlir_runtime._close_cached_runtime_sessions()
    with ttmlir_runtime._COMPILED_ARTIFACT_CACHE_LOCK:
        ttmlir_runtime._COMPILED_ARTIFACT_CACHE.clear()
        ttmlir_runtime._COMPILED_ARTIFACT_COMPILE_LOCKS.clear()


def _clone_state_blocks(blocks: list[object]) -> list[object]:
    return [jnp.asarray(np.asarray(block)).copy() for block in blocks]


def _parse_device_ids(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return (0,)
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_mesh_shape(value: str | None, *, n_devices: int) -> tuple[int, int]:
    if value is None or not value.strip():
        return (1, n_devices)
    normalized = value.lower().replace("x", ",")
    dims = tuple(int(part.strip()) for part in normalized.split(",") if part.strip())
    if len(dims) != 2:
        raise ValueError("TT_THRML_TEST_MESH_SHAPE must have exactly two dimensions.")
    return dims[0], dims[1]


def _maybe_peak_device_memory_bytes(ttnn_proxy, device) -> int | None:
    query_names = (
        "get_peak_allocated_memory_bytes",
        "get_peak_memory_usage",
        "get_device_memory_usage",
        "get_memory_usage",
    )
    for query_name in query_names:
        query = getattr(ttnn_proxy, query_name, None)
        if not callable(query):
            continue
        try:
            value = query(device)
        except TypeError:
            try:
                value = query()
            except Exception:
                continue
        except Exception:
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, dict):
            for key in ("peak_bytes", "peak", "bytes"):
                if isinstance(value.get(key), int):
                    return value[key]
    return None


def _transfer_delta(
    start: TransferSnapshot,
    end: TransferSnapshot,
) -> tuple[int, int]:
    return (
        end.host_to_device_transfers - start.host_to_device_transfers,
        end.device_to_host_transfers - start.device_to_host_transfers,
    )


def _run_single_job(
    *,
    case: HardwareCase,
    backend,
    options,
    ttnn_proxy: ProfilingTTNNProxy,
    peak_memory_device,
    profiler_devices: tuple[object, ...],
    artifact_root: Path,
    key,
    schedule: SamplingSchedule,
    phase_label: str,
) -> RunMetrics:
    transfer_start = ttnn_proxy.snapshot_transfers()
    peak_before = _maybe_peak_device_memory_bytes(ttnn_proxy, peak_memory_device)
    report_root = artifact_root / "generated" / "ttnn" / "reports"
    report_name = f"{case.name}_{phase_label}"
    with (
        _configure_ttnn_reports(report_root=report_root, report_name=report_name) as resolve_ttnn_report_path,
        _count_runtime_transfers(ttnn_proxy.transfers),
        _count_ttmlir_dispatches() as dispatch_counts,
    ):
        _maybe_tracy_signpost(
            header=f"tt-thrml wormhole {phase_label}",
            message=case.name,
        )
        start = time.perf_counter()
        outputs = tt_thrml.sample_states(
            key,
            case.program,
            schedule,
            _clone_state_blocks(case.init_state_free),
            _clone_state_blocks(case.state_clamp),
            case.nodes_to_sample,
            backend=backend,
            options=options,
        )
        wall_seconds = time.perf_counter() - start
        _maybe_read_device_profiler(ttnn_proxy, profiler_devices)
        ttnn_report_path = resolve_ttnn_report_path()
    peak_after = _maybe_peak_device_memory_bytes(ttnn_proxy, peak_memory_device)
    transfer_end = ttnn_proxy.snapshot_transfers()
    host_to_device, device_to_host = _transfer_delta(transfer_start, transfer_end)
    assert len(outputs) == len(case.nodes_to_sample)
    peak_values = [value for value in (peak_before, peak_after) if value is not None]
    return RunMetrics(
        wall_seconds=wall_seconds,
        total_sample_count=int(schedule.n_samples),
        dispatch_count=sum(dispatch_counts.values()),
        dispatch_count_by_family={key: value for key, value in dispatch_counts.items() if value},
        host_to_device_transfers=host_to_device,
        device_to_host_transfers=device_to_host,
        peak_device_memory_bytes=max(peak_values) if peak_values else None,
        profile=options.profiler.snapshot(),
        official_artifacts=_discover_official_artifacts(
            artifact_root=artifact_root,
            ttnn_report_path=ttnn_report_path,
        ),
    )


def _run_many_jobs(
    *,
    case: HardwareCase,
    backend,
    options,
    ttnn_proxy: ProfilingTTNNProxy,
    peak_memory_device,
    profiler_devices: tuple[object, ...],
    artifact_root: Path,
    keys,
    schedule: SamplingSchedule,
    phase_label: str,
) -> RunMetrics:
    init_state_frees = [_clone_state_blocks(case.init_state_free) for _ in keys]
    state_clamps = [_clone_state_blocks(case.state_clamp) for _ in keys]
    transfer_start = ttnn_proxy.snapshot_transfers()
    peak_before = _maybe_peak_device_memory_bytes(ttnn_proxy, peak_memory_device)
    report_root = artifact_root / "generated" / "ttnn" / "reports"
    report_name = f"{case.name}_{phase_label}"
    with (
        _configure_ttnn_reports(report_root=report_root, report_name=report_name) as resolve_ttnn_report_path,
        _count_runtime_transfers(ttnn_proxy.transfers),
        _count_ttmlir_dispatches() as dispatch_counts,
    ):
        _maybe_tracy_signpost(
            header=f"tt-thrml wormhole {phase_label}",
            message=case.name,
        )
        start = time.perf_counter()
        outputs = tt_thrml.sample_states_many(
            keys,
            case.program,
            schedule,
            init_state_frees,
            state_clamps,
            case.nodes_to_sample,
            backend=backend,
            options=options,
        )
        wall_seconds = time.perf_counter() - start
        _maybe_read_device_profiler(ttnn_proxy, profiler_devices)
        ttnn_report_path = resolve_ttnn_report_path()
    peak_after = _maybe_peak_device_memory_bytes(ttnn_proxy, peak_memory_device)
    transfer_end = ttnn_proxy.snapshot_transfers()
    host_to_device, device_to_host = _transfer_delta(transfer_start, transfer_end)
    assert len(outputs) == len(keys)
    peak_values = [value for value in (peak_before, peak_after) if value is not None]
    return RunMetrics(
        wall_seconds=wall_seconds,
        total_sample_count=int(schedule.n_samples) * len(keys),
        dispatch_count=sum(dispatch_counts.values()),
        dispatch_count_by_family={key: value for key, value in dispatch_counts.items() if value},
        host_to_device_transfers=host_to_device,
        device_to_host_transfers=device_to_host,
        peak_device_memory_bytes=max(peak_values) if peak_values else None,
        profile=options.profiler.snapshot(),
        official_artifacts=_discover_official_artifacts(
            artifact_root=artifact_root,
            ttnn_report_path=ttnn_report_path,
        ),
    )


def _estimate_case_metrics(
    *,
    case: HardwareCase,
    runner: Callable[..., RunMetrics],
    ttnn_proxy: ProfilingTTNNProxy,
    peak_memory_device,
    profiler_devices: tuple[object, ...],
    artifact_root: Path,
    backend,
    keys,
) -> dict[str, object]:
    _clear_ttmlir_caches()
    cold = runner(
        case=case,
        backend=backend,
        options=tt_thrml.ExecutionOptions(
            profiler=ThreadSafeRuntimeProfile(),
            profile_sync=True,
        ),
        ttnn_proxy=ttnn_proxy,
        peak_memory_device=peak_memory_device,
        profiler_devices=profiler_devices,
        artifact_root=artifact_root,
        keys=keys,
        schedule=case.single_sample_schedule,
        phase_label="cold",
    )
    warm = runner(
        case=case,
        backend=backend,
        options=tt_thrml.ExecutionOptions(
            profiler=ThreadSafeRuntimeProfile(),
            profile_sync=True,
        ),
        ttnn_proxy=ttnn_proxy,
        peak_memory_device=peak_memory_device,
        profiler_devices=profiler_devices,
        artifact_root=artifact_root,
        keys=keys,
        schedule=case.single_sample_schedule,
        phase_label="first_sample",
    )
    steady = runner(
        case=case,
        backend=backend,
        options=tt_thrml.ExecutionOptions(
            profiler=ThreadSafeRuntimeProfile(),
            profile_sync=True,
        ),
        ttnn_proxy=ttnn_proxy,
        peak_memory_device=peak_memory_device,
        profiler_devices=profiler_devices,
        artifact_root=artifact_root,
        keys=keys,
        schedule=case.steady_state_schedule,
        phase_label="steady_state",
    )

    peak_memory_values = [
        value
        for value in (
            cold.peak_device_memory_bytes,
            warm.peak_device_memory_bytes,
            steady.peak_device_memory_bytes,
        )
        if value is not None
    ]
    return {
        "compile_time_seconds": max(cold.wall_seconds - warm.wall_seconds, 0.0),
        "compile_time_method": "cold_single_sample_minus_warm_single_sample",
        "first_sample_latency_seconds": warm.wall_seconds,
        "steady_state_samples_per_second": (
            steady.total_sample_count / steady.wall_seconds if steady.wall_seconds > 0 else 0.0
        ),
        "transfer_count_method": "ttnn_bridge_plus_ttmlir_runtime_host_boundary",
        "dispatch_count_first_sample": warm.dispatch_count,
        "dispatch_count_steady_state": steady.dispatch_count,
        "dispatch_count_by_family_first_sample": warm.dispatch_count_by_family,
        "dispatch_count_by_family_steady_state": steady.dispatch_count_by_family,
        "host_to_device_transfers_first_sample": warm.host_to_device_transfers,
        "device_to_host_transfers_first_sample": warm.device_to_host_transfers,
        "host_to_device_transfers_steady_state": steady.host_to_device_transfers,
        "device_to_host_transfers_steady_state": steady.device_to_host_transfers,
        "peak_device_memory_bytes": max(peak_memory_values) if peak_memory_values else None,
        "profile_first_sample": warm.profile,
        "profile_steady_state": steady.profile,
        "official_artifacts_first_sample": warm.official_artifacts,
        "official_artifacts_steady_state": steady.official_artifacts,
    }


def _base_spin_weights(n_nodes: int, *, start: float, stop: float) -> jax.Array:
    return jnp.linspace(start, stop, n_nodes, dtype=jnp.float32)


def _categorical_pair_weights(n_pairs: int, *, n_categories: int) -> jax.Array:
    pair_index = jnp.arange(n_pairs, dtype=jnp.float32).reshape(n_pairs, 1, 1)
    row = jnp.arange(n_categories, dtype=jnp.float32).reshape(1, n_categories, 1)
    col = jnp.arange(n_categories, dtype=jnp.float32).reshape(1, 1, n_categories)
    return 0.15 * jnp.sin(pair_index + row) + 0.1 * jnp.cos(pair_index - col)


def _make_spin_case(*, n_nodes: int = 16) -> HardwareCase:
    nodes = [SpinNode() for _ in range(n_nodes)]
    free_blocks = [Block(nodes[0::2]), Block(nodes[1::2])]
    init_state_free = [
        jax.random.bernoulli(block_key, 0.5, (len(block.nodes),)).astype(jnp.bool_)
        for block_key, block in zip(
            jax.random.split(jax.random.key(1101), len(free_blocks)),
            free_blocks,
            strict=True,
        )
    ]
    return HardwareCase(
        name="spin_single_device",
        program=FactorSamplingProgram(
            BlockGibbsSpec(free_blocks, []),
            [SpinGibbsConditional(), SpinGibbsConditional()],
            [
                SpinEBMFactor([Block(nodes)], _base_spin_weights(n_nodes, start=0.35, stop=-0.2)),
                SpinEBMFactor(
                    [Block(nodes[:-1]), Block(nodes[1:])],
                    _base_spin_weights(n_nodes - 1, start=0.25, stop=-0.15),
                ),
                SpinEBMFactor(
                    [Block(nodes[:-2]), Block(nodes[2:])],
                    _base_spin_weights(n_nodes - 2, start=-0.1, stop=0.2),
                ),
            ],
            [],
        ),
        init_state_free=init_state_free,
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
        single_sample_schedule=SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=2),
        steady_state_schedule=SamplingSchedule(n_warmup=0, n_samples=8, steps_per_sample=2),
    )


def _make_categorical_case(*, n_nodes: int = 8, n_categories: int = 4) -> HardwareCase:
    nodes = [CategoricalNode() for _ in range(n_nodes)]
    free_blocks = [Block(nodes[0::2]), Block(nodes[1::2])]
    init_state_free = [
        jax.random.randint(
            block_key,
            shape=(len(block.nodes),),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        )
        for block_key, block in zip(
            jax.random.split(jax.random.key(2202), len(free_blocks)),
            free_blocks,
            strict=True,
        )
    ]
    bias_weights = 0.2 * jnp.sin(
        jnp.arange(n_nodes * n_categories, dtype=jnp.float32).reshape(n_nodes, n_categories)
    )
    sampler = CategoricalGibbsConditional(n_categories)
    return HardwareCase(
        name="categorical_single_device",
        program=FactorSamplingProgram(
            BlockGibbsSpec(free_blocks, []),
            [sampler, sampler],
            [
                CategoricalEBMFactor([Block(nodes)], bias_weights),
                CategoricalEBMFactor(
                    [Block(nodes[:-1]), Block(nodes[1:])],
                    _categorical_pair_weights(n_nodes - 1, n_categories=n_categories),
                ),
                CategoricalEBMFactor(
                    [Block(nodes[:-2]), Block(nodes[2:])],
                    _categorical_pair_weights(n_nodes - 2, n_categories=n_categories),
                ),
            ],
            [],
        ),
        init_state_free=init_state_free,
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
        single_sample_schedule=SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=2),
        steady_state_schedule=SamplingSchedule(n_warmup=0, n_samples=6, steps_per_sample=2),
    )


def _make_gaussian_case(*, n_nodes: int = 16) -> HardwareCase:
    nodes = [ContinuousNode() for _ in range(n_nodes)]
    free_blocks = [Block(nodes[0::2]), Block(nodes[1::2])]
    init_state_free = [
        0.1 * jax.random.normal(block_key, (len(block.nodes),), dtype=jnp.float32)
        for block_key, block in zip(
            jax.random.split(jax.random.key(3303), len(free_blocks)),
            free_blocks,
            strict=True,
        )
    ]
    return HardwareCase(
        name="gaussian_single_device",
        program=FactorSamplingProgram(
            BlockGibbsSpec(
                free_blocks,
                [],
                {ContinuousNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)},
            ),
            [GaussianConditional(), GaussianConditional()],
            [
                LinearFactor(_base_spin_weights(n_nodes, start=0.12, stop=-0.18), Block(nodes)),
                QuadraticFactor(
                    0.8 + 0.05 * jnp.cos(jnp.arange(n_nodes, dtype=jnp.float32)),
                    Block(nodes),
                ),
                CouplingFactor(
                    _base_spin_weights(n_nodes - 1, start=0.06, stop=-0.04),
                    (Block(nodes[:-1]), Block(nodes[1:])),
                ),
                CouplingFactor(
                    _base_spin_weights(n_nodes - 2, start=-0.03, stop=0.02),
                    (Block(nodes[:-2]), Block(nodes[2:])),
                ),
            ],
            [],
        ),
        init_state_free=init_state_free,
        state_clamp=[],
        nodes_to_sample=[Block(nodes)],
        single_sample_schedule=SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=2),
        steady_state_schedule=SamplingSchedule(n_warmup=0, n_samples=6, steps_per_sample=2),
    )


def _make_mixed_case(*, n_motifs: int = 4, n_categories: int = 4) -> HardwareCase:
    spin_nodes = [SpinNode() for _ in range(n_motifs)]
    categorical_nodes = [CategoricalNode() for _ in range(n_motifs)]
    continuous_nodes = [ContinuousNode() for _ in range(n_motifs)]
    free_super_blocks = [
        (
            Block(spin_nodes[0::2]),
            Block(categorical_nodes[0::2]),
            Block(continuous_nodes[0::2]),
        ),
        (
            Block(spin_nodes[1::2]),
            Block(categorical_nodes[1::2]),
            Block(continuous_nodes[1::2]),
        ),
    ]
    init_keys = jax.random.split(jax.random.key(4404), 6)
    init_state_free = [
        jax.random.bernoulli(init_keys[0], 0.5, (len(spin_nodes[0::2]),)).astype(jnp.bool_),
        jax.random.randint(
            init_keys[1],
            shape=(len(categorical_nodes[0::2]),),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        ),
        0.1 * jax.random.normal(init_keys[2], (len(continuous_nodes[0::2]),), dtype=jnp.float32),
        jax.random.bernoulli(init_keys[3], 0.5, (len(spin_nodes[1::2]),)).astype(jnp.bool_),
        jax.random.randint(
            init_keys[4],
            shape=(len(categorical_nodes[1::2]),),
            minval=0,
            maxval=n_categories,
            dtype=jnp.uint8,
        ),
        0.1 * jax.random.normal(init_keys[5], (len(continuous_nodes[1::2]),), dtype=jnp.float32),
    ]
    return HardwareCase(
        name="mixed_single_device",
        program=FactorSamplingProgram(
            BlockGibbsSpec(
                free_super_blocks,
                [],
                {
                    SpinNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool_),
                    CategoricalNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8),
                    ContinuousNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
                },
            ),
            [
                SpinGibbsConditional(),
                CategoricalGibbsConditional(n_categories),
                GaussianConditional(),
                SpinGibbsConditional(),
                CategoricalGibbsConditional(n_categories),
                GaussianConditional(),
            ],
            [
                SpinEBMFactor([Block(spin_nodes)], _base_spin_weights(n_motifs, start=0.2, stop=-0.15)),
                SpinEBMFactor(
                    [Block(spin_nodes[:-1]), Block(spin_nodes[1:])],
                    _base_spin_weights(n_motifs - 1, start=0.15, stop=-0.08),
                ),
                CategoricalEBMFactor(
                    [Block(categorical_nodes)],
                    0.15
                    * jnp.sin(
                        jnp.arange(n_motifs * n_categories, dtype=jnp.float32).reshape(
                            n_motifs, n_categories
                        )
                    ),
                ),
                CategoricalEBMFactor(
                    [Block(categorical_nodes[:-1]), Block(categorical_nodes[1:])],
                    _categorical_pair_weights(n_motifs - 1, n_categories=n_categories),
                ),
                DiscreteEBMFactor(
                    [Block(spin_nodes)],
                    [Block(categorical_nodes)],
                    0.1
                    * jnp.cos(
                        jnp.arange(n_motifs * n_categories, dtype=jnp.float32).reshape(
                            n_motifs, n_categories
                        )
                    ),
                ),
                LinearFactor(_base_spin_weights(n_motifs, start=0.1, stop=-0.12), Block(continuous_nodes)),
                QuadraticFactor(
                    0.85 + 0.04 * jnp.sin(jnp.arange(n_motifs, dtype=jnp.float32)),
                    Block(continuous_nodes),
                ),
                CouplingFactor(
                    _base_spin_weights(n_motifs - 1, start=0.05, stop=-0.03),
                    (Block(continuous_nodes[:-1]), Block(continuous_nodes[1:])),
                ),
            ],
            [],
        ),
        init_state_free=init_state_free,
        state_clamp=[],
        nodes_to_sample=[
            Block(spin_nodes),
            Block(categorical_nodes),
            Block(continuous_nodes),
        ],
        single_sample_schedule=SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=2),
        steady_state_schedule=SamplingSchedule(n_warmup=0, n_samples=4, steps_per_sample=2),
    )


@dataclass(frozen=True)
class WormholeTestEnv:
    ttnn: object
    system_desc_path: Path
    build_dir: Path
    device_ids: tuple[int, ...]
    mesh_shape: tuple[int, int]


def _maybe_extend_ttmlir_python_path(build_dir: Path) -> None:
    candidates: list[str] = []
    for candidate in (build_dir / "python_packages", build_dir / "tools"):
        if not candidate.exists():
            continue
        candidate_str = str(candidate.resolve())
        if candidate_str in sys.path:
            continue
        candidates.append(candidate_str)
    for candidate_str in reversed(candidates):
        sys.path.insert(0, candidate_str)


@pytest.fixture(scope="module")
def wormhole_env() -> WormholeTestEnv:
    system_desc = os.environ.get("SYSTEM_DESC_PATH")
    build_dir = os.environ.get("TTMLIR_BUILD_DIR")
    if not system_desc:
        pytest.skip("SYSTEM_DESC_PATH is required for wormhole smoke/perf tests.")
    if not build_dir:
        pytest.skip("TTMLIR_BUILD_DIR is required for wormhole smoke/perf tests.")
    resolved_build_dir = Path(build_dir).resolve()
    _maybe_extend_ttmlir_python_path(resolved_build_dir)
    ttnn = pytest.importorskip("ttnn")
    device_ids = _parse_device_ids(os.environ.get("TT_THRML_TEST_DEVICE_IDS"))
    mesh_device_ids = device_ids[:2] if len(device_ids) >= 2 else device_ids
    mesh_shape = _parse_mesh_shape(
        os.environ.get("TT_THRML_TEST_MESH_SHAPE"),
        n_devices=len(mesh_device_ids),
    )
    return WormholeTestEnv(
        ttnn=ttnn,
        system_desc_path=Path(system_desc).resolve(),
        build_dir=resolved_build_dir,
        device_ids=device_ids,
        mesh_shape=mesh_shape,
    )


def _record_metrics(record_property, metrics: dict[str, object]) -> None:
    payload = json.dumps(metrics, sort_keys=True)
    record_property("wormhole_metrics", payload)
    print(payload)


def _run_single_device_case(
    *,
    case: HardwareCase,
    env: WormholeTestEnv,
    artifact_root: Path,
) -> dict[str, object]:
    ttnn_proxy = ProfilingTTNNProxy(env.ttnn, transfers=TransferCounter())
    device = tt_thrml.open_device(ttnn_proxy, device_id=env.device_ids[0])
    try:
        backend = tt_thrml.make_ttmlir_backend_binding(
            ttnn_proxy,
            device,
            system_desc_path=env.system_desc_path,
            artifact_root=artifact_root,
            build_dir=env.build_dir,
        )
        metrics = _estimate_case_metrics(
            case=case,
            runner=lambda keys=None, **kwargs: _run_single_job(
                key=jax.random.key(17),
                **kwargs,
            ),
            ttnn_proxy=ttnn_proxy,
            peak_memory_device=device,
            profiler_devices=(device,),
            artifact_root=artifact_root,
            backend=backend,
            keys=None,
        )
        return {
            "mode": "single_device",
            "case": case.name,
            "device_ids": [env.device_ids[0]],
            **metrics,
        }
    finally:
        tt_thrml.close_devices(ttnn_proxy, (device,))
        _clear_ttmlir_caches()


def _run_mesh_case(
    *,
    case: HardwareCase,
    env: WormholeTestEnv,
    artifact_root: Path,
) -> dict[str, object]:
    if len(env.device_ids) < 2:
        pytest.skip("MeshDevice replicated mode requires at least two TT devices.")
    mesh_device_ids = env.device_ids[:2]
    ttnn_proxy = ProfilingTTNNProxy(env.ttnn, transfers=TransferCounter())
    mesh_device = tt_thrml.open_mesh_device(
        ttnn_proxy,
        mesh_shape=env.mesh_shape,
        device_ids=mesh_device_ids,
    )
    try:
        backend = tt_thrml.make_ttmlir_backend_binding(
            ttnn_proxy,
            mesh_device,
            system_desc_path=env.system_desc_path,
            artifact_root=artifact_root,
            build_dir=env.build_dir,
        )
        metrics = _estimate_case_metrics(
            case=case,
            runner=lambda keys=None, **kwargs: _run_single_job(
                key=jax.random.key(23),
                **kwargs,
            ),
            ttnn_proxy=ttnn_proxy,
            peak_memory_device=mesh_device,
            profiler_devices=(mesh_device,),
            artifact_root=artifact_root,
            backend=backend,
            keys=None,
        )
        return {
            "mode": "mesh_replicated",
            "case": case.name,
            "device_ids": list(mesh_device_ids),
            "mesh_shape": list(env.mesh_shape),
            **metrics,
        }
    finally:
        tt_thrml.close_mesh_device(ttnn_proxy, mesh_device)
        _clear_ttmlir_caches()


def _run_multi_device_case(
    *,
    case: HardwareCase,
    env: WormholeTestEnv,
    artifact_root: Path,
) -> dict[str, object]:
    if len(env.device_ids) < 2:
        pytest.skip("Multi-device independent chains require at least two TT devices.")
    worker_device_ids = env.device_ids[:2]
    ttnn_proxy = ProfilingTTNNProxy(env.ttnn, transfers=TransferCounter())
    devices = tt_thrml.open_devices(ttnn_proxy, device_ids=worker_device_ids)
    try:
        backend = tt_thrml.make_ttmlir_backend_binding(
            ttnn_proxy,
            devices,
            system_desc_path=env.system_desc_path,
            artifact_root=artifact_root,
            build_dir=env.build_dir,
        )
        keys = [jax.random.key(100 + index) for index in range(len(worker_device_ids))]
        metrics = _estimate_case_metrics(
            case=case,
            runner=_run_many_jobs,
            ttnn_proxy=ttnn_proxy,
            peak_memory_device=devices[0],
            profiler_devices=tuple(devices),
            artifact_root=artifact_root,
            backend=backend,
            keys=keys,
        )
        return {
            "mode": "multi_device_independent_chains",
            "case": case.name,
            "device_ids": list(worker_device_ids),
            "job_count": len(keys),
            **metrics,
        }
    finally:
        tt_thrml.close_devices(ttnn_proxy, devices)
        _clear_ttmlir_caches()


@pytest.mark.parametrize(
    "case_factory",
    [_make_spin_case, _make_categorical_case, _make_gaussian_case, _make_mixed_case],
    ids=["spin_single_device", "categorical_single_device", "gaussian_single_device", "mixed_single_device"],
)
def test_single_device_wormhole_smoke_perf(
    wormhole_env: WormholeTestEnv,
    tmp_path: Path,
    record_property,
    case_factory,
):
    case = case_factory()
    metrics = _run_single_device_case(
        case=case,
        env=wormhole_env,
        artifact_root=tmp_path / case.name,
    )
    assert metrics["dispatch_count_first_sample"] > 0
    assert metrics["steady_state_samples_per_second"] > 0
    _record_metrics(record_property, metrics)


def test_mesh_device_replicated_wormhole_smoke_perf(
    wormhole_env: WormholeTestEnv,
    tmp_path: Path,
    record_property,
):
    case = _make_mixed_case()
    metrics = _run_mesh_case(
        case=case,
        env=wormhole_env,
        artifact_root=tmp_path / "mesh_replicated_mixed",
    )
    assert metrics["dispatch_count_first_sample"] > 0
    assert metrics["steady_state_samples_per_second"] > 0
    _record_metrics(record_property, metrics)


def test_multi_device_independent_chains_wormhole_smoke_perf(
    wormhole_env: WormholeTestEnv,
    tmp_path: Path,
    record_property,
):
    case = _make_mixed_case()
    metrics = _run_multi_device_case(
        case=case,
        env=wormhole_env,
        artifact_root=tmp_path / "multi_device_independent_chains_mixed",
    )
    assert metrics["dispatch_count_first_sample"] > 0
    assert metrics["steady_state_samples_per_second"] > 0
    _record_metrics(record_property, metrics)
