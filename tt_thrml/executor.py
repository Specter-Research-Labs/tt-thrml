"""Fused executor with device-resident global state and bulk RNG.

Each sweep, for each THRML sampling group, invokes a single compiled kernel
    (global_state, *rng_slices) -> new_global_state
so all compute (gather, gamma, sampling, slice-update) happens inside the
flatbuffer. The host keeps a pointer to the global_state tensor between
kernel invocations and never reads it back mid-sweep.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockSamplingProgram, SamplingSchedule

from .compiler import compile_program
from .core import BulkRNGBuffers, CompiledProgram, Family, TTMLIRConfig
from .rng import generate_bulk_rng, generate_bulk_rng_for_schedule, make_rng_spec, slice_rng_for_sweep


class Executor:
    def __init__(
        self,
        ttnn,
        device,
        program: BlockSamplingProgram,
        config: TTMLIRConfig,
        compiled: CompiledProgram | None = None,
        *,
        n_sweeps: int = 100,
        profile: bool = False,
    ):
        self.ttnn = ttnn
        self.device = device
        self.program = program
        self.config = config
        self.n_sweeps = n_sweeps
        self.profile = profile
        self.compiled = (
            compiled if compiled is not None else compile_program(ttnn, device, program, config, n_sweeps=n_sweeps)
        )

        self._global_state: object | None = None
        self._state_loaded = False
        self._rng_buffers: BulkRNGBuffers | None = None
        self._sweep_counter = 0
        self._rng_n_sweeps = 0

        self._tt_runtime: Any | None = None
        self._runtime_utils: Any | None = None
        self._runtime_device: Any | None = None
        self._binary_cache: dict[Path, Any] = {}
        self._timing_log: list[tuple[str, float, float]] = []

    @property
    def state_is_loaded(self) -> bool:
        return self._state_loaded

    def load_state(self, state_free, state_clamp) -> None:
        """Concatenate block states in block order and upload as one tensor."""
        n_free = self.compiled.n_free_blocks
        block_arrays: list[np.ndarray] = []

        for i, state in enumerate(state_free):
            block_arrays.append(self._block_state_to_global_chunk(i, state))
        for i, state in enumerate(state_clamp):
            block_arrays.append(self._block_state_to_global_chunk(n_free + i, state))

        if len(block_arrays) != len(self.compiled.blocks):
            raise ValueError(f"Expected {len(self.compiled.blocks)} block states, got {len(block_arrays)}")

        import torch  # type: ignore[reportMissingImports]

        global_arr = np.concatenate(block_arrays).astype(np.float32)
        tensor = torch.from_numpy(global_arr.copy()).contiguous()
        self._global_state = self.ttnn.from_torch(
            tensor,
            dtype=self.compiled.state_dtype,
            layout=self.compiled.layout,
            device=self.device,
        )
        self._state_loaded = True

    def _block_state_to_global_chunk(self, block_index: int, state) -> np.ndarray:
        spec = self.compiled.blocks[block_index].spec
        arr = np.asarray(state)
        if spec.family == Family.SPIN:
            return np.where(arr.astype(np.float32) > 0, 1.0, -1.0).reshape(spec.n_nodes)
        if spec.family == Family.CATEGORICAL:
            return arr.astype(np.float32).reshape(spec.n_nodes)
        return arr.astype(np.float32).reshape(spec.n_nodes)

    def prepare_rng(self, key) -> None:
        self._rng_buffers = generate_bulk_rng(
            key,
            self.compiled.rng_spec,
            self.ttnn,
            self.device,
            state_dtype=self.compiled.state_dtype,
            layout=self.compiled.layout,
        )
        self._sweep_counter = 0
        self._rng_n_sweeps = self.compiled.rng_spec.n_sweeps

    def run_sweep(self) -> None:
        if not self._state_loaded:
            raise RuntimeError("State must be loaded before running sweep")
        if self._rng_buffers is None:
            raise RuntimeError("RNG must be prepared before running sweep")
        if self._sweep_counter >= self._rng_n_sweeps:
            raise RuntimeError("RNG buffer exhausted - call prepare_rng again")

        sweep_idx = self._sweep_counter

        for group in self.compiled.sampling_groups:
            self._global_state = self._run_sampling_group(group, sweep_idx)

        self._sweep_counter += 1

    def run_warmup(self, n_warmup: int) -> None:
        for _ in range(n_warmup):
            self.run_sweep()

    def _run_block_kernel(self, block_index: int, sweep_idx: int) -> object:
        block = self.compiled.blocks[block_index]
        spec = block.spec
        if block.kernel_artifact is None:
            raise RuntimeError("Block kernel artifact is not compiled; use sampling groups.")
        rng_buffers = self._rng_buffers
        if rng_buffers is None:
            raise RuntimeError("RNG must be prepared before running sweep")
        rng_slice = slice_rng_for_sweep(self.ttnn, rng_buffers, sweep_idx, block_index, spec.family)
        outputs = self._run_compiled_kernel(
            block.kernel_artifact,
            [self._global_state, rng_slice],
            family_label=spec.family.value,
        )
        if len(outputs) != 1:
            raise RuntimeError(f"Expected 1 output from kernel, got {len(outputs)}")
        return outputs[0]

    def _run_sampling_group(self, group, sweep_idx: int) -> object:
        rng_buffers = self._rng_buffers
        if rng_buffers is None:
            raise RuntimeError("RNG must be prepared before running sweep")
        inputs = [self._global_state]
        family_labels = []
        for block_index in group.block_indices:
            spec = self.compiled.blocks[block_index].spec
            family_labels.append(spec.family.value)
            inputs.append(
                slice_rng_for_sweep(
                    self.ttnn,
                    rng_buffers,
                    sweep_idx,
                    block_index,
                    spec.family,
                )
            )
        outputs = self._run_compiled_kernel(
            group.kernel_artifact,
            inputs,
            family_label="+".join(family_labels),
        )
        if len(outputs) != 1:
            raise RuntimeError(f"Expected 1 output from group kernel, got {len(outputs)}")
        return outputs[0]

    def _ensure_runtime(self) -> None:
        if self._tt_runtime is not None:
            return
        try:
            import ttrt.runtime as tt_runtime  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError("ttrt.runtime not available") from exc
        self._tt_runtime = tt_runtime
        self._runtime_utils = tt_runtime
        self._runtime_device = tt_runtime.create_runtime_device_from_ttnn(self.device)

    def _get_binary(self, artifact_path: Path) -> Any:
        cached = self._binary_cache.get(artifact_path)
        if cached is not None:
            return cached
        from ttrt.common.util import Binary, FileManager, Logger  # type: ignore[reportMissingImports]

        if self._tt_runtime is None:
            raise RuntimeError("TTRT runtime is not initialized")
        logger = Logger()
        binary = Binary(logger, FileManager(logger), str(artifact_path))
        self._tt_runtime.set_compatible_device_runtime(binary.fbb)
        self._binary_cache[artifact_path] = binary
        return binary

    def _run_compiled_kernel(self, artifact_path: Path, inputs: list[object], family_label: str = "") -> list[object]:
        self._ensure_runtime()
        binary = self._get_binary(artifact_path)
        tt_runtime = self._tt_runtime
        runtime_utils = self._runtime_utils
        runtime_device = self._runtime_device
        if tt_runtime is None or runtime_utils is None or runtime_device is None:
            raise RuntimeError("TTRT runtime is not initialized")

        runtime_inputs = []
        for i, tensor in enumerate(inputs):
            rt = runtime_utils.create_runtime_tensor_from_ttnn(tensor, True)
            layout = tt_runtime.get_layout(binary.fbb, 0, i)
            runtime_inputs.append(tt_runtime.to_layout(rt, runtime_device, layout, True))

        try:
            if self.profile:
                self.ttnn.synchronize_device(self.device)
                self.ttnn.start_tracy_zone(__file__, f"dispatch:{family_label}", 0xFF8800)
                t0 = time.perf_counter()
                runtime_outputs = tt_runtime.submit(runtime_device, binary.fbb, 0, runtime_inputs)
                t1 = time.perf_counter()
                self.ttnn.stop_tracy_zone(f"dispatch:{family_label}")
                self.ttnn.start_tracy_zone(__file__, f"kernel:{family_label}", 0x0088FF)
                self.ttnn.synchronize_device(self.device)
                t2 = time.perf_counter()
                self.ttnn.stop_tracy_zone(f"kernel:{family_label}")
                self._timing_log.append((family_label, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0))
            else:
                runtime_outputs = tt_runtime.submit(runtime_device, binary.fbb, 0, runtime_inputs)
            outputs = [runtime_utils.get_ttnn_tensor_from_runtime_tensor(o) for o in runtime_outputs]
        finally:
            for rt in runtime_inputs:
                tt_runtime.deallocate_tensor(rt, force=True)

        return outputs

    def observe(self, blocks: Sequence[Block]) -> dict:
        if self._global_state is None:
            return {}
        global_arr = self._read_global_array()
        starts = self.compiled.block_global_starts
        block_id_to_index = {id(b): i for i, b in enumerate(self.program.gibbs_spec.blocks)}
        result = {}
        for block in blocks:
            block_index = block_id_to_index.get(id(block))
            if block_index is None:
                continue
            spec = self.compiled.blocks[block_index].spec
            start = starts[block_index]
            chunk = global_arr[start : start + spec.n_nodes]
            result[block] = self._global_chunk_to_block_state(chunk, spec.family)
        return result

    def _read_global_array(self) -> np.ndarray:
        return self.ttnn.to_torch(self._global_state).cpu().numpy().reshape(-1)

    def _read_state_lists(self) -> tuple[list, list]:
        """Unpack device state into THRML-compatible (state_free, state_clamped) lists."""
        global_arr = self._read_global_array()
        starts = self.compiled.block_global_starts
        n_free = self.compiled.n_free_blocks
        state_free, state_clamped = [], []
        for block_index, block in enumerate(self.compiled.blocks):
            spec = block.spec
            start = starts[block_index]
            chunk = global_arr[start : start + spec.n_nodes]
            arr = jnp.array(self._global_chunk_to_block_state(chunk, spec.family))
            if block_index < n_free:
                state_free.append(arr)
            else:
                state_clamped.append(arr)
        return state_free, state_clamped

    def _prepare_rng_for_n_sweeps(self, key, n_sweeps: int) -> None:
        spec = make_rng_spec(self.compiled.blocks[: self.compiled.n_free_blocks], n_sweeps)
        self._rng_buffers = generate_bulk_rng(
            key,
            spec,
            self.ttnn,
            self.device,
            state_dtype=self.compiled.state_dtype,
            layout=self.compiled.layout,
        )
        self._sweep_counter = 0
        self._rng_n_sweeps = n_sweeps

    def _prepare_rng_thrml_compat(self, key, schedule, n_total: int) -> None:
        spec = make_rng_spec(self.compiled.blocks[: self.compiled.n_free_blocks], n_total)
        self._rng_buffers = generate_bulk_rng_for_schedule(
            key,
            spec,
            schedule,
            self.compiled.n_free_blocks,
            self.ttnn,
            self.device,
            state_dtype=self.compiled.state_dtype,
            layout=self.compiled.layout,
        )
        self._sweep_counter = 0
        self._rng_n_sweeps = n_total

    def sample_states(
        self,
        key,
        schedule: SamplingSchedule,
        nodes_to_sample: Sequence[Block],
        *,
        init_state_free,
        state_clamp=(),
    ):
        """Collect raw states for nodes_to_sample over schedule.n_samples iterations.

        Mirrors thrml.sample_states: wraps sample_with_observation using StateObserver.
        """
        from thrml.observers import StateObserver

        f_observe = StateObserver(list(nodes_to_sample))
        _, results = self.sample_with_observation(
            key,
            schedule,
            f_observe,
            init_state_free=init_state_free,
            state_clamp=state_clamp,
        )
        return results

    def sample_with_observation(
        self,
        key,
        schedule: SamplingSchedule,
        f_observe,
        *,
        init_state_free,
        state_clamp=(),
        observation_carry_init=None,
    ):
        """Run full sampling schedule, calling f_observe after each recorded sample.

        Mirrors thrml.sample_with_observation: sample 0 is taken after warmup,
        samples 1..n_samples-1 after each steps_per_sample group.
        Returns (final_observer_carry, stacked_observations).
        """
        n_total = schedule.n_warmup + max(0, schedule.n_samples - 1) * schedule.steps_per_sample
        self.load_state(list(init_state_free), list(state_clamp))
        self._prepare_rng_thrml_compat(key, schedule, n_total)

        self.run_warmup(schedule.n_warmup)

        carry = f_observe.init() if observation_carry_init is None else observation_carry_init
        state_free, state_clamped = self._read_state_lists()
        carry, obs = f_observe(self.program, state_free, state_clamped, carry, jnp.array(0))

        if schedule.n_samples <= 1:
            stacked = None if obs is None else jax.tree.map(lambda x: x[None], obs)
            return carry, stacked

        all_observations = [obs]
        for i in range(1, schedule.n_samples):
            for _ in range(schedule.steps_per_sample):
                self.run_sweep()
            state_free, state_clamped = self._read_state_lists()
            carry, obs = f_observe(self.program, state_free, state_clamped, carry, jnp.array(i))
            all_observations.append(obs)

        if all(o is None for o in all_observations):
            return carry, None

        stacked = jax.tree.map(lambda *xs: jnp.stack(list(xs)), *all_observations)
        return carry, stacked

    def timing_summary(self) -> dict:
        """Return per-family dispatch and kernel timing stats (ms) from profiled runs.

        Only populated when profile=True. Keys are family names; each value is a dict
        with 'n', 'dispatch_mean_ms', 'dispatch_total_ms', 'kernel_mean_ms', 'kernel_total_ms'.
        """
        groups: dict[str, list[tuple[float, float]]] = {}
        for family_label, dispatch_ms, kernel_ms in self._timing_log:
            groups.setdefault(family_label, []).append((dispatch_ms, kernel_ms))
        summary = {}
        for family_label, entries in sorted(groups.items()):
            dispatches = [d for d, _ in entries]
            kernels = [k for _, k in entries]
            summary[family_label] = {
                "n": len(entries),
                "dispatch_mean_ms": sum(dispatches) / len(dispatches),
                "dispatch_total_ms": sum(dispatches),
                "kernel_mean_ms": sum(kernels) / len(kernels),
                "kernel_total_ms": sum(kernels),
            }
        return summary

    def _global_chunk_to_block_state(self, chunk: np.ndarray, family: Family) -> np.ndarray:
        if family == Family.SPIN:
            return (chunk > 0).astype(bool)
        if family == Family.CATEGORICAL:
            return np.round(chunk).astype(np.int32)
        return chunk.astype(np.float32)


def make_executor(
    ttnn,
    device,
    program: BlockSamplingProgram,
    config: TTMLIRConfig,
    *,
    n_sweeps: int = 100,
    profile: bool = False,
) -> Executor:
    return Executor(ttnn, device, program, config, n_sweeps=n_sweeps, profile=profile)
