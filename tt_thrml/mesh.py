"""Multi-device mesh support for fused TT-THRML executor."""

from __future__ import annotations

from typing import Any, Iterable, cast

from thrml.block_sampling import BlockSamplingProgram

from .core import TTMLIRConfig
from .executor import Executor


def mesh_device_ids(device) -> tuple[int, ...]:
    get_ids = getattr(device, "get_device_ids", None)
    if callable(get_ids):
        return tuple(int(d) for d in cast(Iterable[Any], get_ids()))
    return ()


def mesh_size(device) -> int:
    ids = mesh_device_ids(device)
    return len(ids) if ids else 1


def mesh_barrier(ttnn, device) -> None:
    ttnn.synchronize_device(device)


class MeshExecutor(Executor):
    """Fused executor with mesh synchronization between sampling groups."""

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
            mesh_barrier(self.ttnn, self.device)

        self._sweep_counter += 1


def make_mesh_executor(
    ttnn,
    device,
    program: BlockSamplingProgram,
    config: TTMLIRConfig,
    *,
    n_sweeps: int = 100,
) -> MeshExecutor:
    return MeshExecutor(ttnn, device, program, config, n_sweeps=n_sweeps)
