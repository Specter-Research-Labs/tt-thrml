"""Multi-device mesh support for fused TT-THRML executor."""

from __future__ import annotations

from math import prod

from thrml.block_sampling import BlockSamplingProgram

from .core import TTMLIRConfig
from .executor import Executor


def _mesh_shape(device) -> tuple[int, ...]:
    """Extract mesh shape from device."""
    shape = getattr(device, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        dims = getattr(shape, "dims", None)
        if callable(dims):
            return tuple(int(shape[i]) for i in range(dims()))
        return ()


def mesh_device_ids(device) -> tuple[int, ...]:
    """Get device IDs from mesh device."""
    get_ids = getattr(device, "get_device_ids", None)
    if callable(get_ids):
        return tuple(int(d) for d in get_ids())
    device_id = getattr(device, "id", None)
    if isinstance(device_id, int):
        return (device_id,)
    return ()


def mesh_size(device) -> int:
    """Get number of devices in mesh."""
    ids = mesh_device_ids(device)
    if ids:
        return len(ids)
    shape = _mesh_shape(device)
    if shape:
        return prod(shape)
    return 1


def is_mesh_device(device) -> bool:
    """Check if device is a multi-device mesh."""
    return mesh_size(device) > 1


def mesh_barrier(ttnn, device) -> None:
    """Synchronize across mesh devices."""
    sync = getattr(ttnn, "synchronize_device", None)
    if callable(sync):
        sync(device)


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
        n_free = self.compiled.n_free_blocks

        for group in self.compiled.sampling_order:
            for block_index in group:
                if block_index >= n_free:
                    continue
                self._global_state = self._run_block_kernel(block_index, sweep_idx)
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
    """Create mesh executor for THRML program."""
    return MeshExecutor(ttnn, device, program, config, n_sweeps=n_sweeps)
