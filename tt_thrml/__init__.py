"""Tenstorrent execution backend for THRML.

One fused kernel per sampling group. Bulk RNG uploaded once. Zero host readback mid-sweep.
"""

from .compiler import compile_program
from .conditional_samplers import GaussianConditional
from .core import (
    BulkRNGBuffers,
    CompiledFusedBlock,
    CompiledProgram,
    Family,
    FusedBlockSpec,
    RNGSpec,
    TTMLIRConfig,
    close_device,
    close_devices,
    close_mesh_device,
    device_ids,
    is_mesh_device,
    make_ttmlir_config,
    open_device,
    open_devices,
    open_mesh_device,
)
from .executor import Executor, make_executor
from .mesh import MeshExecutor, make_mesh_executor, mesh_barrier, mesh_device_ids, mesh_size
from .rng import generate_bulk_rng, make_rng_spec, slice_rng_for_sweep


def sample_states(
    key,
    program,
    schedule,
    init_state_free,
    state_clamp,
    nodes_to_sample,
    *,
    ttnn,
    device,
    config: TTMLIRConfig,
):
    """High-level sampling API matching thrml.sample_states."""
    executor = make_executor(ttnn, device, program, config)
    return executor.sample_states(
        key,
        schedule,
        nodes_to_sample,
        init_state_free=init_state_free,
        state_clamp=state_clamp,
    )


def sample_with_observation(
    key,
    program,
    schedule,
    init_state_free,
    state_clamp,
    observation_carry_init,
    f_observe,
    *,
    ttnn,
    device,
    config: TTMLIRConfig,
):
    """High-level observer sampling API matching thrml.sample_with_observation."""
    executor = make_executor(ttnn, device, program, config)
    carry, results = executor.sample_with_observation(
        key,
        schedule,
        f_observe,
        init_state_free=init_state_free,
        state_clamp=state_clamp,
        observation_carry_init=observation_carry_init,
    )
    return carry, results


__all__ = [
    "TTMLIRConfig",
    "make_ttmlir_config",
    "open_device",
    "open_devices",
    "open_mesh_device",
    "close_device",
    "close_mesh_device",
    "close_devices",
    "Executor",
    "make_executor",
    "MeshExecutor",
    "make_mesh_executor",
    "sample_states",
    "sample_with_observation",
    "GaussianConditional",
]
