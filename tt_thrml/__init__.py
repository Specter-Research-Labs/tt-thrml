"""Tenstorrent execution backend for THRML.

One fused kernel per block. Bulk RNG uploaded once. Zero host readback mid-sweep.
"""

from .core import (
    Family,
    TTMLIRConfig,
    FusedBlockSpec,
    CompiledFusedBlock,
    CompiledProgram,
    RNGSpec,
    BulkRNGBuffers,
    make_ttmlir_config,
    open_device,
    open_devices,
    open_mesh_device,
    close_device,
    close_mesh_device,
    close_devices,
    is_mesh_device,
    device_ids,
)

from .compiler import compile_program

from .conditional_samplers import GaussianConditional

from .executor import Executor, make_executor

from .rng import (
    make_rng_spec,
    generate_bulk_rng,
    slice_rng_for_sweep,
)

from .mesh import (
    MeshExecutor,
    make_mesh_executor,
    mesh_size,
    mesh_device_ids,
    mesh_barrier,
)


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
    )
    return carry, results


__all__ = [
    "Family",
    "TTMLIRConfig",
    "FusedBlockSpec",
    "CompiledFusedBlock",
    "CompiledProgram",
    "RNGSpec",
    "BulkRNGBuffers",
    "make_ttmlir_config",
    "open_device",
    "open_devices",
    "open_mesh_device",
    "close_device",
    "close_mesh_device",
    "close_devices",
    "is_mesh_device",
    "device_ids",
    "compile_program",
    "GaussianConditional",
    "Executor",
    "make_executor",
    "make_rng_spec",
    "generate_bulk_rng",
    "slice_rng_for_sweep",
    "MeshExecutor",
    "make_mesh_executor",
    "mesh_size",
    "mesh_device_ids",
    "mesh_barrier",
    "sample_states",
    "sample_with_observation",
]
