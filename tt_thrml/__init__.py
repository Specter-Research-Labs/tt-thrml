"""Tenstorrent execution backend for THRML.

TT-Lang is the primary execution path for supported program shapes. TT-MLIR is
kept as an explicit comparison path while TT-Lang coverage broadens.
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
from .executor import Executor, make_executor, make_ttmlir_executor
from .mesh import MeshExecutor, make_mesh_executor, mesh_barrier, mesh_device_ids, mesh_size
from .rng import generate_bulk_rng, make_rng_spec, slice_rng_for_sweep
from .ttlang_runtime import (
    TTLangDiscreteSweepRuntime,
    make_primary_ttlang_executor,
    make_ttlang_discrete_runtime,
    supports_ttlang_discrete_runtime,
    validate_ttlang_discrete_runtime,
)

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
    "make_ttmlir_executor",
    "MeshExecutor",
    "make_mesh_executor",
    "GaussianConditional",
    "TTLangDiscreteSweepRuntime",
    "make_primary_ttlang_executor",
    "make_ttlang_discrete_runtime",
    "supports_ttlang_discrete_runtime",
    "validate_ttlang_discrete_runtime",
]
