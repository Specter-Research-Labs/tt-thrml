"""Core data structures, configuration, and device helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence


class Family(str, Enum):
    SPIN = "spin"
    CATEGORICAL = "categorical"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class TTMLIRConfig:
    system_desc_path: Path
    artifact_root: Path
    ttmlir_opt: str
    ttmlir_translate: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_desc_path", Path(self.system_desc_path).resolve())
        object.__setattr__(self, "artifact_root", Path(self.artifact_root).resolve())
        object.__setattr__(self, "ttmlir_opt", _normalize_tool_path(self.ttmlir_opt))
        object.__setattr__(self, "ttmlir_translate", _normalize_tool_path(self.ttmlir_translate))

    def cache_key(self) -> str:
        payload = {
            "system_desc_path": str(self.system_desc_path),
            "ttmlir_opt": self.ttmlir_opt,
            "ttmlir_translate": self.ttmlir_translate,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FusedInteractionSpec:
    """One THRML interaction baked into constants for the fused kernel.

    weighted_mask is the elementwise product weights*active_mask, pre-folded so the
    kernel materializes only one constant per interaction. gather_indices is a tuple
    with one (n_nodes, n_terms) index array per source spin of the interaction
    (empty for bias-only interactions).
    """
    weighted_mask: object
    gather_indices: tuple[object, ...]
    n_spin: int
    n_terms: int
    contribution_kind: str


@dataclass(frozen=True)
class FusedBlockSpec:
    """Specification for a fused block kernel."""
    block_index: int
    family: Family
    n_nodes: int
    n_categories: int | None
    block_global_start: int
    total_nodes: int
    interactions: tuple[FusedInteractionSpec, ...]


@dataclass(frozen=True)
class CompiledFusedBlock:
    """Compiled fused kernel for one block."""
    spec: FusedBlockSpec
    kernel_artifact: object


@dataclass(frozen=True)
class CompiledProgram:
    """Compiled THRML program with fused kernels."""
    blocks: tuple[CompiledFusedBlock, ...]
    n_free_blocks: int
    total_nodes: int
    block_global_starts: tuple[int, ...]
    sampling_order: tuple[tuple[int, ...], ...]
    state_dtype: object
    index_dtype: object
    layout: object
    rng_spec: RNGSpec


@dataclass(frozen=True)
class RNGSpec:
    """Specification for bulk RNG buffers."""
    n_sweeps: int
    spin_blocks: tuple[int, ...]
    categorical_blocks: tuple[int, ...]
    gaussian_blocks: tuple[int, ...]
    nodes_per_block: tuple[int, ...]
    categories_per_block: tuple[int | None, ...]


@dataclass(frozen=True)
class BulkRNGBuffers:
    """Pre-generated RNG buffers on device."""
    spin_threshold_logits: object | None
    categorical_gumbel: object | None
    gaussian_noise: object | None
    sweep_offset: int = 0


def _normalize_tool_path(command: str | Path) -> str:
    command_str = str(command)
    command_path = Path(command_str)
    if command_path.is_absolute() or command_path.parent != Path("."):
        return str(command_path.resolve())
    return command_str


def make_ttmlir_config(
    *,
    system_desc_path: Path | str,
    artifact_root: Path | str | None = None,
    build_dir: Path | str | None = None,
    ttmlir_opt: Path | str | None = None,
    ttmlir_translate: Path | str | None = None,
) -> TTMLIRConfig:
    """Create TT-MLIR configuration with sensible defaults."""
    env_build_dir = os.environ.get("TTMLIR_BUILD_DIR")

    if build_dir is not None and (ttmlir_opt is not None or ttmlir_translate is not None):
        raise ValueError("Pass either build_dir or explicit tool paths, not both.")

    if build_dir is None and ttmlir_opt is None and ttmlir_translate is None:
        if env_build_dir is None:
            raise ValueError(
                "TT-MLIR tools must be configured. Pass build_dir, explicit tool paths, "
                "or set TTMLIR_BUILD_DIR."
            )
        build_dir = env_build_dir

    if (ttmlir_opt is None) != (ttmlir_translate is None):
        raise ValueError("Pass both ttmlir_opt and ttmlir_translate together.")

    if build_dir is not None:
        build_dir = Path(build_dir).resolve()
        ttmlir_opt = build_dir / "bin" / "ttmlir-opt"
        ttmlir_translate = build_dir / "bin" / "ttmlir-translate"

    resolved_artifact_root = Path(
        artifact_root if artifact_root is not None
        else Path(tempfile.gettempdir()) / "tt-thrml"
    ).resolve()

    return TTMLIRConfig(
        system_desc_path=Path(system_desc_path).resolve(),
        artifact_root=resolved_artifact_root,
        ttmlir_opt=ttmlir_opt,
        ttmlir_translate=ttmlir_translate,
    )


def open_device(ttnn, *, device_id: int = 0) -> object:
    return ttnn.open_device(device_id=device_id)


def open_devices(ttnn, *, device_ids: Sequence[int]) -> tuple[object, ...]:
    if not device_ids:
        raise ValueError("At least one device id required.")
    return tuple(ttnn.open_device(device_id=int(d)) for d in device_ids)


def open_mesh_device(
    ttnn,
    *,
    mesh_shape: tuple[int, int],
    device_ids: Sequence[int] | None = None,
    num_command_queues: int = 1,
    offset=None,
) -> object:
    kwargs: dict = {
        "mesh_shape": ttnn.MeshShape(*mesh_shape),
        "num_command_queues": num_command_queues,
    }
    if device_ids is not None:
        kwargs["physical_device_ids"] = [int(d) for d in device_ids]
    if offset is not None:
        kwargs["offset"] = offset
    return ttnn.open_mesh_device(**kwargs)


def close_device(ttnn, device) -> None:
    ttnn.close_device(device)


def close_mesh_device(ttnn, device) -> None:
    ttnn.close_mesh_device(device)


def close_devices(ttnn, devices) -> None:
    for device in devices:
        ttnn.close_device(device)


def is_mesh_device(device) -> bool:
    get_ids = getattr(device, "get_device_ids", None)
    return callable(get_ids) and len(get_ids()) > 1


def device_ids(device) -> tuple[int, ...]:
    get_ids = getattr(device, "get_device_ids", None)
    if callable(get_ids):
        return tuple(int(d) for d in get_ids())
    return ()
