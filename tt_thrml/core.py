"""Core data structures and device helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence, cast


class Family(str, Enum):
    SPIN = "spin"
    CATEGORICAL = "categorical"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class FusedInteractionSpec:
    """One THRML interaction baked into constants for the fused kernel.

    weighted_mask is the elementwise product weights*active_mask, pre-folded so the
    kernel materializes only one constant per interaction. gather_indices is a tuple
    with one flat-global (n_nodes, n_terms) index array per source of the interaction.
    """

    weighted_mask: object
    gather_indices: tuple[object, ...]
    n_spin: int
    n_categorical: int
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
    kernel_artifact: Path | None


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
    return len(device_ids(device)) > 1


def device_ids(device) -> tuple[int, ...]:
    get_ids = getattr(device, "get_device_ids", None)
    if callable(get_ids):
        return tuple(int(d) for d in cast(Iterable[Any], get_ids()))
    return ()
