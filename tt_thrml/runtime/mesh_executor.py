from __future__ import annotations

from dataclasses import dataclass

import torch

from .program_executor import TTProgramExecutor
from .mesh_support import (
    _mesh_shape_tuple,
    is_multi_device_mesh,
    mesh_device_ids,
    mesh_device_size,
    shard_tensor_to_mesh_mapper,
)


@dataclass(frozen=True)
class MeshSweepGroupPlan:
    block_indices: tuple[int, ...]
    block_indices_by_owner: tuple[tuple[int, ...], ...]


def build_round_robin_mesh_sweep_plan(
    *,
    n_blocks: int,
    n_owners: int,
    sampling_order,
) -> tuple[MeshSweepGroupPlan, ...]:
    if n_owners <= 0:
        raise ValueError("n_owners must be positive.")
    if n_blocks < 0:
        raise ValueError("n_blocks cannot be negative.")

    owner_by_block_index = tuple(block_index % n_owners for block_index in range(n_blocks))
    group_plans = []
    for sampling_group in sampling_order:
        block_indices = tuple(int(block_index) for block_index in sampling_group)
        by_owner = [[] for _ in range(n_owners)]
        for block_index in block_indices:
            by_owner[owner_by_block_index[block_index]].append(block_index)
        group_plans.append(
            MeshSweepGroupPlan(
                block_indices=block_indices,
                block_indices_by_owner=tuple(
                    tuple(block_indices_for_owner) for block_indices_for_owner in by_owner
                ),
            )
        )
    return tuple(group_plans)


class TTMeshProgramExecutor(TTProgramExecutor):
    """Mesh-backed executor foundation for a single THRML program sweep.

    V1 keeps the normal TTProgramExecutor parameter/state logic on a MeshDevice,
    makes block ownership explicit, and inserts a real TT collective barrier at
    sweep-group boundaries. This gives us a dedicated mesh execution path
    without overloading the single-device executor with mesh-specific policy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mesh_device = self.device
        self.mesh_shape = _mesh_shape_tuple(self.mesh_device)
        self.mesh_device_ids = mesh_device_ids(self.mesh_device)
        self.mesh_device_count = mesh_device_size(self.mesh_device)
        self.block_owner_by_index = tuple(
            block_index % self.mesh_device_count
            for block_index in range(len(self.compiled.blocks))
        )
        self._mesh_barrier_token = None

    def _ensure_mesh_barrier_token(self):
        if self._mesh_barrier_token is not None:
            return self._mesh_barrier_token

        token_shape = (1, 1, 1, self.mesh_device_count)
        mesh_mapper = shard_tensor_to_mesh_mapper(self.ttnn, self.mesh_device, dim=3)
        row_major_layout = getattr(self.ttnn, "ROW_MAJOR_LAYOUT", None)
        bfloat16 = getattr(self.ttnn, "bfloat16", None)
        token = torch.arange(self.mesh_device_count, dtype=torch.float32).reshape(token_shape)
        from_torch_kwargs = {
            "device": self.mesh_device,
            "mesh_mapper": mesh_mapper,
        }
        if row_major_layout is not None:
            from_torch_kwargs["layout"] = row_major_layout
        if bfloat16 is not None:
            from_torch_kwargs["dtype"] = bfloat16
        self._mesh_barrier_token = self.ttnn.from_torch(token, **from_torch_kwargs)
        return self._mesh_barrier_token

    def _mesh_group_barrier(self, group_index: int):
        token = self._ensure_mesh_barrier_token()
        all_gather = getattr(self.ttnn, "all_gather", None)
        if callable(all_gather):
            gathered = self._profile_call(
                f"run_sweep.group{group_index}.mesh_barrier",
                lambda: all_gather(token, dim=3),
            )
            deallocate = getattr(gathered, "deallocate", None)
            if callable(deallocate):
                deallocate(True)
            return

        experimental = getattr(self.ttnn, "experimental", None)
        all_gather_async = (
            None if experimental is None else getattr(experimental, "all_gather_async", None)
        )
        if callable(all_gather_async):
            gathered = self._profile_call(
                f"run_sweep.group{group_index}.mesh_barrier",
                lambda: all_gather_async(token, dim=3),
            )
            deallocate = getattr(gathered, "deallocate", None)
            if callable(deallocate):
                deallocate(True)
            return

        synchronize_device = getattr(self.ttnn, "synchronize_device", None)
        if callable(synchronize_device):
            self._profile_call(
                f"run_sweep.group{group_index}.mesh_barrier",
                lambda: synchronize_device(self.mesh_device),
            )

    def _after_sampling_group(self, group_index: int, sampling_group) -> None:
        del sampling_group
        self._mesh_group_barrier(group_index)
__all__ = [
    "MeshSweepGroupPlan",
    "TTMeshProgramExecutor",
    "build_round_robin_mesh_sweep_plan",
    "is_multi_device_mesh",
    "mesh_device_size",
]
