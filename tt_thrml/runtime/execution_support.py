from __future__ import annotations

from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
import numpy as np


def stack_observer_history(history):
    if not history:
        return []
    if history[0] is None:
        return None
    if len(history) == 1:
        return jax.tree.map(lambda value: value[None], history[0])
    return jax.tree.map(lambda *values: jnp.stack(values, axis=0), *history)


@dataclass
class RuntimeProfileStage:
    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    def record(self, seconds: float) -> None:
        self.count += 1
        self.total_seconds += float(seconds)
        self.max_seconds = max(self.max_seconds, float(seconds))


@dataclass
class RuntimeProfile:
    stages: dict[str, RuntimeProfileStage] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        entry = self.stages.setdefault(stage, RuntimeProfileStage())
        entry.record(seconds)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            stage: {
                "count": entry.count,
                "total_seconds": entry.total_seconds,
                "mean_seconds": (
                    entry.total_seconds / entry.count if entry.count else 0.0
                ),
                "max_seconds": entry.max_seconds,
            }
            for stage, entry in sorted(self.stages.items())
        }


@dataclass(frozen=True)
class PreparedRunRandoms:
    iteration_keys: tuple[object, ...]
    sample_keys: tuple[tuple[object, ...], ...]
    prepared_randoms_by_block: dict[int, object]


@dataclass(frozen=True)
class PreparedScheduleRandoms:
    run_randoms: PreparedRunRandoms
    warmup_count: int
    steps_per_sample: int
    sample_interval_offsets: tuple[int, ...]


@dataclass(frozen=True)
class CompiledObservationPlan:
    blocks: tuple["Block", ...]
    direct_views: tuple[tuple[int, "CompiledStateView"], ...]
    gathered_views: tuple[tuple[int, "CompiledStateView"], ...]
    direct_view_groups: tuple[tuple[tuple[int, "CompiledStateView"], ...], ...]
    gathered_view_groups: tuple[
        tuple[int, tuple[tuple[int, "CompiledStateView"], ...]], ...
    ]


@dataclass(frozen=True)
class CompiledMomentObserverPlan:
    observation_plan: CompiledObservationPlan
    flat_node_count: int
    flat_to_type_slices_list: tuple[np.ndarray, ...]
    flat_to_full_moment_slices: tuple[np.ndarray, ...]
    expansion_indices: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class LoadedObservationJob:
    key: object
    state_free: object
    state_clamp: object
    observation_carry_init: object
    clamp_group_id: object | None = None


@dataclass(frozen=True)
class LoadedStateJob:
    key: object
    state_free: object
    state_clamp: object
    clamp_group_id: object | None = None


__all__ = [
    "CompiledMomentObserverPlan",
    "CompiledObservationPlan",
    "LoadedObservationJob",
    "LoadedStateJob",
    "RuntimeProfile",
    "RuntimeProfileStage",
    "PreparedRunRandoms",
    "PreparedScheduleRandoms",
    "stack_observer_history",
]
