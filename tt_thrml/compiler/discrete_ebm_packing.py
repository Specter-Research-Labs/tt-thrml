from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from .interaction_lowering import lower_interaction_contribution
from ..runtime_config import CATEGORICAL_PARAMETER_FAMILY, SPIN_PARAMETER_FAMILY


@dataclass(frozen=True)
class PackedSpinGammaBatch:
    interaction_indices: tuple[int, ...]
    n_spin: int
    tail_shape: tuple[int, ...]
    weights: torch.Tensor
    active_mask: torch.Tensor
    spin_conditions: tuple[torch.Tensor, ...]
    categorical_conditions: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PackedCategoricalThetaBatch:
    interaction_indices: tuple[int, ...]
    n_spin: int
    n_categories: int
    tail_shape: tuple[int, ...]
    weights: torch.Tensor
    active_mask: torch.Tensor
    spin_conditions: tuple[torch.Tensor, ...]
    categorical_conditions: tuple[torch.Tensor, ...]


def _interaction_metadata(
    interaction: object,
    *,
    parameter_family=None,
) -> tuple[int, np.ndarray]:
    contribution = lower_interaction_contribution(
        interaction,
        parameter_family=parameter_family,
    )
    return contribution.n_spin, contribution.weights


def _concat_instance_axis(
    values: Sequence[np.ndarray], *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    concatenated = np.concatenate([np.asarray(value) for value in values], axis=1)
    tensor = torch.from_numpy(concatenated)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.unsqueeze(0)


def _partition_entries_by_instances(
    entries: Sequence[
        tuple[int, np.ndarray, np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]]
    ],
    max_instances_per_batch: int | None,
):
    if max_instances_per_batch is None:
        return [list(entries)]

    if max_instances_per_batch <= 0:
        raise ValueError("max_instances_per_batch must be positive when provided.")

    partitions = []
    current_partition = []
    current_instances = 0

    for entry in entries:
        entry_instances = int(np.asarray(entry[2]).shape[1])
        if (
            current_partition
            and current_instances + entry_instances > max_instances_per_batch
        ):
            partitions.append(current_partition)
            current_partition = []
            current_instances = 0

        current_partition.append(entry)
        current_instances += entry_instances

    if current_partition:
        partitions.append(current_partition)

    return partitions


def pack_spin_gamma_batches(
    interactions: Iterable[object],
    active_flags: Iterable[np.ndarray],
    states: Iterable[Sequence[np.ndarray]],
    *,
    max_instances_per_batch: int | None = None,
) -> list[PackedSpinGammaBatch]:
    grouped: dict[
        tuple[int, tuple[int, ...]],
        list[
            tuple[
                int, np.ndarray, np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]
            ]
        ],
    ] = {}

    for index, (interaction, active, interaction_states) in enumerate(
        zip(interactions, active_flags, states)
    ):
        n_spin, weights = _interaction_metadata(
            interaction,
            parameter_family=SPIN_PARAMETER_FAMILY,
        )
        active_np = np.asarray(active)
        spin_states = [np.asarray(state) for state in interaction_states[:n_spin]]
        categorical_states = [
            np.asarray(state) for state in interaction_states[n_spin:]
        ]
        key = (n_spin, tuple(int(size) for size in weights.shape[2:]))
        grouped.setdefault(key, []).append(
            (index, weights, active_np, spin_states, categorical_states)
        )

    batches = []
    for (n_spin, tail_shape), entries in grouped.items():
        for partition in _partition_entries_by_instances(
            entries, max_instances_per_batch
        ):
            weights = _concat_instance_axis(
                [entry[1] for entry in partition], dtype=torch.float32
            )
            active_mask = _concat_instance_axis(
                [entry[2] for entry in partition], dtype=torch.float32
            )
            spin_conditions = tuple(
                _concat_instance_axis(
                    [entry[3][spin_idx] for entry in partition], dtype=torch.int32
                )
                for spin_idx in range(n_spin)
            )
            categorical_conditions = tuple(
                _concat_instance_axis(
                    [entry[4][cat_idx] for entry in partition], dtype=torch.int64
                )
                for cat_idx in range(len(tail_shape))
            )

            batches.append(
                PackedSpinGammaBatch(
                    interaction_indices=tuple(entry[0] for entry in partition),
                    n_spin=n_spin,
                    tail_shape=tail_shape,
                    weights=weights,
                    active_mask=active_mask,
                    spin_conditions=spin_conditions,
                    categorical_conditions=categorical_conditions,
                )
            )

    return batches


def pack_categorical_theta_batches(
    interactions: Iterable[object],
    active_flags: Iterable[np.ndarray],
    states: Iterable[Sequence[np.ndarray]],
    *,
    max_instances_per_batch: int | None = None,
) -> list[PackedCategoricalThetaBatch]:
    grouped: dict[
        tuple[int, int, tuple[int, ...]],
        list[
            tuple[
                int, np.ndarray, np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray]
            ]
        ],
    ] = {}

    for index, (interaction, active, interaction_states) in enumerate(
        zip(interactions, active_flags, states)
    ):
        n_spin, weights = _interaction_metadata(
            interaction,
            parameter_family=CATEGORICAL_PARAMETER_FAMILY,
        )
        active_np = np.asarray(active)
        spin_states = [np.asarray(state) for state in interaction_states[:n_spin]]
        categorical_states = [
            np.asarray(state) for state in interaction_states[n_spin:]
        ]
        key = (
            n_spin,
            int(weights.shape[2]),
            tuple(int(size) for size in weights.shape[3:]),
        )
        grouped.setdefault(key, []).append(
            (index, weights, active_np, spin_states, categorical_states)
        )

    batches = []
    for (n_spin, n_categories, tail_shape), entries in grouped.items():
        for partition in _partition_entries_by_instances(
            entries, max_instances_per_batch
        ):
            weights = _concat_instance_axis(
                [entry[1] for entry in partition], dtype=torch.float32
            )
            active_mask = _concat_instance_axis(
                [entry[2] for entry in partition], dtype=torch.float32
            )
            spin_conditions = tuple(
                _concat_instance_axis(
                    [entry[3][spin_idx] for entry in partition], dtype=torch.int32
                )
                for spin_idx in range(n_spin)
            )
            categorical_conditions = tuple(
                _concat_instance_axis(
                    [entry[4][cat_idx] for entry in partition], dtype=torch.int64
                )
                for cat_idx in range(len(tail_shape))
            )

            batches.append(
                PackedCategoricalThetaBatch(
                    interaction_indices=tuple(entry[0] for entry in partition),
                    n_spin=n_spin,
                    n_categories=n_categories,
                    tail_shape=tail_shape,
                    weights=weights,
                    active_mask=active_mask,
                    spin_conditions=spin_conditions,
                    categorical_conditions=categorical_conditions,
                )
            )

    return batches
