from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from ..runtime_config import (
    GAUSSIAN_PARAMETER_FAMILY,
    SPIN_PARAMETER_FAMILY,
    ParameterFamily,
    normalize_parameter_family,
    parameter_family_spec,
)

_INTERACTION_LOWERERS: dict[
    type,
    Callable[[object, ParameterFamily | None], object],
] = {}


@dataclass(frozen=True)
class InteractionContribution:
    parameter_family: ParameterFamily
    n_spin: int
    weights: np.ndarray
    contribution_kind: str = "default"


@dataclass(frozen=True)
class LoweredBlockInteraction:
    contribution: InteractionContribution
    active_mask: np.ndarray
    source_global_inds: tuple[int, ...]
    source_global_slices: tuple[np.ndarray, ...]
    n_categories: int | None
    tail_shape: tuple[int, ...]


def _coerce_interaction_contribution(
    contribution: InteractionContribution,
    *,
    parameter_family: ParameterFamily | None,
) -> InteractionContribution:
    contribution_family = normalize_parameter_family(contribution.parameter_family)
    expected_family = (
        None if parameter_family is None else normalize_parameter_family(parameter_family)
    )
    if expected_family is not None and contribution_family != expected_family:
        raise TypeError(
            "Interaction lowering parameter family mismatch: "
            f"expected {expected_family.value!r}, got {contribution_family.value!r}."
        )

    if isinstance(contribution.n_spin, bool) or not isinstance(contribution.n_spin, int):
        raise TypeError("Interaction lowering n_spin must be an integer.")
    n_spin = int(contribution.n_spin)
    if n_spin < 0:
        raise ValueError("Interaction lowering n_spin must be non-negative.")

    weights = np.asarray(contribution.weights)
    family_spec = parameter_family_spec(contribution_family)
    minimum_ndim = family_spec.interaction_minimum_ndim
    if weights.ndim < minimum_ndim:
        raise ValueError(
            "Interaction lowering weights must have at least "
            f"{minimum_ndim} dimensions for {contribution_family.value}."
        )
    return InteractionContribution(
        parameter_family=contribution_family,
        n_spin=n_spin,
        weights=weights,
        contribution_kind=str(contribution.contribution_kind),
    )


def register_interaction_lowerer(
    interaction_type: type,
    lowerer: Callable[[object, ParameterFamily | None], InteractionContribution],
) -> None:
    _INTERACTION_LOWERERS[interaction_type] = lowerer


def clear_interaction_lowerers() -> None:
    _INTERACTION_LOWERERS.clear()


def _registered_interaction_contribution(
    interaction: object,
    *,
    parameter_family: ParameterFamily | None,
) -> InteractionContribution | None:
    for interaction_type in type(interaction).__mro__:
        lowerer = _INTERACTION_LOWERERS.get(interaction_type)
        if lowerer is None:
            continue
        contribution = lowerer(interaction, parameter_family)
        if not isinstance(contribution, InteractionContribution):
            raise TypeError(
                "Registered interaction lowerers must return InteractionContribution."
            )
        return _coerce_interaction_contribution(
            contribution,
            parameter_family=parameter_family,
        )
    return None


def lower_interaction_contribution(
    interaction: object,
    *,
    parameter_family: ParameterFamily | str | None = None,
) -> InteractionContribution:
    normalized_family = (
        None if parameter_family is None else normalize_parameter_family(parameter_family)
    )
    lowering_factory = getattr(interaction, "tt_interaction_contribution", None)
    if callable(lowering_factory):
        contribution = lowering_factory(parameter_family=normalized_family)
        if not isinstance(contribution, InteractionContribution):
            raise TypeError(
                "tt_interaction_contribution() must return InteractionContribution."
            )
        return _coerce_interaction_contribution(
            contribution,
            parameter_family=normalized_family,
        )

    registered = _registered_interaction_contribution(
        interaction,
        parameter_family=normalized_family,
    )
    if registered is not None:
        return registered

    if isinstance(interaction, Mapping):
        n_spin = int(interaction["n_spin"])
        weights = np.asarray(interaction["weights"])
        family = normalized_family or SPIN_PARAMETER_FAMILY
        return _coerce_interaction_contribution(
            InteractionContribution(
                parameter_family=family,
                n_spin=n_spin,
                weights=weights,
            ),
            parameter_family=normalized_family,
        )

    if normalized_family == GAUSSIAN_PARAMETER_FAMILY:
        if hasattr(interaction, "weights"):
            return _coerce_interaction_contribution(
                InteractionContribution(
                    parameter_family=GAUSSIAN_PARAMETER_FAMILY,
                    n_spin=0,
                    weights=np.asarray(getattr(interaction, "weights")),
                    contribution_kind="linear",
                ),
                parameter_family=normalized_family,
            )
        if hasattr(interaction, "inverse_weights"):
            inverse_weights = np.asarray(getattr(interaction, "inverse_weights"))
            return _coerce_interaction_contribution(
                InteractionContribution(
                    parameter_family=GAUSSIAN_PARAMETER_FAMILY,
                    n_spin=0,
                    weights=np.reciprocal(inverse_weights),
                    contribution_kind="precision",
                ),
                parameter_family=normalized_family,
            )

    if hasattr(interaction, "n_spin") and hasattr(interaction, "weights"):
        family = normalized_family or SPIN_PARAMETER_FAMILY
        return _coerce_interaction_contribution(
            InteractionContribution(
                parameter_family=family,
                n_spin=int(getattr(interaction, "n_spin")),
                weights=np.asarray(getattr(interaction, "weights")),
            ),
            parameter_family=normalized_family,
        )

    raise TypeError(
        "Unsupported interaction type. Expected either "
        "tt_interaction_contribution(), a registered interaction "
        "lowerer, a mapping with n_spin/weights, an object with n_spin and weights "
        "attributes, or a supported gaussian interaction shape."
    )


def lower_block_interactions(
    program,
    block_index: int,
    *,
    parameter_family: ParameterFamily | str | None = None,
) -> tuple[LoweredBlockInteraction, ...]:
    lowered = []
    for interaction, active_mask, source_global_inds, source_global_slices in zip(
        program.per_block_interactions[block_index],
        program.per_block_interaction_active[block_index],
        program.per_block_interaction_global_inds[block_index],
        program.per_block_interaction_global_slices[block_index],
    ):
        contribution = lower_interaction_contribution(
            interaction,
            parameter_family=parameter_family,
        )
        family_spec = parameter_family_spec(contribution.parameter_family)
        n_categories = (
            None
            if family_spec.categorical_axis is None
            else int(contribution.weights.shape[family_spec.categorical_axis])
        )
        tail_shape = tuple(
            int(size)
            for size in contribution.weights.shape[family_spec.interaction_tail_axis_start :]
        )

        lowered.append(
            LoweredBlockInteraction(
                contribution=contribution,
                active_mask=np.asarray(active_mask),
                source_global_inds=tuple(
                    int(global_index) for global_index in source_global_inds
                ),
                source_global_slices=tuple(
                    np.asarray(source_slice)
                    for source_slice in source_global_slices
                ),
                n_categories=n_categories,
                tail_shape=tail_shape,
            )
        )
    return tuple(lowered)


__all__ = [
    "GAUSSIAN_PARAMETER_FAMILY",
    "InteractionContribution",
    "LoweredBlockInteraction",
    "ParameterFamily",
    "clear_interaction_lowerers",
    "lower_block_interactions",
    "lower_interaction_contribution",
    "register_interaction_lowerer",
]
