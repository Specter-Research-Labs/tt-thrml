from __future__ import annotations

import math
from typing import Sequence

import torch


def _signed_spin_values(condition: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.where(
        condition.to(torch.bool), torch.ones_like(like), -torch.ones_like(like)
    )


def _spin_product(
    spin_conditions: Sequence[torch.Tensor], like: torch.Tensor
) -> torch.Tensor:
    product = torch.ones_like(like)
    for condition in spin_conditions:
        product = product * _signed_spin_values(condition, like)
    return product


def _checked_prod(shape: Sequence[int]) -> int:
    return math.prod(shape) if shape else 1


def _flatten_tail_index(
    categorical_conditions: Sequence[torch.Tensor], tail_shape: Sequence[int]
) -> torch.Tensor | None:
    if len(categorical_conditions) != len(tail_shape):
        raise ValueError(
            "The number of categorical conditions must match the trailing weight dimensions."
        )

    if not categorical_conditions:
        return None

    flat_index = torch.zeros_like(categorical_conditions[0], dtype=torch.int64)
    stride = 1
    for condition, size in zip(reversed(categorical_conditions), reversed(tail_shape)):
        flat_index = flat_index + condition.to(torch.int64) * stride
        stride *= size
    return flat_index


def select_last_dim_expected(
    values: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    selector = torch.nn.functional.one_hot(
        index.squeeze(-1).to(torch.int64),
        num_classes=values.shape[-1],
    ).to(values.dtype)
    return torch.sum(values * selector, dim=-1, keepdim=True)


def _select_gamma_weights(
    weights: torch.Tensor,
    active_mask: torch.Tensor,
    categorical_conditions: Sequence[torch.Tensor],
) -> torch.Tensor:
    tail_shape = weights.shape[active_mask.ndim :]
    if categorical_conditions:
        flat_index = _flatten_tail_index(categorical_conditions, tail_shape)
        flat_weights = weights.reshape(*active_mask.shape, _checked_prod(tail_shape))
        return select_last_dim_expected(flat_weights, flat_index.unsqueeze(-1)).squeeze(-1)

    if tail_shape:
        raise ValueError(
            "Categorical conditions are required when weights include trailing categorical axes."
        )

    return weights


def _select_theta_weights(
    weights: torch.Tensor,
    active_mask: torch.Tensor,
    categorical_conditions: Sequence[torch.Tensor],
) -> torch.Tensor:
    category_dim = active_mask.ndim
    tail_shape = weights.shape[category_dim + 1 :]
    if categorical_conditions:
        flat_index = _flatten_tail_index(categorical_conditions, tail_shape)
        n_categories = weights.shape[category_dim]
        flat_weights = weights.reshape(
            *active_mask.shape, n_categories, _checked_prod(tail_shape)
        )
        gather_index = (
            flat_index.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(*active_mask.shape, n_categories, 1)
        )
        return select_last_dim_expected(flat_weights, gather_index).squeeze(-1)

    if tail_shape:
        raise ValueError(
            "Categorical conditions are required when weights include trailing categorical axes."
        )

    return weights


def spin_gamma_dense_expected(
    weights: torch.Tensor,
    active_mask: torch.Tensor,
    spin_conditions: Sequence[torch.Tensor],
    categorical_conditions: Sequence[torch.Tensor] = (),
) -> torch.Tensor:
    gathered = _select_gamma_weights(weights, active_mask, categorical_conditions)
    product = _spin_product(spin_conditions, active_mask.to(gathered.dtype))
    return torch.sum(
        gathered * active_mask.to(gathered.dtype) * product, dim=-1, keepdim=True
    )


def categorical_theta_dense_expected(
    weights: torch.Tensor,
    active_mask: torch.Tensor,
    spin_conditions: Sequence[torch.Tensor],
    categorical_conditions: Sequence[torch.Tensor],
) -> torch.Tensor:
    selected = _select_theta_weights(weights, active_mask, categorical_conditions)
    spin_scale = _spin_product(
        spin_conditions, active_mask.to(selected.dtype)
    ) * active_mask.to(selected.dtype)
    return torch.sum(selected * spin_scale.unsqueeze(-1), dim=active_mask.ndim - 1)
