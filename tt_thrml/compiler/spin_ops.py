from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .ttnn_kernels import select_last_dim_expected


@dataclass(frozen=True)
class SpinGammaInputs:
    flat_weights: object
    flat_index: object | None
    interaction_scale: object | None
    n_nodes: int
    n_interactions: int


class SpinGammaOp(Protocol):
    def __call__(self, *, ttnn, device, inputs: SpinGammaInputs) -> object: ...


def _select_reference_tail(
    *,
    ttnn,
    device,
    flat_weights: object,
    flat_index: object | None,
):
    if flat_index is None:
        return flat_weights

    selected_host = select_last_dim_expected(
        ttnn.to_torch(flat_weights),
        ttnn.to_torch(flat_index).to(torch.int64),
    )
    return ttnn.from_torch(
        selected_host,
        dtype=getattr(flat_weights, "dtype", None),
        layout=getattr(flat_weights, "layout", None),
        device=device,
    )


def dense_spin_gamma_op(
    *,
    ttnn,
    device,
    inputs: SpinGammaInputs,
):
    selected = _select_reference_tail(
        ttnn=ttnn,
        device=device,
        flat_weights=inputs.flat_weights,
        flat_index=inputs.flat_index,
    )

    weighted = selected
    if inputs.interaction_scale is not None:
        weighted = ttnn.multiply(selected, inputs.interaction_scale)

    batch_size = int(weighted.shape[0])
    target_partial_shape = (batch_size, 1, inputs.n_nodes, 1)
    if inputs.n_interactions == 1:
        if tuple(weighted.shape) == target_partial_shape:
            return weighted
        return ttnn.reshape(weighted, target_partial_shape)

    weighted_4d = weighted
    if len(tuple(weighted.shape)) != 4:
        weighted_4d = ttnn.reshape(
            weighted,
            (batch_size, 1, inputs.n_nodes, inputs.n_interactions),
        )
    return ttnn.sum(weighted_4d, dim=3, keepdim=True)
