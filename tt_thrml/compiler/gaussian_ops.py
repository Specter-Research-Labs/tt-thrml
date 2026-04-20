from __future__ import annotations

from dataclasses import dataclass
import math

import jax
from jax import numpy as jnp
import numpy as np
import torch

from .ttnn_kernels import select_last_dim_expected


@dataclass(frozen=True)
class GaussianCanonicalInputs:
    flat_weights: object
    flat_index: object | None
    interaction_scale: object | None
    n_nodes: int
    n_interactions: int
    contribution_kind: str


def dense_gaussian_canonical_op(*, ttnn, device, inputs: GaussianCanonicalInputs):
    selected = inputs.flat_weights
    if inputs.flat_index is not None:
        selected_host = select_last_dim_expected(
            ttnn.to_torch(inputs.flat_weights),
            ttnn.to_torch(inputs.flat_index).to(torch.int64),
        )
        selected = ttnn.from_torch(
            selected_host,
            dtype=getattr(inputs.flat_weights, "dtype", None),
            layout=getattr(inputs.flat_weights, "layout", None),
            device=device,
        )

    weighted = (
        selected if inputs.interaction_scale is None else ttnn.multiply(selected, inputs.interaction_scale)
    )
    batch_size = int(weighted.shape[0])
    weighted_4d = weighted
    if len(tuple(weighted.shape)) != 4:
        weighted_4d = ttnn.reshape(
            weighted,
            (batch_size, 1, inputs.n_nodes, inputs.n_interactions),
        )
    partial = ttnn.sum(weighted_4d, dim=3, keepdim=True)
    zeros = ttnn.full(
        [int(partial.shape[0]), 1, inputs.n_nodes, 1],
        fill_value=0.0,
        dtype=getattr(partial, "dtype", getattr(ttnn, "bfloat16")),
        layout=getattr(inputs.flat_weights, "layout", None),
        device=device,
    )
    if inputs.contribution_kind == "linear":
        return ttnn.concat([partial, zeros], dim=-1)
    if inputs.contribution_kind == "precision":
        return ttnn.concat([zeros, partial], dim=-1)
    raise TypeError(
        "dense_gaussian_canonical_op only supports linear and precision "
        f"contribution kinds, got {inputs.contribution_kind!r}."
    )


def gaussian_noise_tensor(key, *, n_nodes: int) -> torch.Tensor:
    noise = np.asarray(
        jax.random.normal(key, shape=(n_nodes,), dtype=jnp.float32),
        dtype=np.float32,
    )
    return torch.from_numpy(noise.copy()).reshape(1, 1, n_nodes, 1)


def gather_gaussian_tail_weights_jax(
    weights: jax.Array,
    categorical_states: tuple[jax.Array, ...],
):
    tail_shape = tuple(int(size) for size in weights.shape[3:])
    if categorical_states:
        if len(categorical_states) != len(tail_shape):
            raise TypeError(
                "Gaussian categorical tail state count must match the trailing "
                "weight dimensions."
            )
        flat_index = jnp.zeros_like(jnp.asarray(categorical_states[0]), dtype=jnp.int32)
        stride = 1
        for state, size in zip(reversed(categorical_states), reversed(tail_shape), strict=True):
            flat_index = flat_index + jnp.asarray(state, dtype=jnp.int32) * int(stride)
            stride *= int(size)
        flat_weights = jnp.reshape(
            weights,
            (*weights.shape[:3], math.prod(tail_shape) if tail_shape else 1),
        )
        selector = jax.nn.one_hot(
            flat_index.astype(jnp.int32),
            flat_weights.shape[-1],
            dtype=flat_weights.dtype,
        )
        return jnp.sum(flat_weights * selector, axis=-1)

    if tail_shape:
        raise TypeError(
            "Gaussian categorical tail weights require categorical state inputs."
        )
    return weights
