from __future__ import annotations

import jax
from jax import numpy as jnp
import numpy as np
import torch


def signed_spin_tensor(values: np.ndarray) -> torch.Tensor:
    bool_values = np.asarray(values, dtype=np.bool_)
    signed = np.where(bool_values, 1.0, -1.0).astype(np.float32)
    return torch.from_numpy(signed).reshape(1, 1, 1, -1)


def bool_state_from_signed_torch(values: torch.Tensor, *, dtype) -> jax.Array:
    bool_values = values.detach().cpu().to(torch.float32).reshape(-1).numpy() > 0
    return jnp.asarray(bool_values, dtype=dtype)


def categorical_index_tensor(values: np.ndarray) -> torch.Tensor:
    categorical = np.asarray(values).astype(np.int32, copy=False)
    return torch.from_numpy(categorical.copy()).reshape(1, 1, 1, -1)

def categorical_state_from_index_torch(values: torch.Tensor, *, dtype) -> jax.Array:
    categorical = values.detach().cpu().to(torch.int64).reshape(-1).numpy()
    return jnp.asarray(categorical, dtype=dtype)


def seed_from_jax_key(key) -> int:
    key_words = jax.random.key_data(key).astype(jnp.uint32)
    return int(int(key_words[0]) ^ int(key_words[1]))


def spin_threshold_logits_tensor(key, *, n_nodes: int) -> torch.Tensor:
    uniform = jax.random.uniform(
        key,
        shape=(int(n_nodes),),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    threshold_logits = jnp.log(uniform) - jnp.log1p(-uniform)
    threshold_logits_np = np.asarray(threshold_logits, dtype=np.float32).copy()
    return torch.from_numpy(threshold_logits_np).reshape(1, 1, int(n_nodes), 1)


def spin_threshold_logits_batch_tensor(keys, *, n_nodes: int) -> torch.Tensor:
    thresholds = []
    for key in keys:
        uniform = jax.random.uniform(
            key,
            shape=(int(n_nodes),),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        thresholds.append(
            np.asarray(jnp.log(uniform) - jnp.log1p(-uniform), dtype=np.float32)
        )
    if not thresholds:
        return torch.zeros((0, 1, int(n_nodes), 1), dtype=torch.float32)
    return torch.from_numpy(np.stack(thresholds, axis=0).copy()).reshape(
        len(thresholds), 1, int(n_nodes), 1
    )


def stack_sample_history(history):
    if not history:
        return []

    stacked = []
    for block_index in range(len(history[0])):
        samples = [entry[block_index] for entry in history]
        if len(samples) == 1:
            stacked.append(jax.tree.map(lambda value: value[None], samples[0]))
        else:
            stacked.append(
                jax.tree.map(lambda *values: jnp.stack(values, axis=0), *samples)
            )
    return stacked
