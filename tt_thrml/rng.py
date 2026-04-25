"""Bulk RNG buffer generation and management.

Per-block per-sweep tensors are shaped to match the compiled kernel's
rng_slice input:
  spin:         (n_nodes, 1)
  categorical:  (n_nodes, n_categories)
  gaussian:     (n_nodes,)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .core import BulkRNGBuffers, Family, RNGSpec


def make_rng_spec(blocks: tuple, n_sweeps: int) -> RNGSpec:
    spin_blocks, categorical_blocks, gaussian_blocks = [], [], []
    nodes_per_block, categories_per_block = [], []
    for block in blocks:
        spec = block.spec
        nodes_per_block.append(spec.n_nodes)
        categories_per_block.append(spec.n_categories)
        if spec.family == Family.SPIN:
            spin_blocks.append(spec.block_index)
        elif spec.family == Family.CATEGORICAL:
            categorical_blocks.append(spec.block_index)
        elif spec.family == Family.GAUSSIAN:
            gaussian_blocks.append(spec.block_index)

    return RNGSpec(
        n_sweeps=n_sweeps,
        spin_blocks=tuple(spin_blocks),
        categorical_blocks=tuple(categorical_blocks),
        gaussian_blocks=tuple(gaussian_blocks),
        nodes_per_block=tuple(nodes_per_block),
        categories_per_block=tuple(categories_per_block),
    )


def _upload(ttnn, device, arr: np.ndarray, state_dtype, layout) -> object:
    import torch  # type: ignore[reportMissingImports]

    tensor = torch.from_numpy(arr.astype(np.float32).copy()).contiguous()
    return ttnn.from_torch(tensor, dtype=state_dtype, layout=layout, device=device)


def generate_bulk_rng(
    key,
    rng_spec: RNGSpec,
    ttnn,
    device,
    *,
    state_dtype,
    layout,
) -> BulkRNGBuffers:
    key, spin_key, cat_key, gauss_key = jax.random.split(key, 4)

    spin_logits = (
        _generate_spin_rng(spin_key, rng_spec, ttnn, device, state_dtype, layout) if rng_spec.spin_blocks else None
    )
    categorical_gumbel = (
        _generate_categorical_rng(cat_key, rng_spec, ttnn, device, state_dtype, layout)
        if rng_spec.categorical_blocks
        else None
    )
    gaussian_noise = (
        _generate_gaussian_rng(gauss_key, rng_spec, ttnn, device, state_dtype, layout)
        if rng_spec.gaussian_blocks
        else None
    )

    return BulkRNGBuffers(
        spin_threshold_logits=spin_logits,
        categorical_gumbel=categorical_gumbel,
        gaussian_noise=gaussian_noise,
    )


def _generate_spin_rng(key, rng_spec, ttnn, device, state_dtype, layout):
    result = {}
    keys = jax.random.split(key, len(rng_spec.spin_blocks) * rng_spec.n_sweeps)
    key_idx = 0
    for block_idx in rng_spec.spin_blocks:
        n_nodes = rng_spec.nodes_per_block[block_idx]
        sweep_tensors = []
        for _ in range(rng_spec.n_sweeps):
            u = jax.random.uniform(keys[key_idx], shape=(n_nodes,), minval=1e-7, maxval=1.0 - 1e-7)
            logits = jnp.log(u / (1.0 - u))
            arr = np.asarray(logits, dtype=np.float32).reshape(n_nodes, 1)
            sweep_tensors.append(_upload(ttnn, device, arr, state_dtype, layout))
            key_idx += 1
        result[block_idx] = sweep_tensors
    return result


def _generate_categorical_rng(key, rng_spec, ttnn, device, state_dtype, layout):
    result = {}
    keys = jax.random.split(key, len(rng_spec.categorical_blocks) * rng_spec.n_sweeps)
    key_idx = 0
    for block_idx in rng_spec.categorical_blocks:
        n_nodes = rng_spec.nodes_per_block[block_idx]
        n_categories = rng_spec.categories_per_block[block_idx] or 2
        sweep_tensors = []
        for _ in range(rng_spec.n_sweeps):
            gumbel = jax.random.gumbel(keys[key_idx], shape=(n_nodes, n_categories), dtype=jnp.float32)
            arr = np.asarray(gumbel, dtype=np.float32).reshape(n_nodes, n_categories)
            sweep_tensors.append(_upload(ttnn, device, arr, state_dtype, layout))
            key_idx += 1
        result[block_idx] = sweep_tensors
    return result


def _generate_gaussian_rng(key, rng_spec, ttnn, device, state_dtype, layout):
    result = {}
    keys = jax.random.split(key, len(rng_spec.gaussian_blocks) * rng_spec.n_sweeps)
    key_idx = 0
    for block_idx in rng_spec.gaussian_blocks:
        n_nodes = rng_spec.nodes_per_block[block_idx]
        sweep_tensors = []
        for _ in range(rng_spec.n_sweeps):
            noise = jax.random.normal(keys[key_idx], shape=(n_nodes,), dtype=jnp.float32)
            arr = np.asarray(noise, dtype=np.float32).reshape(n_nodes)
            sweep_tensors.append(_upload(ttnn, device, arr, state_dtype, layout))
            key_idx += 1
        result[block_idx] = sweep_tensors
    return result


def generate_bulk_rng_for_schedule(
    key,
    rng_spec: RNGSpec,
    schedule,
    n_free_blocks: int,
    ttnn,
    device,
    *,
    state_dtype,
    layout,
) -> BulkRNGBuffers:
    """Generate bulk RNG matching THRML's exact sample_with_observation key derivation.

    THRML splits the initial key as (sample_key, warmup_key) = split(key, 2), then:
    - warmup: split(warmup_key, n_warmup) → sweep keys
    - samples: split(sample_key, n_samples-1) → outer → split(outer, steps_per_sample) → sweeps
    Each sweep key is split into n_free_blocks per-block keys, and block i uses keys[i].
    This matches jax.random.normal/gumbel calls exactly, giving sample-by-sample parity.
    """
    sample_key, warmup_key = jax.random.split(key, 2)

    sweep_keys = []
    if schedule.n_warmup > 0:
        for wk in jax.random.split(warmup_key, schedule.n_warmup):
            sweep_keys.append(wk)
    if schedule.n_samples > 1:
        for outer in jax.random.split(sample_key, schedule.n_samples - 1):
            for sk in jax.random.split(outer, schedule.steps_per_sample):
                sweep_keys.append(sk)

    free_spin = [bi for bi in rng_spec.spin_blocks if bi < n_free_blocks]
    free_cat = [bi for bi in rng_spec.categorical_blocks if bi < n_free_blocks]
    free_gauss = [bi for bi in rng_spec.gaussian_blocks if bi < n_free_blocks]

    spin_result = {bi: [] for bi in free_spin}
    cat_result = {bi: [] for bi in free_cat}
    gauss_result = {bi: [] for bi in free_gauss}

    for sweep_key in sweep_keys:
        per_block_keys = jax.random.split(sweep_key, n_free_blocks)

        for bi in free_spin:
            bk = jax.random.split(per_block_keys[bi], 2)[0]
            n = rng_spec.nodes_per_block[bi]
            u = jax.random.uniform(bk, shape=(n,))
            logits = jnp.log(u / (1.0 - u))
            arr = np.asarray(logits, dtype=np.float32).reshape(n, 1)
            spin_result[bi].append(_upload(ttnn, device, arr, state_dtype, layout))

        for bi in free_cat:
            bk = jax.random.split(per_block_keys[bi], 2)[0]
            n = rng_spec.nodes_per_block[bi]
            n_cat = rng_spec.categories_per_block[bi] or 2
            gumbel = jax.random.gumbel(bk, shape=(n, n_cat), dtype=jnp.float32)
            arr = np.asarray(gumbel, dtype=np.float32).reshape(n, n_cat)
            cat_result[bi].append(_upload(ttnn, device, arr, state_dtype, layout))

        for bi in free_gauss:
            bk = jax.random.split(per_block_keys[bi], 2)[0]
            n = rng_spec.nodes_per_block[bi]
            noise = jax.random.normal(bk, shape=(n,), dtype=jnp.float32)
            arr = np.asarray(noise, dtype=np.float32).reshape(n)
            gauss_result[bi].append(_upload(ttnn, device, arr, state_dtype, layout))

    return BulkRNGBuffers(
        spin_threshold_logits=spin_result if spin_result else None,
        categorical_gumbel=cat_result if cat_result else None,
        gaussian_noise=gauss_result if gauss_result else None,
    )


def slice_rng_for_sweep(
    ttnn,
    rng_buffers: BulkRNGBuffers,
    sweep_index: int,
    block_index: int,
    family: Family,
) -> object:
    if family == Family.SPIN:
        if rng_buffers.spin_threshold_logits is None:
            raise RuntimeError("Spin RNG buffer is not available.")
        return rng_buffers.spin_threshold_logits[block_index][sweep_index]
    if family == Family.CATEGORICAL:
        if rng_buffers.categorical_gumbel is None:
            raise RuntimeError("Categorical RNG buffer is not available.")
        return rng_buffers.categorical_gumbel[block_index][sweep_index]
    if family == Family.GAUSSIAN:
        if rng_buffers.gaussian_noise is None:
            raise RuntimeError("Gaussian RNG buffer is not available.")
        return rng_buffers.gaussian_noise[block_index][sweep_index]
    raise ValueError(f"Unknown family: {family}")
