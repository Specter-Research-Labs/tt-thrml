from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import jax
from jax import numpy as jnp
import numpy as np
import torch

from .device_contract import HostFallbackError, raise_host_fallback_disabled
from .ttnn_kernels import select_last_dim_expected

from ..runtime.runtime_utils import seed_from_jax_key
from ..tensor_specs import _first_available_attr


def tail_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= int(size)
    return tuple(reversed(strides))


def _ttnn_cast(*, ttnn, value: object, dtype: object):
    current_dtype = getattr(value, "dtype", None)
    if current_dtype == dtype:
        return value

    typecast = getattr(ttnn, "typecast", None)
    if callable(typecast):
        cast_value = typecast(value, dtype=dtype)
    else:
        to_dtype = getattr(ttnn, "to_dtype", None)
        if not callable(to_dtype):
            raise TypeError(
                "TT backend cannot cast categorical index tensors to the required dtype "
                f"{dtype!r}."
            )
        cast_value = to_dtype(value, dtype)

    if getattr(cast_value, "dtype", None) != dtype:
        raise TypeError(
            "TT backend returned the wrong dtype while casting categorical index tensors: "
            f"expected {dtype!r}, got {getattr(cast_value, 'dtype', None)!r}."
        )
    return cast_value


def categorical_flat_index_tensor_device(
    *,
    ttnn,
    device,
    categorical_sources: Sequence[object],
    categorical_tail_strides: Sequence[int],
    layout: object,
    index_dtype: object,
) -> object | None:
    if not categorical_sources:
        return None

    source_shape = getattr(categorical_sources[0], "shape", None)
    if source_shape is None:
        raise TypeError("Categorical source tensors must expose a shape.")

    if len(categorical_sources) == 1 and int(categorical_tail_strides[0]) == 1:
        return _ttnn_cast(
            ttnn=ttnn,
            value=categorical_sources[0],
            dtype=index_dtype,
        )

    flat_index = None
    for source, stride in zip(categorical_sources, categorical_tail_strides):
        source_index = _ttnn_cast(ttnn=ttnn, value=source, dtype=index_dtype)
        if int(stride) == 1:
            scaled = source_index
        else:
            stride_tensor = ttnn.full(
                list(source_shape),
                fill_value=int(stride),
                dtype=index_dtype,
                layout=layout,
                device=device,
            )
            scaled = ttnn.multiply(source_index, stride_tensor)
        flat_index = scaled if flat_index is None else ttnn.add(flat_index, scaled)

    if flat_index is None:
        return None

    return _ttnn_cast(ttnn=ttnn, value=flat_index, dtype=index_dtype)


def categorical_tail_index_tensor_device(
    *,
    ttnn,
    device,
    categorical_sources: Sequence[object],
    categorical_tail_strides: Sequence[int],
    n_nodes: int,
    n_interactions: int,
    n_categories: int | None,
    layout: object,
    index_dtype: object,
) -> object | None:
    flat_index = categorical_flat_index_tensor_device(
        ttnn=ttnn,
        device=device,
        categorical_sources=categorical_sources,
        categorical_tail_strides=categorical_tail_strides,
        layout=layout,
        index_dtype=index_dtype,
    )
    if flat_index is None:
        return None

    if n_categories is None:
        target_shape = (1, 1, n_nodes, n_interactions, 1)
        if tuple(getattr(flat_index, "shape", ())) == target_shape:
            return flat_index
        return ttnn.reshape(flat_index, target_shape)

    expanded = ttnn.reshape(flat_index, (1, n_nodes * n_interactions, 1, 1))
    if n_categories > 1:
        expanded = ttnn.repeat(expanded, (1, 1, n_categories, 1))
    return expanded


@dataclass(frozen=True)
class CategoricalThetaInputs:
    flat_weights: object
    flat_index: object | None
    interaction_scale: object
    n_nodes: int
    n_interactions: int
    n_categories: int


class CategoricalThetaOp(Protocol):
    def __call__(self, *, ttnn, device, inputs: CategoricalThetaInputs) -> object: ...


@dataclass(frozen=True)
class TTNNCategoricalSamplingPlan:
    n_users: int
    n_categories: int
    padded_categories: int
    input_indices: object
    output_indices: object
    k: object
    p: object
    temp: object


class CategoricalSampler(Protocol):
    def __call__(
        self,
        *,
        ttnn,
        device,
        logits: object,
        key,
        output_dtype,
        plan: TTNNCategoricalSamplingPlan | None,
        gumbel_noise: object | None = None,
    ) -> object: ...


def _device_layout_for_like(ttnn, value: object):
    layout = getattr(value, "layout", None)
    if layout is not None:
        return layout
    return _first_available_attr(ttnn, "ROW_MAJOR_LAYOUT", "TILE_LAYOUT")


def _device_dtype_for_like(ttnn, value: object):
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return dtype
    return getattr(ttnn, "bfloat16")


def _maybe_to_layout(ttnn, value: object, layout: object):
    current_layout = getattr(value, "layout", None)
    if current_layout == layout:
        return value

    to_layout = getattr(ttnn, "to_layout", None)
    if not callable(to_layout):
        return value

    try:
        return to_layout(value, layout)
    except TypeError:
        return to_layout(value, layout=layout)


def dense_categorical_theta_op(
    *,
    ttnn,
    device,
    inputs: CategoricalThetaInputs,
):
    selected = inputs.flat_weights
    if inputs.flat_index is not None:
        raise_host_fallback_disabled(
            "native categorical tail selection",
            remedy=(
                "Use the TT-MLIR parameter-kernel backend for categorical blocks with "
                "indexed categorical tails."
            ),
        )

    scaled = ttnn.multiply(selected, inputs.interaction_scale)
    partial = ttnn.reshape(
        ttnn.sum(
            ttnn.reshape(
                scaled,
                (1, inputs.n_nodes, inputs.n_interactions, inputs.n_categories),
            ),
            dim=2,
            keepdim=False,
        ),
        (1, 1, inputs.n_nodes, inputs.n_categories),
    )
    return partial


def categorical_interaction_scale_tensor_device(
    *,
    ttnn,
    active_mask: object,
    spin_sources: Sequence[object],
    n_nodes: int,
    n_interactions: int,
    active_mask_is_all_ones: bool = False,
    skip_flatten_if_aligned: bool = False,
):
    interaction_scale = None if active_mask_is_all_ones else active_mask
    target_shape = (1, n_nodes * n_interactions, 1, 1)
    for gathered_spin in spin_sources:
        if skip_flatten_if_aligned and tuple(getattr(gathered_spin, "shape", ())) == target_shape:
            flat_spin = gathered_spin
        else:
            flat_spin = ttnn.reshape(
                gathered_spin,
                target_shape,
            )
        interaction_scale = (
            flat_spin
            if interaction_scale is None
            else ttnn.multiply(interaction_scale, flat_spin)
        )
    if interaction_scale is None:
        return active_mask
    return interaction_scale


def supports_ttnn_categorical_sampling(
    ttnn,
    *,
    n_users: int,
    n_categories: int,
) -> bool:
    return (
        callable(getattr(ttnn, "sampling", None))
        and n_users <= 32
        and n_categories <= 32
    )


def compile_ttnn_categorical_sampling_plan(
    *,
    ttnn,
    device,
    n_users: int,
    n_categories: int,
) -> TTNNCategoricalSamplingPlan | None:
    if not supports_ttnn_categorical_sampling(
        ttnn, n_users=n_users, n_categories=n_categories
    ):
        return None

    padded_categories = 32
    index_dtype = _first_available_attr(ttnn, "uint32", "int32")
    index_values = (
        torch.arange(padded_categories, dtype=torch.int64)
        .reshape(1, 1, 1, padded_categories)
        .repeat(1, 1, 32, 1)
    )
    k_values = torch.full((32,), fill_value=n_categories, dtype=torch.int64)
    p_values = torch.ones(32, dtype=torch.float32)
    temp_values = torch.ones(32, dtype=torch.float32)

    return TTNNCategoricalSamplingPlan(
        n_users=n_users,
        n_categories=n_categories,
        padded_categories=padded_categories,
        input_indices=ttnn.from_torch(
            index_values,
            dtype=index_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        ),
        output_indices=ttnn.from_torch(
            torch.arange(n_users, dtype=torch.int64).reshape(1, 1, 1, n_users),
            dtype=index_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        ),
        k=ttnn.from_torch(
            k_values,
            dtype=_first_available_attr(ttnn, "uint32", "int32"),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        ),
        p=ttnn.from_torch(
            p_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        ),
        temp=ttnn.from_torch(
            temp_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        ),
    )


def exact_ttnn_categorical_sampler(
    *,
    ttnn,
    device,
    logits: object,
    key,
    output_dtype,
    plan: TTNNCategoricalSamplingPlan | None,
    gumbel_noise: object | None = None,
):
    del output_dtype, plan
    if gumbel_noise is None:
        raise HostFallbackError(
            "exact_ttnn_categorical_sampler requires explicit gumbel noise; "
            "implicit host-side categorical RNG is disabled."
        )
    argmax = getattr(ttnn, "argmax", None)
    if not callable(argmax):
        raise TypeError(
            "TT backend must expose argmax() for exact TT categorical sampling."
        )

    shape = getattr(logits, "shape", None)
    if shape is None or len(shape) != 4:
        raise TypeError("Categorical logits must expose a rank-4 TT tensor shape.")

    n_users = int(shape[2])
    n_categories = int(shape[3])
    noisy_logits = ttnn.add(logits, gumbel_noise)
    sampled = argmax(noisy_logits, dim=-1, keepdim=False)
    sampled = ttnn.reshape(sampled, (1, 1, 1, n_users))
    sampled = _maybe_to_layout(
        ttnn,
        sampled,
        getattr(ttnn, "TILE_LAYOUT", _device_layout_for_like(ttnn, logits)),
    )
    sampled = _ttnn_cast(
        ttnn=ttnn,
        value=sampled,
        dtype=_first_available_attr(ttnn, "uint32", "int32"),
    )
    return sampled


def categorical_gumbel_noise_tensor(keys, *, n_users: int, n_categories: int) -> torch.Tensor:
    gumbel = np.asarray(
        [
            jax.random.gumbel(
                key,
                shape=(int(n_users), int(n_categories)),
                dtype=jnp.float32,
            )
            for key in keys
        ],
        dtype=np.float32,
    )
    return torch.from_numpy(gumbel.copy()).reshape(
        len(keys), 1, int(n_users), int(n_categories)
    )


def ttnn_categorical_sampler(
    *,
    ttnn,
    device,
    logits: object,
    key,
    output_dtype,
    plan: TTNNCategoricalSamplingPlan | None,
    gumbel_noise: object | None = None,
) -> object:
    del gumbel_noise, output_dtype
    if plan is None:
        raise ValueError(
            "TTNN categorical sampling requires a compiled TTNN sampling plan."
        )
    sampling = getattr(ttnn, "sampling", None)
    if not callable(sampling):
        raise TypeError("TT backend must expose sampling() for TTNN categorical sampling.")

    padded_logits = logits
    current_categories = plan.n_categories
    if plan.padded_categories > current_categories:
        padded_logits = ttnn.concat(
            [
                padded_logits,
                ttnn.full(
                    [1, 1, plan.n_users, plan.padded_categories - current_categories],
                    fill_value=-1.0e4,
                    dtype=ttnn.bfloat16,
                    layout=_first_available_attr(ttnn, "TILE_LAYOUT", "ROW_MAJOR_LAYOUT"),
                    device=device,
                ),
            ],
            dim=-1,
        )

    if plan.n_users < 32:
        padded_logits = ttnn.concat(
            [
                padded_logits,
                ttnn.full(
                    [1, 1, 32 - plan.n_users, plan.padded_categories],
                    fill_value=0.0,
                    dtype=ttnn.bfloat16,
                    layout=_first_available_attr(ttnn, "TILE_LAYOUT", "ROW_MAJOR_LAYOUT"),
                    device=device,
                ),
            ],
            dim=2,
        )

    sampled = sampling(
        padded_logits,
        plan.input_indices,
        k=plan.k,
        p=plan.p,
        temp=plan.temp,
        seed=seed_from_jax_key(key),
    )
    sampled = ttnn.reshape(sampled, (1, 1, 1, -1))
    sampled = ttnn.gather(sampled, -1, index=plan.output_indices)
    return _ttnn_cast(
        ttnn=ttnn,
        value=ttnn.reshape(sampled, (1, 1, 1, plan.n_users)),
        dtype=_first_available_attr(ttnn, "uint32", "int32"),
    )
