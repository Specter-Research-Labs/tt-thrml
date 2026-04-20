from __future__ import annotations

import math
from typing import Sequence

from jax import numpy as jnp
import numpy as np
import torch

from thrml.block_management import Block, get_node_locations, verify_block_state

from ..compiler.sampler_lowering import CompiledSamplerStateSpec
from .compiled_program import CompiledBlock, CompiledStateView, node_kind_from_template
from .mesh_support import (
    canonical_replica_to_torch,
    is_multi_device_mesh,
    replicate_tensor_to_mesh_mapper,
)
from .runtime_utils import (
    bool_state_from_signed_torch,
    categorical_index_tensor,
    categorical_state_from_index_torch,
    signed_spin_tensor,
)


def _host_flat_torch_tensor(array: np.ndarray, *, batch_size: int | None = None) -> torch.Tensor:
    if batch_size is None:
        return torch.from_numpy(array.reshape(1, -1).copy()).reshape(1, 1, 1, -1)
    return torch.from_numpy(array.reshape(batch_size, -1).copy()).reshape(
        batch_size, 1, 1, -1
    )


def _torch_flat_host_numpy(
    value: torch.Tensor,
    *,
    torch_dtype,
    batch_size: int | None = None,
) -> np.ndarray:
    host_value = value.detach().cpu().to(torch_dtype)
    if batch_size is None:
        return host_value.reshape(-1).numpy()
    return host_value.reshape(batch_size, -1).numpy()


def _float32_host_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _int32_host_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def device_tensor_for_state(executor, node_kind: str, value) -> torch.Tensor:
    array = np.asarray(value)
    if node_kind == "spin":
        return signed_spin_tensor(array)
    if node_kind == "categorical":
        return categorical_index_tensor(array)
    if node_kind == "continuous":
        return _host_flat_torch_tensor(_float32_host_array(array))
    raise ValueError(f"Unsupported node kind: {node_kind}")


def device_dtype_for_node_kind(executor, node_kind: str):
    if node_kind == "spin":
        return executor.compiled.spin_state_dtype
    if node_kind == "categorical":
        return executor.compiled.categorical_state_dtype
    if node_kind == "continuous":
        return executor.compiled.spin_state_dtype
    raise ValueError(f"Unsupported node kind: {node_kind}")


def device_tensor_for_state_batch(
    executor,
    node_kind: str,
    values,
) -> torch.Tensor:
    array = np.asarray(values)
    if array.ndim <= 1:
        return device_tensor_for_state(executor, node_kind, array)

    batch_size = int(array.shape[0])
    flat_values = array.reshape(batch_size, -1)
    if node_kind == "spin":
        signed = np.where(flat_values.astype(np.bool_), 1.0, -1.0).astype(np.float32)
        return _host_flat_torch_tensor(signed, batch_size=batch_size)
    if node_kind == "categorical":
        return _host_flat_torch_tensor(
            _int32_host_array(flat_values),
            batch_size=batch_size,
        )
    if node_kind == "continuous":
        return _host_flat_torch_tensor(
            _float32_host_array(flat_values),
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported node kind: {node_kind}")


def ensure_tensor_batch_size(executor, value, batch_size: int):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0 or batch_size <= 1:
        return value

    current_batch_size = int(shape[0])
    if current_batch_size == batch_size:
        return value
    if current_batch_size != 1:
        raise ValueError(
            f"Cannot expand tensor with leading batch {current_batch_size} to {batch_size}."
        )

    cache_key = (id(value), batch_size)
    cached = executor._expanded_batch_tensor_cache.get(cache_key)
    if cached is not None:
        return cached

    repeat_sizes = (batch_size,) + (1,) * (len(shape) - 1)
    expanded = executor.ttnn.repeat(value, repeat_sizes)
    executor._expanded_batch_tensor_cache[cache_key] = expanded
    return expanded


def state_from_device_torch(
    executor,
    node_kind: str,
    value: torch.Tensor,
    *,
    dtype,
):
    if node_kind == "spin":
        return bool_state_from_signed_torch(value, dtype=dtype)
    if node_kind == "categorical":
        return categorical_state_from_index_torch(value, dtype=dtype)
    if node_kind == "continuous":
        return jnp.asarray(
            _torch_flat_host_numpy(value, torch_dtype=torch.float32),
            dtype=dtype,
        )
    raise ValueError(f"Unsupported node kind: {node_kind}")


def state_numpy_from_device_torch(
    executor,
    node_kind: str,
    value: torch.Tensor,
    *,
    dtype,
) -> np.ndarray:
    if node_kind == "spin":
        bool_values = _torch_flat_host_numpy(value, torch_dtype=torch.float32) > 0
        return np.asarray(bool_values, dtype=dtype)
    if node_kind == "categorical":
        categorical = _torch_flat_host_numpy(value, torch_dtype=torch.int64)
        return np.asarray(categorical, dtype=dtype)
    if node_kind == "continuous":
        continuous = _torch_flat_host_numpy(value, torch_dtype=torch.float32)
        return np.asarray(continuous, dtype=dtype)
    raise ValueError(f"Unsupported node kind: {node_kind}")


def state_batch_numpy_from_device_torch(
    executor,
    node_kind: str,
    value: torch.Tensor,
    *,
    dtype,
) -> np.ndarray:
    batch_size = int(value.shape[0])
    if node_kind == "spin":
        bool_values = _torch_flat_host_numpy(
            value,
            torch_dtype=torch.float32,
            batch_size=batch_size,
        ) > 0
        return np.asarray(bool_values, dtype=dtype)
    if node_kind == "categorical":
        categorical = _torch_flat_host_numpy(
            value,
            torch_dtype=torch.int64,
            batch_size=batch_size,
        )
        return np.asarray(categorical, dtype=dtype)
    if node_kind == "continuous":
        continuous = _torch_flat_host_numpy(
            value,
            torch_dtype=torch.float32,
            batch_size=batch_size,
        )
        return np.asarray(continuous, dtype=dtype)
    raise ValueError(f"Unsupported node kind: {node_kind}")


def sampler_state_spec(
    executor,
    block_index: int,
) -> CompiledSamplerStateSpec | None:
    return executor.compiled.blocks[block_index].sampler_lowering.sampler_state_spec


def device_tensor_for_sampler_state_spec(
    executor,
    spec: CompiledSamplerStateSpec,
    value,
) -> torch.Tensor:
    array = np.asarray(value, dtype=spec.output_dtype)
    if array.shape != spec.shape:
        raise ValueError(
            "Sampler state does not match compiled shape "
            f"{spec.shape}; got {array.shape}."
        )
    return _host_flat_torch_tensor(array)


def device_tensor_for_sampler_state_spec_batch(
    executor,
    spec: CompiledSamplerStateSpec,
    values,
) -> torch.Tensor:
    array = np.asarray(values, dtype=spec.output_dtype)
    if array.ndim == len(spec.shape):
        array = np.expand_dims(array, axis=0)
    expected_shape = (int(array.shape[0]), *spec.shape)
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            "Batched sampler state does not match compiled shape "
            f"{expected_shape}; got {tuple(array.shape)}."
        )
    batch_size = int(array.shape[0])
    return _host_flat_torch_tensor(array, batch_size=batch_size)


def sample_is_device_tensor(executor, sample) -> bool:
    shape = getattr(sample, "shape", None)
    return shape is not None and len(shape) == 4


def maybe_to_layout(executor, value, layout):
    current_layout = getattr(value, "layout", None)
    if current_layout == layout:
        return value

    to_layout = getattr(executor.ttnn, "to_layout", None)
    if not callable(to_layout):
        return value

    try:
        return to_layout(value, layout)
    except TypeError:
        return to_layout(value, layout=layout)


def device_tensor_to_torch(executor, value) -> torch.Tensor:
    if is_multi_device_mesh(executor.device):
        return canonical_replica_to_torch(executor.ttnn, executor.device, value)
    return executor.ttnn.to_torch(value)


def _device_upload_kwargs(executor, *, dtype, layout) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "dtype": dtype,
        "layout": layout,
        "device": executor.device,
    }
    if is_multi_device_mesh(executor.device):
        kwargs["mesh_mapper"] = replicate_tensor_to_mesh_mapper(
            executor.ttnn,
            executor.device,
        )
    return kwargs


def coerce_rank4_ttnn_tensor(
    executor,
    value,
    *,
    target_shape: tuple[int, int, int, int],
    target_dtype,
    layout,
    host_tensor_fn,
):
    if sample_is_device_tensor(executor, value):
        current_layout = getattr(value, "layout", None)
        current_dtype = getattr(value, "dtype", None)
        if (
            tuple(value.shape) == target_shape
            and current_layout == layout
            and current_dtype == target_dtype
        ):
            return value

        reshaped = (
            value
            if tuple(value.shape) == target_shape
            else executor.ttnn.reshape(value, target_shape)
        )
        reshaped = maybe_to_layout(executor, reshaped, layout)
        current_dtype = getattr(reshaped, "dtype", None)
        if current_dtype != target_dtype:
            typecast = getattr(executor.ttnn, "typecast", None)
            if callable(typecast):
                reshaped = typecast(reshaped, dtype=target_dtype)
            else:
                to_dtype = getattr(executor.ttnn, "to_dtype", None)
                if callable(to_dtype):
                    reshaped = to_dtype(reshaped, target_dtype)

        current_layout = getattr(reshaped, "layout", None)
        current_dtype = getattr(reshaped, "dtype", None)
        if (
            tuple(getattr(reshaped, "shape", ())) == target_shape
            and (current_layout is None or current_layout == layout)
            and (current_dtype is None or current_dtype == target_dtype)
        ):
            return reshaped

        host_tensor = device_tensor_to_torch(executor, reshaped)
        return executor.ttnn.from_torch(
            host_tensor.contiguous(),
            **_device_upload_kwargs(
                executor,
                dtype=target_dtype,
                layout=layout,
            ),
        )

    return executor.ttnn.from_torch(
        host_tensor_fn(),
        **_device_upload_kwargs(
            executor,
            dtype=target_dtype,
            layout=layout,
        ),
    )


def sampler_state_from_device_torch(
    executor,
    spec: CompiledSamplerStateSpec,
    value: torch.Tensor,
):
    batch_size = int(value.shape[0])
    if batch_size != 1:
        raise TypeError("copy_sampler_states() does not support batched sampler state.")
    flat_value = value.detach().cpu().reshape(-1)
    dtype_kind = np.dtype(spec.output_dtype).kind
    if dtype_kind == "f":
        flat_value = flat_value.to(torch.float32)
    elif dtype_kind in ("b", "i", "u"):
        flat_value = flat_value.to(torch.int64)
    reshaped = np.asarray(
        flat_value.numpy().reshape(spec.shape if spec.shape else ()),
        dtype=spec.output_dtype,
    )
    return jnp.asarray(reshaped, dtype=spec.output_dtype)


def coerce_sampler_state_to_device_tensor(
    executor,
    spec: CompiledSamplerStateSpec,
    sampler_state,
):
    batch_size = (
        int(sampler_state.shape[0])
        if sample_is_device_tensor(executor, sampler_state)
        else 1
    )
    return coerce_rank4_ttnn_tensor(
        executor,
        sampler_state,
        target_shape=(batch_size, 1, 1, math.prod(spec.shape) or 1),
        target_dtype=spec.device_dtype,
        layout=spec.layout,
        host_tensor_fn=lambda: device_tensor_for_sampler_state_spec(
            executor,
            spec,
            sampler_state,
        ),
    )


def ensure_block_state_storage(executor) -> None:
    if len(executor._block_state_slots) != len(executor.compiled.state_views):
        executor._block_state_slots = [None] * len(executor.compiled.state_views)
        executor._block_state_slots_row_major = [None] * len(executor.compiled.state_views)
    if len(executor._global_state_slots) != len(executor.compiled.global_slots):
        executor._global_state_slots = [None] * len(executor.compiled.global_slots)


def ensure_sampler_state_storage(executor) -> None:
    n_samplers = len(executor.program.samplers)
    if executor._sampler_states is None or len(executor._sampler_states) != n_samplers:
        executor._sampler_states = [None] * n_samplers
    if len(executor._sampler_state_slots) != n_samplers:
        executor._sampler_state_slots = [None] * n_samplers


def clear_global_source_slot_cache(executor) -> None:
    if executor._global_state_slots:
        executor._global_state_slots = [None] * len(executor._global_state_slots)


def default_sampler_states(executor) -> list[object]:
    return [sampler.init() for sampler in executor.program.samplers]


def set_sampler_states_internal(
    executor,
    sampler_states: Sequence[object],
    *,
    sampler_state_batch_size: int = 1,
) -> None:
    if len(sampler_states) != len(executor.program.samplers):
        raise ValueError("sampler_states must match the number of free-block samplers.")

    ensure_sampler_state_storage(executor)
    normalized_host_states = list(sampler_states)
    for block_index, sampler_state in enumerate(sampler_states):
        spec = sampler_state_spec(executor, block_index)
        if spec is None:
            executor._sampler_state_slots[block_index] = None
            continue

        normalized_host_states[block_index] = jnp.asarray(
            np.asarray(sampler_state, dtype=spec.output_dtype),
            dtype=spec.output_dtype,
        )
        if sampler_state_batch_size <= 1:
            device_tensor = device_tensor_for_sampler_state_spec(
                executor, spec, sampler_state
            )
        else:
            device_tensor = device_tensor_for_sampler_state_spec_batch(
                executor,
                spec,
                np.repeat(
                    np.asarray(sampler_state, dtype=spec.output_dtype)[None, ...],
                    sampler_state_batch_size,
                    axis=0,
                ),
            )

        executor._sampler_state_slots[block_index] = executor.ttnn.from_torch(
            device_tensor,
            **_device_upload_kwargs(
                executor,
                dtype=spec.device_dtype,
                layout=spec.layout,
            ),
        )

    executor._sampler_states = normalized_host_states


def ensure_sampler_states(executor) -> None:
    needs_reset = executor._sampler_states is None or len(executor._sampler_states) != len(
        executor.program.samplers
    )
    if not needs_reset and len(executor._sampler_state_slots) != len(executor.program.samplers):
        needs_reset = True
    if not needs_reset:
        for block_index in range(len(executor.program.samplers)):
            if (
                sampler_state_spec(executor, block_index) is not None
                and executor._sampler_state_slots[block_index] is None
            ):
                needs_reset = True
                break
    if needs_reset:
        set_sampler_states_internal(executor, default_sampler_states(executor))


def set_sampler_states(executor, sampler_states: Sequence[object]) -> None:
    set_sampler_states_internal(executor, sampler_states)


def copy_sampler_states(executor) -> list[object]:
    ensure_sampler_states(executor)
    assert executor._sampler_states is not None
    copied_states = list(executor._sampler_states)
    for block_index in range(len(copied_states)):
        spec = sampler_state_spec(executor, block_index)
        if spec is None:
            continue
        slot = executor._sampler_state_slots[block_index]
        if slot is None:
            raise RuntimeError(
                f"Sampler-state slot for block {block_index} was not initialized."
            )
        copied_states[block_index] = sampler_state_from_device_torch(
            executor,
            spec,
            device_tensor_to_torch(executor, slot),
        )
    return copied_states


def sampler_state_for_block(executor, block_index: int):
    ensure_sampler_states(executor)
    assert executor._sampler_states is not None
    spec = sampler_state_spec(executor, block_index)
    if spec is None:
        return executor._sampler_states[block_index]
    slot = executor._sampler_state_slots[block_index]
    if slot is None:
        raise RuntimeError(
            f"Sampler-state slot for block {block_index} was not initialized."
        )
    return slot


def store_sampler_state_for_block(executor, block_index: int, sampler_state) -> None:
    ensure_sampler_states(executor)
    assert executor._sampler_states is not None
    spec = sampler_state_spec(executor, block_index)
    if spec is None:
        executor._sampler_states[block_index] = sampler_state
        return
    executor._sampler_state_slots[block_index] = coerce_sampler_state_to_device_tensor(
        executor,
        spec,
        sampler_state,
    )


def upload_state_slice(executor, views: Sequence[CompiledStateView], values):
    return [
        executor.ttnn.from_torch(
            device_tensor_for_state(executor, view.node_kind, slot_value),
            **_device_upload_kwargs(
                executor,
                dtype=device_dtype_for_node_kind(executor, view.node_kind),
                layout=executor.compiled.state_layout,
            ),
        )
        for view, slot_value in zip(views, values, strict=True)
    ]


def upload_state_slice_batch(executor, views: Sequence[CompiledStateView], values):
    return [
        executor.ttnn.from_torch(
            device_tensor_for_state_batch(executor, view.node_kind, slot_value),
            **_device_upload_kwargs(
                executor,
                dtype=device_dtype_for_node_kind(executor, view.node_kind),
                layout=executor.compiled.state_layout,
            ),
        )
        for view, slot_value in zip(views, values, strict=True)
    ]


def load_free_state(executor, state_free) -> None:
    verify_block_state(
        executor.program.gibbs_spec.free_blocks,
        state_free,
        executor._shape_dtypes_cache,
        -1,
    )
    ensure_block_state_storage(executor)
    n_free_blocks = len(executor.program.gibbs_spec.free_blocks)
    uploaded = executor._profile_call(
        "load_state.upload_free_blocks",
        lambda: upload_state_slice(
            executor,
            executor.compiled.state_views[:n_free_blocks],
            state_free,
        ),
    )
    for block_index, tensor in enumerate(uploaded):
        executor._block_state_slots[block_index] = tensor
        executor._block_state_slots_row_major[block_index] = None
    clear_global_source_slot_cache(executor)
    set_sampler_states_internal(executor, default_sampler_states(executor))


def load_free_state_batch(executor, state_free_batch: Sequence[object]) -> None:
    if not state_free_batch:
        raise ValueError("load_free_state_batch() requires at least one state.")

    for state_free in state_free_batch:
        verify_block_state(
            executor.program.gibbs_spec.free_blocks,
            state_free,
            executor._shape_dtypes_cache,
            -1,
        )

    ensure_block_state_storage(executor)
    n_free_blocks = len(executor.program.gibbs_spec.free_blocks)
    views = executor.compiled.state_views[:n_free_blocks]
    stacked_values = [
        np.stack(
            [np.asarray(state_free[block_index]) for state_free in state_free_batch],
            axis=0,
        )
        for block_index in range(n_free_blocks)
    ]
    uploaded = executor._profile_call(
        "load_state.upload_free_blocks_batch",
        lambda: upload_state_slice_batch(executor, views, stacked_values),
    )
    for block_index, tensor in enumerate(uploaded):
        executor._block_state_slots[block_index] = tensor
        executor._block_state_slots_row_major[block_index] = None
    clear_global_source_slot_cache(executor)
    set_sampler_states_internal(
        executor,
        default_sampler_states(executor),
        sampler_state_batch_size=len(state_free_batch),
    )


def load_clamp_state(executor, state_clamp) -> None:
    verify_block_state(
        executor.program.gibbs_spec.clamped_blocks,
        state_clamp,
        executor._shape_dtypes_cache,
        -1,
    )
    ensure_block_state_storage(executor)
    n_free_blocks = len(executor.program.gibbs_spec.free_blocks)
    uploaded = executor._profile_call(
        "load_state.upload_clamp_blocks",
        lambda: upload_state_slice(
            executor,
            executor.compiled.state_views[n_free_blocks:],
            state_clamp,
        ),
    )
    for offset, tensor in enumerate(uploaded, start=n_free_blocks):
        executor._block_state_slots[offset] = tensor
        executor._block_state_slots_row_major[offset] = None
    clear_global_source_slot_cache(executor)


def load_clamp_state_batch(executor, state_clamp, *, batch_size: int) -> None:
    verify_block_state(
        executor.program.gibbs_spec.clamped_blocks,
        state_clamp,
        executor._shape_dtypes_cache,
        -1,
    )
    ensure_block_state_storage(executor)
    n_free_blocks = len(executor.program.gibbs_spec.free_blocks)
    views = executor.compiled.state_views[n_free_blocks:]
    stacked_values = [
        np.broadcast_to(np.asarray(slot_value), (batch_size, *np.asarray(slot_value).shape)).copy()
        for slot_value in state_clamp
    ]
    uploaded = executor._profile_call(
        "load_state.upload_clamp_blocks_batch",
        lambda: upload_state_slice_batch(executor, views, stacked_values),
    )
    for offset, tensor in enumerate(uploaded, start=n_free_blocks):
        executor._block_state_slots[offset] = tensor
        executor._block_state_slots_row_major[offset] = None
    clear_global_source_slot_cache(executor)


def load_state(executor, state_free, state_clamp) -> None:
    load_free_state(executor, state_free)
    load_clamp_state(executor, state_clamp)


def state_is_loaded(executor) -> bool:
    return bool(executor._block_state_slots) and all(
        slot is not None for slot in executor._block_state_slots
    )


def require_state(executor) -> None:
    if not state_is_loaded(executor):
        raise RuntimeError("Call load_state() before running the TT executor.")


def view_for_block(executor, block: Block) -> CompiledStateView:
    block_index = executor._program_block_id_to_index.get(id(block))
    if block_index is not None:
        return executor.compiled.state_views[block_index]

    template_sd = executor.program.gibbs_spec.node_shape_struct[block.node_type]
    node_kind = node_kind_from_template(template_sd)
    global_slot_index, positions = get_node_locations(block, executor.program.gibbs_spec)
    positions_np = np.asarray(positions, dtype=np.int32).copy()
    host_gather_index = torch.from_numpy(positions_np).reshape(1, 1, 1, -1).to(
        torch.int64
    )
    gather_index = executor.ttnn.from_torch(
        host_gather_index,
        **_device_upload_kwargs(
            executor,
            dtype=executor.compiled.index_dtype,
            layout=executor.compiled.state_layout,
        ),
    )
    return CompiledStateView(
        block_index=-1,
        global_slot_index=int(global_slot_index),
        node_kind=node_kind,
        n_nodes=len(block.nodes),
        output_dtype=template_sd.dtype,
        positions=positions_np,
        gather_index=gather_index,
        host_gather_index=host_gather_index,
    )


def source_slot_torch_for_global_slot(executor, global_slot_index: int) -> torch.Tensor:
    if executor._enable_global_source_slot_cache:
        cached = executor._global_state_slots[global_slot_index]
        if cached is not None:
            return cached

    slot_meta = executor.compiled.global_slots[global_slot_index]
    tensors = [
        device_tensor_to_torch(executor, executor._block_state_slots[block_index])
        for block_index in slot_meta.block_indices
    ]
    source_slot = tensors[0] if len(tensors) == 1 else torch.concat(tensors, dim=-1)
    if not source_slot.is_floating_point():
        source_slot = source_slot.to(torch.int64)
    if executor._enable_global_source_slot_cache:
        executor._global_state_slots[global_slot_index] = source_slot
    return source_slot


def gather_block_torch_from_source_slot(
    executor,
    view: CompiledStateView,
    *,
    source_slot: torch.Tensor,
) -> torch.Tensor:
    return torch.gather(source_slot, -1, view.host_gather_index)


def gather_block_torch_from_source_slot_batch(
    executor,
    view: CompiledStateView,
    *,
    source_slot: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(source_slot.shape[0])
    return torch.gather(
        source_slot,
        -1,
        view.host_gather_index.expand(batch_size, -1, -1, -1),
    )


def gather_block_torch(executor, view: CompiledStateView) -> torch.Tensor:
    if view.block_index >= 0:
        return device_tensor_to_torch(executor, executor._block_state_slots[view.block_index])
    return gather_block_torch_from_source_slot(
        executor,
        view,
        source_slot=source_slot_torch_for_global_slot(executor, view.global_slot_index),
    )


def row_major_block_state(executor, block_index: int):
    cached = executor._block_state_slots_row_major[block_index]
    if cached is not None:
        return cached

    if block_index not in executor.compiled.row_major_cache_block_indices:
        return maybe_to_layout(
            executor,
            executor._block_state_slots[block_index],
            executor.compiled.categorical_layout,
        )

    row_major_state = executor._profile_call(
        f"block_state.row_major_cache.block{block_index}",
        lambda: maybe_to_layout(
            executor,
            executor._block_state_slots[block_index],
            executor.compiled.categorical_layout,
        ),
    )
    executor._block_state_slots_row_major[block_index] = row_major_state
    return row_major_state


def coerce_sample_to_block_state_tensor(
    executor,
    block: CompiledBlock,
    sample,
    *,
    layout,
):
    batch_size = int(sample.shape[0]) if sample_is_device_tensor(executor, sample) else 1
    return coerce_rank4_ttnn_tensor(
        executor,
        sample,
        target_shape=(batch_size, 1, 1, block.n_nodes),
        target_dtype=device_dtype_for_node_kind(executor, block.state_view.node_kind),
        layout=layout,
        host_tensor_fn=lambda: device_tensor_for_state(
            executor,
            block.state_view.node_kind,
            sample,
        ),
    )


def write_block_state(executor, block_index: int, new_state) -> int:
    def _write():
        block = executor.compiled.blocks[block_index]
        view = block.state_view
        tile_state = executor._profile_call(
            f"block_state.write.block{block_index}.tile",
            lambda: coerce_sample_to_block_state_tensor(
                executor,
                block,
                new_state,
                layout=executor.compiled.state_layout,
            ),
        )
        executor._block_state_slots[view.block_index] = tile_state
        executor._block_state_slots_row_major[view.block_index] = None
        clear_global_source_slot_cache(executor)
        return view.global_slot_index

    return executor._profile_call("block_state.write", _write)


def source_slot_numpy_batch_piece(executor, block_index: int) -> np.ndarray:
    slot = device_tensor_to_torch(executor, executor._block_state_slots[block_index])
    batch_size = int(slot.shape[0])
    node_kind = executor.compiled.state_views[block_index].node_kind
    if node_kind == "spin":
        return _torch_flat_host_numpy(
            slot,
            torch_dtype=torch.float32,
            batch_size=batch_size,
        ) > 0
    if node_kind == "categorical":
        return _torch_flat_host_numpy(
            slot,
            torch_dtype=torch.int64,
            batch_size=batch_size,
        )
    if node_kind == "continuous":
        return _torch_flat_host_numpy(
            slot,
            torch_dtype=torch.float32,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported node kind: {node_kind}")
