from __future__ import annotations

import jax
from jax import numpy as jnp
import numpy as np
import torch

from .runtime import restore_output_tensor, run_flatbuffer, supports_direct_ttnn_inputs


def ttnn_input_as_torch(ttnn, tensor, *, dtype):
    return ttnn.to_torch(tensor).to(dtype).contiguous()


def optional_ttnn_input_as_torch(ttnn, tensor, *, dtype):
    if tensor is None:
        return None
    return ttnn_input_as_torch(ttnn, tensor, dtype=dtype)


def default_float32_scale(flat_weights: torch.Tensor, flat_index: torch.Tensor | None):
    template = flat_index if flat_index is not None else flat_weights
    return torch.ones_like(template, dtype=torch.float32)


def reference_output_spec(ttnn, reference):
    output_layout = getattr(
        reference,
        "layout",
        getattr(ttnn, "TILE_LAYOUT", getattr(ttnn, "ROW_MAJOR_LAYOUT", None)),
    )
    output_dtype = getattr(reference, "dtype", getattr(ttnn, "bfloat16"))
    return output_layout, output_dtype


def prefer_direct_device_output(ttnn, reference) -> bool:
    output_layout, _output_dtype = reference_output_spec(ttnn, reference)
    row_major_layout = getattr(ttnn, "ROW_MAJOR_LAYOUT", None)
    if row_major_layout is not None and output_layout == row_major_layout:
        return False
    return True


def execute_single_output_flatbuffer(
    *,
    config,
    ttnn,
    device,
    flatbuffer_path,
    input_tensors,
    output_reference,
    op_name: str,
    direct_input_tensors=None,
    run_flatbuffer_fn=run_flatbuffer,
    supports_direct_ttnn_inputs_fn=supports_direct_ttnn_inputs,
):
    runtime_inputs = list(input_tensors)
    if (
        direct_input_tensors is not None
        and supports_direct_ttnn_inputs_fn(device=device)
    ):
        runtime_inputs = list(direct_input_tensors)

    use_direct_device_output = prefer_direct_device_output(ttnn, output_reference)

    execution = run_flatbuffer_fn(
        config,
        flatbuffer_path=flatbuffer_path,
        input_tensors=runtime_inputs,
        device=device,
        prefer_device_output=use_direct_device_output,
    )
    if len(execution.outputs) != 1:
        raise RuntimeError(
            f"Expected one output tensor from tt-mlir {op_name}, "
            f"got {len(execution.outputs)}."
        )

    output_layout, output_dtype = reference_output_spec(ttnn, output_reference)
    return restore_output_tensor(
        ttnn,
        device=device,
        output=execution.outputs[0],
        output_dtype=output_dtype,
        output_layout=output_layout,
    )


def torch_to_jax(value: torch.Tensor, *, dtype) -> jax.Array:
    return jnp.asarray(value.detach().cpu().numpy(), dtype=dtype)


def numpy_to_torch(value, *, dtype, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value, dtype=dtype).copy())
    if shape is not None:
        return tensor.reshape(*shape)
    return tensor
