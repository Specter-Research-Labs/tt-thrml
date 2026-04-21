from __future__ import annotations

import jax
from jax import numpy as jnp
import numpy as np
import torch

from ..device_contract import raise_host_fallback_disabled
from ...tensor_specs import _first_available_attr
from .runtime import (
    restore_output_tensor,
    run_flatbuffer,
    supports_direct_ttnn_inputs,
    supports_direct_ttnn_outputs,
)


def ttnn_input_as_torch(ttnn, tensor, *, dtype):
    del ttnn
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise TypeError("TTNN tensors must expose a shape for TT-MLIR compilation.")
    return torch.zeros(tuple(int(dim) for dim in shape), dtype=dtype).contiguous()


def optional_ttnn_input_as_torch(ttnn, tensor, *, dtype):
    if tensor is None:
        return None
    return ttnn_input_as_torch(ttnn, tensor, dtype=dtype)


def default_float32_scale(flat_weights: torch.Tensor, flat_index: torch.Tensor | None):
    template = flat_index if flat_index is not None else flat_weights
    return torch.ones_like(template, dtype=torch.float32)


def default_scale_tensor_like(ttnn, *, device, flat_weights, flat_index=None):
    full = getattr(ttnn, "full", None)
    if not callable(full):
        raise_host_fallback_disabled(
            "TT-MLIR direct default scale construction",
            remedy=(
                "Provide explicit interaction_scale tensors, or use a TTNN build "
                "that exposes ttnn.full for direct device execution."
            ),
        )
    template = flat_index if flat_index is not None else flat_weights
    layout = getattr(
        template,
        "layout",
        getattr(
            flat_weights,
            "layout",
            _first_available_attr(ttnn, "ROW_MAJOR_LAYOUT", "TILE_LAYOUT"),
        ),
    )
    dtype = getattr(flat_weights, "dtype", _first_available_attr(ttnn, "bfloat16"))
    return full(
        list(tensor_shape(template)),
        fill_value=1.0,
        dtype=dtype,
        layout=layout,
        device=device,
    )


def reference_output_spec(ttnn, reference):
    output_layout = getattr(
        reference,
        "layout",
        _first_available_attr(ttnn, "TILE_LAYOUT", "ROW_MAJOR_LAYOUT"),
    )
    output_dtype = getattr(reference, "dtype", _first_available_attr(ttnn, "bfloat16"))
    return output_layout, output_dtype


def tensor_shape(value) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"Expected tensor-like value with shape metadata, got {type(value)!r}.")
    return tuple(int(dim) for dim in tuple(shape))


def normalized_dtype_name(dtype) -> str | None:
    if dtype is None:
        return None
    text = str(dtype).lower()
    if "bfloat16" in text or text == "bf16":
        return "bfloat16"
    if "float32" in text or text.endswith("float"):
        return "float32"
    if "uint32" in text:
        return "uint32"
    if "int32" in text:
        return "int32"
    if "int64" in text:
        return "int64"
    return text.replace("torch.", "").replace("dtype.", "")


def abstract_jax_input(shape: tuple[int, ...], *, dtype) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(tuple(int(dim) for dim in shape), _jax_dtype(dtype))


def prefer_direct_device_output(ttnn, reference) -> bool:
    del ttnn, reference
    return True


def execute_single_output_flatbuffer(
    *,
    config,
    ttnn,
    device,
    flatbuffer_path,
    output_reference,
    input_tensors=None,
    op_name: str,
    direct_input_tensors=None,
    input_tensors_factory=None,
    allow_host_input_tensors: bool = False,
    run_flatbuffer_fn=run_flatbuffer,
    supports_direct_ttnn_inputs_fn=supports_direct_ttnn_inputs,
    supports_direct_ttnn_outputs_fn=supports_direct_ttnn_outputs,
):
    runtime_inputs = None if input_tensors is None else list(input_tensors)
    if direct_input_tensors is not None:
        if supports_direct_ttnn_inputs_fn(device=device):
            runtime_inputs = list(direct_input_tensors)
        else:
            raise_host_fallback_disabled(
                f"{op_name} TT-MLIR input materialization",
                remedy="Use a runtime with direct TTNN bridge support.",
            )
    elif not allow_host_input_tensors:
        raise_host_fallback_disabled(
            f"{op_name} TT-MLIR input materialization",
            remedy=(
                "Pass direct TTNN tensors to the runtime bridge. Host placeholder "
                "inputs are only valid for explicit compiler/debug tests."
            ),
        )
    elif runtime_inputs is None:
        if input_tensors_factory is None:
            raise RuntimeError(
                f"{op_name} requires input_tensors or input_tensors_factory."
            )
        runtime_inputs = list(input_tensors_factory())

    use_direct_device_output = prefer_direct_device_output(ttnn, output_reference)
    if use_direct_device_output and not supports_direct_ttnn_outputs_fn(device=device):
        raise_host_fallback_disabled(
            f"{op_name} TT-MLIR output restoration",
            remedy="Use a runtime with direct TTNN output bridge support.",
        )

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
    if _is_torch_tensor(execution.outputs[0]):
        raise_host_fallback_disabled(
            f"{op_name} TT-MLIR output restoration",
            remedy="Use a runtime with direct TTNN output bridge support.",
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


def _is_torch_tensor(value) -> bool:
    tensor_type = getattr(torch, "Tensor", None)
    if tensor_type is None or tensor_type is object:
        return False
    return isinstance(value, tensor_type)


def _jax_dtype(dtype):
    name = normalized_dtype_name(dtype)
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "uint32":
        return jnp.uint32
    if name == "int32":
        return jnp.int32
    if name == "int64":
        return jnp.int64
    raise TypeError(f"Unsupported JAX placeholder dtype {dtype!r}.")
