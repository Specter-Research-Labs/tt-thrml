from __future__ import annotations

from dataclasses import dataclass
import sys
import types

import numpy as np
import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.float32 = "float32"
    torch_stub.int64 = "int64"

    class _FakeTorchArray:
        def __init__(self, value):
            self.value = np.asarray(value)

        def reshape(self, *shape):
            target_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
            return _FakeTorchArray(self.value.reshape(target_shape))

        def repeat(self, *sizes):
            repeat_sizes = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], tuple) else sizes
            return _FakeTorchArray(np.tile(self.value, repeat_sizes))

    torch_stub.arange = lambda *args, dtype=None: _FakeTorchArray(
        np.arange(*args, dtype=np.int64)
    )
    torch_stub.full = lambda shape, *, fill_value, dtype=None: _FakeTorchArray(
        np.full(shape, fill_value=fill_value, dtype=np.float32)
    )
    torch_stub.ones = lambda shape, *, dtype=None: _FakeTorchArray(
        np.ones(shape, dtype=np.float32)
    )
    torch_stub.from_numpy = lambda value: value
    torch_stub.zeros = lambda *args, **kwargs: ("zeros", args, kwargs)
    torch_stub.as_tensor = lambda value: value
    sys.modules["torch"] = torch_stub

from tt_thrml.compiler.categorical_ops import (
    _ttnn_cast,
    categorical_flat_index_tensor_device,
    compile_ttnn_categorical_sampling_plan,
    exact_ttnn_categorical_sampler,
    ttnn_categorical_sampler,
)


@dataclass
class TinyTensor:
    value: np.ndarray
    dtype: object
    layout: object

    @property
    def shape(self):
        return self.value.shape


class TinyTTNN:
    ROW_MAJOR_LAYOUT = "row_major"
    uint32 = "uint32"
    int32 = "int32"

    def __init__(self):
        self.typecast_calls: list[object] = []

    def typecast(self, value, *, dtype):
        self.typecast_calls.append(dtype)
        return TinyTensor(
            value=np.asarray(value.value, dtype=np.uint32),
            dtype=dtype,
            layout=value.layout,
        )

    def full(self, shape, *, fill_value, dtype=None, layout=None, device=None):
        del device
        return TinyTensor(
            value=np.full(shape, fill_value=fill_value, dtype=np.uint32),
            dtype=dtype,
            layout=layout,
        )

    def multiply(self, lhs, rhs):
        return TinyTensor(
            value=np.asarray(lhs.value, dtype=np.uint32) * np.asarray(rhs.value, dtype=np.uint32),
            dtype=lhs.dtype,
            layout=lhs.layout,
        )

    def add(self, lhs, rhs):
        return TinyTensor(
            value=np.asarray(lhs.value, dtype=np.uint32) + np.asarray(rhs.value, dtype=np.uint32),
            dtype=lhs.dtype,
            layout=lhs.layout,
        )


class PlanTTNN:
    bfloat16 = "bfloat16"
    uint32 = "uint32"
    int32 = "int32"
    ROW_MAJOR_LAYOUT = "row_major"

    def from_torch(self, value, *, dtype=None, layout=None, device=None):
        del device
        return types.SimpleNamespace(value=value, dtype=dtype, layout=layout)

    def sampling(self, *args, **kwargs):
        raise AssertionError("sampling() should not be called while compiling the plan")


def _row_major_index_tensor(values) -> TinyTensor:
    return TinyTensor(
        value=np.asarray(values, dtype=np.uint32),
        dtype=TinyTTNN.uint32,
        layout=TinyTTNN.ROW_MAJOR_LAYOUT,
    )


def test_categorical_flat_index_keeps_index_dtype_for_row_major_sources():
    ttnn = TinyTTNN()
    first = _row_major_index_tensor([[[[1]], [[0]]]])
    second = _row_major_index_tensor([[[[2]], [[1]]]])

    flat_index = categorical_flat_index_tensor_device(
        ttnn=ttnn,
        device="fake",
        categorical_sources=(first, second),
        categorical_tail_strides=(3, 1),
        layout=TinyTTNN.ROW_MAJOR_LAYOUT,
        index_dtype=TinyTTNN.uint32,
    )

    np.testing.assert_array_equal(
        flat_index.value,
        np.asarray([[[[5]], [[1]]]], dtype=np.uint32),
    )
    assert flat_index.dtype == TinyTTNN.uint32
    assert ttnn.typecast_calls == []


def test_ttnn_cast_requires_real_cast_support():
    with pytest.raises(TypeError, match="cannot cast categorical index tensors"):
        _ttnn_cast(
            ttnn=object(),
            value=TinyTensor(
                value=np.asarray([1], dtype=np.uint32),
                dtype="wrong",
                layout=TinyTTNN.ROW_MAJOR_LAYOUT,
            ),
            dtype=TinyTTNN.uint32,
        )


def test_ttnn_categorical_sampler_requires_explicit_sampling_plan():
    with pytest.raises(ValueError, match="requires a compiled TTNN sampling plan"):
        ttnn_categorical_sampler(
            ttnn=TinyTTNN(),
            device="fake",
            logits=object(),
            key=object(),
            output_dtype="u32",
            plan=None,
        )


def test_exact_ttnn_categorical_sampler_requires_argmax():
    with pytest.raises(TypeError, match="must expose argmax"):
        exact_ttnn_categorical_sampler(
            ttnn=TinyTTNN(),
            device="fake",
            logits=object(),
            key=object(),
            output_dtype="u32",
            plan=None,
        )


def test_ttnn_categorical_sampling_plan_uses_unsigned_index_dtype():
    fake_ttnn = PlanTTNN()

    plan = compile_ttnn_categorical_sampling_plan(
        ttnn=fake_ttnn,
        device="fake",
        n_users=4,
        n_categories=3,
    )

    assert plan is not None
    assert plan.input_indices.dtype == fake_ttnn.uint32
    assert plan.output_indices.dtype == fake_ttnn.uint32
