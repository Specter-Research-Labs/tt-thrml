from __future__ import annotations

import sys
from types import SimpleNamespace
import types

import numpy as np
import pytest

from tt_thrml.compiler.device_contract import HostFallbackError

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    sys.modules["torch"] = fake_torch

from tt_thrml.runtime import state_runtime


class _FakeDeviceTensor:
    def __init__(self, shape):
        self.shape = shape


class _FakeTorchTensor:
    def __init__(self, shape):
        self.shape = shape


def _make_executor():
    return SimpleNamespace(
        ttnn=SimpleNamespace(
            Tensor=_FakeDeviceTensor,
        )
    )


def test_sample_is_device_tensor_rejects_rank4_numpy_arrays():
    executor = _make_executor()
    sample = np.zeros((1, 1, 1, 4), dtype=np.float32)

    assert state_runtime.sample_is_device_tensor(executor, sample) is False


def test_sample_is_device_tensor_rejects_arbitrary_rank4_objects():
    executor = _make_executor()
    sample = SimpleNamespace(shape=(1, 1, 1, 4))

    assert state_runtime.sample_is_device_tensor(executor, sample) is False


def test_sample_is_device_tensor_accepts_declared_ttnn_tensor_type():
    executor = _make_executor()
    sample = _FakeDeviceTensor((1, 1, 1, 4))

    assert state_runtime.sample_is_device_tensor(executor, sample) is True


def test_sample_is_device_tensor_accepts_namespaced_ttnn_tensor_type():
    executor = SimpleNamespace(
        ttnn=SimpleNamespace(
            tensor=SimpleNamespace(Tensor=_FakeDeviceTensor),
        )
    )
    sample = _FakeDeviceTensor((1, 1, 1, 4))

    assert state_runtime.sample_is_device_tensor(executor, sample) is True


def test_sample_is_device_tensor_accepts_torch_tensor_shape_when_available(monkeypatch):
    executor = SimpleNamespace(ttnn=SimpleNamespace())
    monkeypatch.setattr(state_runtime, "torch", SimpleNamespace(Tensor=_FakeTorchTensor))
    sample = _FakeTorchTensor((1, 1, 1, 4))

    assert state_runtime.sample_is_device_tensor(executor, sample) is True


def test_coerce_rank4_ttnn_tensor_raises_when_device_coercion_would_fallback_to_host():
    class _FakeTypedDeviceTensor(_FakeDeviceTensor):
        def __init__(self, shape, *, layout, dtype):
            super().__init__(shape)
            self.layout = layout
            self.dtype = dtype

    fake_tensor = _FakeTypedDeviceTensor(
        (1, 1, 1, 4),
        layout="tile",
        dtype="wrong-dtype",
    )
    executor = SimpleNamespace(
        device="fake:0",
        ttnn=SimpleNamespace(
            Tensor=_FakeTypedDeviceTensor,
            reshape=lambda value, shape: value,
        ),
    )

    with pytest.raises(HostFallbackError, match="rank-4 TTNN tensor coercion"):
        state_runtime.coerce_rank4_ttnn_tensor(
            executor,
            fake_tensor,
            target_shape=(1, 1, 1, 4),
            target_dtype="expected-dtype",
            layout="row-major",
            host_tensor_fn=lambda: (_ for _ in ()).throw(
                AssertionError("host_tensor_fn should not be used for device tensors")
            ),
        )
