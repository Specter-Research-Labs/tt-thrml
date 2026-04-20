from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.float32 = "float32"
    torch_stub.int64 = "int64"
    torch_stub.from_numpy = lambda value: value
    torch_stub.zeros = lambda *args, **kwargs: ("zeros", args, kwargs)
    torch_stub.as_tensor = lambda value: value
    sys.modules["torch"] = torch_stub

from tt_thrml.runtime.family_handlers import _compute_spin_interaction_partial


class _FakeTTNN:
    @staticmethod
    def reshape(value, shape):
        return np.reshape(value, shape)

    @staticmethod
    def multiply(lhs, rhs):
        return np.multiply(lhs, rhs)


def test_spin_interaction_partial_supports_continuous_tails():
    captured = {}

    def record_spin_gamma(*, ttnn, device, inputs):
        del ttnn, device
        captured["inputs"] = inputs
        return "spin-gamma"

    executor = SimpleNamespace(
        ttnn=_FakeTTNN(),
        device="fake-device",
        compiled=SimpleNamespace(state_layout="tile", index_dtype="u32"),
        _block_state_slots=[np.zeros((1, 1, 1, 2), dtype=np.float32)],
        _parameter_kernel_op_for_block=lambda block: record_spin_gamma,
        _profile_call=lambda _stage, fn: fn(),
        _ensure_tensor_batch_size=lambda tensor, batch_size: tensor,
    )
    block = SimpleNamespace(
        block_index=0,
        n_nodes=2,
        n_interactions=1,
        state_view=SimpleNamespace(block_index=0),
    )
    interaction = SimpleNamespace(
        tail_shape=(),
        active_mask_is_all_ones=True,
        active_mask=None,
        flat_weights="flat-weights",
        n_interactions=1,
        parameter_scale_shape_tail=(1, 2, 1),
        categorical_tail_strides=(),
    )
    spin_source = np.array([[[[2.0], [3.0]]]], dtype=np.float32)
    continuous_source = np.array([[[[5.0], [7.0]]]], dtype=np.float32)

    result = _compute_spin_interaction_partial(
        executor,
        block,
        interaction,
        (spin_source,),
        (),
        (continuous_source,),
    )

    assert result == "spin-gamma"
    assert captured["inputs"].flat_index is None
    np.testing.assert_allclose(
        captured["inputs"].interaction_scale,
        np.array([[[[10.0], [21.0]]]], dtype=np.float32),
    )
