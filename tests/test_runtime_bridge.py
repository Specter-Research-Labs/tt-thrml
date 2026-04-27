from __future__ import annotations

from types import SimpleNamespace

import pytest

from tt_thrml.executor import _resolve_runtime_bridge


def _bridge_module() -> SimpleNamespace:
    return SimpleNamespace(
        create_runtime_device_from_ttnn=lambda device: ("device", device),
        create_runtime_tensor_from_ttnn=lambda tensor, borrow: ("tensor", tensor, borrow),
        get_ttnn_tensor_from_runtime_tensor=lambda tensor: ("ttnn", tensor),
    )


def test_resolve_runtime_bridge_accepts_top_level_bindings():
    tt_runtime = _bridge_module()

    assert _resolve_runtime_bridge(tt_runtime) is tt_runtime


def test_resolve_runtime_bridge_accepts_internal_utils_bindings():
    runtime_utils = _bridge_module()
    tt_runtime = SimpleNamespace(_ttmlir_runtime=SimpleNamespace(utils=runtime_utils))

    assert _resolve_runtime_bridge(tt_runtime) is runtime_utils


def test_resolve_runtime_bridge_rejects_missing_bindings():
    with pytest.raises(RuntimeError, match="TTNN runtime bridge APIs"):
        _resolve_runtime_bridge(SimpleNamespace())
