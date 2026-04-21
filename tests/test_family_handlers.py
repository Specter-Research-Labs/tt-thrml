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

from tt_thrml.tensor_specs import PhysicalTensorSpec
from tt_thrml.runtime.compiled_program import (
    CompiledGaussianFamilyRuntime,
    CompiledInteractionGroup,
)
from tt_thrml.runtime.family_handlers import (
    _spin_parameters_to_host,
    GaussianPreparedRandom,
    _compute_categorical_interaction_group_partial,
    _compute_spin_interaction_group_partial,
    _compute_spin_interaction_partial,
    _sample_gaussian_parameters,
)
from tt_thrml.compiler.spin_ops import SPIN_PARAMETER_TO_GAMMA_SCALE


class _FakeTTNN:
    @staticmethod
    def reshape(value, shape):
        return np.reshape(value, shape)

    @staticmethod
    def multiply(lhs, rhs):
        return np.multiply(lhs, rhs)

    @staticmethod
    def concat(values, dim=0):
        return np.concatenate(values, axis=dim)


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
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 1),
            layout="tile",
            dtype="bf16",
        ),
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


def test_spin_interaction_group_partial_launches_once_for_compatible_interactions():
    captured = {}
    gather_calls = []

    def record_spin_gamma(*, ttnn, device, inputs):
        del ttnn, device
        captured["inputs"] = inputs
        return "batched-spin-gamma"

    interaction_a = SimpleNamespace(
        tail_shape=(),
        active_mask_is_all_ones=True,
        active_mask=None,
        flat_weights=np.ones((1, 1, 2, 1), dtype=np.float32),
        n_interactions=1,
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 1),
            layout="tile",
            dtype="bf16",
        ),
        categorical_tail_strides=(),
    )
    interaction_b = SimpleNamespace(
        tail_shape=(),
        active_mask_is_all_ones=True,
        active_mask=None,
        flat_weights=2.0 * np.ones((1, 1, 2, 1), dtype=np.float32),
        n_interactions=1,
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 1),
            layout="tile",
            dtype="bf16",
        ),
        categorical_tail_strides=(),
    )
    spin_source_a = np.array([[[[2.0], [3.0]]]], dtype=np.float32)
    spin_source_b = np.array([[[[5.0], [7.0]]]], dtype=np.float32)

    executor = SimpleNamespace(
        ttnn=_FakeTTNN(),
        device="fake-device",
        compiled=SimpleNamespace(state_layout="tile", index_dtype="u32"),
        _block_state_slots=[np.zeros((1, 1, 1, 2), dtype=np.float32)],
        _parameter_kernel_op_for_block=lambda block: record_spin_gamma,
        _profile_call=lambda _stage, fn: fn(),
        _ensure_tensor_batch_size=lambda tensor, batch_size: tensor,
        _gather_interaction_sources=lambda block, interaction: (
            gather_calls.append(interaction),
            ((spin_source_a,) if interaction is interaction_a else (spin_source_b,)),
            (),
            (),
        )[1:],
    )
    block = SimpleNamespace(
        block_index=0,
        n_nodes=2,
        n_interactions=1,
        state_view=SimpleNamespace(block_index=0),
    )

    result = _compute_spin_interaction_group_partial(
        executor,
        block,
        (interaction_a, interaction_b),
    )

    assert result == "batched-spin-gamma"
    assert gather_calls == [interaction_a, interaction_b]
    assert captured["inputs"].n_interactions == 2
    np.testing.assert_allclose(
        captured["inputs"].flat_weights,
        np.array([[[[1.0, 2.0], [1.0, 2.0]]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        captured["inputs"].interaction_scale,
        np.array([[[[2.0, 5.0], [3.0, 7.0]]]], dtype=np.float32),
    )


def test_spin_interaction_group_partial_keeps_no_tail_groups_unindexed():
    captured = {}

    def record_spin_gamma(*, ttnn, device, inputs):
        del ttnn, device
        captured["inputs"] = inputs
        return "grouped-spin-gamma"

    interaction_a = SimpleNamespace(
        tail_shape=(),
        active_mask_is_all_ones=True,
        active_mask=None,
        flat_weights=np.ones((1, 1, 2, 1), dtype=np.float32),
        n_interactions=1,
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 1),
            layout="tile",
            dtype="bf16",
        ),
        categorical_tail_strides=(),
    )
    interaction_b = SimpleNamespace(
        tail_shape=(),
        active_mask_is_all_ones=True,
        active_mask=None,
        flat_weights=2.0 * np.ones((1, 1, 2, 1), dtype=np.float32),
        n_interactions=1,
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 1),
            layout="tile",
            dtype="bf16",
        ),
        categorical_tail_strides=(),
    )
    interaction_group = CompiledInteractionGroup(
        interactions=(interaction_a, interaction_b),
        n_interactions=2,
        flat_weights=np.array([[[[1.0, 2.0], [1.0, 2.0]]]], dtype=np.float32),
        active_mask=np.ones((1, 1, 2, 2), dtype=np.float32),
        flat_indices=None,
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 2),
            layout="tile",
            dtype="bf16",
        ),
        flat_weights_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 2),
            layout="tile",
            dtype="bf16",
        ),
        active_mask_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 2),
            layout="tile",
            dtype="bf16",
        ),
    )
    spin_source_a = np.array([[[[2.0], [3.0]]]], dtype=np.float32)
    spin_source_b = np.array([[[[5.0], [7.0]]]], dtype=np.float32)

    executor = SimpleNamespace(
        ttnn=_FakeTTNN(),
        device="fake-device",
        compiled=SimpleNamespace(state_layout="tile", index_dtype="u32"),
        _block_state_slots=[np.zeros((1, 1, 1, 2), dtype=np.float32)],
        _parameter_kernel_op_for_block=lambda block: record_spin_gamma,
        _profile_call=lambda _stage, fn: fn(),
        _ensure_tensor_batch_size=lambda tensor, batch_size: tensor,
        _gather_interaction_sources=lambda block, interaction: (
            ((spin_source_a,) if interaction is interaction_a else (spin_source_b,)),
            (),
            (),
        ),
    )
    block = SimpleNamespace(
        block_index=0,
        n_nodes=2,
        n_interactions=1,
        state_view=SimpleNamespace(block_index=0),
    )

    result = _compute_spin_interaction_group_partial(
        executor,
        block,
        interaction_group,
    )

    assert result == "grouped-spin-gamma"
    assert captured["inputs"].flat_index is None
    np.testing.assert_allclose(
        captured["inputs"].interaction_scale,
        np.array([[[[2.0, 5.0], [3.0, 7.0]]]], dtype=np.float32),
    )


def test_spin_parameters_to_host_uses_named_inverse_scale(monkeypatch):
    executor = SimpleNamespace()
    block = SimpleNamespace()
    parameters = np.array([[[[2.0], [-4.0]]]], dtype=np.float32)

    monkeypatch.setattr(
        "tt_thrml.runtime.family_handlers._state_runtime.device_tensor_to_torch",
        lambda _executor, _parameters: parameters,
    )

    host_parameters = _spin_parameters_to_host(executor, block, parameters)

    np.testing.assert_allclose(
        host_parameters,
        np.array([1.0, -2.0], dtype=np.float32),
    )
    assert SPIN_PARAMETER_TO_GAMMA_SCALE == 2.0


def test_categorical_interaction_group_partial_keeps_node_major_order():
    captured = {}
    gather_calls = []

    def record_categorical_theta(*, ttnn, device, inputs):
        del ttnn, device
        captured["inputs"] = inputs
        return "batched-categorical-theta"

    interaction_a = SimpleNamespace(
        tail_shape=(3,),
        active_mask_is_all_ones=False,
        active_mask=np.array([[[[11.0]], [[12.0]]]], dtype=np.float32),
        flat_weights=np.array(
            [
                [
                    [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
                    [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0], [400.0, 500.0, 600.0]],
                ]
            ],
            dtype=np.float32,
        ).reshape(1, 2, 3, 3),
        n_interactions=1,
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 1, 1),
            layout="tile",
            dtype="bf16",
        ),
        categorical_tail_strides=(1,),
        fused_static_theta_bias=False,
        use_single_node_fused_theta_scale_fast_path=False,
        fused_static_theta_prefix=None,
    )
    interaction_b = SimpleNamespace(
        tail_shape=(3,),
        active_mask_is_all_ones=False,
        active_mask=np.array([[[[21.0]], [[22.0]]]], dtype=np.float32),
        flat_weights=np.array(
            [
                [
                    [[7.0, 8.0, 9.0], [70.0, 80.0, 90.0], [700.0, 800.0, 900.0]],
                    [[13.0, 14.0, 15.0], [130.0, 140.0, 150.0], [1300.0, 1400.0, 1500.0]],
                ]
            ],
            dtype=np.float32,
        ).reshape(1, 2, 3, 3),
        n_interactions=1,
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 1, 1),
            layout="tile",
            dtype="bf16",
        ),
        categorical_tail_strides=(1,),
        fused_static_theta_bias=False,
        use_single_node_fused_theta_scale_fast_path=False,
        fused_static_theta_prefix=None,
    )
    categorical_source_a = np.array([[[[3]], [[4]]]], dtype=np.uint32)
    categorical_source_b = np.array([[[[5]], [[6]]]], dtype=np.uint32)

    executor = SimpleNamespace(
        ttnn=_FakeTTNN(),
        device="fake-device",
        compiled=SimpleNamespace(categorical_layout="tile", index_dtype=np.uint32),
        _block_state_slots=[np.zeros((1, 1, 1, 2), dtype=np.float32)],
        _parameter_kernel_op_for_block=lambda block: record_categorical_theta,
        _profile_call=lambda _stage, fn: fn(),
        _gather_interaction_sources=lambda block, interaction: (
            gather_calls.append(interaction),
            (),
            (
                (categorical_source_a,)
                if interaction is interaction_a
                else (categorical_source_b,)
            ),
            (),
        )[1:],
    )
    block = SimpleNamespace(
        block_index=0,
        n_nodes=2,
        n_categories=3,
        state_view=SimpleNamespace(block_index=0),
    )

    result = _compute_categorical_interaction_group_partial(
        executor,
        block,
        (interaction_a, interaction_b),
    )

    assert result == "batched-categorical-theta"
    assert gather_calls == [interaction_a, interaction_b]
    assert captured["inputs"].n_interactions == 2
    np.testing.assert_allclose(
        captured["inputs"].flat_weights,
        np.array(
            [
                [
                    [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
                    [[7.0, 8.0, 9.0], [70.0, 80.0, 90.0], [700.0, 800.0, 900.0]],
                    [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0], [400.0, 500.0, 600.0]],
                    [[13.0, 14.0, 15.0], [130.0, 140.0, 150.0], [1300.0, 1400.0, 1500.0]],
                ]
            ],
            dtype=np.float32,
        ).reshape(1, 4, 3, 3),
    )
    np.testing.assert_array_equal(
        captured["inputs"].flat_index,
        np.array([[[[3]], [[4]], [[5]], [[6]]]], dtype=np.uint32),
    )
    np.testing.assert_allclose(
        captured["inputs"].interaction_scale,
        np.array([[[[11.0]], [[21.0]], [[12.0]], [[22.0]]]], dtype=np.float32),
    )


class _GaussianFakeTTNN:
    @staticmethod
    def reshape(value, shape):
        return np.reshape(value, shape)

    @staticmethod
    def multiply(lhs, rhs):
        return np.multiply(lhs, rhs)

    @staticmethod
    def sum(value, *, dim, keepdim):
        return np.sum(value, axis=dim, keepdims=keepdim)

    @staticmethod
    def full(shape, *, fill_value, dtype=None, layout=None, device=None):
        del layout, device
        return np.full(shape, fill_value, dtype=dtype or np.float32)

    @staticmethod
    def gt(lhs, rhs):
        return np.greater(lhs, rhs)

    @staticmethod
    def where(condition, lhs, rhs):
        return np.where(condition, lhs, rhs)

    @staticmethod
    def reciprocal(value):
        return np.reciprocal(value)

    @staticmethod
    def sqrt(value):
        return np.sqrt(value)

    @staticmethod
    def add(lhs, rhs):
        return np.add(lhs, rhs)


def test_sample_gaussian_parameters_clamps_non_positive_precision():
    runtime = CompiledGaussianFamilyRuntime(
        zero_parameters=np.zeros((1, 1, 2, 2), dtype=np.float32),
        linear_selector=np.array([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=np.float32),
        precision_selector=np.array([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=np.float32),
        parameter_spec=PhysicalTensorSpec(
            shape_tail=(1, 2, 2),
            layout="tile",
            dtype=np.float32,
        ),
    )
    executor = SimpleNamespace(
        ttnn=_GaussianFakeTTNN(),
        device="fake-device",
        _ensure_tensor_batch_size=lambda tensor, batch_size: tensor,
        _profile_call=lambda _stage, fn: fn(),
    )
    block = SimpleNamespace(
        block_index=0,
        n_nodes=2,
        family_runtime=runtime,
    )
    parameters = np.array(
        [[[[2.0, 0.0], [4.0, -3.0]]]],
        dtype=np.float32,
    )
    prepared_random = GaussianPreparedRandom(
        gaussian_noise=np.zeros((1, 1, 2, 1), dtype=np.float32)
    )

    sample = _sample_gaussian_parameters(
        executor,
        block,
        key="unused",
        parameters=parameters,
        prepared_random=prepared_random,
    )

    assert sample.shape == (1, 1, 1, 2)
    assert np.isfinite(sample).all()
    np.testing.assert_allclose(
        sample.reshape(-1),
        np.array([2.0e6, 4.0e6], dtype=np.float32),
        rtol=1e-6,
        atol=1e-3,
    )
