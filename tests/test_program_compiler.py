from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.float32 = "float32"
    torch_stub.int64 = "int64"
    torch_stub.from_numpy = lambda value: value
    torch_stub.zeros = lambda *args, **kwargs: ("zeros", args, kwargs)
    torch_stub.as_tensor = lambda value: value
    sys.modules["torch"] = torch_stub

from tt_thrml.compiler import program_compiler
from tt_thrml.runtime.compiled_program import CompiledInteraction, CompiledInteractionExecution, CompiledInteractionGroup
from tt_thrml.runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    ParameterKernelBackend,
    SPIN_PARAMETER_FAMILY,
)
from tt_thrml.tensor_specs import (
    categorical_active_mask_tensor_spec,
    categorical_weight_tensor_spec,
    interaction_scale_tensor_spec,
    spin_gaussian_weight_tensor_spec,
)


def test_compile_block_rejects_ttnn_fallback_categorical_backend(monkeypatch):
    fake_program = SimpleNamespace(
        gibbs_spec=SimpleNamespace(
            free_blocks=[SimpleNamespace(nodes=("n0",))],
        )
    )
    fake_state_view = SimpleNamespace(n_nodes=1)
    fake_context = SimpleNamespace(
        ttnn="fake-ttnn",
        device="fake:0",
        state_layout="tile",
        categorical_layout="row_major",
        spin_state_dtype="bf16",
        categorical_state_dtype="u32",
        index_dtype="u32",
    )

    monkeypatch.setattr(
        program_compiler,
        "resolve_sampler_lowering_config",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        program_compiler,
        "compile_sampler_lowering",
        lambda *args, **kwargs: SimpleNamespace(
            parameter_family=CATEGORICAL_PARAMETER_FAMILY,
            n_categories=3,
        ),
    )

    with pytest.raises(TypeError, match="require TT-MLIR parameter kernels"):
        program_compiler._compile_block(
            program=fake_program,
            block_index=0,
            sampler=object(),
            state_view=fake_state_view,
            global_slots=[],
            state_views=[],
            row_major_cache_block_indices=set(),
            context=fake_context,
            parameter_kernel_backends={
                CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.NATIVE
            },
        )


def test_sampler_lowering_accepts_clean_tt_parameter_family_hook():
    from tt_thrml.compiler import sampler_lowering

    class CleanSampler:
        def tt_parameter_family(self):
            return SPIN_PARAMETER_FAMILY

    assert (
        sampler_lowering.declared_parameter_family(CleanSampler())
        is SPIN_PARAMETER_FAMILY
    )


def test_compile_block_builds_gaussian_selectors_instead_of_index_tensors(monkeypatch):
    class FakeTTNN:
        def full(self, shape, *, fill_value, dtype=None, layout=None, device=None):
            return SimpleNamespace(
                value=fill_value,
                shape=tuple(shape),
                dtype=dtype,
                layout=layout,
                device=device,
            )

        def from_torch(self, value, *, dtype=None, layout=None, device=None):
            return SimpleNamespace(
                value=value,
                dtype=dtype,
                layout=layout,
                device=device,
            )

    fake_program = SimpleNamespace(
        gibbs_spec=SimpleNamespace(
            free_blocks=[SimpleNamespace(nodes=("n0", "n1"))],
        )
    )
    fake_state_view = SimpleNamespace(n_nodes=2, output_dtype="float32")
    fake_context = SimpleNamespace(
        ttnn=FakeTTNN(),
        device="fake:0",
        state_layout="tile",
        categorical_layout="row_major",
        spin_state_dtype="bf16",
        categorical_state_dtype="u32",
        index_dtype="u32",
    )

    monkeypatch.setattr(
        program_compiler,
        "resolve_sampler_lowering_config",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        program_compiler,
        "compile_sampler_lowering",
        lambda *args, **kwargs: SimpleNamespace(
            parameter_family=GAUSSIAN_PARAMETER_FAMILY,
            n_categories=None,
        ),
    )
    monkeypatch.setattr(
        program_compiler,
        "lower_block_interactions",
        lambda *args, **kwargs: [],
    )

    compiled = program_compiler._compile_block(
        program=fake_program,
        block_index=0,
        sampler=object(),
        state_view=fake_state_view,
        global_slots=[],
        state_views=[],
        row_major_cache_block_indices=set(),
        context=fake_context,
        parameter_kernel_backends=None,
    )

    linear_selector = compiled.family_runtime.linear_selector.value
    precision_selector = compiled.family_runtime.precision_selector.value

    assert linear_selector.shape == (1, 1, 2, 2)
    assert precision_selector.shape == (1, 1, 2, 2)
    assert linear_selector.tolist() == [[[[1.0, 0.0], [1.0, 0.0]]]]
    assert precision_selector.tolist() == [[[[0.0, 1.0], [0.0, 1.0]]]]


class _GroupingTTNN:
    def concat(self, values, dim=0):
        return np.concatenate(values, axis=dim)

    def reshape(self, value, shape):
        return np.reshape(value, shape)

    def full(self, shape, *, fill_value, dtype=None, layout=None, device=None):
        del layout, device
        return np.full(shape, fill_value, dtype=np.float32 if dtype is None else dtype)


def test_group_compiled_interactions_merges_mixed_spin_tail_sizes():
    context = SimpleNamespace(ttnn=_GroupingTTNN(), device="fake:0")
    interactions = (
        CompiledInteraction(
            contribution_kind="default",
            n_interactions=1,
            tail_shape=(),
            categorical_tail_strides=(),
            execution=CompiledInteractionExecution((), (), ()),
            flat_weights=np.ones((1, 1, 2, 1), dtype=np.float32),
            active_mask=np.ones((1, 1, 2, 1), dtype=np.float32),
            active_mask_is_all_ones=True,
            parameter_spec=interaction_scale_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                has_tail=False,
                layout="tile",
                dtype=np.float32,
            ),
            flat_weights_spec=spin_gaussian_weight_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                tail_size=1,
                layout="tile",
                dtype=np.float32,
            ),
            active_mask_spec=interaction_scale_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                has_tail=False,
                layout="tile",
                dtype=np.float32,
            ),
            fused_static_theta_bias=False,
            use_single_node_fused_theta_scale_fast_path=False,
            fused_static_theta_prefix=None,
        ),
        CompiledInteraction(
            contribution_kind="default",
            n_interactions=1,
            tail_shape=(3,),
            categorical_tail_strides=(1,),
            execution=CompiledInteractionExecution((), (), ()),
            flat_weights=np.ones((1, 1, 2, 1, 3), dtype=np.float32),
            active_mask=np.ones((1, 1, 2, 1, 1), dtype=np.float32),
            active_mask_is_all_ones=True,
            parameter_spec=interaction_scale_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                has_tail=True,
                layout="tile",
                dtype=np.float32,
            ),
            flat_weights_spec=spin_gaussian_weight_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                tail_size=3,
                layout="tile",
                dtype=np.float32,
            ),
            active_mask_spec=interaction_scale_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                has_tail=True,
                layout="tile",
                dtype=np.float32,
            ),
            fused_static_theta_bias=False,
            use_single_node_fused_theta_scale_fast_path=False,
            fused_static_theta_prefix=None,
        ),
    )

    grouped = program_compiler._group_compiled_interactions(
        interactions=interactions,
        context=context,
        parameter_family=SPIN_PARAMETER_FAMILY,
    )

    assert len(grouped) == 1
    assert isinstance(grouped[0], CompiledInteractionGroup)
    assert grouped[0].flat_weights.shape == (1, 1, 2, 2, 3)
    assert grouped[0].active_mask.shape == (1, 1, 2, 2, 1)
    assert np.all(grouped[0].flat_weights[:, :, :, 0, 1:] == 0.0)


def test_group_compiled_interactions_merges_mixed_categorical_tail_sizes():
    context = SimpleNamespace(ttnn=_GroupingTTNN(), device="fake:0")
    interactions = (
        CompiledInteraction(
            contribution_kind="default",
            n_interactions=1,
            tail_shape=(),
            categorical_tail_strides=(),
            execution=CompiledInteractionExecution((), (), ()),
            flat_weights=np.ones((1, 2, 3, 1), dtype=np.float32),
            active_mask=np.ones((1, 2, 1, 1), dtype=np.float32),
            active_mask_is_all_ones=True,
            parameter_spec=interaction_scale_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                has_tail=False,
                layout="row_major",
                dtype=np.float32,
            ),
            flat_weights_spec=categorical_weight_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                n_categories=3,
                tail_size=1,
                layout="row_major",
                dtype=np.float32,
            ),
            active_mask_spec=categorical_active_mask_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                layout="row_major",
                dtype=np.float32,
            ),
            fused_static_theta_bias=False,
            use_single_node_fused_theta_scale_fast_path=False,
            fused_static_theta_prefix=None,
        ),
        CompiledInteraction(
            contribution_kind="default",
            n_interactions=1,
            tail_shape=(3,),
            categorical_tail_strides=(1,),
            execution=CompiledInteractionExecution((), (), ()),
            flat_weights=np.ones((1, 2, 3, 3), dtype=np.float32),
            active_mask=np.ones((1, 2, 1, 1), dtype=np.float32),
            active_mask_is_all_ones=True,
            parameter_spec=interaction_scale_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                has_tail=True,
                layout="row_major",
                dtype=np.float32,
            ),
            flat_weights_spec=categorical_weight_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                n_categories=3,
                tail_size=3,
                layout="row_major",
                dtype=np.float32,
            ),
            active_mask_spec=categorical_active_mask_tensor_spec(
                n_nodes=2,
                n_interactions=1,
                layout="row_major",
                dtype=np.float32,
            ),
            fused_static_theta_bias=False,
            use_single_node_fused_theta_scale_fast_path=False,
            fused_static_theta_prefix=None,
        ),
    )

    grouped = program_compiler._group_compiled_interactions(
        interactions=interactions,
        context=context,
        parameter_family=CATEGORICAL_PARAMETER_FAMILY,
    )

    assert len(grouped) == 1
    assert isinstance(grouped[0], CompiledInteractionGroup)
    assert grouped[0].flat_weights.shape == (1, 4, 3, 3)
    assert grouped[0].active_mask.shape == (1, 4, 1, 1)
    assert np.all(grouped[0].flat_weights[:, :2, :, 1:] == 0.0)
