from __future__ import annotations

import sys
import types
from types import SimpleNamespace

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
from tt_thrml.runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    ParameterKernelBackend,
    SPIN_PARAMETER_FAMILY,
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
