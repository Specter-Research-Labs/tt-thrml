from __future__ import annotations

import pytest

pytest.importorskip("torch")

import tt_thrml.runtime.backend_executor as backend_executor

from tt_thrml.runtime_config import (
    ParameterKernelBackend,
    SPIN_PARAMETER_FAMILY,
    make_backend_binding,
)


class _FakeProgram:
    pass


def test_borrow_executor_reuses_compiled_program_across_backend_executor_keys(monkeypatch):
    backend_executor.clear_compiled_program_cache()
    monkeypatch.setattr(backend_executor, "program_supported_by_executor", lambda program: True)
    init_calls = []

    class RecordingExecutor:
        def __init__(
            self,
            *,
            ttnn,
            device,
            program,
            compiled=None,
            parameter_kernel_ops=None,
            parameter_kernel_backends=None,
            profiler=None,
            profile_sync=False,
            progress=None,
        ):
            del profiler, profile_sync, progress
            self.ttnn = ttnn
            self.device = device
            self.program = program
            self.parameter_kernel_ops = dict(parameter_kernel_ops or {})
            self.parameter_kernel_backends = dict(parameter_kernel_backends or {})
            self.compiled = compiled if compiled is not None else object()
            init_calls.append(
                {
                    "compiled": compiled,
                    "parameter_kernel_ops": self.parameter_kernel_ops,
                    "parameter_kernel_backends": self.parameter_kernel_backends,
                    "executor": self,
                }
            )

    monkeypatch.setattr(backend_executor, "TTProgramExecutor", RecordingExecutor)

    ttnn = object()
    device = object()
    base_spin_op = object()
    overlay_spin_op = object()
    program = _FakeProgram()
    base_backend = make_backend_binding(
        ttnn,
        device,
        parameter_kernel_ops={SPIN_PARAMETER_FAMILY: base_spin_op},
        parameter_kernel_backends={SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM},
    )
    overlay_backend = base_backend.with_parameter_kernel_op(
        SPIN_PARAMETER_FAMILY,
        overlay_spin_op,
    )

    try:
        with backend_executor.borrow_executor(program=program, backend=base_backend) as first:
            pass
        with backend_executor.borrow_executor(program=program, backend=base_backend) as second:
            pass
        with backend_executor.borrow_executor(program=program, backend=overlay_backend) as third:
            pass
    finally:
        backend_executor.clear_compiled_program_cache()

    assert first is second
    assert third is not first
    assert len(init_calls) == 2
    assert init_calls[0]["compiled"] is None
    assert init_calls[0]["parameter_kernel_ops"] == {
        SPIN_PARAMETER_FAMILY: base_spin_op,
    }
    assert init_calls[0]["parameter_kernel_backends"] == {
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM,
    }
    assert init_calls[1]["compiled"] is first.compiled
    assert init_calls[1]["parameter_kernel_ops"] == {
        SPIN_PARAMETER_FAMILY: overlay_spin_op,
    }
    assert init_calls[1]["parameter_kernel_backends"] == {
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM,
    }
