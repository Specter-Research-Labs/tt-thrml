from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import tt_thrml
import tt_thrml.api as tt_api

from tests.ttnn_test_utils import FakeTTNN
from tt_thrml.runtime_config import (
    BackendBinding,
    CATEGORICAL_PARAMETER_FAMILY,
    ExecutionOptions,
    ParameterFamily,
    ParameterKernelBackend,
    SPIN_PARAMETER_FAMILY,
    make_backend_binding,
)


SUPPORTED_API_EXPORTS = [
    "clear_compiled_program_cache",
    "sample_states",
    "sample_states_many",
    "sample_with_observation",
    "sample_with_observation_many",
]

SUPPORTED_ROOT_EXPORTS = [
    "BackendBinding",
    "ExecutionOptions",
    "ParameterFamily",
    "ParameterKernelBackend",
    "TTMLIRConfig",
    "close_devices",
    "close_mesh_device",
    "clear_compiled_program_cache",
    "make_backend_binding",
    "make_ttmlir_backend_binding",
    "make_ttmlir_config",
    "make_ttmlir_parameter_kernel_backends",
    "make_ttmlir_parameter_kernel_ops",
    "open_device",
    "open_devices",
    "open_mesh_device",
    "sample_states",
    "sample_states_many",
    "sample_with_observation",
    "sample_with_observation_many",
]

DELETED_ROOT_NAMES = (
    "Block",
    "BlockSamplingProgram",
    "SamplingSchedule",
    "FactorSamplingProgram",
    "NativeInteraction",
    "block_management",
    "conditional_samplers",
    "factor",
    "interaction",
    "models",
    "observers",
    "pgm",
)


def _make_backend(
    *,
    devices=("fake:0",),
    parameter_kernel_ops=None,
    parameter_kernel_backends=None,
) -> BackendBinding:
    device = devices[0] if len(devices) == 1 else devices
    return make_backend_binding(
        FakeTTNN(),
        device,
        parameter_kernel_ops=parameter_kernel_ops,
        parameter_kernel_backends=parameter_kernel_backends,
    )


def _make_schedule():
    return SimpleNamespace(n_samples=2)


def _make_state(value: int):
    return [np.asarray([value], dtype=np.int32)]


def test_root_exports_only_backend_and_execution_surface():
    assert tt_api.__all__ == SUPPORTED_API_EXPORTS
    assert tt_thrml.__all__[1:] == SUPPORTED_ROOT_EXPORTS
    assert tt_thrml.BackendBinding is BackendBinding
    assert tt_thrml.ExecutionOptions is ExecutionOptions
    assert tt_thrml.ParameterFamily is ParameterFamily
    assert tt_thrml.ParameterKernelBackend is ParameterKernelBackend
    assert tt_thrml.make_backend_binding is make_backend_binding
    assert tt_thrml.clear_compiled_program_cache is tt_api.clear_compiled_program_cache
    assert tt_thrml.sample_states is tt_api.sample_states
    assert tt_thrml.sample_states_many is tt_api.sample_states_many
    assert tt_thrml.sample_with_observation is tt_api.sample_with_observation
    assert tt_thrml.sample_with_observation_many is tt_api.sample_with_observation_many

    for deleted_name in DELETED_ROOT_NAMES:
        assert deleted_name not in tt_thrml.__all__


def test_clear_compiled_program_cache_delegates_to_backend_executor(monkeypatch):
    calls = []

    monkeypatch.setattr(
        tt_api,
        "_clear_compiled_program_cache",
        lambda: calls.append("cleared"),
    )

    tt_api.clear_compiled_program_cache()

    assert calls == ["cleared"]


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: tt_api.sample_states(
            "key",
            SimpleNamespace(name="program"),
            _make_schedule(),
            _make_state(1),
            _make_state(2),
            ["node"],
        ),
        lambda: tt_api.sample_states_many(
            ["k0"],
            SimpleNamespace(name="program"),
            _make_schedule(),
            [_make_state(1)],
            [_make_state(2)],
            ["node"],
        ),
        lambda: tt_api.sample_with_observation(
            "key",
            SimpleNamespace(name="program"),
            _make_schedule(),
            _make_state(1),
            _make_state(2),
            {"carry": 0},
            object(),
        ),
        lambda: tt_api.sample_with_observation_many(
            ["k0"],
            SimpleNamespace(name="program"),
            _make_schedule(),
            [_make_state(1)],
            [_make_state(2)],
            [{"carry": 0}],
            object(),
        ),
    ],
)
def test_sampling_entrypoints_require_backend_binding(invoke):
    with pytest.raises(TypeError, match="BackendBinding"):
        invoke()


def test_sampling_entrypoints_require_execution_options_instances():
    with pytest.raises(TypeError, match="ExecutionOptions"):
        tt_api.sample_states(
            "key",
            SimpleNamespace(name="program"),
            _make_schedule(),
            _make_state(1),
            _make_state(2),
            ["node"],
            backend=_make_backend(),
            options="bad-options",
        )


def test_sample_states_passes_backend_binding_options_and_progress(monkeypatch):
    program = SimpleNamespace(name="program")
    schedule = _make_schedule()
    init_state_free = _make_state(1)
    state_clamp = _make_state(2)
    backend = _make_backend(
        parameter_kernel_ops={SPIN_PARAMETER_FAMILY: "spin-op"},
        parameter_kernel_backends={SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM},
    )
    progress_messages = []
    options = ExecutionOptions(
        profiler="profiler",
        profile_sync=True,
        progress=progress_messages.append,
    )
    recorded = {}
    expected = ["state-history"]

    class RecordingExecutor:
        def sample_states(
            self,
            key,
            schedule,
            *,
            nodes_to_sample,
            init_state_free,
            state_clamp,
        ):
            recorded["sample_states"] = {
                "key": key,
                "schedule": schedule,
                "nodes_to_sample": list(nodes_to_sample),
                "init_state_free": init_state_free,
                "state_clamp": state_clamp,
            }
            return expected

    @contextmanager
    def fake_borrow_executor(*, program, backend, options):
        recorded["borrow_executor"] = {
            "program": program,
            "backend": backend,
            "options": options,
        }
        yield RecordingExecutor()

    monkeypatch.setattr(tt_api, "_borrow_executor", fake_borrow_executor)

    actual = tt_api.sample_states(
        "key-0",
        program,
        schedule,
        init_state_free,
        state_clamp,
        ["spin-node"],
        backend=backend,
        options=options,
    )

    assert actual == expected
    assert recorded["borrow_executor"]["program"] is program
    assert recorded["borrow_executor"]["backend"] is backend
    assert recorded["borrow_executor"]["options"] is options
    assert recorded["borrow_executor"]["backend"].parameter_kernel_ops == {
        SPIN_PARAMETER_FAMILY: "spin-op",
    }
    assert recorded["sample_states"] == {
        "key": "key-0",
        "schedule": schedule,
        "nodes_to_sample": ["spin-node"],
        "init_state_free": init_state_free,
        "state_clamp": state_clamp,
    }
    assert progress_messages == [
        "[tt-thrml.api] sample_states.executor_init:start",
        "[tt-thrml.api] sample_states.executor_init:done",
        "[tt-thrml.api] sample_states.run:start",
        "[tt-thrml.api] sample_states.run:done",
    ]


def test_sample_states_many_builds_loaded_jobs_and_uses_coordinator(monkeypatch):
    recorded = {}
    expected = [["state-history"]]

    class RecordingCoordinator:
        def sample_loaded_states_many(
            self,
            *,
            schedule,
            jobs,
            nodes_to_sample,
            preserve_clamp_groups,
        ):
            recorded["call"] = {
                "schedule": schedule,
                "jobs": jobs,
                "nodes_to_sample": list(nodes_to_sample),
                "preserve_clamp_groups": preserve_clamp_groups,
            }
            return expected

    def fake_make_coordinator(*, program, backend, options):
        recorded["coordinator"] = {
            "program": program,
            "backend": backend,
            "options": options,
        }
        return RecordingCoordinator()

    monkeypatch.setattr(tt_api, "make_coordinator", fake_make_coordinator)

    actual = tt_api.sample_states_many(
        ["k0", "k1"],
        SimpleNamespace(name="program"),
        _make_schedule(),
        [_make_state(1), _make_state(3)],
        [_make_state(2), _make_state(4)],
        ["node"],
        backend=_make_backend(),
    )

    assert actual == expected
    assert recorded["coordinator"]["program"].name == "program"
    assert [job.key for job in recorded["call"]["jobs"]] == ["k0", "k1"]
    np.testing.assert_array_equal(recorded["call"]["jobs"][0].state_free[0], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(recorded["call"]["jobs"][1].state_free[0], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(recorded["call"]["jobs"][0].state_clamp[0], np.asarray([2], dtype=np.int32))
    np.testing.assert_array_equal(recorded["call"]["jobs"][1].state_clamp[0], np.asarray([4], dtype=np.int32))
    assert recorded["call"]["nodes_to_sample"] == ["node"]
    assert recorded["call"]["preserve_clamp_groups"] is False


def test_sample_with_observation_passes_backend_binding_options_and_progress(monkeypatch):
    program = SimpleNamespace(name="program")
    schedule = _make_schedule()
    init_chain_state = _make_state(3)
    state_clamp = _make_state(4)
    observer = object()
    backend = _make_backend(
        parameter_kernel_ops={CATEGORICAL_PARAMETER_FAMILY: "theta-op"},
        parameter_kernel_backends={
            CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM
        },
    )
    progress_messages = []
    options = ExecutionOptions(progress=progress_messages.append)
    recorded = {"events": []}
    expected = ({"moments": 1}, {"carry": 2})

    class RecordingExecutor:
        def load_state(self, state_free, state_clamp):
            recorded["events"].append(("load_state", state_free, state_clamp))

        def sample_loaded_observation(
            self,
            key,
            schedule,
            *,
            state_clamp,
            observation_carry_init,
            f_observe,
        ):
            recorded["events"].append(
                (
                    "sample_loaded_observation",
                    key,
                    schedule,
                    state_clamp,
                    observation_carry_init,
                    f_observe,
                )
            )
            return expected

    @contextmanager
    def fake_borrow_executor(*, program, backend, options):
        recorded["borrow_executor"] = {
            "program": program,
            "backend": backend,
            "options": options,
        }
        yield RecordingExecutor()

    monkeypatch.setattr(tt_api, "_borrow_executor", fake_borrow_executor)

    actual = tt_api.sample_with_observation(
        "key-1",
        program,
        schedule,
        init_chain_state,
        state_clamp,
        {"carry": 0},
        observer,
        backend=backend,
        options=options,
    )

    assert actual == expected
    assert recorded["borrow_executor"]["backend"].parameter_kernel_ops == {
        CATEGORICAL_PARAMETER_FAMILY: "theta-op",
    }
    assert recorded["events"][0] == ("load_state", init_chain_state, state_clamp)
    assert recorded["events"][1][0] == "sample_loaded_observation"
    assert progress_messages == [
        "[tt-thrml.api] sample_with_observation.executor_init:start",
        "[tt-thrml.api] sample_with_observation.executor_init:done",
        "[tt-thrml.api] sample_with_observation.load_state:start",
        "[tt-thrml.api] sample_with_observation.load_state:done",
        "[tt-thrml.api] sample_with_observation.run:start",
        "[tt-thrml.api] sample_with_observation.run:done",
    ]


def test_sample_with_observation_many_builds_loaded_jobs_and_uses_coordinator(
    monkeypatch,
):
    recorded = {}
    expected = ([{"moments": 1}], [{"carry": 2}])

    class RecordingCoordinator:
        def sample_loaded_observation_many(
            self,
            *,
            schedule,
            jobs,
            f_observe,
            preserve_clamp_groups,
        ):
            recorded["call"] = {
                "schedule": schedule,
                "jobs": jobs,
                "f_observe": f_observe,
                "preserve_clamp_groups": preserve_clamp_groups,
            }
            return expected

    def fake_make_coordinator(*, program, backend, options):
        recorded["coordinator"] = {
            "program": program,
            "backend": backend,
            "options": options,
        }
        return RecordingCoordinator()

    monkeypatch.setattr(tt_api, "make_coordinator", fake_make_coordinator)

    actual = tt_api.sample_with_observation_many(
        ["k0"],
        SimpleNamespace(name="program"),
        _make_schedule(),
        [_make_state(1)],
        [_make_state(2)],
        [{"carry": 0}],
        object(),
        backend=_make_backend(),
    )

    assert actual == expected
    assert recorded["coordinator"]["program"].name == "program"
    assert [job.key for job in recorded["call"]["jobs"]] == ["k0"]
    np.testing.assert_array_equal(recorded["call"]["jobs"][0].state_free[0], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(recorded["call"]["jobs"][0].state_clamp[0], np.asarray([2], dtype=np.int32))
    assert [job.observation_carry_init for job in recorded["call"]["jobs"]] == [
        {"carry": 0}
    ]
    assert [job.clamp_group_id for job in recorded["call"]["jobs"]] == [0]
    assert recorded["call"]["preserve_clamp_groups"] is True
