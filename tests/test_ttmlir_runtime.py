from pathlib import Path
from contextlib import contextmanager
import sys
import types

import pytest

from tt_thrml.compiler.ttmlir import runtime as ttmlir_runtime
from tt_thrml.compiler.ttmlir.runtime import make_ttmlir_config


def test_ttmlir_config_normalizes_paths_and_defaults_tools(tmp_path: Path):
    config = ttmlir_runtime.TTMLIRConfig(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
    )

    assert config.system_desc_path == (tmp_path / "system_desc.ttsys").resolve()
    assert config.artifact_root == (tmp_path / "artifacts").resolve()
    assert config.ttmlir_opt == "ttmlir-opt"
    assert config.ttmlir_translate == "ttmlir-translate"


def test_make_ttmlir_config_can_derive_compiler_tools_from_build_dir(tmp_path: Path):
    build_dir = tmp_path / "tt-mlir" / "build-py312-stablehlo"

    config = make_ttmlir_config(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
        build_dir=build_dir,
    )

    assert config.system_desc_path == (tmp_path / "system_desc.ttsys").resolve()
    assert config.artifact_root == (tmp_path / "artifacts").resolve()
    assert config.ttmlir_opt == str((build_dir / "bin" / "ttmlir-opt").resolve())
    assert config.ttmlir_translate == str(
        (build_dir / "bin" / "ttmlir-translate").resolve()
    )


def test_make_ttmlir_config_can_use_build_dir_from_environment(
    monkeypatch, tmp_path: Path
):
    build_dir = tmp_path / "tt-mlir" / "build-py312-stablehlo"
    monkeypatch.setenv("TTMLIR_BUILD_DIR", str(build_dir))

    config = make_ttmlir_config(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
    )

    assert config.ttmlir_opt == str((build_dir / "bin" / "ttmlir-opt").resolve())
    assert config.ttmlir_translate == str(
        (build_dir / "bin" / "ttmlir-translate").resolve()
    )


def test_make_ttmlir_config_requires_explicit_system_desc_path(tmp_path: Path):
    with pytest.raises(ValueError, match="system_desc_path is required"):
        make_ttmlir_config(artifact_root=tmp_path / "artifacts")


def test_make_ttmlir_config_requires_explicit_compiler_tools(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("TTMLIR_BUILD_DIR", raising=False)

    with pytest.raises(ValueError, match="TT-MLIR compiler tools must be configured"):
        make_ttmlir_config(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
        )


def test_make_ttmlir_config_rejects_mixed_config_and_explicit_fields(tmp_path: Path):
    config = ttmlir_runtime.TTMLIRConfig(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
    )

    with pytest.raises(ValueError, match="Pass either `config=`"):
        make_ttmlir_config(config=config, artifact_root=tmp_path / "other-artifacts")


def test_make_ttmlir_config_rejects_build_dir_and_explicit_tools(tmp_path: Path):
    with pytest.raises(ValueError, match="Pass either `build_dir=`"):
        make_ttmlir_config(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=tmp_path / "artifacts",
            build_dir=tmp_path / "build",
            ttmlir_opt="/tmp/ttmlir-opt",
        )


def test_borrow_runtime_session_separates_cache_by_device_runtime(monkeypatch, tmp_path):
    class FakeTTModule:
        class MeshDeviceOptions:
            def __init__(self):
                self.mesh_shape = None
                self.device_ids = []

        def __init__(self):
            self.current_device_runtime = None
            self.set_runtime_calls = []
            self.open_calls = []

        def set_current_device_runtime(self, runtime):
            self.current_device_runtime = runtime
            self.set_runtime_calls.append(runtime)

        def open_mesh_device(self, mesh_options):
            call = (
                self.current_device_runtime,
                tuple(mesh_options.mesh_shape),
                tuple(mesh_options.device_ids),
            )
            self.open_calls.append(call)
            return f"device-{len(self.open_calls)}"

        def close_mesh_device(self, device):
            return None

        class FabricConfig:
            DISABLED = "disabled"

        def set_fabric_config(self, config):
            return None

    fake_tt_runtime = FakeTTModule()
    monkeypatch.setattr(ttmlir_runtime, "_import_ttrt_runtime", lambda: fake_tt_runtime)
    ttmlir_runtime._RUNTIME_SESSION_CACHE.clear()

    config = ttmlir_runtime.TTMLIRConfig(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
    )

    with ttmlir_runtime.borrow_runtime_session(
        config,
        mesh_shape=(1, 1),
        device_ids=[7],
        device_runtime="ttnn",
    ) as session_a:
        assert session_a.device == "device-1"

    with ttmlir_runtime.borrow_runtime_session(
        config,
        mesh_shape=(1, 1),
        device_ids=[7],
        device_runtime="ttnn",
    ) as session_b:
        assert session_b.device == "device-1"

    with ttmlir_runtime.borrow_runtime_session(
        config,
        mesh_shape=(1, 1),
        device_ids=[3],
        device_runtime="ttmetal",
    ) as session_c:
        assert session_c.device == "device-2"

    assert fake_tt_runtime.set_runtime_calls == ["ttnn", "ttmetal"]
    assert fake_tt_runtime.open_calls == [
        ("ttnn", (1, 1), (7,)),
        ("ttmetal", (1, 1), (3,)),
    ]
    ttmlir_runtime._RUNTIME_SESSION_CACHE.clear()


def test_run_flatbuffer_uses_compatible_runtime(monkeypatch, tmp_path):
    class FakeShard:
        def __init__(self, name):
            self.name = name

        def get_dtype(self):
            return "float32"

        def __repr__(self):
            return self.name

    class FakeTTModule:
        class DataType:
            Float32 = "float32"

        def __init__(self):
            self.compatible_binary = None
            self.current_device_runtime = "unset"
            self.layouts = []
            self.deallocated = []

        def set_compatible_device_runtime(self, binary_fbb):
            self.compatible_binary = binary_fbb
            self.current_device_runtime = "compatible-runtime"

        def get_current_device_runtime(self):
            return self.current_device_runtime

        def set_current_device_runtime(self, runtime):
            self.current_device_runtime = runtime

        def get_layout(self, fbb, program_index, input_index):
            self.layouts.append((fbb, program_index, input_index))
            return f"layout-{input_index}"

        def to_layout(self, host_tensor, device, layout, borrow):
            return (host_tensor, device, layout, borrow)

        def submit(self, device, fbb, program_index, runtime_inputs):
            return ["runtime-output"]

        def wait(self, runtime_outputs):
            return None

        def to_host(self, runtime_output, untilize=True):
            return [FakeShard("shard-0")]

        def deallocate_tensor(self, runtime_tensor, force=True):
            self.deallocated.append((runtime_tensor, force))

    class FakeBinary:
        def __init__(self, logger, file_manager, file_path):
            self.fbb = "fake-flatbuffer"
            self.file_path = file_path

        def get_program(self, index):
            assert index == 0
            return types.SimpleNamespace(mesh_shape=(1, 1))

    fake_tt_runtime = FakeTTModule()

    captured = {}

    class FakeDevice:
        def id(self):
            return 7

    @contextmanager
    def fake_borrow_runtime_session(config, *, mesh_shape, device_ids=None, device_runtime=None):
        captured["config"] = config
        captured["mesh_shape"] = mesh_shape
        captured["device_ids"] = device_ids
        captured["device_runtime"] = device_runtime
        yield types.SimpleNamespace(tt_runtime=fake_tt_runtime, device="fake-device")

    monkeypatch.setattr(
        ttmlir_runtime,
        "borrow_runtime_session",
        fake_borrow_runtime_session,
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_create_runtime_input",
        lambda tt_runtime, tensor: f"host:{tensor}",
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_is_torch_tensor",
        lambda tensor: True,
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_runtime_tensor_to_torch",
        lambda tt_runtime, runtime_tensor: f"converted:{runtime_tensor}",
    )
    monkeypatch.setattr(ttmlir_runtime, "_import_ttrt_runtime", lambda: fake_tt_runtime)
    monkeypatch.setattr(
        ttmlir_runtime,
        "_import_ttrt_util_types",
        lambda: (FakeBinary, lambda logger: "file-manager", lambda: "logger"),
    )

    config = ttmlir_runtime.TTMLIRConfig(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
    )
    result = ttmlir_runtime.run_flatbuffer(
        config,
        flatbuffer_path=tmp_path / "kernel.ttnn",
        input_tensors=["input-a", "input-b"],
        device=FakeDevice(),
    )

    assert fake_tt_runtime.compatible_binary == "fake-flatbuffer"
    assert captured == {
        "config": config,
        "mesh_shape": (1, 1),
        "device_ids": (7,),
        "device_runtime": "compatible-runtime",
    }
    assert result.outputs == ("converted:shard-0",)
    assert result.runtime_output_dtypes == ("float32",)
    assert fake_tt_runtime.current_device_runtime == "unset"


def test_run_flatbuffer_reuses_existing_ttnn_device_when_bridge_available(
    monkeypatch, tmp_path
):
    class FakeShard:
        def __repr__(self):
            return "fake-shard"

        def get_dtype(self):
            return "float32"

    class FakeTTModule:
        class DataType:
            Float32 = "float32"

        def __init__(self):
            self.compatible_binary = None
            self.current_device_runtime = "unset"
            self.runtime_devices = []
            self.layouts = []
            self.runtime_tensor_inputs = []
            self.deallocated = []

        def set_compatible_device_runtime(self, binary_fbb):
            self.compatible_binary = binary_fbb
            self.current_device_runtime = "compatible-runtime"

        def get_current_device_runtime(self):
            return self.current_device_runtime

        def set_current_device_runtime(self, runtime):
            self.current_device_runtime = runtime

        def create_runtime_device_from_ttnn(self, device):
            self.runtime_devices.append(device)
            return "runtime-device"

        def get_ttnn_tensor_from_runtime_tensor(self, runtime_tensor):
            return f"device:{runtime_tensor}"

        def get_layout(self, fbb, program_index, input_index):
            self.layouts.append((fbb, program_index, input_index))
            return f"layout-{input_index}"

        def to_layout(self, runtime_tensor, device, layout, borrow):
            return (runtime_tensor, device, layout, borrow)

        def submit(self, device, fbb, program_index, runtime_inputs):
            self.runtime_tensor_inputs.append(tuple(runtime_inputs))
            return ["runtime-output"]

        def to_host(self, runtime_output, untilize=True):
            return [FakeShard()]

        def deallocate_tensor(self, runtime_tensor, force=True):
            self.deallocated.append((runtime_tensor, force))

    class FakeBinary:
        def __init__(self, logger, file_manager, file_path):
            self.fbb = "fake-flatbuffer"
            self.file_path = file_path

        def get_program(self, index):
            assert index == 0
            return types.SimpleNamespace(mesh_shape=(1, 1))

    fake_tt_runtime = FakeTTModule()

    def fail_borrow_runtime_session(*args, **kwargs):
        raise AssertionError("borrow_runtime_session should not be used")

    monkeypatch.setattr(
        ttmlir_runtime,
        "borrow_runtime_session",
        fail_borrow_runtime_session,
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_create_runtime_input",
        lambda tt_runtime, tensor: f"host:{tensor}",
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_is_torch_tensor",
        lambda tensor: True,
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_runtime_tensor_to_torch",
        lambda tt_runtime, runtime_tensor: f"converted:{runtime_tensor}",
    )
    monkeypatch.setattr(ttmlir_runtime, "_import_ttrt_runtime", lambda: fake_tt_runtime)
    monkeypatch.setattr(
        ttmlir_runtime,
        "_import_ttrt_util_types",
        lambda: (FakeBinary, lambda logger: "file-manager", lambda: "logger"),
    )

    config = ttmlir_runtime.TTMLIRConfig(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
    )
    device = object()

    result = ttmlir_runtime.run_flatbuffer(
        config,
        flatbuffer_path=tmp_path / "kernel.ttnn",
        input_tensors=["input-a", "input-b"],
        device=device,
    )

    assert fake_tt_runtime.compatible_binary == "fake-flatbuffer"
    assert fake_tt_runtime.runtime_devices == [device]
    assert fake_tt_runtime.layouts == [
        ("fake-flatbuffer", 0, 0),
        ("fake-flatbuffer", 0, 1),
    ]
    assert fake_tt_runtime.runtime_tensor_inputs == [
        (
            ("host:input-a", "runtime-device", "layout-0", True),
            ("host:input-b", "runtime-device", "layout-1", True),
        )
    ]
    assert result.outputs == ("converted:fake-shard",)
    assert result.runtime_output_dtypes == ("float32",)


def test_run_flatbuffer_uses_direct_ttnn_tensor_bridge_when_available(
    monkeypatch, tmp_path
):
    class FakeTTNNTensor:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return self.name

    class FakeShard:
        def __repr__(self):
            return "fake-shard"

        def get_dtype(self):
            return "float32"

    class FakeTTModule:
        class DataType:
            Float32 = "float32"

        def __init__(self):
            self.compatible_binary = None
            self.current_device_runtime = "unset"
            self.runtime_devices = []
            self.direct_input_tensors = []
            self.layouts = []
            self.runtime_tensor_inputs = []

        def set_compatible_device_runtime(self, binary_fbb):
            self.compatible_binary = binary_fbb
            self.current_device_runtime = "compatible-runtime"

        def get_current_device_runtime(self):
            return self.current_device_runtime

        def set_current_device_runtime(self, runtime):
            self.current_device_runtime = runtime

        def create_runtime_device_from_ttnn(self, device):
            self.runtime_devices.append(device)
            return "runtime-device"

        def create_runtime_tensor_from_ttnn(self, tensor, borrow):
            self.direct_input_tensors.append((tensor, borrow))
            return f"device-input:{tensor}"

        def get_layout(self, fbb, program_index, input_index):
            self.layouts.append((fbb, program_index, input_index))
            return f"layout-{input_index}"

        def to_layout(self, runtime_tensor, device, layout, borrow):
            return (runtime_tensor, device, layout, borrow)

        def submit(self, device, fbb, program_index, runtime_inputs):
            self.runtime_tensor_inputs.append(tuple(runtime_inputs))
            return ["runtime-output"]

        def to_host(self, runtime_output, untilize=True):
            return [FakeShard()]

        def deallocate_tensor(self, runtime_tensor, force=True):
            return None

    class FakeBinary:
        def __init__(self, logger, file_manager, file_path):
            self.fbb = "fake-flatbuffer"
            self.file_path = file_path

        def get_program(self, index):
            assert index == 0
            return types.SimpleNamespace(mesh_shape=(1, 1))

    fake_tt_runtime = FakeTTModule()
    input_a = FakeTTNNTensor("input-a")
    input_b = FakeTTNNTensor("input-b")

    def fail_borrow_runtime_session(*args, **kwargs):
        raise AssertionError("borrow_runtime_session should not be used")

    monkeypatch.setattr(
        ttmlir_runtime,
        "borrow_runtime_session",
        fail_borrow_runtime_session,
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_is_torch_tensor",
        lambda tensor: False,
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_runtime_tensor_to_torch",
        lambda tt_runtime, runtime_tensor: f"converted:{runtime_tensor}",
    )
    monkeypatch.setattr(ttmlir_runtime, "_import_ttrt_runtime", lambda: fake_tt_runtime)
    monkeypatch.setattr(
        ttmlir_runtime,
        "_import_ttrt_util_types",
        lambda: (FakeBinary, lambda logger: "file-manager", lambda: "logger"),
    )

    config = ttmlir_runtime.TTMLIRConfig(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
    )
    device = object()

    result = ttmlir_runtime.run_flatbuffer(
        config,
        flatbuffer_path=tmp_path / "kernel.ttnn",
        input_tensors=[input_a, input_b],
        device=device,
    )

    assert fake_tt_runtime.compatible_binary == "fake-flatbuffer"
    assert fake_tt_runtime.runtime_devices == [device]
    assert fake_tt_runtime.direct_input_tensors == [
        (input_a, True),
        (input_b, True),
    ]
    assert fake_tt_runtime.layouts == [
        ("fake-flatbuffer", 0, 0),
        ("fake-flatbuffer", 0, 1),
    ]
    assert fake_tt_runtime.runtime_tensor_inputs == [
        (
            ("device-input:input-a", "runtime-device", "layout-0", True),
            ("device-input:input-b", "runtime-device", "layout-1", True),
        )
    ]
    assert result.outputs == ("converted:fake-shard",)
    assert result.runtime_output_dtypes == ("float32",)


def test_run_flatbuffer_can_return_device_outputs_when_requested(
    monkeypatch, tmp_path
):
    class FakeRuntimeOutput:
        def get_dtype(self):
            return "float32"

        def __repr__(self):
            return "runtime-output"

    class FakeTTModule:
        class DataType:
            Float32 = "float32"

        def __init__(self):
            self.compatible_binary = None
            self.current_device_runtime = "unset"
            self.runtime_devices = []
            self.layouts = []
            self.runtime_tensor_inputs = []
            self.deallocated = []

        def set_compatible_device_runtime(self, binary_fbb):
            self.compatible_binary = binary_fbb
            self.current_device_runtime = "compatible-runtime"

        def get_current_device_runtime(self):
            return self.current_device_runtime

        def set_current_device_runtime(self, runtime):
            self.current_device_runtime = runtime

        def create_runtime_device_from_ttnn(self, device):
            self.runtime_devices.append(device)
            return "runtime-device"

        def get_ttnn_tensor_from_runtime_tensor(self, runtime_tensor):
            return f"device:{runtime_tensor}"

        def get_layout(self, fbb, program_index, input_index):
            self.layouts.append((fbb, program_index, input_index))
            return f"layout-{input_index}"

        def to_layout(self, runtime_tensor, device, layout, borrow):
            return (runtime_tensor, device, layout, borrow)

        def submit(self, device, fbb, program_index, runtime_inputs):
            self.runtime_tensor_inputs.append(tuple(runtime_inputs))
            return [FakeRuntimeOutput()]

        def deallocate_tensor(self, runtime_tensor, force=True):
            self.deallocated.append((runtime_tensor, force))

    class FakeBinary:
        def __init__(self, logger, file_manager, file_path):
            self.fbb = "fake-flatbuffer"
            self.file_path = file_path

        def get_program(self, index):
            assert index == 0
            return types.SimpleNamespace(mesh_shape=(1, 1))

    fake_tt_runtime = FakeTTModule()

    def fail_borrow_runtime_session(*args, **kwargs):
        raise AssertionError("borrow_runtime_session should not be used")

    monkeypatch.setattr(
        ttmlir_runtime,
        "borrow_runtime_session",
        fail_borrow_runtime_session,
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_create_runtime_input",
        lambda tt_runtime, tensor: f"host:{tensor}",
    )
    monkeypatch.setattr(
        ttmlir_runtime,
        "_is_torch_tensor",
        lambda tensor: True,
    )
    monkeypatch.setattr(ttmlir_runtime, "_import_ttrt_runtime", lambda: fake_tt_runtime)
    monkeypatch.setattr(
        ttmlir_runtime,
        "_import_ttrt_util_types",
        lambda: (FakeBinary, lambda logger: "file-manager", lambda: "logger"),
    )

    config = ttmlir_runtime.TTMLIRConfig(
        system_desc_path=tmp_path / "system_desc.ttsys",
        artifact_root=tmp_path / "artifacts",
    )
    device = object()

    result = ttmlir_runtime.run_flatbuffer(
        config,
        flatbuffer_path=tmp_path / "kernel.ttnn",
        input_tensors=["input-a", "input-b"],
        device=device,
        prefer_device_output=True,
    )

    assert result.outputs == ("device:runtime-output",)
    assert result.runtime_output_dtypes == ("float32",)


def test_resolve_runtime_utils_module_falls_back_to_private_utils_submodule(
    monkeypatch,
):
    fake_runtime = types.ModuleType("ttrt.runtime")
    fake_private_runtime = types.SimpleNamespace(
        utils=types.SimpleNamespace(create_runtime_device_from_ttnn=lambda device: device)
    )
    fake_runtime._ttmlir_runtime = fake_private_runtime

    monkeypatch.setitem(sys.modules, "ttrt.runtime", fake_runtime)
    monkeypatch.setitem(
        sys.modules,
        "ttrt",
        types.SimpleNamespace(runtime=fake_runtime),
    )

    resolved = ttmlir_runtime._resolve_runtime_utils_module(fake_runtime)

    assert resolved is fake_private_runtime.utils
