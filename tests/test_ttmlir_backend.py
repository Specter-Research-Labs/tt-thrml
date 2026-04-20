from pathlib import Path
import sys
import types

from tt_thrml.runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    GAUSSIAN_PARAMETER_FAMILY,
    BackendBinding,
    ParameterKernelBackend,
    SPIN_PARAMETER_FAMILY,
)


def _install_stub(module_name, factory_name, tag, calls=None):
    module = types.ModuleType(module_name)

    def _factory(**kwargs):
        if calls is not None:
            calls.append((tag, kwargs))
        return (tag, kwargs["config"].artifact_root)

    setattr(module, factory_name, _factory)
    sys.modules[module_name] = module


def test_make_ttmlir_parameter_kernels_build_family_registry(
    monkeypatch,
    tmp_path: Path,
):
    from tt_thrml.compiler.ttmlir import backend

    calls = []
    artifact_root = tmp_path / "artifacts"
    build_dir = tmp_path / "build"
    system_desc_path = tmp_path / "system_desc.ttsys"

    _install_stub(
        "tt_thrml.compiler.ttmlir.categorical_theta",
        "make_ttmlir_categorical_theta_op",
        "theta",
        calls,
    )
    _install_stub(
        "tt_thrml.compiler.ttmlir.gaussian_canonical",
        "make_ttmlir_gaussian_canonical_op",
        "gaussian",
        calls,
    )
    _install_stub(
        "tt_thrml.compiler.ttmlir.spin_gamma",
        "make_ttmlir_spin_gamma_op",
        "gamma",
        calls,
    )

    shared_kwargs = {
        "build_dir": build_dir,
        "system_desc_path": system_desc_path,
        "artifact_root": artifact_root,
    }

    parameter_kernel_ops = backend.make_ttmlir_parameter_kernel_ops(**shared_kwargs)
    parameter_kernel_backends = backend.make_ttmlir_parameter_kernel_backends()
    config = backend.make_ttmlir_config(**shared_kwargs)

    assert parameter_kernel_ops == {
        CATEGORICAL_PARAMETER_FAMILY: ("theta", artifact_root),
        GAUSSIAN_PARAMETER_FAMILY: ("gaussian", artifact_root),
        SPIN_PARAMETER_FAMILY: ("gamma", artifact_root),
    }
    assert parameter_kernel_backends == {
        CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        GAUSSIAN_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
    }
    assert calls == [
        ("theta", {"config": config}),
        ("gaussian", {"config": config}),
        ("gamma", {"config": config}),
    ]
    assert backend.__all__ == [
        "TTMLIRConfig",
        "make_ttmlir_backend_binding",
        "make_ttmlir_config",
        "make_ttmlir_parameter_kernel_backends",
        "make_ttmlir_parameter_kernel_ops",
    ]
    assert not hasattr(backend, "TTMLIRParameterOps")
    assert not hasattr(backend, "configure_default_ttmlir_backend")
    assert not hasattr(backend, "make_ttmlir_parameter_ops")


def test_make_ttmlir_config_collects_paths_and_artifacts(tmp_path: Path):
    from tt_thrml.compiler.ttmlir import backend

    build_dir = tmp_path / "tt-mlir" / "build"
    config = backend.make_ttmlir_config(
        system_desc_path=tmp_path / "system_desc.ttsys",
        build_dir=build_dir,
        artifact_root=tmp_path / "artifacts",
    )

    assert isinstance(config, backend.TTMLIRConfig)
    assert config.system_desc_path == (tmp_path / "system_desc.ttsys").resolve()
    assert config.artifact_root == (tmp_path / "artifacts").resolve()
    assert config.ttmlir_opt == str((build_dir / "bin" / "ttmlir-opt").resolve())
    assert config.ttmlir_translate == str(
        (build_dir / "bin" / "ttmlir-translate").resolve()
    )


def test_tt_thrml_root_exports_ttmlir_config_and_parameter_kernel_ops(tmp_path: Path):
    import tt_thrml

    artifact_root = tmp_path / "artifacts"
    tt_thrml.__dict__.pop("make_ttmlir_backend_binding", None)
    tt_thrml.__dict__.pop("make_ttmlir_parameter_kernel_ops", None)
    tt_thrml.__dict__.pop("make_ttmlir_parameter_kernel_backends", None)
    tt_thrml.__dict__.pop("make_ttmlir_config", None)
    tt_thrml.__dict__.pop("TTMLIRConfig", None)

    _install_stub(
        "tt_thrml.compiler.ttmlir.categorical_theta",
        "make_ttmlir_categorical_theta_op",
        "theta",
    )
    _install_stub(
        "tt_thrml.compiler.ttmlir.gaussian_canonical",
        "make_ttmlir_gaussian_canonical_op",
        "gaussian",
    )
    _install_stub(
        "tt_thrml.compiler.ttmlir.spin_gamma",
        "make_ttmlir_spin_gamma_op",
        "gamma",
    )

    config = tt_thrml.make_ttmlir_config(
        system_desc_path=tmp_path / "system_desc.ttsys",
        build_dir=tmp_path / "tt-mlir" / "build",
        artifact_root=artifact_root,
    )
    parameter_kernel_ops = tt_thrml.make_ttmlir_parameter_kernel_ops(config=config)
    parameter_kernel_backends = tt_thrml.make_ttmlir_parameter_kernel_backends()

    assert tt_thrml.TTMLIRConfig is not None
    assert callable(tt_thrml.make_ttmlir_backend_binding)
    assert callable(tt_thrml.make_ttmlir_config)
    assert callable(tt_thrml.make_ttmlir_parameter_kernel_ops)
    assert callable(tt_thrml.make_ttmlir_parameter_kernel_backends)
    assert parameter_kernel_ops == {
        CATEGORICAL_PARAMETER_FAMILY: ("theta", artifact_root),
        GAUSSIAN_PARAMETER_FAMILY: ("gaussian", artifact_root),
        SPIN_PARAMETER_FAMILY: ("gamma", artifact_root),
    }
    assert parameter_kernel_backends == {
        CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        GAUSSIAN_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
    }
    assert not hasattr(tt_thrml, "configure_default_ttmlir_backend")
    assert not hasattr(tt_thrml, "make_ttmlir_parameter_ops")


def test_make_ttmlir_backend_binding_builds_complete_backend_binding(tmp_path: Path):
    from tt_thrml.compiler.ttmlir import backend

    artifact_root = tmp_path / "artifacts"
    _install_stub(
        "tt_thrml.compiler.ttmlir.categorical_theta",
        "make_ttmlir_categorical_theta_op",
        "theta",
    )
    _install_stub(
        "tt_thrml.compiler.ttmlir.gaussian_canonical",
        "make_ttmlir_gaussian_canonical_op",
        "gaussian",
    )
    _install_stub(
        "tt_thrml.compiler.ttmlir.spin_gamma",
        "make_ttmlir_spin_gamma_op",
        "gamma",
    )

    binding = backend.make_ttmlir_backend_binding(
        "fake-ttnn",
        "fake:0",
        config=backend.TTMLIRConfig(
            system_desc_path=tmp_path / "system_desc.ttsys",
            artifact_root=artifact_root,
        ),
    )

    assert isinstance(binding, BackendBinding)
    assert binding.devices == ("fake:0",)
    assert binding.parameter_kernel_backends == {
        CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        GAUSSIAN_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
    }
    assert binding.parameter_kernel_ops == {
        CATEGORICAL_PARAMETER_FAMILY: ("theta", artifact_root),
        GAUSSIAN_PARAMETER_FAMILY: ("gaussian", artifact_root),
        SPIN_PARAMETER_FAMILY: ("gamma", artifact_root),
    }
