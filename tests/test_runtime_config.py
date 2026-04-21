import pytest
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode

import tt_thrml
import tt_thrml.runtime_config as runtime_config
from tt_thrml.fingerprint import backend_object_fingerprint, program_fingerprint, stable_fingerprint
from tt_thrml.runtime_config import (
    BackendBinding,
    CATEGORICAL_PARAMETER_FAMILY,
    ExecutionOptions,
    ParameterFamily,
    ParameterKernelBackend,
    SPIN_PARAMETER_FAMILY,
    make_backend_binding,
    merge_parameter_kernel_backends,
    normalize_execution_options,
    normalize_parameter_family,
    normalize_parameter_kernel_backend,
    parameter_family_spec,
    require_backend,
)


def test_make_backend_binding_normalizes_devices_and_primary_device():
    binding = make_backend_binding("fake-ttnn", ["fake:0", "fake:1"])

    assert isinstance(binding, BackendBinding)
    assert binding.ttnn == "fake-ttnn"
    assert binding.devices == ("fake:0", "fake:1")
    assert binding.primary_device == "fake:0"


def test_backend_binding_parameter_kernel_overlay_merges_families():
    binding = make_backend_binding(
        "fake-ttnn",
        "fake:0",
        parameter_kernel_ops={SPIN_PARAMETER_FAMILY: "old-gamma"},
        parameter_kernel_backends={SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM},
    )

    updated = (
        binding.with_parameter_kernel_backends(
            {CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR}
        )
        .with_parameter_kernel_ops({CATEGORICAL_PARAMETER_FAMILY: "theta-op"})
        .with_parameter_kernel_op(SPIN_PARAMETER_FAMILY, "new-gamma")
    )

    assert binding.parameter_kernel_ops == {
        SPIN_PARAMETER_FAMILY: "old-gamma",
    }
    assert binding.parameter_kernel_backends == {
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM,
    }
    assert updated.parameter_kernel_ops == {
        CATEGORICAL_PARAMETER_FAMILY: "theta-op",
        SPIN_PARAMETER_FAMILY: "new-gamma",
    }
    assert updated.parameter_kernel_backends == {
        CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM,
    }


def test_backend_binding_cache_key_helpers_reuse_normalized_parameter_cache_key():
    ttnn = object()
    devices = [object(), object()]
    spin_op = object()
    theta_op = object()
    binding = make_backend_binding(
        ttnn,
        devices,
        parameter_kernel_ops=[
            (SPIN_PARAMETER_FAMILY, spin_op),
            (CATEGORICAL_PARAMETER_FAMILY, theta_op),
        ],
        parameter_kernel_backends=[
            (SPIN_PARAMETER_FAMILY, ParameterKernelBackend.CUSTOM),
            (CATEGORICAL_PARAMETER_FAMILY, ParameterKernelBackend.TTMLIR),
        ],
    )
    reordered = make_backend_binding(
        ttnn,
        devices,
        parameter_kernel_ops=[
            (CATEGORICAL_PARAMETER_FAMILY, theta_op),
            (SPIN_PARAMETER_FAMILY, spin_op),
        ],
        parameter_kernel_backends=[
            (CATEGORICAL_PARAMETER_FAMILY, ParameterKernelBackend.TTMLIR),
            (SPIN_PARAMETER_FAMILY, ParameterKernelBackend.CUSTOM),
        ],
    )
    expected_parameter_kernel_backend_key = (
        (CATEGORICAL_PARAMETER_FAMILY.value, ParameterKernelBackend.TTMLIR.value),
        (SPIN_PARAMETER_FAMILY.value, ParameterKernelBackend.CUSTOM.value),
    )
    expected_parameter_kernel_cache_key = (
        (
            CATEGORICAL_PARAMETER_FAMILY.value,
            backend_object_fingerprint(theta_op),
        ),
        (
            SPIN_PARAMETER_FAMILY.value,
            backend_object_fingerprint(spin_op),
        ),
    )

    assert binding.cache_key == reordered.cache_key
    assert binding.cache_key == (
        backend_object_fingerprint(ttnn),
        (
            backend_object_fingerprint(devices[0]),
            backend_object_fingerprint(devices[1]),
        ),
        expected_parameter_kernel_backend_key,
        expected_parameter_kernel_cache_key,
    )
    assert binding.device_cache_key(devices[1]) == (
        backend_object_fingerprint(ttnn),
        backend_object_fingerprint(devices[1]),
        expected_parameter_kernel_backend_key,
    )
    assert binding.executor_cache_key(devices[0]) == (
        backend_object_fingerprint(ttnn),
        backend_object_fingerprint(devices[0]),
        expected_parameter_kernel_backend_key,
        expected_parameter_kernel_cache_key,
    )


def test_parameter_family_normalization_and_specs_are_typed():
    assert normalize_parameter_family("spin_natural_parameter") is ParameterFamily.SPIN
    assert normalize_parameter_family(ParameterFamily.CATEGORICAL) is ParameterFamily.CATEGORICAL
    assert normalize_parameter_kernel_backend("ttmlir") is ParameterKernelBackend.TTMLIR
    assert parameter_family_spec(ParameterFamily.GAUSSIAN).sampler_kind == "continuous"
    assert parameter_family_spec(ParameterFamily.CATEGORICAL).categorical_axis == 2
    assert merge_parameter_kernel_backends(
        {SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM},
        base_parameter_kernel_backends={CATEGORICAL_PARAMETER_FAMILY: "ttmlir"},
    ) == {
        CATEGORICAL_PARAMETER_FAMILY: ParameterKernelBackend.TTMLIR,
        SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM,
    }


def test_backend_binding_requires_explicit_backend_for_attached_parameter_ops():
    with pytest.raises(ValueError, match="parameter_kernel_backends"):
        make_backend_binding(
            "fake-ttnn",
            "fake:0",
            parameter_kernel_ops={SPIN_PARAMETER_FAMILY: "gamma"},
        )

    with pytest.raises(ValueError, match="TTNN fallback parameter"):
        make_backend_binding(
            "fake-ttnn",
            "fake:0",
            parameter_kernel_ops={SPIN_PARAMETER_FAMILY: "gamma"},
            parameter_kernel_backends={SPIN_PARAMETER_FAMILY: ParameterKernelBackend.NATIVE},
        )


def test_require_backend_accepts_backend_binding_only():
    binding = make_backend_binding("fake-ttnn", "fake:0")

    assert require_backend(binding) is binding

    with pytest.raises(TypeError, match="BackendBinding"):
        require_backend(None)

    with pytest.raises(TypeError, match="BackendBinding"):
        require_backend(object())


def test_execution_options_normalization_and_cacheability():
    default_options = normalize_execution_options()
    progress_options = ExecutionOptions(progress=lambda _message: None)
    profiler_options = ExecutionOptions(profiler=object())
    sync_options = ExecutionOptions(profile_sync=True)

    assert default_options == ExecutionOptions()
    assert default_options.cacheable is True
    assert normalize_execution_options(progress_options) is progress_options
    assert progress_options.cacheable is False
    assert profiler_options.cacheable is False
    assert sync_options.cacheable is False

    with pytest.raises(TypeError, match="ExecutionOptions"):
        normalize_execution_options(object())


def test_runtime_config_surface_matches_backend_binding_end_state():
    assert runtime_config.BackendBinding is BackendBinding
    assert runtime_config.ExecutionOptions is ExecutionOptions
    assert runtime_config.make_backend_binding is make_backend_binding
    assert not hasattr(runtime_config, "clear_default_backend")
    assert not hasattr(runtime_config, "configure_default_backend")
    assert not hasattr(runtime_config, "get_default_backend")
    assert not hasattr(runtime_config, "resolve_backend")


def test_fingerprint_accepts_node_types_and_programs_with_node_type_metadata():
    node = CategoricalNode()
    program = FactorSamplingProgram(
        BlockGibbsSpec(
            [Block([node])],
            [],
            {},
        ),
        [CategoricalGibbsConditional(3)],
        [CategoricalEBMFactor([Block([node])], [[0.1, -0.1, 0.2]])],
        [],
    )

    assert stable_fingerprint(CategoricalNode) == stable_fingerprint(CategoricalNode)
    assert isinstance(program_fingerprint(program), str)


def test_import_tt_thrml_without_torch_keeps_minimal_root_surface(monkeypatch):
    real_import_module = tt_thrml.import_module

    def fake_import_module(name, package=None):
        if name == ".api":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import_module(name, package)

    tt_thrml.__dict__.pop("sample_states", None)
    monkeypatch.setattr(tt_thrml, "import_module", fake_import_module)

    assert tt_thrml.BackendBinding.__name__ == "BackendBinding"
    assert tt_thrml.ExecutionOptions.__name__ == "ExecutionOptions"
    assert callable(tt_thrml.make_backend_binding)
    assert tt_thrml.ParameterFamily is ParameterFamily
    assert tt_thrml.ParameterKernelBackend is ParameterKernelBackend
    assert callable(tt_thrml.open_device)
    assert callable(tt_thrml.open_devices)
    assert callable(tt_thrml.close_devices)
    assert callable(tt_thrml.make_ttmlir_parameter_kernel_ops)
    assert callable(tt_thrml.make_ttmlir_parameter_kernel_backends)
    assert callable(tt_thrml.make_ttmlir_backend_binding)
    assert not hasattr(tt_thrml, "clear_default_backend")
    assert not hasattr(tt_thrml, "configure_default_backend")
    assert not hasattr(tt_thrml, "sample_states")
