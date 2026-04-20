"""Tenstorrent execution companion for upstream THRML."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tt-thrml")
except PackageNotFoundError:
    __version__ = "0+local"

__all__ = [
    "__version__",
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

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "BackendBinding": (".runtime_config", "BackendBinding"),
    "ExecutionOptions": (".runtime_config", "ExecutionOptions"),
    "ParameterFamily": (".runtime_config", "ParameterFamily"),
    "ParameterKernelBackend": (".runtime_config", "ParameterKernelBackend"),
    "make_backend_binding": (".runtime_config", "make_backend_binding"),
    "close_devices": (".device_open", "close_devices"),
    "close_mesh_device": (".device_open", "close_mesh_device"),
    "open_device": (".device_open", "open_device"),
    "open_devices": (".device_open", "open_devices"),
    "open_mesh_device": (".device_open", "open_mesh_device"),
    "clear_compiled_program_cache": (".api", "clear_compiled_program_cache"),
    "TTMLIRConfig": (".compiler.ttmlir.backend", "TTMLIRConfig"),
    "make_ttmlir_backend_binding": (
        ".compiler.ttmlir.backend",
        "make_ttmlir_backend_binding",
    ),
    "make_ttmlir_config": (".compiler.ttmlir.backend", "make_ttmlir_config"),
    "sample_states": (".api", "sample_states"),
    "sample_states_many": (".api", "sample_states_many"),
    "sample_with_observation": (".api", "sample_with_observation"),
    "sample_with_observation_many": (".api", "sample_with_observation_many"),
    "make_ttmlir_parameter_kernel_ops": (
        ".compiler.ttmlir.backend",
        "make_ttmlir_parameter_kernel_ops",
    ),
    "make_ttmlir_parameter_kernel_backends": (
        ".compiler.ttmlir.backend",
        "make_ttmlir_parameter_kernel_backends",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or ""
        if missing_module in {"jax", "torch"}:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        raise

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
