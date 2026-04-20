"""Runtime internals for TT-backed execution."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "LoadedObservationJob",
    "LoadedStateJob",
    "RuntimeProfile",
    "TTProgramExecutor",
    "program_supported_by_executor",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "LoadedObservationJob": (".execution_support", "LoadedObservationJob"),
    "LoadedStateJob": (".execution_support", "LoadedStateJob"),
    "RuntimeProfile": (".execution_support", "RuntimeProfile"),
    "TTProgramExecutor": (".program_executor", "TTProgramExecutor"),
    "program_supported_by_executor": (
        "tt_thrml.compiler.sampler_lowering",
        "program_supported_by_executor",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
