from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TypeAlias

from .fingerprint import backend_object_fingerprint

BackendDevice: TypeAlias = object
BackendDevices: TypeAlias = BackendDevice | tuple[BackendDevice, ...] | list[BackendDevice]


class ParameterFamily(str, Enum):
    SPIN = "spin_natural_parameter"
    CATEGORICAL = "categorical_logits"
    GAUSSIAN = "gaussian_canonical"


@dataclass(frozen=True)
class ParameterFamilySpec:
    family: ParameterFamily
    sampler_kind: str
    output_node_kind: str
    random_source_kind: str
    interaction_minimum_ndim: int
    interaction_tail_axis_start: int
    categorical_axis: int | None = None


SPIN_PARAMETER_FAMILY = ParameterFamily.SPIN
CATEGORICAL_PARAMETER_FAMILY = ParameterFamily.CATEGORICAL
GAUSSIAN_PARAMETER_FAMILY = ParameterFamily.GAUSSIAN


class ParameterKernelBackend(str, Enum):
    NATIVE = "native"
    TTMLIR = "ttmlir"
    CUSTOM = "custom"

_PARAMETER_FAMILY_SPECS: dict[ParameterFamily, ParameterFamilySpec] = {
    ParameterFamily.SPIN: ParameterFamilySpec(
        family=ParameterFamily.SPIN,
        sampler_kind="spin",
        output_node_kind="spin",
        random_source_kind="spin_threshold_logits",
        interaction_minimum_ndim=2,
        interaction_tail_axis_start=2,
    ),
    ParameterFamily.CATEGORICAL: ParameterFamilySpec(
        family=ParameterFamily.CATEGORICAL,
        sampler_kind="categorical",
        output_node_kind="categorical",
        random_source_kind="categorical_gumbels",
        interaction_minimum_ndim=3,
        interaction_tail_axis_start=3,
        categorical_axis=2,
    ),
    ParameterFamily.GAUSSIAN: ParameterFamilySpec(
        family=ParameterFamily.GAUSSIAN,
        sampler_kind="continuous",
        output_node_kind="continuous",
        random_source_kind="gaussian_normals",
        interaction_minimum_ndim=2,
        interaction_tail_axis_start=3,
    ),
}

ParameterKernelFamily: TypeAlias = ParameterFamily | str
ParameterKernelOps: TypeAlias = (
    Mapping[ParameterKernelFamily, object]
    | Iterable[tuple[ParameterKernelFamily, object]]
)
ParameterKernelBackendFamily: TypeAlias = ParameterFamily | str
ParameterKernelBackends: TypeAlias = (
    Mapping[ParameterKernelBackendFamily, ParameterKernelBackend | str]
    | Iterable[tuple[ParameterKernelBackendFamily, ParameterKernelBackend | str]]
)


def normalize_parameter_family(parameter_family: ParameterKernelFamily) -> ParameterFamily:
    if isinstance(parameter_family, ParameterFamily):
        return parameter_family
    if not isinstance(parameter_family, str):
        raise TypeError(
            "parameter_family must be provided as a string or ParameterFamily."
        )
    try:
        return ParameterFamily(parameter_family)
    except ValueError as exc:
        raise ValueError(f"Unsupported parameter family: {parameter_family!r}.") from exc


def parameter_family_spec(
    parameter_family: ParameterKernelFamily,
) -> ParameterFamilySpec:
    return _PARAMETER_FAMILY_SPECS[normalize_parameter_family(parameter_family)]


def normalize_parameter_kernel_backend(
    parameter_kernel_backend: ParameterKernelBackend | str,
) -> ParameterKernelBackend:
    if isinstance(parameter_kernel_backend, ParameterKernelBackend):
        return parameter_kernel_backend
    if not isinstance(parameter_kernel_backend, str):
        raise TypeError(
            "parameter kernel backend must be provided as a string or "
            "ParameterKernelBackend."
        )
    try:
        return ParameterKernelBackend(parameter_kernel_backend)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported parameter kernel backend: {parameter_kernel_backend!r}."
        ) from exc


def normalize_backend_devices(device: BackendDevices) -> tuple[BackendDevice, ...]:
    if isinstance(device, tuple):
        return device
    if isinstance(device, list):
        return tuple(device)
    return (device,)


def _normalize_parameter_kernel_ops_items(
    parameter_kernel_ops: ParameterKernelOps | None = None,
) -> tuple[tuple[ParameterFamily, object], ...]:
    merged: dict[ParameterFamily, object] = {}
    if parameter_kernel_ops is not None:
        items = (
            parameter_kernel_ops.items()
            if isinstance(parameter_kernel_ops, Mapping)
            else parameter_kernel_ops
        )
        for family, op in items:
            if op is None:
                continue
            merged[normalize_parameter_family(family)] = op
    return tuple(sorted(merged.items(), key=lambda item: item[0].value))


def _normalize_parameter_kernel_backends_items(
    parameter_kernel_backends: ParameterKernelBackends | None = None,
) -> tuple[tuple[ParameterFamily, ParameterKernelBackend], ...]:
    merged: dict[ParameterFamily, ParameterKernelBackend] = {}
    if parameter_kernel_backends is not None:
        items = (
            parameter_kernel_backends.items()
            if isinstance(parameter_kernel_backends, Mapping)
            else parameter_kernel_backends
        )
        for family, backend in items:
            merged[normalize_parameter_family(family)] = (
                normalize_parameter_kernel_backend(backend)
            )
    return tuple(sorted(merged.items(), key=lambda item: item[0].value))


def merge_parameter_kernel_ops(
    parameter_kernel_ops: ParameterKernelOps | None = None,
    *,
    base_parameter_kernel_ops: ParameterKernelOps | None = None,
) -> dict[ParameterFamily, object]:
    merged = dict(_normalize_parameter_kernel_ops_items(base_parameter_kernel_ops))
    merged.update(_normalize_parameter_kernel_ops_items(parameter_kernel_ops))
    return merged


def merge_parameter_kernel_backends(
    parameter_kernel_backends: ParameterKernelBackends | None = None,
    *,
    base_parameter_kernel_backends: ParameterKernelBackends | None = None,
) -> dict[ParameterFamily, ParameterKernelBackend]:
    merged = dict(
        _normalize_parameter_kernel_backends_items(base_parameter_kernel_backends)
    )
    merged.update(_normalize_parameter_kernel_backends_items(parameter_kernel_backends))
    return merged


def resolve_parameter_kernel_backend(
    parameter_family: ParameterKernelFamily,
    parameter_kernel_backends: ParameterKernelBackends | None = None,
) -> ParameterKernelBackend:
    family = normalize_parameter_family(parameter_family)
    backends = dict(_normalize_parameter_kernel_backends_items(parameter_kernel_backends))
    return backends.get(family, ParameterKernelBackend.NATIVE)


@dataclass(frozen=True)
class BackendBinding:
    ttnn: object
    devices: tuple[BackendDevice, ...]
    _parameter_kernel_ops: tuple[tuple[ParameterFamily, object], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    _parameter_kernel_backends: tuple[
        tuple[ParameterFamily, ParameterKernelBackend], ...
    ] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        normalized_devices = normalize_backend_devices(self.devices)
        if not normalized_devices:
            raise ValueError("TT backend binding must include at least one device.")
        object.__setattr__(self, "devices", normalized_devices)
        object.__setattr__(
            self,
            "_parameter_kernel_ops",
            _normalize_parameter_kernel_ops_items(self._parameter_kernel_ops),
        )
        object.__setattr__(
            self,
            "_parameter_kernel_backends",
            _normalize_parameter_kernel_backends_items(self._parameter_kernel_backends),
        )
        parameter_kernel_ops = dict(self._parameter_kernel_ops)
        parameter_kernel_backends = dict(self._parameter_kernel_backends)
        missing_backends = sorted(
            family.value
            for family in parameter_kernel_ops
            if family not in parameter_kernel_backends
        )
        if missing_backends:
            raise ValueError(
                "parameter kernel ops require explicit parameter_kernel_backends "
                f"entries for families: {missing_backends!r}."
            )
        invalid_ttnn_fallback_overrides = sorted(
            family.value
            for family, backend in parameter_kernel_backends.items()
            if backend is ParameterKernelBackend.NATIVE and family in parameter_kernel_ops
        )
        if invalid_ttnn_fallback_overrides:
            raise ValueError(
                "parameter kernel ops cannot be attached to TTNN fallback parameter "
                f"kernel families: {invalid_ttnn_fallback_overrides!r}."
            )

    @property
    def primary_device(self) -> BackendDevice:
        return self.devices[0]

    @property
    def parameter_kernel_ops(self) -> dict[ParameterFamily, object]:
        return dict(self._parameter_kernel_ops)

    @property
    def parameter_kernel_backends(self) -> dict[ParameterFamily, ParameterKernelBackend]:
        return dict(self._parameter_kernel_backends)

    def with_parameter_kernel_ops(
        self,
        parameter_kernel_ops: ParameterKernelOps | None = None,
    ):
        merged = merge_parameter_kernel_ops(
            parameter_kernel_ops,
            base_parameter_kernel_ops=self._parameter_kernel_ops,
        )
        normalized = tuple(sorted(merged.items(), key=lambda item: item[0].value))
        if normalized == self._parameter_kernel_ops:
            return self
        return replace(self, _parameter_kernel_ops=normalized)

    def with_parameter_kernel_backends(
        self,
        parameter_kernel_backends: ParameterKernelBackends | None = None,
    ):
        merged = merge_parameter_kernel_backends(
            parameter_kernel_backends,
            base_parameter_kernel_backends=self._parameter_kernel_backends,
        )
        normalized = tuple(sorted(merged.items(), key=lambda item: item[0].value))
        if normalized == self._parameter_kernel_backends:
            return self
        return replace(self, _parameter_kernel_backends=normalized)

    def with_parameter_kernel_op(
        self,
        parameter_family: ParameterKernelFamily,
        parameter_kernel_op,
    ):
        return self.with_parameter_kernel_ops(
            {normalize_parameter_family(parameter_family): parameter_kernel_op}
        )

    def with_parameter_kernel_backend(
        self,
        parameter_family: ParameterKernelBackendFamily,
        parameter_kernel_backend: ParameterKernelBackend | str,
    ):
        return self.with_parameter_kernel_backends(
            {
                normalize_parameter_family(parameter_family): normalize_parameter_kernel_backend(
                    parameter_kernel_backend
                )
            }
        )

    @property
    def parameter_kernel_backend_key(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (family.value, backend.value)
            for family, backend in self._parameter_kernel_backends
        )

    @property
    def parameter_kernel_op_key(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (family.value, backend_object_fingerprint(op))
            for family, op in self._parameter_kernel_ops
        )

    @property
    def semantic_ttnn_key(self) -> str:
        return backend_object_fingerprint(self.ttnn)

    @property
    def device_key(self) -> tuple[str, ...]:
        return tuple(backend_object_fingerprint(device) for device in self.devices)

    @property
    def cache_key(
        self,
    ) -> tuple[
        str,
        tuple[str, ...],
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str], ...],
    ]:
        return (
            self.semantic_ttnn_key,
            self.device_key,
            self.parameter_kernel_backend_key,
            self.parameter_kernel_op_key,
        )

    def device_cache_key(
        self,
        device: BackendDevice | None = None,
    ) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        resolved_device = self.primary_device if device is None else device
        ttnn_key, _device_ids, parameter_kernel_backend_key, _parameter_kernel_op_key = (
            self.cache_key
        )
        return (
            ttnn_key,
            backend_object_fingerprint(resolved_device),
            parameter_kernel_backend_key,
        )

    def executor_cache_key(
        self,
        device: BackendDevice | None = None,
    ) -> tuple[
        str,
        str,
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str], ...],
    ]:
        resolved_device = self.primary_device if device is None else device
        (
            ttnn_key,
            _device_ids,
            parameter_kernel_backend_key,
            parameter_kernel_op_key,
        ) = self.cache_key
        return (
            ttnn_key,
            backend_object_fingerprint(resolved_device),
            parameter_kernel_backend_key,
            parameter_kernel_op_key,
        )


@dataclass(frozen=True)
class ExecutionOptions:
    profiler: object | None = None
    profile_sync: bool = False
    progress: Callable[[str], None] | None = None

    @property
    def cacheable(self) -> bool:
        return self.profiler is None and self.progress is None and not self.profile_sync


def make_backend_binding(
    ttnn,
    device,
    *,
    parameter_kernel_ops: ParameterKernelOps | None = None,
    parameter_kernel_backends: ParameterKernelBackends | None = None,
) -> BackendBinding:
    return BackendBinding(
        ttnn=ttnn,
        devices=normalize_backend_devices(device),
        _parameter_kernel_ops=_normalize_parameter_kernel_ops_items(parameter_kernel_ops),
        _parameter_kernel_backends=_normalize_parameter_kernel_backends_items(
            parameter_kernel_backends
        ),
    )


def require_backend(backend: BackendBinding | None) -> BackendBinding:
    if not isinstance(backend, BackendBinding):
        raise TypeError("backend must be provided as a BackendBinding.")
    return backend


def normalize_execution_options(
    options: ExecutionOptions | None = None,
) -> ExecutionOptions:
    if options is None:
        return ExecutionOptions()
    if not isinstance(options, ExecutionOptions):
        raise TypeError("options must be an ExecutionOptions instance.")
    return options
