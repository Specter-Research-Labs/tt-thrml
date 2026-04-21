from __future__ import annotations

import atexit
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import TYPE_CHECKING

from ..device_contract import raise_host_fallback_disabled

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TTMLIRConfig:
    system_desc_path: Path
    artifact_root: Path
    ttmlir_opt: str | Path = "ttmlir-opt"
    ttmlir_translate: str | Path = "ttmlir-translate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_desc_path", Path(self.system_desc_path).resolve())
        object.__setattr__(self, "artifact_root", Path(self.artifact_root).resolve())
        object.__setattr__(self, "ttmlir_opt", _normalize_tool_command(self.ttmlir_opt))
        object.__setattr__(
            self,
            "ttmlir_translate",
            _normalize_tool_command(self.ttmlir_translate),
        )

    def stable_cache_key(self) -> str:
        payload = {
            "system_desc_path": str(self.system_desc_path),
            "ttmlir_opt": self.ttmlir_opt,
            "ttmlir_translate": self.ttmlir_translate,
        }
        return _stable_cache_key(payload)


@dataclass(frozen=True)
class TTMLIRCompiledArtifact:
    artifact_dir: Path
    stablehlo_path: Path
    ttir_path: Path
    ttnn_path: Path
    flatbuffer_path: Path
    stablehlo_to_ttir_command: tuple[str, ...]
    ttir_to_ttnn_command: tuple[str, ...]
    ttnn_to_flatbuffer_command: tuple[str, ...]


@dataclass(frozen=True)
class TTMLIRExecutionResult:
    outputs: tuple[object, ...]
    runtime_output_dtypes: tuple[str, ...]


@dataclass
class TTMLIRRuntimeSession:
    tt_runtime: object
    device: object
    mesh_shape: tuple[int, ...]
    device_ids: tuple[int, ...] | None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


_RUNTIME_SESSION_CACHE: dict[
    tuple[str, str, tuple[int, ...], tuple[int, ...] | None, object], TTMLIRRuntimeSession
] = {}
_RUNTIME_SESSION_CACHE_LOCK = threading.Lock()
_COMPILED_ARTIFACT_CACHE: dict[
    tuple[str, TTMLIRConfig, object], TTMLIRCompiledArtifact
] = {}
_COMPILED_ARTIFACT_CACHE_LOCK = threading.Lock()
_COMPILED_ARTIFACT_COMPILE_LOCKS: dict[
    tuple[str, TTMLIRConfig, object], threading.Lock
] = {}


def _import_torch():
    import torch

    return torch


def _normalize_tool_command(command: str | Path) -> str:
    command_str = str(command)
    command_path = Path(command_str)
    if command_path.is_absolute() or command_path.parent != Path("."):
        return str(command_path.resolve())
    return command_str


def _stable_cache_key(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _command_exists(command: str) -> bool:
    command_path = Path(command)
    if command_path.is_absolute() or command_path.parent != Path("."):
        return command_path.exists()
    return shutil.which(command) is not None


def make_ttmlir_config(
    *,
    config: TTMLIRConfig | None = None,
    system_desc_path: Path | str | None = None,
    artifact_root: Path | str | None = None,
    build_dir: Path | str | None = None,
    ttmlir_opt: Path | str | None = None,
    ttmlir_translate: Path | str | None = None,
) -> TTMLIRConfig:
    env_build_dir = os.environ.get("TTMLIR_BUILD_DIR")

    if config is not None:
        if (
            system_desc_path is not None
            or artifact_root is not None
            or build_dir is not None
            or ttmlir_opt is not None
            or ttmlir_translate is not None
        ):
            raise ValueError(
                "Pass either `config=` or explicit TT-MLIR config fields, not both."
            )
        return config

    if system_desc_path is None:
        raise ValueError(
            "system_desc_path is required. tt-thrml expects TT-MLIR artifacts "
            "and runtime to be provisioned by the environment."
        )

    if build_dir is not None and (ttmlir_opt is not None or ttmlir_translate is not None):
        raise ValueError(
            "Pass either `build_dir=` or explicit compiler tool paths, not both."
        )

    if build_dir is None and ttmlir_opt is None and ttmlir_translate is None:
        if env_build_dir is None:
            raise ValueError(
                "TT-MLIR compiler tools must be configured explicitly. Pass "
                "`build_dir=...`, pass explicit `ttmlir_opt=` and "
                "`ttmlir_translate=`, or set `TTMLIR_BUILD_DIR`."
            )
        build_dir = env_build_dir

    if (ttmlir_opt is None) != (ttmlir_translate is None):
        raise ValueError(
            "Pass both `ttmlir_opt=` and `ttmlir_translate=` together."
        )

    if build_dir is not None:
        build_dir = Path(build_dir).resolve()
        ttmlir_opt = build_dir / "bin" / "ttmlir-opt"
        ttmlir_translate = build_dir / "bin" / "ttmlir-translate"

    resolved_artifact_root = Path(
        artifact_root
        if artifact_root is not None
        else Path(tempfile.gettempdir()) / "tt-thrml-ttmlir"
    ).resolve()
    return TTMLIRConfig(
        system_desc_path=Path(system_desc_path).resolve(),
        artifact_root=resolved_artifact_root,
        ttmlir_opt=ttmlir_opt,
        ttmlir_translate=ttmlir_translate,
    )


def stablehlo_text(lowered) -> str:
    if hasattr(lowered, "as_text"):
        return lowered.as_text(dialect="stablehlo")
    return str(lowered.compiler_ir(dialect="stablehlo"))


def _run_cli(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        text=True,
    )


def require_stablehlo_cli(ttmlir_opt: str) -> None:
    if not _command_exists(ttmlir_opt):
        raise FileNotFoundError(
            "Missing TT-MLIR compiler tool `ttmlir-opt`. Install the prebuilt "
            "toolchain/container, pass `build_dir=...`, or set `TTMLIR_BUILD_DIR`."
        )
    help_result = subprocess.run(
        [ttmlir_opt, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "--stablehlo-to-ttir-pipeline" not in help_result.stdout:
        raise RuntimeError(
            "This tt-mlir build does not expose the StableHLO frontend "
            "(`--stablehlo-to-ttir-pipeline`). Rebuild with "
            "`-DTTMLIR_ENABLE_STABLEHLO=ON`."
        )


def ensure_system_desc(config: TTMLIRConfig) -> None:
    if config.system_desc_path.exists():
        return
    raise FileNotFoundError(
        "Missing TT-MLIR system descriptor at "
        f"{config.system_desc_path}. Generate it ahead of time with "
        "`ttrt query --save-artifacts` or provide the correct path."
    )


def compile_stablehlo_to_flatbuffer(
    config: TTMLIRConfig,
    *,
    stablehlo_module_text: str,
    artifact_dir: Path,
    base_name: str,
) -> TTMLIRCompiledArtifact:
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not _command_exists(config.ttmlir_translate):
        raise FileNotFoundError(
            "Missing TT-MLIR translator tool `ttmlir-translate`. Install the "
            "prebuilt toolchain/container, pass `build_dir=...`, or set "
            "`TTMLIR_BUILD_DIR`."
        )

    require_stablehlo_cli(config.ttmlir_opt)
    ensure_system_desc(config)

    stablehlo_path = artifact_dir / f"{base_name}.stablehlo.mlir"
    ttir_path = artifact_dir / f"{base_name}.ttir.mlir"
    ttnn_path = artifact_dir / f"{base_name}.ttnn.mlir"
    flatbuffer_path = artifact_dir / f"{base_name}.ttnn"
    stablehlo_path.write_text(stablehlo_module_text)

    stablehlo_to_ttir_command = [
        config.ttmlir_opt,
        "--stablehlo-to-ttir-pipeline",
        str(stablehlo_path),
        "-o",
        str(ttir_path),
    ]
    ttir_to_ttnn_command = [
        config.ttmlir_opt,
        (
            "--ttir-to-ttnn-backend-pipeline="
            "enable-cpu-hoisted-const-eval=false "
            f"system-desc-path={config.system_desc_path}"
        ),
        str(ttir_path),
        "-o",
        str(ttnn_path),
    ]
    ttnn_to_flatbuffer_command = [
        config.ttmlir_translate,
        "--ttnn-to-flatbuffer",
        str(ttnn_path),
        "-o",
        str(flatbuffer_path),
    ]

    _run_cli(stablehlo_to_ttir_command)
    _run_cli(ttir_to_ttnn_command)
    _run_cli(ttnn_to_flatbuffer_command)

    return TTMLIRCompiledArtifact(
        artifact_dir=artifact_dir,
        stablehlo_path=stablehlo_path,
        ttir_path=ttir_path,
        ttnn_path=ttnn_path,
        flatbuffer_path=flatbuffer_path,
        stablehlo_to_ttir_command=tuple(stablehlo_to_ttir_command),
        ttir_to_ttnn_command=tuple(ttir_to_ttnn_command),
        ttnn_to_flatbuffer_command=tuple(ttnn_to_flatbuffer_command),
    )


def _signature_cache_key(signature: object) -> str:
    stable_cache_key = getattr(signature, "stable_cache_key", None)
    if not callable(stable_cache_key):
        raise TypeError(
            "TT-MLIR artifact signatures must define a callable "
            "`stable_cache_key()` helper."
        )
    return stable_cache_key()


def _compiled_artifact_cache_dir(
    config: TTMLIRConfig,
    *,
    family: str,
    signature: object,
) -> Path:
    return (
        config.artifact_root
        / family
        / config.stable_cache_key()
        / _signature_cache_key(signature)
    )


def _compiled_artifact_cache_lock(
    cache_key: tuple[str, TTMLIRConfig, object],
) -> threading.Lock:
    with _COMPILED_ARTIFACT_CACHE_LOCK:
        return _COMPILED_ARTIFACT_COMPILE_LOCKS.setdefault(cache_key, threading.Lock())


def get_or_compile_cached_artifact(
    config: TTMLIRConfig,
    *,
    family: str,
    signature: object,
    base_name_prefix: str,
    stablehlo_module_text_factory: Callable[[], str],
    compile_fn: Callable[..., TTMLIRCompiledArtifact] = compile_stablehlo_to_flatbuffer,
) -> TTMLIRCompiledArtifact:
    cache_key = (family, config, signature)
    with _COMPILED_ARTIFACT_CACHE_LOCK:
        artifact = _COMPILED_ARTIFACT_CACHE.get(cache_key)
    if artifact is not None:
        return artifact

    with _compiled_artifact_cache_lock(cache_key):
        with _COMPILED_ARTIFACT_CACHE_LOCK:
            artifact = _COMPILED_ARTIFACT_CACHE.get(cache_key)
        if artifact is not None:
            return artifact

        signature_cache_key = _signature_cache_key(signature)
        artifact = compile_fn(
            config,
            stablehlo_module_text=stablehlo_module_text_factory(),
            artifact_dir=_compiled_artifact_cache_dir(
                config,
                family=family,
                signature=signature,
            ),
            base_name=f"{base_name_prefix}_{signature_cache_key}",
        )
        with _COMPILED_ARTIFACT_CACHE_LOCK:
            _COMPILED_ARTIFACT_CACHE[cache_key] = artifact
        return artifact


def _runtime_dtype_to_torch_dtype(tt_runtime, dtype):
    torch = _import_torch()
    mapping = {
        tt_runtime.DataType.Float32: torch.float32,
        tt_runtime.DataType.Float16: torch.float16,
        tt_runtime.DataType.BFloat16: torch.bfloat16,
        tt_runtime.DataType.UInt32: torch.uint32,
        tt_runtime.DataType.UInt16: torch.uint16,
        tt_runtime.DataType.UInt8: torch.uint8,
        tt_runtime.DataType.Int32: torch.int32,
        tt_runtime.DataType.Bool: torch.bool,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported runtime dtype: {dtype}")
    return mapping[dtype]


def _torch_dtype_to_runtime_dtype(tt_runtime, dtype):
    torch = _import_torch()
    mapping = {
        torch.float32: tt_runtime.DataType.Float32,
        torch.float16: tt_runtime.DataType.Float16,
        torch.bfloat16: tt_runtime.DataType.BFloat16,
        torch.uint32: tt_runtime.DataType.UInt32,
        torch.uint16: tt_runtime.DataType.UInt16,
        torch.uint8: tt_runtime.DataType.UInt8,
        torch.int32: tt_runtime.DataType.Int32,
        torch.bool: tt_runtime.DataType.Bool,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


def _create_runtime_input(tt_runtime, tensor):
    tensor = tensor.contiguous()
    return tt_runtime.create_borrowed_host_tensor(
        tensor.data_ptr(),
        list(tensor.shape),
        list(tensor.stride()),
        tensor.element_size(),
        _torch_dtype_to_runtime_dtype(tt_runtime, tensor.dtype),
    )


def _runtime_tensor_to_torch(tt_runtime, runtime_tensor):
    torch = _import_torch()
    data_buffer = bytearray(runtime_tensor.get_data_buffer())
    shape = tuple(runtime_tensor.get_shape())
    dtype = _runtime_dtype_to_torch_dtype(tt_runtime, runtime_tensor.get_dtype())
    if len(data_buffer) == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(data_buffer, dtype=dtype).reshape(shape).clone()


def _is_torch_tensor(value) -> bool:
    try:
        torch = _import_torch()
    except ModuleNotFoundError:
        return False
    tensor_type = getattr(torch, "Tensor", None)
    if tensor_type is None or tensor_type is object:
        return False
    return isinstance(value, tensor_type)


def _normalized_dtype_name(dtype) -> str | None:
    if dtype is None:
        return None
    text = str(dtype).lower()
    if "bfloat16" in text or text == "bf16":
        return "bfloat16"
    if "float32" in text or text.endswith("float"):
        return "float32"
    if "uint32" in text:
        return "uint32"
    if "int32" in text:
        return "int32"
    if "int64" in text:
        return "int64"
    return text.replace("torch.", "").replace("dtype.", "")


def _dtype_matches(actual, expected) -> bool:
    if actual == expected:
        return True
    actual_name = _normalized_dtype_name(actual)
    expected_name = _normalized_dtype_name(expected)
    return actual_name is not None and actual_name == expected_name


def _resolve_runtime_utils_module(tt_runtime):
    if callable(getattr(tt_runtime, "create_runtime_device_from_ttnn", None)):
        return tt_runtime

    try:
        from ttrt.runtime import _ttmlir_runtime
    except (ImportError, ModuleNotFoundError):
        return None

    return getattr(_ttmlir_runtime, "utils", None)


def _supports_existing_ttnn_device_bridge(tt_runtime, device) -> bool:
    runtime_utils = _resolve_runtime_utils_module(tt_runtime)
    return device is not None and runtime_utils is not None and callable(
        getattr(runtime_utils, "create_runtime_device_from_ttnn", None)
    )


def _supports_direct_device_inputs(runtime_utils) -> bool:
    return callable(getattr(runtime_utils, "create_runtime_tensor_from_ttnn", None))


def _supports_direct_device_outputs(runtime_utils) -> bool:
    return callable(getattr(runtime_utils, "get_ttnn_tensor_from_runtime_tensor", None))


def supports_direct_ttnn_inputs(*, device=None) -> bool:
    if device is None:
        return False

    try:
        tt_runtime = _import_ttrt_runtime()
    except RuntimeError:
        return False

    runtime_utils = _resolve_runtime_utils_module(tt_runtime)
    return (
        runtime_utils is not None
        and callable(getattr(runtime_utils, "create_runtime_device_from_ttnn", None))
        and _supports_direct_device_inputs(runtime_utils)
    )


def supports_direct_ttnn_outputs(*, device=None) -> bool:
    if device is None:
        return False

    try:
        tt_runtime = _import_ttrt_runtime()
    except RuntimeError:
        return False

    runtime_utils = _resolve_runtime_utils_module(tt_runtime)
    return (
        runtime_utils is not None
        and callable(getattr(runtime_utils, "create_runtime_device_from_ttnn", None))
        and _supports_direct_device_outputs(runtime_utils)
    )


def _prepare_runtime_input_for_device_execution(
    tt_runtime,
    *,
    runtime_utils,
    executable,
    runtime_device,
    input_index: int,
    tensor,
):
    if _is_torch_tensor(tensor):
        runtime_tensor = _create_runtime_input(tt_runtime, tensor)
    else:
        create_runtime_tensor_from_ttnn = getattr(
            runtime_utils, "create_runtime_tensor_from_ttnn", None
        )
        if not callable(create_runtime_tensor_from_ttnn):
            raise TypeError(
                "ttrt.runtime does not expose `create_runtime_tensor_from_ttnn`, "
                "so TTNN tensors cannot be used for direct device execution."
            )
        runtime_tensor = create_runtime_tensor_from_ttnn(tensor, True)

    layout = tt_runtime.get_layout(executable, 0, input_index)
    return tt_runtime.to_layout(runtime_tensor, runtime_device, layout, True)


def _runtime_outputs_to_torch(tt_runtime, runtime_outputs):
    outputs = []
    runtime_output_dtypes = []
    pending_outputs = list(runtime_outputs)
    try:
        for runtime_output in pending_outputs:
            output_shards = tt_runtime.to_host(runtime_output, untilize=True)
            if len(output_shards) != 1:
                raise RuntimeError(
                    f"Expected one output shard, got {len(output_shards)}"
                )
            runtime_output_dtypes.append(str(output_shards[0].get_dtype()))
            outputs.append(_runtime_tensor_to_torch(tt_runtime, output_shards[0]))
            tt_runtime.deallocate_tensor(runtime_output, force=True)
        pending_outputs = []
    finally:
        for runtime_output in pending_outputs:
            tt_runtime.deallocate_tensor(runtime_output, force=True)

    return tuple(outputs), tuple(runtime_output_dtypes)


def _runtime_outputs_to_device_tensors(tt_runtime, runtime_utils, runtime_outputs):
    get_ttnn_tensor_from_runtime_tensor = getattr(
        runtime_utils, "get_ttnn_tensor_from_runtime_tensor", None
    )
    if not callable(get_ttnn_tensor_from_runtime_tensor):
        raise TypeError(
            "ttrt.runtime does not expose `get_ttnn_tensor_from_runtime_tensor`, "
            "so direct TTNN output restoration is unavailable."
        )

    outputs = []
    runtime_output_dtypes = []
    pending_outputs = list(runtime_outputs)
    try:
        while pending_outputs:
            runtime_output = pending_outputs.pop(0)
            runtime_output_dtypes.append(str(runtime_output.get_dtype()))
            # The bridge helper transfers the runtime tensor into a TTNN tensor.
            outputs.append(get_ttnn_tensor_from_runtime_tensor(runtime_output))
    except Exception:
        for runtime_output in pending_outputs:
            tt_runtime.deallocate_tensor(runtime_output, force=True)
        raise

    return tuple(outputs), tuple(runtime_output_dtypes)


def restore_output_tensor(
    ttnn,
    *,
    device,
    output,
    output_dtype,
    output_layout,
):
    if _is_torch_tensor(output):
        raise_host_fallback_disabled(
            "TT-MLIR output restoration",
            remedy="Use a runtime with direct TTNN output bridge support.",
        )

    restored = output
    current_dtype = getattr(restored, "dtype", None)
    if (
        output_dtype is not None
        and current_dtype is not None
        and not _dtype_matches(current_dtype, output_dtype)
    ):
        typecast = getattr(ttnn, "typecast", None)
        if callable(typecast):
            restored = typecast(restored, dtype=output_dtype)
        else:
            to_dtype = getattr(ttnn, "to_dtype", None)
            if callable(to_dtype):
                restored = to_dtype(restored, output_dtype)

    current_layout = getattr(restored, "layout", None)
    if output_layout is not None and current_layout is not None and current_layout != output_layout:
        to_layout = getattr(ttnn, "to_layout", None)
        if callable(to_layout):
            restored = to_layout(restored, output_layout)

    final_layout = getattr(restored, "layout", None)
    if output_layout is not None and final_layout is not None and final_layout != output_layout:
        raise_host_fallback_disabled(
            "TT-MLIR output layout restoration",
            remedy=(
                "Expose ttnn.to_layout for direct TTNN tensors or compile the kernel "
                "to produce the requested output layout."
            ),
        )

    final_dtype = getattr(restored, "dtype", None)
    if (
        output_dtype is not None
        and final_dtype is not None
        and not _dtype_matches(final_dtype, output_dtype)
    ):
        raise_host_fallback_disabled(
            "TT-MLIR output dtype restoration",
            remedy=(
                "Expose ttnn.typecast/to_dtype for direct TTNN tensors or compile the "
                "kernel to produce the requested output dtype."
            ),
        )

    return restored


def _import_ttrt_runtime():
    try:
        import ttrt.runtime as tt_runtime
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "tt-thrml expects `ttrt.runtime` to be installed by the environment. "
            "Use the Tenstorrent runtime container/toolchain instead of relying on "
            "library-side sys.path discovery."
        ) from exc
    return tt_runtime


def _import_ttrt_util_types():
    try:
        from ttrt.common.util import Binary, FileManager, Logger
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "tt-thrml expects TTRT Python utilities to be installed by the "
            "environment. Use the Tenstorrent runtime container/toolchain instead "
            "of relying on library-side sys.path discovery."
        ) from exc
    return Binary, FileManager, Logger


@contextmanager
def _compatible_device_runtime(tt_runtime, executable):
    get_current_device_runtime = getattr(tt_runtime, "get_current_device_runtime", None)
    set_current_device_runtime = getattr(tt_runtime, "set_current_device_runtime", None)
    previous_runtime = (
        get_current_device_runtime() if callable(get_current_device_runtime) else None
    )
    tt_runtime.set_compatible_device_runtime(executable)
    try:
        yield (
            get_current_device_runtime()
            if callable(get_current_device_runtime)
            else None
        )
    finally:
        if previous_runtime is not None and callable(set_current_device_runtime):
            set_current_device_runtime(previous_runtime)


def _is_retryable_mesh_open_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "Failed to pin pages for hugepage" in message
        or "Cannot allocate memory" in message
    )


def _open_mesh_device_with_retry(tt_runtime, mesh_options):
    max_attempts = max(int(os.environ.get("TTMLIR_OPEN_DEVICE_ATTEMPTS", "3")), 1)
    backoff_seconds = float(
        os.environ.get("TTMLIR_OPEN_DEVICE_BACKOFF_SECONDS", "2.0")
    )
    jitter_seconds = (os.getpid() % 11) * 0.1

    for attempt in range(1, max_attempts + 1):
        try:
            return tt_runtime.open_mesh_device(mesh_options)
        except RuntimeError as exc:
            if not _is_retryable_mesh_open_error(exc) or attempt == max_attempts:
                raise
            time.sleep(backoff_seconds * attempt + jitter_seconds)


def _normalize_mesh_shape(mesh_shape) -> tuple[int, ...]:
    return tuple(int(dim) for dim in mesh_shape)


def _normalize_device_ids(device_ids) -> tuple[int, ...] | None:
    if device_ids is None:
        return None
    normalized = tuple(int(device_id) for device_id in device_ids)
    if not normalized:
        return None
    return normalized


def _resolve_runtime_device_ids(device) -> tuple[int, ...] | None:
    if device is None:
        return None

    get_device_ids = getattr(device, "get_device_ids", None)
    if callable(get_device_ids):
        return _normalize_device_ids(get_device_ids())

    get_device_id = getattr(device, "id", None)
    if callable(get_device_id):
        return _normalize_device_ids((get_device_id(),))

    device_id = getattr(device, "device_id", None)
    if device_id is not None:
        return _normalize_device_ids((device_id,))

    return None


def _close_runtime_session(session: TTMLIRRuntimeSession) -> None:
    tt_runtime = session.tt_runtime
    try:
        tt_runtime.close_mesh_device(session.device)
    finally:
        tt_runtime.set_fabric_config(tt_runtime.FabricConfig.DISABLED)


def _close_cached_runtime_sessions() -> None:
    with _RUNTIME_SESSION_CACHE_LOCK:
        sessions = tuple(_RUNTIME_SESSION_CACHE.values())
        _RUNTIME_SESSION_CACHE.clear()
    for session in sessions:
        _close_runtime_session(session)


atexit.register(_close_cached_runtime_sessions)


@contextmanager
def borrow_runtime_session(
    config: TTMLIRConfig,
    *,
    mesh_shape,
    device_ids=None,
    device_runtime=None,
):
    tt_runtime = _import_ttrt_runtime()

    normalized_mesh_shape = _normalize_mesh_shape(mesh_shape)
    normalized_device_ids = _normalize_device_ids(device_ids)
    cache_key = (
        str(config.system_desc_path),
        normalized_mesh_shape,
        normalized_device_ids,
        device_runtime,
    )
    with _RUNTIME_SESSION_CACHE_LOCK:
        session = _RUNTIME_SESSION_CACHE.get(cache_key)
        if session is None:
            mesh_options = tt_runtime.MeshDeviceOptions()
            mesh_options.mesh_shape = normalized_mesh_shape
            if normalized_device_ids is not None:
                mesh_options.device_ids = list(normalized_device_ids)
            if device_runtime is not None:
                tt_runtime.set_current_device_runtime(device_runtime)
            session = TTMLIRRuntimeSession(
                tt_runtime=tt_runtime,
                device=_open_mesh_device_with_retry(tt_runtime, mesh_options),
                mesh_shape=normalized_mesh_shape,
                device_ids=normalized_device_ids,
            )
            _RUNTIME_SESSION_CACHE[cache_key] = session

    with session.lock:
        yield session


def run_flatbuffer(
    config: TTMLIRConfig,
    *,
    flatbuffer_path: Path,
    input_tensors: list[object],
    device=None,
    prefer_device_output: bool = False,
) -> TTMLIRExecutionResult:
    tt_runtime = _import_ttrt_runtime()
    Binary, FileManager, Logger = _import_ttrt_util_types()

    logger = Logger()
    file_manager = FileManager(logger)
    binary = Binary(logger, file_manager, str(flatbuffer_path))
    with _compatible_device_runtime(tt_runtime, binary.fbb) as device_runtime:
        if not _supports_existing_ttnn_device_bridge(tt_runtime, device):
            raise_host_fallback_disabled(
                "TT-MLIR runtime device bridge",
                remedy="Use a runtime that exposes direct TTNN bridge helpers.",
            )
        runtime_utils = _resolve_runtime_utils_module(tt_runtime)
        if runtime_utils is None:
            raise RuntimeError(
                "TT-MLIR runtime bridge unexpectedly missing after support check."
            )
        if not _supports_direct_device_outputs(runtime_utils):
            raise_host_fallback_disabled(
                "TT-MLIR runtime output bridge",
                remedy="Use a runtime that exposes direct TTNN output bridge helpers.",
            )

        runtime_device = runtime_utils.create_runtime_device_from_ttnn(device)
        runtime_inputs = []
        try:
            for input_index, tensor in enumerate(input_tensors):
                runtime_inputs.append(
                    _prepare_runtime_input_for_device_execution(
                        tt_runtime,
                        runtime_utils=runtime_utils,
                        executable=binary.fbb,
                        runtime_device=runtime_device,
                        input_index=input_index,
                        tensor=tensor,
                    )
                )

            runtime_outputs = tt_runtime.submit(
                runtime_device, binary.fbb, 0, runtime_inputs
            )
            outputs, runtime_output_dtypes = _runtime_outputs_to_device_tensors(
                tt_runtime, runtime_utils, runtime_outputs
            )
        finally:
            for runtime_input in runtime_inputs:
                tt_runtime.deallocate_tensor(runtime_input, force=True)

    return TTMLIRExecutionResult(
        outputs=outputs,
        runtime_output_dtypes=runtime_output_dtypes,
    )
