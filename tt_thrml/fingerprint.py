from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _is_real_torch_tensor(value: object) -> bool:
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return False
    tensor_type = getattr(torch, "Tensor", None)
    if tensor_type is None or tensor_type is object:
        return False
    return isinstance(value, tensor_type)


def _normalize(value: Any, *, _seen: set[int] | None = None):
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, type):
        return {
            "type": "python_type",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if isinstance(value, Enum):
        return {"type": "enum", "class": type(value).__name__, "value": value.value}
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": tuple(int(dim) for dim in value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if _is_real_torch_tensor(value):
        import torch

        detached = value.detach().cpu().contiguous()
        return {
            "type": "torch",
            "dtype": str(detached.dtype),
            "shape": tuple(int(dim) for dim in detached.shape),
            "sha256": hashlib.sha256(detached.numpy().tobytes()).hexdigest(),
        }
    object_id = id(value)
    if object_id in _seen:
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "cycle": True,
        }
    _seen.add(object_id)
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [
                (_normalize(key, _seen=_seen), _normalize(inner_value, _seen=_seen))
                for key, inner_value in sorted(value.items(), key=lambda item: repr(item[0]))
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "items": [_normalize(item, _seen=_seen) for item in value],
        }
    if isinstance(value, set):
        return {
            "type": "set",
            "items": sorted((_normalize(item, _seen=_seen) for item in value), key=repr),
        }
    if is_dataclass(value):
        return {
            "type": type(value).__name__,
            "fields": {
                field.name: _normalize(getattr(value, field.name), _seen=_seen)
                for field in fields(value)
            },
        }
    if hasattr(value, "__dict__"):
        public_items = {
            key: inner_value
            for key, inner_value in vars(value).items()
            if not key.startswith("_")
        }
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": _normalize(public_items, _seen=_seen),
        }
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def stable_fingerprint(value: Any) -> str:
    normalized = _normalize(value)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def program_fingerprint(program: object) -> str:
    return stable_fingerprint(program)


def backend_object_fingerprint(value: object) -> str:
    return stable_fingerprint(value)


__all__ = [
    "backend_object_fingerprint",
    "program_fingerprint",
    "stable_fingerprint",
]
