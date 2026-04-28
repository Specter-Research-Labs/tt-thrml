"""Tenstorrent execution backend for THRML.

TT-Lang is the primary execution path for supported program shapes.
"""

from .api import make_executor
from .core import (
    close_device,
    close_devices,
    close_mesh_device,
    device_ids,
    is_mesh_device,
    open_device,
    open_devices,
    open_mesh_device,
)

__all__ = [
    "open_device",
    "open_devices",
    "open_mesh_device",
    "close_device",
    "close_mesh_device",
    "close_devices",
    "make_executor",
    "device_ids",
    "is_mesh_device",
]
