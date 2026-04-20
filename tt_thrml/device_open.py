from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence


def _dispatch_core_config(
    ttnn,
    *,
    dispatch_core_type: str,
    dispatch_core_axis: str,
):
    dispatch_core_type_enum = getattr(ttnn, "DispatchCoreType", None)
    dispatch_core_axis_enum = getattr(ttnn, "DispatchCoreAxis", None)
    dispatch_core_config_ctor = getattr(ttnn, "DispatchCoreConfig", None)
    if (
        dispatch_core_type_enum is None
        or dispatch_core_axis_enum is None
        or dispatch_core_config_ctor is None
    ):
        return None

    dispatch_core_type_value = getattr(dispatch_core_type_enum, dispatch_core_type)
    dispatch_core_axis_value = getattr(dispatch_core_axis_enum, dispatch_core_axis)
    return dispatch_core_config_ctor(dispatch_core_type_value, dispatch_core_axis_value)


def _mesh_shape_value(ttnn, mesh_shape):
    if mesh_shape is None:
        return None
    if isinstance(mesh_shape, tuple):
        mesh_shape_ctor = getattr(ttnn, "MeshShape", None)
        if callable(mesh_shape_ctor):
            return mesh_shape_ctor(*mesh_shape)
    return mesh_shape


def open_device(
    ttnn,
    *,
    device_id: int = 0,
    dispatch_core_type: str = "ETH",
    dispatch_core_axis: str = "ROW",
):
    dispatch_core_config = _dispatch_core_config(
        ttnn,
        dispatch_core_type=dispatch_core_type,
        dispatch_core_axis=dispatch_core_axis,
    )
    if hasattr(ttnn, "CreateDevice") and not hasattr(ttnn, "open_device"):
        create_kwargs = {
            "device_id": device_id,
            "l1_small_size": ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
            "trace_region_size": ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
        }
        if dispatch_core_config is not None:
            create_kwargs["dispatch_core_config"] = dispatch_core_config
        return ttnn.CreateDevice(
            **create_kwargs,
        )

    try:
        return ttnn.open_device(device_id=device_id)
    except TypeError:
        device_api = ttnn._ttnn.device
        open_kwargs = {
            "device_id": device_id,
            "l1_small_size": device_api.DEFAULT_L1_SMALL_SIZE,
            "trace_region_size": device_api.DEFAULT_TRACE_REGION_SIZE,
            "worker_l1_size": device_api.DEFAULT_WORKER_L1_SIZE,
        }
        if dispatch_core_config is not None:
            open_kwargs["dispatch_core_config"] = dispatch_core_config
        return ttnn.open_device(**open_kwargs)


def open_devices(
    ttnn,
    *,
    device_ids: Sequence[int],
    dispatch_core_type: str = "ETH",
    dispatch_core_axis: str = "ROW",
) -> tuple[object, ...]:
    normalized_ids = tuple(int(device_id) for device_id in device_ids)
    if not normalized_ids:
        raise ValueError("open_devices() requires at least one device id.")

    if len(normalized_ids) > 1 and hasattr(ttnn, "CreateDevices"):
        dispatch_core_config = _dispatch_core_config(
            ttnn,
            dispatch_core_type=dispatch_core_type,
            dispatch_core_axis=dispatch_core_axis,
        )
        create_kwargs = {
            "device_ids": list(normalized_ids),
            "l1_small_size": ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
            "trace_region_size": ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
        }
        if dispatch_core_config is not None:
            create_kwargs["dispatch_core_config"] = dispatch_core_config
        created = ttnn.CreateDevices(**create_kwargs)
        if isinstance(created, dict):
            return tuple(created[device_id] for device_id in normalized_ids)
        return tuple(created)

    return tuple(
        open_device(
            ttnn,
            device_id=device_id,
            dispatch_core_type=dispatch_core_type,
            dispatch_core_axis=dispatch_core_axis,
        )
        for device_id in normalized_ids
    )


def open_mesh_device(
    ttnn,
    *,
    mesh_shape,
    device_ids: Sequence[int] | None = None,
    dispatch_core_type: str = "ETH",
    dispatch_core_axis: str = "ROW",
    num_command_queues: int = 1,
    offset=None,
):
    open_mesh = getattr(ttnn, "open_mesh_device", None)
    if not callable(open_mesh):
        raise TypeError("TT backend does not expose open_mesh_device().")

    dispatch_core_config = _dispatch_core_config(
        ttnn,
        dispatch_core_type=dispatch_core_type,
        dispatch_core_axis=dispatch_core_axis,
    )
    open_kwargs = {
        "mesh_shape": _mesh_shape_value(ttnn, mesh_shape),
        "l1_small_size": ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
        "trace_region_size": ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
        "num_command_queues": num_command_queues,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
    }
    if dispatch_core_config is not None:
        open_kwargs["dispatch_core_config"] = dispatch_core_config
    if offset is not None:
        open_kwargs["offset"] = offset
    normalized_ids = None if device_ids is None else [int(device_id) for device_id in device_ids]
    if normalized_ids:
        open_kwargs["physical_device_ids"] = normalized_ids
    return open_mesh(**open_kwargs)


def close_mesh_device(ttnn, mesh_device) -> None:
    close_mesh = getattr(ttnn, "close_mesh_device", None)
    if callable(close_mesh):
        close_mesh(mesh_device)
        return
    close_device = getattr(ttnn, "close_device", None)
    if callable(close_device):
        close_device(mesh_device)


def close_devices(ttnn, devices: object | Sequence[object] | Mapping[int, object]) -> None:
    if isinstance(devices, Mapping):
        close_many_devices = getattr(ttnn, "CloseDevices", None)
        if callable(close_many_devices):
            close_many_devices(devices)
            return
        normalized = tuple(devices.values())
    elif isinstance(devices, (tuple, list)):
        normalized = tuple(devices)
    else:
        normalized = (devices,)

    if not normalized:
        return

    close_device = getattr(ttnn, "close_device", None)
    if callable(close_device):
        for device in normalized:
            close_device(device)
