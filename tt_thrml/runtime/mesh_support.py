from __future__ import annotations

from math import prod


def _mesh_shape_tuple(device) -> tuple[int, ...]:
    shape = getattr(device, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        dims = getattr(shape, "dims", None)
        if callable(dims):
            return tuple(int(shape[index]) for index in range(dims()))
        return ()


def mesh_device_ids(device) -> tuple[int, ...]:
    get_device_ids = getattr(device, "get_device_ids", None)
    if callable(get_device_ids):
        return tuple(int(device_id) for device_id in get_device_ids())
    device_id = getattr(device, "id", None)
    if isinstance(device_id, int):
        return (device_id,)
    return ()


def mesh_device_size(device) -> int:
    device_ids = mesh_device_ids(device)
    if device_ids:
        return len(device_ids)
    mesh_shape = _mesh_shape_tuple(device)
    if mesh_shape:
        return prod(mesh_shape)
    return 1


def is_multi_device_mesh(device) -> bool:
    return mesh_device_size(device) > 1


def replicate_tensor_to_mesh_mapper(ttnn, mesh_device):
    replicate_mapper = getattr(ttnn, "replicate_tensor_to_mesh_mapper", None)
    if callable(replicate_mapper):
        return replicate_mapper(mesh_device)

    replicate_tensor_to_mesh = getattr(ttnn, "ReplicateTensorToMesh", None)
    if callable(replicate_tensor_to_mesh):
        return replicate_tensor_to_mesh(mesh_device)

    create_mesh_mapper = getattr(ttnn, "create_mesh_mapper", None)
    mesh_mapper_config = getattr(ttnn, "MeshMapperConfig", None)
    placement_replicate = getattr(ttnn, "PlacementReplicate", None)
    mesh_shape_ctor = getattr(ttnn, "MeshShape", None)
    if (
        callable(create_mesh_mapper)
        and mesh_mapper_config is not None
        and callable(placement_replicate)
        and callable(mesh_shape_ctor)
    ):
        return create_mesh_mapper(
            mesh_device,
            mesh_mapper_config(
                [placement_replicate()],
                mesh_shape_ctor(1, mesh_device_size(mesh_device)),
            ),
        )
    raise TypeError("TT backend does not expose a replicate-to-mesh mapper helper.")


def shard_tensor_to_mesh_mapper(ttnn, mesh_device, *, dim: int):
    shard_mapper = getattr(ttnn, "shard_tensor_to_mesh_mapper", None)
    if callable(shard_mapper):
        return shard_mapper(mesh_device, dim)

    shard_tensor_to_mesh = getattr(ttnn, "ShardTensorToMesh", None)
    if callable(shard_tensor_to_mesh):
        return shard_tensor_to_mesh(mesh_device, dim=dim)

    create_mesh_mapper = getattr(ttnn, "create_mesh_mapper", None)
    mesh_mapper_config = getattr(ttnn, "MeshMapperConfig", None)
    placement_replicate = getattr(ttnn, "PlacementReplicate", None)
    placement_shard = getattr(ttnn, "PlacementShard", None)
    mesh_shape_ctor = getattr(ttnn, "MeshShape", None)
    if (
        callable(create_mesh_mapper)
        and mesh_mapper_config is not None
        and callable(placement_replicate)
        and callable(placement_shard)
        and callable(mesh_shape_ctor)
    ):
        return create_mesh_mapper(
            mesh_device,
            mesh_mapper_config(
                [placement_replicate(), placement_shard(dim)],
                mesh_shape_ctor(1, mesh_device_size(mesh_device)),
            ),
        )
    raise TypeError("TT backend does not expose a shard-to-mesh mapper helper.")


def concat_mesh_to_tensor_composer(ttnn, mesh_device, *, dim: int):
    concat_composer = getattr(ttnn, "concat_mesh_to_tensor_composer", None)
    if callable(concat_composer):
        return concat_composer(mesh_device, dim)

    concat_mesh_to_tensor = getattr(ttnn, "ConcatMeshToTensor", None)
    if callable(concat_mesh_to_tensor):
        return concat_mesh_to_tensor(mesh_device, dim=dim)

    create_mesh_composer = getattr(ttnn, "create_mesh_composer", None)
    mesh_composer_config = getattr(ttnn, "MeshComposerConfig", None)
    mesh_shape_ctor = getattr(ttnn, "MeshShape", None)
    if (
        callable(create_mesh_composer)
        and mesh_composer_config is not None
        and callable(mesh_shape_ctor)
    ):
        return create_mesh_composer(
            mesh_device,
            mesh_composer_config(
                [dim],
                mesh_shape_ctor(1, mesh_device_size(mesh_device)),
            ),
        )
    raise TypeError("TT backend does not expose a concat-from-mesh composer helper.")


def canonical_replica_to_torch(ttnn, mesh_device, value):
    if not is_multi_device_mesh(mesh_device):
        return ttnn.to_torch(value)

    mesh_composer = concat_mesh_to_tensor_composer(ttnn, mesh_device, dim=0)
    host_value = ttnn.to_torch(value, mesh_composer=mesh_composer)
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        return host_value

    replica_batch = int(shape[0])
    return host_value[:replica_batch].contiguous()


__all__ = [
    "canonical_replica_to_torch",
    "concat_mesh_to_tensor_composer",
    "is_multi_device_mesh",
    "mesh_device_ids",
    "mesh_device_size",
    "replicate_tensor_to_mesh_mapper",
    "shard_tensor_to_mesh_mapper",
]
