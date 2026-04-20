from __future__ import annotations

import sys
import types

import numpy as np
import pytest

class _FakeTorchArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def reshape(self, *shape):
        target_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return _FakeTorchArray(self.value.reshape(target_shape))

    def repeat(self, *sizes):
        repeat_sizes = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], tuple) else sizes
        return _FakeTorchArray(np.tile(self.value, repeat_sizes))

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    @property
    def shape(self):
        return self.value.shape

    def tolist(self):
        return self.value.tolist()


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.float32 = "float32"
    torch_stub.int64 = "int64"
    torch_stub.arange = lambda *args, dtype=None, **kwargs: _FakeTorchArray(
        np.arange(*args, dtype=np.int64)
    )
    torch_stub.from_numpy = lambda value: _FakeTorchArray(value)
    torch_stub.full = lambda shape, *, fill_value, dtype=None, **kwargs: _FakeTorchArray(
        np.full(shape, fill_value=fill_value, dtype=np.float32)
    )
    torch_stub.ones = lambda shape, *, dtype=None, **kwargs: _FakeTorchArray(
        np.ones(shape, dtype=np.float32)
    )
    torch_stub.zeros = lambda *args, **kwargs: ("zeros", args, kwargs)
    torch_stub.as_tensor = lambda value: _FakeTorchArray(value)
    sys.modules["torch"] = torch_stub


import tt_thrml.runtime.backend_executor as backend_executor
import tt_thrml.runtime.state_runtime as state_runtime
from tt_thrml.runtime.mesh_executor import (
    build_round_robin_mesh_sweep_plan,
    is_multi_device_mesh,
)
from tt_thrml.runtime.mesh_support import canonical_replica_to_torch
from tt_thrml.runtime_config import make_backend_binding


class _FakeProgram:
    pass


class _FakeMeshDevice:
    shape = (1, 4)

    def get_device_ids(self):
        return (0, 1, 2, 3)


class _FakeUnitDevice:
    shape = (1, 1)

    def get_device_ids(self):
        return (0,)


class _FakeMeshMapper:
    def __init__(self, kind, mesh_device, *, dim=None):
        self.kind = kind
        self.mesh_device = mesh_device
        self.dim = dim


class _FakeMeshComposer:
    def __init__(self, mesh_device, *, dim):
        self.mesh_device = mesh_device
        self.dim = dim


class _FakeHostTensor:
    def __init__(self, rows):
        self.rows = [tuple(row) for row in rows]

    @property
    def shape(self):
        if not self.rows:
            return (0, 0)
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _FakeHostTensor(self.rows[item])
        raise TypeError(f"Unsupported index {item!r}")

    def contiguous(self):
        return self


class _FakeMeshTensor:
    def __init__(self, shards, *, shape):
        self.shards = tuple(tuple(tuple(row) for row in shard) for shard in shards)
        self.shape = shape


class _MeshAwareFakeTTNN:
    def __init__(self):
        self.last_from_torch_kwargs = None
        self.last_to_torch_kwargs = None

    def replicate_tensor_to_mesh_mapper(self, mesh_device):
        return _FakeMeshMapper("replicate", mesh_device)

    def concat_mesh_to_tensor_composer(self, mesh_device, dim):
        return _FakeMeshComposer(mesh_device, dim=dim)

    def from_torch(self, value, **kwargs):
        self.last_from_torch_kwargs = kwargs
        mesh_mapper = kwargs.get("mesh_mapper")
        if mesh_mapper is None:
            return value
        mesh_size = len(mesh_mapper.mesh_device.get_device_ids())
        return _FakeMeshTensor([value] * mesh_size, shape=(1, 1, 1, 4))

    def to_torch(self, value, **kwargs):
        self.last_to_torch_kwargs = kwargs
        mesh_composer = kwargs.get("mesh_composer")
        if mesh_composer is None:
            return value
        rows = []
        for shard in value.shards:
            rows.extend(shard)
        return _FakeHostTensor(rows)


def test_is_multi_device_mesh_detects_mesh_devices():
    assert is_multi_device_mesh(_FakeMeshDevice())
    assert not is_multi_device_mesh(_FakeUnitDevice())
    assert not is_multi_device_mesh(object())


def test_build_round_robin_mesh_sweep_plan_partitions_blocks_per_owner():
    plan = build_round_robin_mesh_sweep_plan(
        n_blocks=6,
        n_owners=3,
        sampling_order=((0, 1, 2), (3, 4, 5)),
    )

    assert [group.block_indices for group in plan] == [
        (0, 1, 2),
        (3, 4, 5),
    ]
    assert [group.block_indices_by_owner for group in plan] == [
        ((0,), (1,), (2,)),
        ((3,), (4,), (5,)),
    ]


def test_canonical_replica_to_torch_returns_one_replica_for_replicated_mesh_tensor():
    ttnn = _MeshAwareFakeTTNN()
    mesh_tensor = _FakeMeshTensor(
        shards=(
            ((1, 2, 3), (4, 5, 6)),
            ((1, 2, 3), (4, 5, 6)),
            ((1, 2, 3), (4, 5, 6)),
            ((1, 2, 3), (4, 5, 6)),
        ),
        shape=(2, 3),
    )

    host_tensor = canonical_replica_to_torch(ttnn, _FakeMeshDevice(), mesh_tensor)

    assert host_tensor.rows == [(1, 2, 3), (4, 5, 6)]
    assert ttnn.last_to_torch_kwargs["mesh_composer"].dim == 0


def test_coerce_rank4_ttnn_tensor_replicates_host_uploads_to_mesh():
    ttnn = _MeshAwareFakeTTNN()
    executor = types.SimpleNamespace(
        ttnn=ttnn,
        device=_FakeMeshDevice(),
    )

    state_runtime.coerce_rank4_ttnn_tensor(
        executor,
        object(),
        target_shape=(1, 1, 1, 4),
        target_dtype="bf16",
        layout="tile",
        host_tensor_fn=lambda: "host-state",
    )

    mesh_mapper = ttnn.last_from_torch_kwargs["mesh_mapper"]
    assert mesh_mapper.kind == "replicate"
    assert mesh_mapper.mesh_device.get_device_ids() == (0, 1, 2, 3)


def test_borrow_executor_uses_mesh_executor_for_multi_device_mesh(monkeypatch):
    backend_executor.clear_compiled_program_cache()
    monkeypatch.setattr(backend_executor, "program_supported_by_executor", lambda program: True)
    init_calls = []

    class RecordingExecutor:
        def __init__(self, **kwargs):
            init_calls.append(("unit", kwargs["device"]))
            self.compiled = object()

    class RecordingMeshExecutor:
        def __init__(self, **kwargs):
            init_calls.append(("mesh", kwargs["device"]))
            self.compiled = object()

    monkeypatch.setattr(backend_executor, "TTProgramExecutor", RecordingExecutor)
    monkeypatch.setattr(backend_executor, "TTMeshProgramExecutor", RecordingMeshExecutor)

    try:
        with backend_executor.borrow_executor(
            program=_FakeProgram(),
            backend=make_backend_binding(object(), _FakeUnitDevice()),
        ):
            pass
        with backend_executor.borrow_executor(
            program=_FakeProgram(),
            backend=make_backend_binding(object(), _FakeMeshDevice()),
        ):
            pass
    finally:
        backend_executor.clear_compiled_program_cache()

    assert init_calls[0][0] == "unit"
    assert isinstance(init_calls[0][1], _FakeUnitDevice)
    assert init_calls[1][0] == "mesh"
    assert isinstance(init_calls[1][1], _FakeMeshDevice)
