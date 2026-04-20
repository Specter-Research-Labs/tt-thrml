import tt_thrml


class _FakeOpenTTNN:
    class DispatchCoreType:
        ETH = "eth"
        WORKER = "worker"

    class DispatchCoreAxis:
        ROW = "row"
        COL = "col"

    class DispatchCoreConfig:
        def __init__(self, dispatch_core_type, dispatch_core_axis):
            self.dispatch_core_type = dispatch_core_type
            self.dispatch_core_axis = dispatch_core_axis

    class _TTNNNamespace:
        class device:
            DEFAULT_L1_SMALL_SIZE = 1
            DEFAULT_TRACE_REGION_SIZE = 2
            DEFAULT_WORKER_L1_SIZE = 3

    _ttnn = _TTNNNamespace()

    def __init__(self):
        self.open_device_calls = []
        self.create_devices_calls = []

    def open_device(self, *, device_id):
        self.open_device_calls.append(device_id)
        return f"open:{device_id}"

    def CreateDevices(
        self,
        *,
        device_ids,
        l1_small_size,
        trace_region_size,
        dispatch_core_config,
        worker_l1_size,
    ):
        self.create_devices_calls.append(
            {
                "device_ids": list(device_ids),
                "l1_small_size": l1_small_size,
                "trace_region_size": trace_region_size,
                "dispatch_core_config": dispatch_core_config,
                "worker_l1_size": worker_l1_size,
            }
        )
        return {device_id: f"mesh:{device_id}" for device_id in device_ids}


class _FakeIndependentOpenTTNN:
    DispatchCoreType = _FakeOpenTTNN.DispatchCoreType
    DispatchCoreAxis = _FakeOpenTTNN.DispatchCoreAxis
    DispatchCoreConfig = _FakeOpenTTNN.DispatchCoreConfig
    _ttnn = _FakeOpenTTNN._TTNNNamespace()

    def __init__(self):
        self.open_device_calls = []

    def open_device(self, *, device_id):
        self.open_device_calls.append(device_id)
        return f"open:{device_id}"


class _FakeMeshOpenTTNN:
    DispatchCoreType = _FakeOpenTTNN.DispatchCoreType
    DispatchCoreAxis = _FakeOpenTTNN.DispatchCoreAxis
    DispatchCoreConfig = _FakeOpenTTNN.DispatchCoreConfig

    class MeshShape:
        def __init__(self, *dims):
            self.dims = tuple(dims)

    class _TTNNNamespace:
        class device:
            DEFAULT_L1_SMALL_SIZE = 1
            DEFAULT_TRACE_REGION_SIZE = 2
            DEFAULT_WORKER_L1_SIZE = 3

    _ttnn = _TTNNNamespace()

    def __init__(self):
        self.open_mesh_calls = []
        self.close_mesh_calls = []

    def open_mesh_device(self, **kwargs):
        self.open_mesh_calls.append(kwargs)
        return {
            "mesh_shape": tuple(kwargs["mesh_shape"].dims),
            "physical_device_ids": tuple(kwargs.get("physical_device_ids", ())),
        }

    def close_mesh_device(self, mesh_device):
        self.close_mesh_calls.append(mesh_device)


class _FakeLegacyOpenTTNN:
    class _TTNNNamespace:
        class device:
            DEFAULT_L1_SMALL_SIZE = 1
            DEFAULT_TRACE_REGION_SIZE = 2
            DEFAULT_WORKER_L1_SIZE = 3

    _ttnn = _TTNNNamespace()

    def __init__(self):
        self.open_device_calls = []

    def open_device(self, *, device_id):
        self.open_device_calls.append(device_id)
        return f"legacy-open:{device_id}"


def test_open_devices_defaults_to_create_devices_for_multi_open():
    fake_ttnn = _FakeOpenTTNN()

    devices = tt_thrml.open_devices(fake_ttnn, device_ids=(0, 1, 2, 3))

    assert devices == ("mesh:0", "mesh:1", "mesh:2", "mesh:3")
    assert fake_ttnn.open_device_calls == []
    assert len(fake_ttnn.create_devices_calls) == 1


def test_open_devices_forwards_explicit_dispatch_config():
    fake_ttnn = _FakeOpenTTNN()

    devices = tt_thrml.open_devices(
        fake_ttnn,
        device_ids=(0, 1, 2, 3),
        dispatch_core_type="WORKER",
        dispatch_core_axis="COL",
    )

    assert devices == ("mesh:0", "mesh:1", "mesh:2", "mesh:3")
    assert fake_ttnn.open_device_calls == []
    assert len(fake_ttnn.create_devices_calls) == 1
    dispatch_config = fake_ttnn.create_devices_calls[0]["dispatch_core_config"]
    assert dispatch_config.dispatch_core_type == "worker"
    assert dispatch_config.dispatch_core_axis == "col"


def test_open_devices_falls_back_to_independent_opens_when_create_devices_is_missing():
    fake_ttnn = _FakeIndependentOpenTTNN()

    devices = tt_thrml.open_devices(fake_ttnn, device_ids=(0, 1, 2, 3))

    assert devices == ("open:0", "open:1", "open:2", "open:3")
    assert fake_ttnn.open_device_calls == [0, 1, 2, 3]


def test_open_device_handles_legacy_ttnn_without_dispatch_core_types():
    fake_ttnn = _FakeLegacyOpenTTNN()

    device = tt_thrml.open_device(fake_ttnn, device_id=7)

    assert device == "legacy-open:7"
    assert fake_ttnn.open_device_calls == [7]


def test_open_mesh_device_forwards_mesh_shape_device_ids_and_dispatch_config():
    fake_ttnn = _FakeMeshOpenTTNN()

    mesh_device = tt_thrml.open_mesh_device(
        fake_ttnn,
        mesh_shape=(1, 4),
        device_ids=(0, 1, 2, 3),
        dispatch_core_type="WORKER",
        dispatch_core_axis="COL",
        num_command_queues=2,
    )

    assert mesh_device == {
        "mesh_shape": (1, 4),
        "physical_device_ids": (0, 1, 2, 3),
    }
    assert len(fake_ttnn.open_mesh_calls) == 1
    call = fake_ttnn.open_mesh_calls[0]
    assert tuple(call["mesh_shape"].dims) == (1, 4)
    assert call["physical_device_ids"] == [0, 1, 2, 3]
    assert call["num_command_queues"] == 2
    assert call["dispatch_core_config"].dispatch_core_type == "worker"
    assert call["dispatch_core_config"].dispatch_core_axis == "col"


def test_close_devices_accepts_single_mesh_device_object():
    fake_ttnn = _FakeMeshOpenTTNN()
    mesh_device = {"mesh": (1, 4)}

    tt_thrml.close_mesh_device(fake_ttnn, mesh_device)
    tt_thrml.close_devices(fake_ttnn, mesh_device)

    assert fake_ttnn.close_mesh_calls == [mesh_device]
