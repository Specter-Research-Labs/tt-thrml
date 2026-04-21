import tt_thrml


class _FakeTTNN:
    class MeshShape:
        def __init__(self, *dims):
            self.dims = dims

    def __init__(self):
        self.open_device_calls = []
        self.close_device_calls = []
        self.open_mesh_calls = []
        self.close_mesh_calls = []

    def open_device(self, *, device_id):
        self.open_device_calls.append(device_id)
        return f"device:{device_id}"

    def close_device(self, device):
        self.close_device_calls.append(device)

    def open_mesh_device(self, **kwargs):
        self.open_mesh_calls.append(kwargs)
        return f"mesh:{kwargs['mesh_shape'].dims}"

    def close_mesh_device(self, device):
        self.close_mesh_calls.append(device)


def test_open_device_passes_device_id():
    ttnn = _FakeTTNN()
    device = tt_thrml.open_device(ttnn, device_id=3)
    assert device == "device:3"
    assert ttnn.open_device_calls == [3]


def test_open_device_defaults_to_zero():
    ttnn = _FakeTTNN()
    tt_thrml.open_device(ttnn)
    assert ttnn.open_device_calls == [0]


def test_open_devices_opens_each_id():
    ttnn = _FakeTTNN()
    devices = tt_thrml.open_devices(ttnn, device_ids=(0, 1, 2))
    assert devices == ("device:0", "device:1", "device:2")
    assert ttnn.open_device_calls == [0, 1, 2]


def test_open_devices_rejects_empty_ids():
    ttnn = _FakeTTNN()
    try:
        tt_thrml.open_devices(ttnn, device_ids=())
        assert False, "should have raised"
    except ValueError:
        pass


def test_open_mesh_device_constructs_mesh_shape_and_forwards_args():
    ttnn = _FakeTTNN()
    tt_thrml.open_mesh_device(
        ttnn,
        mesh_shape=(2, 4),
        device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        num_command_queues=2,
    )
    assert len(ttnn.open_mesh_calls) == 1
    call = ttnn.open_mesh_calls[0]
    assert call["mesh_shape"].dims == (2, 4)
    assert call["physical_device_ids"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert call["num_command_queues"] == 2


def test_open_mesh_device_without_explicit_ids():
    ttnn = _FakeTTNN()
    tt_thrml.open_mesh_device(ttnn, mesh_shape=(1, 4))
    call = ttnn.open_mesh_calls[0]
    assert "physical_device_ids" not in call
    assert call["mesh_shape"].dims == (1, 4)


def test_close_device_delegates_to_ttnn():
    ttnn = _FakeTTNN()
    tt_thrml.close_device(ttnn, "dev-handle")
    assert ttnn.close_device_calls == ["dev-handle"]


def test_close_mesh_device_delegates_to_ttnn():
    ttnn = _FakeTTNN()
    tt_thrml.close_mesh_device(ttnn, "mesh-handle")
    assert ttnn.close_mesh_calls == ["mesh-handle"]


def test_close_devices_calls_close_device_for_each():
    ttnn = _FakeTTNN()
    tt_thrml.close_devices(ttnn, ["dev-a", "dev-b", "dev-c"])
    assert ttnn.close_device_calls == ["dev-a", "dev-b", "dev-c"]
