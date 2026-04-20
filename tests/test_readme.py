class _FakeTTNN:
    pass


def test_readme_quick_example_uses_explicit_backend_binding(monkeypatch):
    import tt_thrml

    captured = {}

    def fake_sample_states(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        nodes_to_sample,
        *,
        backend=None,
        options=None,
    ):
        captured.update(
            key=key,
            program=program,
            schedule=schedule,
            init_state_free=init_state_free,
            state_clamp=state_clamp,
            nodes_to_sample=nodes_to_sample,
            backend=backend,
            options=options,
        )
        return ["fake-samples"]

    monkeypatch.setitem(tt_thrml.__dict__, "sample_states", fake_sample_states)

    ttnn = _FakeTTNN()
    backend = tt_thrml.make_backend_binding(ttnn, "fake:0")
    key = object()
    program = object()
    schedule = object()
    init_state_free = object()
    state_clamp = [object()]
    nodes_to_sample = [object()]

    samples = tt_thrml.sample_states(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        nodes_to_sample,
        backend=backend,
    )

    assert isinstance(backend, tt_thrml.BackendBinding)
    assert backend.ttnn is ttnn
    assert backend.devices == ("fake:0",)
    assert samples == ["fake-samples"]
    assert captured == {
        "key": key,
        "program": program,
        "schedule": schedule,
        "init_state_free": init_state_free,
        "state_clamp": state_clamp,
        "nodes_to_sample": nodes_to_sample,
        "backend": backend,
        "options": None,
    }
    assert not hasattr(tt_thrml, "configure_default_backend")
