"""Run an existing upstream THRML program on Tenstorrent hardware."""

import jax
import thrml
import ttnn

import tt_thrml


def run_on_tt(*, program, init_state_free, state_clamp, nodes_to_sample):
    schedule = thrml.SamplingSchedule(n_warmup=32, n_samples=64, steps_per_sample=2)
    key = jax.random.key(0)
    device = tt_thrml.open_device(ttnn, device_id=0)
    backend = tt_thrml.make_backend_binding(ttnn, device)
    try:
        return tt_thrml.sample_states(
            key,
            program,
            schedule,
            init_state_free,
            state_clamp,
            [thrml.Block(nodes_to_sample)],
            backend=backend,
        )
    finally:
        tt_thrml.close_devices(ttnn, (device,))
