from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from thrml.block_sampling import SamplingSchedule

from tt_thrml.executor import Executor


class _FakeObserver:
    def init(self):
        return "observer-init"

    def __call__(self, program, state_free, state_clamped, carry, sample_index):
        return f"{carry}:{int(sample_index)}", jnp.asarray(int(sample_index))


class _ExecutorShell:
    program = object()

    def load_state(self, state_free, state_clamp):
        self.loaded = (state_free, state_clamp)

    def _prepare_rng_thrml_compat(self, key, schedule, n_total):
        self.rng = (key, schedule, n_total)

    def run_warmup(self, n_warmup):
        self.warmup = n_warmup

    def _read_state_lists(self):
        return [], []


def test_sample_with_observation_uses_supplied_initial_carry():
    executor = cast(Executor, _ExecutorShell())

    carry, observations = Executor.sample_with_observation(
        executor,
        key=object(),
        schedule=SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=1),
        f_observe=_FakeObserver(),
        init_state_free=[],
        state_clamp=[],
        observation_carry_init="caller-init",
    )

    assert carry == "caller-init:0"
    assert observations is not None
    assert observations.tolist() == [0]
