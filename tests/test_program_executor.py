from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import jax
import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.float32 = "float32"
    torch_stub.int64 = "int64"
    torch_stub.from_numpy = lambda value: value
    torch_stub.zeros = lambda *args, **kwargs: ("zeros", args, kwargs)
    torch_stub.as_tensor = lambda value: value
    sys.modules["torch"] = torch_stub

from tt_thrml.runtime.family_handlers import ParameterFamilyHandler, PARAMETER_FAMILY_HANDLERS
from tt_thrml.runtime.program_executor import TTProgramExecutor
from tt_thrml.runtime_config import (
    CATEGORICAL_PARAMETER_FAMILY,
    SPIN_PARAMETER_FAMILY,
)


def _fake_executor() -> TTProgramExecutor:
    executor = object.__new__(TTProgramExecutor)
    executor.program = SimpleNamespace(
        gibbs_spec=SimpleNamespace(
            free_blocks=[object(), object()],
            sampling_order=((0, 1),),
        )
    )
    executor.compiled = SimpleNamespace(
        blocks=(
            SimpleNamespace(
                block_index=0,
                sampler_lowering=SimpleNamespace(
                    parameter_family=SPIN_PARAMETER_FAMILY,
                    sampler_state_spec=None,
                    parameters_depend_on_sampler_state=False,
                ),
            ),
            SimpleNamespace(
                block_index=1,
                sampler_lowering=SimpleNamespace(
                    parameter_family=CATEGORICAL_PARAMETER_FAMILY,
                    sampler_state_spec=None,
                    parameters_depend_on_sampler_state=False,
                ),
            ),
        ),
    )
    executor._require_state = lambda: None
    executor._profile_call = lambda _stage, fn: fn()
    executor._write_block_state = lambda block_index, new_state: (block_index, new_state)
    executor._after_sampling_group = lambda group_index, sampling_group: None
    executor._sample_block = lambda key, block_index, prepared_random=None: (
        block_index,
        key,
        prepared_random,
    )
    executor._sample_single_block_from_sample_key = lambda sample_key, block_index, sampler_state, prepared_random=None: (
        ("sample", block_index, sample_key, prepared_random),
        sampler_state,
    )
    executor._block_state_slots = [SimpleNamespace(shape=(1, 1, 1, 1))] * 2
    return executor


def test_run_sweep_batch_supports_mixed_stateless_family_programs(monkeypatch):
    executor = _fake_executor()
    sample_calls = []
    executor._sample_block_batch = lambda sample_keys, block_index: sample_calls.append(
        (tuple(sample_keys), block_index)
    ) or ("batch-sample", block_index)

    original = dict(PARAMETER_FAMILY_HANDLERS)
    try:
        PARAMETER_FAMILY_HANDLERS[SPIN_PARAMETER_FAMILY] = ParameterFamilyHandler(
            **{
                **original[SPIN_PARAMETER_FAMILY].__dict__,
                "supports_batch_sampling": True,
            }
        )
        PARAMETER_FAMILY_HANDLERS[CATEGORICAL_PARAMETER_FAMILY] = ParameterFamilyHandler(
            **{
                **original[CATEGORICAL_PARAMETER_FAMILY].__dict__,
                "supports_batch_sampling": True,
            }
        )
        executor.run_sweep_batch((jax.random.PRNGKey(0), jax.random.PRNGKey(1)))
    finally:
        PARAMETER_FAMILY_HANDLERS.clear()
        PARAMETER_FAMILY_HANDLERS.update(original)

    assert len(sample_calls) == 2


def test_run_sweep_batch_rejects_programs_without_batched_family_support():
    executor = _fake_executor()
    with pytest.raises(TypeError, match="batched sampling support"):
        executor.run_sweep_batch((jax.random.PRNGKey(0), jax.random.PRNGKey(1)))


def test_run_blocks_prepares_bulk_randoms_once_per_block(monkeypatch):
    executor = _fake_executor()
    prepare_calls = []
    run_calls = []

    original = dict(PARAMETER_FAMILY_HANDLERS)
    try:
        for family, handler in original.items():
            PARAMETER_FAMILY_HANDLERS[family] = ParameterFamilyHandler(
                **{
                    **handler.__dict__,
                    "prepare_batch_sample_inputs": lambda _executor, block, sample_keys, _family=family: (
                        prepare_calls.append((block.block_index, len(sample_keys), _family)),
                        f"prepared-{block.block_index}",
                    )[1],
                    "select_prepared_random": lambda _executor, _block, prepared_random, iter_index: (
                        prepared_random,
                        iter_index,
                    ),
                }
            )
        executor.run_sweep = lambda key, *, sample_keys=None, prepared_randoms_by_block=None: run_calls.append(
            (key, sample_keys, prepared_randoms_by_block)
        )
        executor.run_blocks(jax.random.PRNGKey(0), n_iters=3)
    finally:
        PARAMETER_FAMILY_HANDLERS.clear()
        PARAMETER_FAMILY_HANDLERS.update(original)

    assert prepare_calls == [
        (0, 3, SPIN_PARAMETER_FAMILY),
        (1, 3, CATEGORICAL_PARAMETER_FAMILY),
    ]
    assert len(run_calls) == 3
    assert run_calls[0][2][0] == ("prepared-0", 0)
    assert run_calls[1][2][1] == ("prepared-1", 1)


def test_run_sweep_prepares_single_iteration_randoms_up_front():
    executor = _fake_executor()
    prepare_calls = []
    sample_calls = []

    original = dict(PARAMETER_FAMILY_HANDLERS)
    try:
        for family, handler in original.items():
            PARAMETER_FAMILY_HANDLERS[family] = ParameterFamilyHandler(
                **{
                    **handler.__dict__,
                    "prepare_batch_sample_inputs": lambda _executor, block, sample_keys, _family=family: (
                        prepare_calls.append((block.block_index, len(sample_keys), _family)),
                        f"prepared-{block.block_index}",
                    )[1],
                    "select_prepared_random": lambda _executor, _block, prepared_random, iter_index: (
                        prepared_random,
                        iter_index,
                    ),
                }
            )
        executor._sample_single_block_from_sample_key = lambda sample_key, block_index, sampler_state, prepared_random=None: (
            sample_calls.append((block_index, sample_key, prepared_random)),
            sampler_state,
        )
        executor.run_sweep(jax.random.PRNGKey(0))
    finally:
        PARAMETER_FAMILY_HANDLERS.clear()
        PARAMETER_FAMILY_HANDLERS.update(original)

    assert prepare_calls == [
        (0, 1, SPIN_PARAMETER_FAMILY),
        (1, 1, CATEGORICAL_PARAMETER_FAMILY),
    ]
    assert sample_calls[0][2] == ("prepared-0", 0)
    assert sample_calls[1][2] == ("prepared-1", 0)
