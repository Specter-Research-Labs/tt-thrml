#!/usr/bin/env python3
"""Run the first THRML-derived TT-Lang categorical/spin plan.

This is the inverse of `run_ttlang_spin_categorical_plan.py`: it lowers the
mixed parity program's first categorical block into a TT-Lang spin-source
categorical plan, then executes the one-hot categorical decision on Wormhole.

The runner is deliberately narrow while the executor contract is still being
proved: one categorical target, one spin source interaction, three categories,
and no score ties.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tt_thrml.example_programs import make_mixed_spin_categorical_gaussian_program
from tt_thrml.ttlang_backend import ExperimentalTTLangExecutor, TTLangCategoricalSpinRun

TILE = 32
N_CATEGORIES = 3


def _require_ttlang():
    import ttl  # type: ignore[reportMissingImports]
    import ttnn  # type: ignore[reportMissingImports]

    return ttl, ttnn


ttl, ttnn = _require_ttlang()


def from_torch(tensor: torch.Tensor, *, device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@ttl.operation(grid=(1, 1))
def thrml_categorical_spin_plan_tile(spin_state, weights, bias, gumbel, one, half, out):
    spin_dfb = ttl.make_dataflow_buffer_like(spin_state, shape=(1, 1), block_count=2)
    weights_dfb = ttl.make_dataflow_buffer_like(weights, shape=(1, 1), block_count=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), block_count=2)
    gumbel_dfb = ttl.make_dataflow_buffer_like(gumbel, shape=(1, 1), block_count=2)
    score0_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    score1_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    score2_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    one_dfb = ttl.make_dataflow_buffer_like(one, shape=(1, 1), block_count=2)
    half_dfb = ttl.make_dataflow_buffer_like(half, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with spin_dfb.wait() as spin_blk:
            with (
                weights_dfb.wait() as weights_blk,
                bias_dfb.wait() as bias_blk,
                gumbel_dfb.wait() as gumbel_blk,
                score0_dfb.reserve() as score0_blk,
            ):
                score0_blk.store(bias_blk + gumbel_blk + spin_blk * weights_blk)
            with (
                weights_dfb.wait() as weights_blk,
                bias_dfb.wait() as bias_blk,
                gumbel_dfb.wait() as gumbel_blk,
                score1_dfb.reserve() as score1_blk,
            ):
                score1_blk.store(bias_blk + gumbel_blk + spin_blk * weights_blk)
            with (
                weights_dfb.wait() as weights_blk,
                bias_dfb.wait() as bias_blk,
                gumbel_dfb.wait() as gumbel_blk,
                score2_dfb.reserve() as score2_blk,
            ):
                score2_blk.store(bias_blk + gumbel_blk + spin_blk * weights_blk)

        with (
            score0_dfb.wait() as score0,
            score1_dfb.wait() as score1,
            score2_dfb.wait() as score2,
            one_dfb.wait() as one_blk,
            half_dfb.wait() as half_blk,
        ):
            gt01 = (ttl.math.sign(score0 - score1) + one_blk) * half_blk
            gt02 = (ttl.math.sign(score0 - score2) + one_blk) * half_blk
            with out_dfb.reserve() as out_blk:
                out_blk.store(gt01 * gt02)

            gt10 = (ttl.math.sign(score1 - score0) + one_blk) * half_blk
            gt12 = (ttl.math.sign(score1 - score2) + one_blk) * half_blk
            with out_dfb.reserve() as out_blk:
                out_blk.store(gt10 * gt12)

            gt20 = (ttl.math.sign(score2 - score0) + one_blk) * half_blk
            gt21 = (ttl.math.sign(score2 - score1) + one_blk) * half_blk
            with out_dfb.reserve() as out_blk:
                out_blk.store(gt20 * gt21)

    @ttl.datamovement()
    def read():
        with spin_dfb.reserve() as spin_blk:
            tx_spin = ttl.copy(spin_state[0, 0], spin_blk)
            tx_spin.wait()
        for category in range(3):
            with weights_dfb.reserve() as weights_blk:
                tx_weights = ttl.copy(weights[category, 0], weights_blk)
                tx_weights.wait()
            with bias_dfb.reserve() as bias_blk:
                tx_bias = ttl.copy(bias[category, 0], bias_blk)
                tx_bias.wait()
            with gumbel_dfb.reserve() as gumbel_blk:
                tx_gumbel = ttl.copy(gumbel[category, 0], gumbel_blk)
                tx_gumbel.wait()
        with one_dfb.reserve() as one_blk:
            tx_one = ttl.copy(one[0, 0], one_blk)
            tx_one.wait()
        with half_dfb.reserve() as half_blk:
            tx_half = ttl.copy(half[0, 0], half_blk)
            tx_half.wait()

    @ttl.datamovement()
    def write():
        for category in range(3):
            with out_dfb.wait() as out_blk:
                tx_out = ttl.copy(out_blk, out[category, 0])
                tx_out.wait()


def _tile_planes(values: list[float]) -> torch.Tensor:
    planes = np.asarray(values, dtype=np.float32)[:, None, None] * np.ones((len(values), TILE, TILE), dtype=np.float32)
    return torch.from_numpy(planes.reshape(len(values) * TILE, TILE)).to(torch.bfloat16)


def _case_state(spin: bool) -> list[np.ndarray]:
    return [
        np.asarray([spin]),
        np.asarray([0], dtype=np.uint8),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
        np.asarray([1], dtype=np.uint8),
        np.asarray([0.0], dtype=np.float32),
    ]


def _run_case(device, run: TTLangCategoricalSpinRun) -> None:
    plan = run.plan
    if len(plan.spin_lanes) != 1 or len(plan.spin_weights[0]) != N_CATEGORIES:
        raise RuntimeError(f"unsupported plan shape: {plan}")

    spin_value = float(run.spin_values[0])
    weight_values = [float(weight) for weight in plan.spin_weights[0]]
    expected = _tile_planes(list(run.expected_one_hot))

    spin_state = from_torch(torch.full((TILE, TILE), spin_value, dtype=torch.bfloat16), device=device)
    weights = from_torch(_tile_planes(weight_values), device=device)
    bias = from_torch(_tile_planes(list(plan.bias)), device=device)
    gumbel = from_torch(_tile_planes(list(run.gumbel)), device=device)
    one = from_torch(torch.ones((TILE, TILE), dtype=torch.bfloat16), device=device)
    half = from_torch(torch.full((TILE, TILE), 0.5, dtype=torch.bfloat16), device=device)
    out = from_torch(torch.zeros((N_CATEGORIES * TILE, TILE), dtype=torch.bfloat16), device=device)

    thrml_categorical_spin_plan_tile(spin_state, weights, bias, gumbel, one, half, out)
    result = ttnn.to_torch(out).to(torch.bfloat16)

    mismatches = result != expected
    print(
        "case",
        {
            "spin_value": spin_value,
            "weights": weight_values,
            "bias": plan.bias,
            "gumbel": run.gumbel,
            "expected_category": run.expected_category,
        },
    )
    print("result", result)
    print("expected", expected)
    if mismatches.any():
        first = mismatches.nonzero()[0].tolist()
        print("mismatches", int(mismatches.sum().item()))
        print(
            "first mismatch",
            first,
            "result",
            result[first[0], first[1]],
            "expected",
            expected[first[0], first[1]],
        )
        raise AssertionError("TT-Lang categorical spin plan mismatch")


def main() -> None:
    executor = ExperimentalTTLangExecutor(make_mixed_spin_categorical_gaussian_program())

    device = ttnn.open_device(device_id=0)
    try:
        for spin, gumbel in ((True, (0.0, 0.0, 0.0)), (False, (-0.5, 0.0, 0.25))):
            run = executor.materialize_categorical_spin_run(
                0,
                _case_state(spin),
                gumbel=gumbel,
            )
            _run_case(device, run)

        print("PASS: TT-Lang THRML categorical spin plan")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
