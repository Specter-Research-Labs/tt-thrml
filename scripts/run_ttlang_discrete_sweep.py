#!/usr/bin/env python3
"""Run one narrow TT-Lang discrete sweep over device-resident THRML state."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tt_thrml.example_programs import make_mixed_spin_categorical_gaussian_program
from tt_thrml.ttlang_backend import ExperimentalTTLangExecutor

TILE = 32
N_LANES = 10
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
def copy_state_10(inp, out):
    lane_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=10)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def read():
        for lane in range(10):
            with lane_dfb.reserve() as lane_blk:
                tx_in = ttl.copy(inp[lane, 0], lane_blk)
                tx_in.wait()

    @ttl.datamovement()
    def write():
        for lane in range(10):
            with lane_dfb.wait() as lane_blk:
                tx_out = ttl.copy(lane_blk, out[lane, 0])
                tx_out.wait()


def _define_spin_update(source_start: int, output_lane: int):
    @ttl.operation(grid=(1, 1))
    def spin_update(state, weights, bias, threshold_logits, half, out):
        cat_state_dfb = ttl.make_dataflow_buffer_like(state, shape=(1, 1), block_count=2)
        weights_dfb = ttl.make_dataflow_buffer_like(weights, shape=(1, 1), block_count=2)
        bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), block_count=2)
        threshold_dfb = ttl.make_dataflow_buffer_like(threshold_logits, shape=(1, 1), block_count=2)
        half_dfb = ttl.make_dataflow_buffer_like(half, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as out_blk:
                with bias_dfb.wait() as bias_blk:
                    out_blk.store(bias_blk)
                for _ in range(3):
                    with cat_state_dfb.wait() as cat_state_blk, weights_dfb.wait() as weights_blk:
                        out_blk += cat_state_blk * weights_blk
                with threshold_dfb.wait() as threshold_blk, half_dfb.wait() as half_blk:
                    decision = ttl.math.sign(out_blk + out_blk - threshold_blk)
                    out_blk.store(ttl.math.sign(decision - half_blk))

        @ttl.datamovement()
        def read():
            with bias_dfb.reserve() as bias_blk:
                tx_bias = ttl.copy(bias[0, 0], bias_blk)
                tx_bias.wait()
            for category in range(3):
                with cat_state_dfb.reserve() as cat_state_blk, weights_dfb.reserve() as weights_blk:
                    tx_state = ttl.copy(state[source_start + category, 0], cat_state_blk)
                    tx_weights = ttl.copy(weights[category, 0], weights_blk)
                    tx_state.wait()
                    tx_weights.wait()
            with threshold_dfb.reserve() as threshold_blk:
                tx_threshold = ttl.copy(threshold_logits[0, 0], threshold_blk)
                tx_threshold.wait()
            with half_dfb.reserve() as half_blk:
                tx_half = ttl.copy(half[0, 0], half_blk)
                tx_half.wait()

        @ttl.datamovement()
        def write():
            with out_dfb.wait() as out_blk:
                tx_out = ttl.copy(out_blk, out[output_lane, 0])
                tx_out.wait()

    return spin_update


def _define_categorical_update(source_lane: int, output_start: int):
    @ttl.operation(grid=(1, 1))
    def categorical_update(state, weights, bias, gumbel, one, half, out):
        spin_dfb = ttl.make_dataflow_buffer_like(state, shape=(1, 1), block_count=2)
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
                tx_spin = ttl.copy(state[source_lane, 0], spin_blk)
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
                    tx_out = ttl.copy(out_blk, out[output_start + category, 0])
                    tx_out.wait()

    return categorical_update


spin_group0 = _define_spin_update(source_start=1, output_lane=0)
categorical_group0 = _define_categorical_update(source_lane=0, output_start=1)
spin_group1 = _define_spin_update(source_start=6, output_lane=5)
categorical_group1 = _define_categorical_update(source_lane=5, output_start=6)


def _tile_planes(values: list[float]) -> torch.Tensor:
    planes = np.asarray(values, dtype=np.float32)[:, None, None] * np.ones((len(values), TILE, TILE), dtype=np.float32)
    return torch.from_numpy(planes.reshape(len(values) * TILE, TILE)).to(torch.bfloat16)


def _state_tiles(lanes: np.ndarray) -> torch.Tensor:
    return _tile_planes([float(value) for value in lanes])


def _initial_state() -> list[np.ndarray]:
    return [
        np.asarray([True]),
        np.asarray([1], dtype=np.uint8),
        np.asarray([0.25], dtype=np.float32),
        np.asarray([False]),
        np.asarray([1], dtype=np.uint8),
        np.asarray([-0.75], dtype=np.float32),
    ]


def _run_discrete_sweep(state_in, state_mid, state_out, constants) -> None:
    copy_state_10(state_in, state_mid)
    spin_group0(
        state_in,
        constants["spin0_weights"],
        constants["spin0_bias"],
        constants["threshold"],
        constants["half"],
        state_mid,
    )
    categorical_group0(
        state_in,
        constants["cat0_weights"],
        constants["cat0_bias"],
        constants["gumbel0"],
        constants["one"],
        constants["half"],
        state_mid,
    )

    copy_state_10(state_mid, state_out)
    spin_group1(
        state_mid,
        constants["spin1_weights"],
        constants["spin1_bias"],
        constants["threshold"],
        constants["half"],
        state_out,
    )
    categorical_group1(
        state_mid,
        constants["cat1_weights"],
        constants["cat1_bias"],
        constants["gumbel1"],
        constants["one"],
        constants["half"],
        state_out,
    )


def _make_constants(device, spin_plans, categorical_plans) -> dict:
    return {
        "one": from_torch(torch.ones((TILE, TILE), dtype=torch.bfloat16), device=device),
        "half": from_torch(torch.full((TILE, TILE), 0.5, dtype=torch.bfloat16), device=device),
        "threshold": from_torch(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device=device),
        "gumbel0": from_torch(torch.zeros((N_CATEGORIES * TILE, TILE), dtype=torch.bfloat16), device=device),
        "gumbel1": from_torch(torch.zeros((N_CATEGORIES * TILE, TILE), dtype=torch.bfloat16), device=device),
        "spin0_weights": from_torch(_tile_planes(list(spin_plans[0].categorical_weights[0])), device=device),
        "spin0_bias": from_torch(torch.full((TILE, TILE), spin_plans[0].bias, dtype=torch.bfloat16), device=device),
        "cat0_weights": from_torch(_tile_planes(list(categorical_plans[1].spin_weights[0])), device=device),
        "cat0_bias": from_torch(_tile_planes(list(categorical_plans[1].bias)), device=device),
        "spin1_weights": from_torch(_tile_planes(list(spin_plans[3].categorical_weights[0])), device=device),
        "spin1_bias": from_torch(torch.full((TILE, TILE), spin_plans[3].bias, dtype=torch.bfloat16), device=device),
        "cat1_weights": from_torch(_tile_planes(list(categorical_plans[4].spin_weights[0])), device=device),
        "cat1_bias": from_torch(_tile_planes(list(categorical_plans[4].bias)), device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, default=0, metavar="N", help="run N timed device-resident sweeps")
    args = parser.parse_args()

    executor = ExperimentalTTLangExecutor(make_mixed_spin_categorical_gaussian_program())
    initial_lanes = executor.encode_state(_initial_state())
    expected_lanes = executor.evaluate_discrete_sweep(
        initial_lanes,
        spin_threshold_logits={0: 0.0, 3: 0.0},
        categorical_gumbel={1: (0.0, 0.0, 0.0), 4: (0.0, 0.0, 0.0)},
    )
    spin_plans = {plan.block_index: plan for plan in executor.spin_categorical_plans}
    categorical_plans = {plan.block_index: plan for plan in executor.categorical_spin_plans}

    device = ttnn.open_device(device_id=0)
    try:
        state0 = from_torch(_state_tiles(initial_lanes), device=device)
        state1 = from_torch(torch.zeros((N_LANES * TILE, TILE), dtype=torch.bfloat16), device=device)
        state2 = from_torch(torch.zeros((N_LANES * TILE, TILE), dtype=torch.bfloat16), device=device)
        constants = _make_constants(device, spin_plans, categorical_plans)

        _run_discrete_sweep(state0, state1, state2, constants)

        result = ttnn.to_torch(state2).to(torch.bfloat16)
        expected = _state_tiles(expected_lanes)
        mismatches = result != expected
        print("result", result)
        print("expected", expected)
        if mismatches.any():
            first = mismatches.nonzero()[0].tolist()
            print("mismatches", int(mismatches.sum().item()))
            print(
                "first mismatch", first, "result", result[first[0], first[1]], "expected", expected[first[0], first[1]]
            )
            raise AssertionError("TT-Lang discrete sweep mismatch")
        print("PASS: TT-Lang THRML discrete sweep")

        if args.benchmark:
            n = int(args.benchmark)
            if n <= 0:
                raise ValueError("--benchmark must be positive")
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            src, mid, dst = state2, state1, state0
            for _ in range(n):
                _run_discrete_sweep(src, mid, dst, constants)
                src, dst = dst, src
            ttnn.synchronize_device(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(
                "benchmark",
                {
                    "sweeps": n,
                    "total_ms": elapsed_ms,
                    "ms_per_sweep": elapsed_ms / n,
                    "dispatches_per_sweep": 6,
                },
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
