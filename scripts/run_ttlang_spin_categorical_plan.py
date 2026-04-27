#!/usr/bin/env python3
"""Run the first THRML-derived TT-Lang spin/categorical plan.

This is an experimental hardware runner. It builds the existing mixed parity
case, lowers the first spin block to a TT-Lang categorical-source plan, then
executes that plan through a small TT-Lang kernel.

The runner is deliberately narrow: one spin target, one categorical source
interaction, three categories. It exists to validate the production lowering
contract before generalizing the TT-Lang executor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.parity._parity_runner import _make_mixed_case
from tt_thrml.ttlang_backend import ExperimentalTTLangExecutor, TTLangSpinCategoricalRun

TILE = 32
N_CATEGORIES = 3
THRESHOLD_LOGIT = 0.0


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
def thrml_spin_categorical_plan_tile(cat_state, weights, bias, threshold_logits, half, out):
    cat_state_dfb = ttl.make_dataflow_buffer_like(cat_state, shape=(1, 1), block_count=2)
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
                tx_state = ttl.copy(cat_state[category, 0], cat_state_blk)
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
            tx_out = ttl.copy(out_blk, out[0, 0])
            tx_out.wait()


def _tile_planes(values: list[float]) -> torch.Tensor:
    planes = np.asarray(values, dtype=np.float32)[:, None, None] * np.ones((len(values), TILE, TILE), dtype=np.float32)
    return torch.from_numpy(planes.reshape(len(values) * TILE, TILE)).to(torch.bfloat16)


def _case_state(category: int) -> list[np.ndarray]:
    return [
        np.asarray([True]),
        np.asarray([category], dtype=np.uint8),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([False]),
        np.asarray([1], dtype=np.uint8),
        np.asarray([0.0], dtype=np.float32),
    ]


def _run_case(device, run: TTLangSpinCategoricalRun) -> None:
    plan = run.plan
    if plan.n_categorical_terms != 1 or len(plan.categorical_weights[0]) != N_CATEGORIES:
        raise RuntimeError(f"unsupported plan shape: {plan}")

    cat_values = list(run.categorical_values[0])
    weight_values = [float(weight) for weight in plan.categorical_weights[0]]
    expected = torch.full((TILE, TILE), run.expected_spin, dtype=torch.bfloat16)

    cat_state = from_torch(_tile_planes(cat_values), device=device)
    weights = from_torch(_tile_planes(weight_values), device=device)
    bias = from_torch(torch.full((TILE, TILE), plan.bias, dtype=torch.bfloat16), device=device)
    threshold = from_torch(torch.full((TILE, TILE), run.threshold_logit, dtype=torch.bfloat16), device=device)
    half = from_torch(torch.full((TILE, TILE), 0.5, dtype=torch.bfloat16), device=device)
    out = from_torch(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device=device)

    thrml_spin_categorical_plan_tile(cat_state, weights, bias, threshold, half, out)
    result = ttnn.to_torch(out).to(torch.bfloat16)

    mismatches = result != expected
    print(
        "case",
        {
            "cat_values": cat_values,
            "weights": weight_values,
            "bias": plan.bias,
            "threshold_logit": run.threshold_logit,
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
        raise AssertionError("TT-Lang spin categorical plan mismatch")


def main() -> None:
    executor = ExperimentalTTLangExecutor(_make_mixed_case().program)

    device = ttnn.open_device(device_id=0)
    try:
        for category in (0, 1):
            run = executor.materialize_spin_categorical_run(
                0,
                _case_state(category),
                threshold_logit=THRESHOLD_LOGIT,
            )
            _run_case(device, run)

        print("PASS: TT-Lang THRML spin categorical plan")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
