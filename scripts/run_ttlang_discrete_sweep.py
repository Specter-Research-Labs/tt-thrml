#!/usr/bin/env python3
"""Run one narrow TT-Lang discrete sweep over device-resident THRML state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tt_thrml.example_programs import make_mixed_spin_categorical_gaussian_program
from tt_thrml.ttlang_backend import ExperimentalTTLangExecutor
from tt_thrml.ttlang_runtime import make_ttlang_discrete_runtime, state_tiles


def _require_ttlang():
    import ttl  # type: ignore[reportMissingImports]
    import ttnn  # type: ignore[reportMissingImports]

    return ttl, ttnn


def _initial_state() -> list[np.ndarray]:
    return [
        np.asarray([True]),
        np.asarray([1], dtype=np.uint8),
        np.asarray([0.25], dtype=np.float32),
        np.asarray([False]),
        np.asarray([1], dtype=np.uint8),
        np.asarray([-0.75], dtype=np.float32),
    ]


def _assert_tiles_equal(result, expected_lanes: np.ndarray, *, label: str) -> None:
    expected = state_tiles(expected_lanes)
    mismatches = result != expected
    if not mismatches.any():
        return
    first = mismatches.nonzero()[0].tolist()
    print("mismatches", int(mismatches.sum().item()))
    print(
        label, "first mismatch", first, "result", result[first[0], first[1]], "expected", expected[first[0], first[1]]
    )
    raise AssertionError(f"TT-Lang discrete sweep mismatch after {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, default=0, metavar="N", help="run N timed device-resident sweeps")
    args = parser.parse_args()

    ttl, ttnn = _require_ttlang()
    executor = ExperimentalTTLangExecutor(make_mixed_spin_categorical_gaussian_program())
    initial_lanes = executor.encode_state(_initial_state())
    sweep_kwargs = {
        "spin_threshold_logits": {0: 0.0, 3: 0.0},
        "categorical_gumbel": {1: (0.0, 0.0, 0.0), 4: (0.0, 0.0, 0.0)},
    }
    expected_lanes = executor.evaluate_discrete_sweep(
        initial_lanes,
        **sweep_kwargs,
    )

    device = ttnn.open_device(device_id=0)
    try:
        runtime = make_ttlang_discrete_runtime(
            ttl=ttl,
            ttnn=ttnn,
            device=device,
            executor=executor,
        )
        runtime.upload_state(initial_lanes)
        runtime.run_sweep()

        result = runtime.materialize_state()
        print("result", result)
        print("expected", state_tiles(expected_lanes))
        _assert_tiles_equal(result, expected_lanes, label="one sweep")
        print("PASS: TT-Lang THRML discrete sweep")

        if args.benchmark:
            n = int(args.benchmark)
            elapsed_ms = runtime.run_sweeps(n)
            expected_after_benchmark = executor.evaluate_discrete_sweeps(expected_lanes, n, **sweep_kwargs)
            _assert_tiles_equal(runtime.materialize_state(), expected_after_benchmark, label=f"{n + 1} sweeps")
            print(
                "benchmark",
                {
                    "sweeps": n,
                    "total_ms": elapsed_ms,
                    "ms_per_sweep": elapsed_ms / n,
                    "dispatches_per_sweep": runtime.dispatches_per_sweep,
                },
            )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
