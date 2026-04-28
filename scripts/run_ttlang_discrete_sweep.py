#!/usr/bin/env python3
"""Run one narrow TT-Lang discrete sweep over device-resident THRML state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tt_thrml
from tt_thrml.example_programs import make_mixed_spin_categorical_gaussian_program
from tt_thrml.ttlang_backend import TTLangProgramPlanner, decode_state
from tt_thrml.ttlang_runtime import state_tiles


def _require_ttnn():
    import ttnn  # type: ignore[reportMissingImports]

    return ttnn


def _initial_state(n_pairs: int) -> list[object]:
    states: list[object] = []
    for pair_index in range(n_pairs):
        states.extend(
            [
                [pair_index % 2 == 0],
                [1],
                [0.25 if pair_index % 2 == 0 else -0.75],
            ]
        )
    return states


def _assert_tiles_equal(result, expected_lanes: object, *, label: str) -> None:
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


def _as_plain_list(value: object) -> object:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _assert_state_lists_equal(result: list[object], expected: list[object], *, label: str) -> None:
    if len(result) != len(expected):
        raise AssertionError(f"TT-Lang decoded state length mismatch after {label}: {len(result)} != {len(expected)}")
    for index, (got, want) in enumerate(zip(result, expected, strict=True)):
        if _as_plain_list(got) != _as_plain_list(want):
            raise AssertionError(f"TT-Lang decoded state mismatch after {label} block {index}: {got} != {want}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, default=0, metavar="N", help="run N timed device-resident sweeps")
    parser.add_argument("--warmup", type=int, default=0, metavar="N", help="run N untimed sweeps before benchmarking")
    parser.add_argument("--seed", type=int, default=0, help="JAX PRNG seed for the THRML chain randomness window")
    parser.add_argument("--pairs", type=int, default=2, help="number of independent spin/categorical/gaussian groups")
    parser.add_argument("--json", action="store_true", help="emit the benchmark record as JSON")
    args = parser.parse_args()

    import jax

    ttnn = _require_ttnn()
    program = make_mixed_spin_categorical_gaussian_program(n_pairs=args.pairs)
    executor = TTLangProgramPlanner(program)
    initial_state = _initial_state(args.pairs)
    initial_lanes = executor.encode_state(initial_state)
    total_sweeps = max(1, int(args.benchmark), 1 + max(0, int(args.warmup)))
    randomness_window = executor.sweep_randomness_window_from_key(jax.random.PRNGKey(args.seed), total_sweeps)
    expected_lanes = executor.evaluate_discrete_sweep(initial_lanes, **randomness_window.sweep(0).as_kwargs())

    device = ttnn.open_device(device_id=0)
    try:
        runtime = tt_thrml.make_executor(ttnn, device, program)
        runtime.load_state(initial_state)
        runtime.set_sweep_randomness_window(randomness_window)
        runtime.run_sweep()

        result = runtime.materialize_state()
        decoded_result, decoded_clamp = runtime.read_state_lists()
        expected_state = decode_state(executor.layout, expected_lanes)
        print("result", result)
        print("expected", state_tiles(expected_lanes))
        _assert_tiles_equal(result, expected_lanes, label="one sweep")
        _assert_state_lists_equal(decoded_result + decoded_clamp, expected_state, label="one sweep")
        print("PASS: TT-Lang THRML discrete sweep")

        if args.benchmark:
            n = int(args.benchmark)
            warmup = int(args.warmup)
            if warmup < 0:
                raise ValueError("--warmup must be non-negative")
            runtime.load_state(initial_state)
            runtime.rewind_sweep_randomness_window()
            precompile_sweeps = max(warmup, n)
            runtime.run_sweeps(precompile_sweeps)
            runtime.load_state(initial_state)
            runtime.rewind_sweep_randomness_window()
            elapsed_ms = runtime.run_sweeps(n)
            benchmark_window = executor.sweep_randomness_window_from_key(jax.random.PRNGKey(args.seed), n)
            expected_after_benchmark = executor.evaluate_discrete_sweeps_with_randomness_window(
                initial_lanes, benchmark_window
            )
            expected_state_after_benchmark = decode_state(executor.layout, expected_after_benchmark)
            decoded_after_benchmark, decoded_clamp_after_benchmark = runtime.read_state_lists()
            _assert_tiles_equal(runtime.materialize_state(), expected_after_benchmark, label=f"{n + 1} sweeps")
            _assert_state_lists_equal(
                decoded_after_benchmark + decoded_clamp_after_benchmark,
                expected_state_after_benchmark,
                label=f"{n + 1} sweeps",
            )
            record = {
                "warmup_sweeps": warmup,
                "measured_sweeps": n,
                "total_ms": elapsed_ms,
                "ms_per_sweep": elapsed_ms / n,
                "dispatches_per_sweep": runtime.dispatches_per_sweep,
            }
            if args.json:
                print(json.dumps(record, sort_keys=True))
            else:
                print("benchmark", record)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
