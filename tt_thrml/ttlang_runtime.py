"""Runtime shell for the first hardware-proven TT-Lang THRML path."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from thrml.block_sampling import BlockSamplingProgram

from .ttlang_backend import ExperimentalTTLangExecutor

TILE = 32
N_CATEGORIES = 3
ttl: Any = None

_SUPPORTED_SAMPLING_ORDER = ((0, 1, 2), (3, 4, 5))
_SUPPORTED_SPIN_BLOCKS = (0, 3)
_SUPPORTED_CATEGORICAL_BLOCKS = (1, 4)
_SPIN_THRESHOLD_CONSTANTS = {0: "threshold0", 3: "threshold1"}
_CATEGORICAL_GUMBEL_CONSTANTS = {1: "gumbel0", 4: "gumbel1"}


def tile_planes(values: list[float]) -> Any:
    torch = _torch()
    planes = np.asarray(values, dtype=np.float32)[:, None, None] * np.ones((len(values), TILE, TILE), dtype=np.float32)
    return torch.from_numpy(planes.reshape(len(values) * TILE, TILE)).to(torch.bfloat16)


def state_tiles(lanes: np.ndarray) -> Any:
    return tile_planes([float(value) for value in lanes])


def validate_ttlang_discrete_runtime(executor: ExperimentalTTLangExecutor) -> None:
    """Validate the narrow program shape currently backed by hardware kernels."""
    if executor.layout.total_lanes != 10:
        raise ValueError(f"TT-Lang discrete runtime expects 10 lanes, got {executor.layout.total_lanes}")
    if executor.sampling_order != _SUPPORTED_SAMPLING_ORDER:
        raise ValueError(f"unsupported TT-Lang sampling order: {executor.sampling_order}")

    spin_plans = {plan.block_index: plan for plan in executor.spin_categorical_plans}
    categorical_plans = {plan.block_index: plan for plan in executor.categorical_spin_plans}
    if tuple(sorted(spin_plans)) != _SUPPORTED_SPIN_BLOCKS:
        raise ValueError(f"unsupported TT-Lang spin plan blocks: {tuple(sorted(spin_plans))}")
    if tuple(sorted(categorical_plans)) != _SUPPORTED_CATEGORICAL_BLOCKS:
        raise ValueError(f"unsupported TT-Lang categorical plan blocks: {tuple(sorted(categorical_plans))}")

    expected_spin_lanes = {0: 0, 3: 5}
    expected_categorical_lanes = {1: (1, 2, 3), 4: (6, 7, 8)}
    for block_index, output_lane in expected_spin_lanes.items():
        plan = spin_plans[block_index]
        if plan.output_lane != output_lane or plan.n_categorical_terms != 1:
            raise ValueError(f"unsupported TT-Lang spin plan shape for block {block_index}")
    for block_index, output_lanes in expected_categorical_lanes.items():
        plan = categorical_plans[block_index]
        if plan.output_lanes != output_lanes or len(plan.spin_lanes) != 1 or plan.n_categories != N_CATEGORIES:
            raise ValueError(f"unsupported TT-Lang categorical plan shape for block {block_index}")


def supports_ttlang_discrete_runtime(program: BlockSamplingProgram) -> bool:
    try:
        validate_ttlang_discrete_runtime(ExperimentalTTLangExecutor(program))
    except ValueError:
        return False
    return True


def make_ttlang_discrete_runtime(
    *, ttl: Any, ttnn: Any, device: Any, executor: ExperimentalTTLangExecutor
) -> "TTLangDiscreteSweepRuntime":
    validate_ttlang_discrete_runtime(executor)
    return TTLangDiscreteSweepRuntime(ttl=ttl, ttnn=ttnn, device=device, executor=executor)


class TTLangDiscreteSweepRuntime:
    """Device-resident executor for the current supported TT-Lang sweep shape.

    This class intentionally wraps only the hardware-proven path: copy the
    pre-group state, then run independent spin and categorical block updates for
    each sampling group. It is a stable executor-shaped surface we can optimize
    behind without changing the runner contract.
    """

    dispatches_per_sweep = 6

    def __init__(self, *, ttl: Any, ttnn: Any, device: Any, executor: ExperimentalTTLangExecutor):
        validate_ttlang_discrete_runtime(executor)
        self.ttl = ttl
        self.ttnn = ttnn
        self.device = device
        self.executor = executor
        self.operations = _define_operations(ttl)

        spin_plans = {plan.block_index: plan for plan in executor.spin_categorical_plans}
        categorical_plans = {plan.block_index: plan for plan in executor.categorical_spin_plans}
        self.constants = self._make_constants(spin_plans, categorical_plans)
        self._state_current: Any | None = None
        self._state_mid: Any | None = None
        self._state_next: Any | None = None

    def upload_state(self, lanes: np.ndarray) -> None:
        torch = _torch()
        self._state_current = self.from_torch(state_tiles(lanes))
        self._state_mid = self.from_torch(
            torch.zeros((self.executor.layout.total_lanes * TILE, TILE), dtype=torch.bfloat16)
        )
        self._state_next = self.from_torch(
            torch.zeros((self.executor.layout.total_lanes * TILE, TILE), dtype=torch.bfloat16)
        )

    def set_sweep_randomness(
        self,
        *,
        spin_threshold_logits: Mapping[int, float] | None = None,
        categorical_gumbel: Mapping[int, Sequence[float]] | None = None,
    ) -> None:
        """Upload deterministic per-block random inputs for the next sweeps."""
        torch = _torch()
        spin_threshold_logits = spin_threshold_logits or {}
        categorical_gumbel = categorical_gumbel or {}

        unsupported_spin_blocks = set(spin_threshold_logits) - set(_SPIN_THRESHOLD_CONSTANTS)
        if unsupported_spin_blocks:
            raise ValueError(f"unsupported TT-Lang spin threshold blocks: {tuple(sorted(unsupported_spin_blocks))}")
        unsupported_categorical_blocks = set(categorical_gumbel) - set(_CATEGORICAL_GUMBEL_CONSTANTS)
        if unsupported_categorical_blocks:
            raise ValueError(
                f"unsupported TT-Lang categorical gumbel blocks: {tuple(sorted(unsupported_categorical_blocks))}"
            )

        for block_index, key in _SPIN_THRESHOLD_CONSTANTS.items():
            value = float(spin_threshold_logits.get(block_index, 0.0))
            self.constants[key] = self.from_torch(torch.full((TILE, TILE), value, dtype=torch.bfloat16))

        for block_index, key in _CATEGORICAL_GUMBEL_CONSTANTS.items():
            values = tuple(float(v) for v in categorical_gumbel.get(block_index, (0.0, 0.0, 0.0)))
            if len(values) != N_CATEGORIES:
                raise ValueError(f"TT-Lang categorical gumbel block {block_index} expects {N_CATEGORIES} values")
            self.constants[key] = self.from_torch(tile_planes(list(values)))

    def run_sweep(self) -> None:
        current, mid, next_state = self._require_state()
        self._run_discrete_sweep(current, mid, next_state)
        self._state_current, self._state_next = next_state, current

    def run_sweeps(self, n_sweeps: int) -> float:
        if n_sweeps <= 0:
            raise ValueError("n_sweeps must be positive")
        self._require_state()
        self.ttnn.synchronize_device(self.device)
        t0 = time.perf_counter()
        for _ in range(n_sweeps):
            self.run_sweep()
        self.ttnn.synchronize_device(self.device)
        return (time.perf_counter() - t0) * 1000.0

    def materialize_state(self) -> Any:
        current, _, _ = self._require_state()
        return self.ttnn.to_torch(current).to(_torch().bfloat16)

    def from_torch(self, tensor: Any):
        return self.ttnn.from_torch(
            tensor,
            dtype=self.ttnn.bfloat16,
            layout=self.ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self.ttnn.DRAM_MEMORY_CONFIG,
        )

    def _make_constants(self, spin_plans: dict[int, Any], categorical_plans: dict[int, Any]) -> dict[str, Any]:
        torch = _torch()
        return {
            "one": self.from_torch(torch.ones((TILE, TILE), dtype=torch.bfloat16)),
            "half": self.from_torch(torch.full((TILE, TILE), 0.5, dtype=torch.bfloat16)),
            "threshold0": self.from_torch(torch.zeros((TILE, TILE), dtype=torch.bfloat16)),
            "threshold1": self.from_torch(torch.zeros((TILE, TILE), dtype=torch.bfloat16)),
            "gumbel0": self.from_torch(torch.zeros((N_CATEGORIES * TILE, TILE), dtype=torch.bfloat16)),
            "gumbel1": self.from_torch(torch.zeros((N_CATEGORIES * TILE, TILE), dtype=torch.bfloat16)),
            "spin0_weights": self.from_torch(tile_planes(list(spin_plans[0].categorical_weights[0]))),
            "spin0_bias": self.from_torch(torch.full((TILE, TILE), spin_plans[0].bias, dtype=torch.bfloat16)),
            "cat0_weights": self.from_torch(tile_planes(list(categorical_plans[1].spin_weights[0]))),
            "cat0_bias": self.from_torch(tile_planes(list(categorical_plans[1].bias))),
            "spin1_weights": self.from_torch(tile_planes(list(spin_plans[3].categorical_weights[0]))),
            "spin1_bias": self.from_torch(torch.full((TILE, TILE), spin_plans[3].bias, dtype=torch.bfloat16)),
            "cat1_weights": self.from_torch(tile_planes(list(categorical_plans[4].spin_weights[0]))),
            "cat1_bias": self.from_torch(tile_planes(list(categorical_plans[4].bias))),
        }

    def _run_discrete_sweep(self, state_in: Any, state_mid: Any, state_out: Any) -> None:
        ops = self.operations
        constants = self.constants
        ops.copy_state_10(state_in, state_mid)
        ops.spin_group0(
            state_in,
            constants["spin0_weights"],
            constants["spin0_bias"],
            constants["threshold0"],
            constants["half"],
            state_mid,
        )
        ops.categorical_group0(
            state_in,
            constants["cat0_weights"],
            constants["cat0_bias"],
            constants["gumbel0"],
            constants["one"],
            constants["half"],
            state_mid,
        )

        ops.copy_state_10(state_mid, state_out)
        ops.spin_group1(
            state_mid,
            constants["spin1_weights"],
            constants["spin1_bias"],
            constants["threshold1"],
            constants["half"],
            state_out,
        )
        ops.categorical_group1(
            state_mid,
            constants["cat1_weights"],
            constants["cat1_bias"],
            constants["gumbel1"],
            constants["one"],
            constants["half"],
            state_out,
        )

    def _require_state(self) -> tuple[Any, Any, Any]:
        if self._state_current is None or self._state_mid is None or self._state_next is None:
            raise RuntimeError("state must be uploaded before running TT-Lang sweeps")
        return self._state_current, self._state_mid, self._state_next


def _torch() -> Any:
    import torch  # type: ignore[reportMissingImports]

    return torch


class _Operations:
    def __init__(self, *, copy_state_10, spin_group0, categorical_group0, spin_group1, categorical_group1):
        self.copy_state_10 = copy_state_10
        self.spin_group0 = spin_group0
        self.categorical_group0 = categorical_group0
        self.spin_group1 = spin_group1
        self.categorical_group1 = categorical_group1


def _define_operations(ttl_module: Any) -> _Operations:
    global ttl
    ttl = ttl_module

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

    def define_spin_update(source_start: int, output_lane: int):
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

    def define_categorical_update(source_lane: int, output_start: int):
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

    return _Operations(
        copy_state_10=copy_state_10,
        spin_group0=define_spin_update(source_start=1, output_lane=0),
        categorical_group0=define_categorical_update(source_lane=0, output_start=1),
        spin_group1=define_spin_update(source_start=6, output_lane=5),
        categorical_group1=define_categorical_update(source_lane=5, output_start=6),
    )
