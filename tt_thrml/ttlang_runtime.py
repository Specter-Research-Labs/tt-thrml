"""Runtime shell for the first hardware-proven TT-Lang THRML path."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from thrml.block_sampling import BlockSamplingProgram

if TYPE_CHECKING:
    from .ttlang_backend import TTLangProgramPlanner

TILE = 32
N_CATEGORIES = 3
ttl: Any = None


def tile_planes(values: list[float]) -> Any:
    torch = _torch()
    planes = torch.as_tensor(values, dtype=torch.float32).reshape(len(values), 1, 1).expand(len(values), TILE, TILE)
    return planes.reshape(len(values) * TILE, TILE).to(torch.bfloat16)


def state_tiles(lanes: Any) -> Any:
    return tile_planes([float(value) for value in lanes])


def validate_ttlang_discrete_runtime(executor: "TTLangProgramPlanner") -> None:
    """Validate the narrow program shape currently backed by hardware kernels."""
    if executor.layout.total_lanes != 10:
        raise ValueError(f"TT-Lang discrete runtime expects 10 lanes, got {executor.layout.total_lanes}")

    spin_plans = {plan.block_index: plan for plan in executor.spin_categorical_plans}
    categorical_plans = {plan.block_index: plan for plan in executor.categorical_spin_plans}
    runtime_groups = _make_runtime_groups(executor, spin_plans, categorical_plans)
    if len(runtime_groups) != 2:
        raise ValueError(f"TT-Lang discrete runtime expects 2 sampling groups, got {len(runtime_groups)}")

    for block_index, plan in spin_plans.items():
        if plan.n_categorical_terms != 1:
            raise ValueError(f"unsupported TT-Lang spin plan shape for block {block_index}")
    for block_index, plan in categorical_plans.items():
        if len(plan.spin_lanes) != 1 or plan.n_categories != N_CATEGORIES:
            raise ValueError(f"unsupported TT-Lang categorical plan shape for block {block_index}")

    planned_blocks = set(spin_plans) | set(categorical_plans)
    for group in runtime_groups:
        if group.spin_block is None or group.categorical_block is None:
            raise ValueError(f"TT-Lang runtime expects one spin and one categorical update per group: {group}")
        supported_group_blocks = {block for block in (group.spin_block, group.categorical_block) if block is not None}
        unexpected_group_blocks = set(group.block_indices) & planned_blocks - supported_group_blocks
        if unexpected_group_blocks:
            raise ValueError(f"unsupported TT-Lang grouped plan blocks: {tuple(sorted(unexpected_group_blocks))}")


def supports_ttlang_discrete_runtime(program: BlockSamplingProgram) -> bool:
    from .ttlang_backend import TTLangProgramPlanner

    try:
        validate_ttlang_discrete_runtime(TTLangProgramPlanner(program))
    except ValueError:
        return False
    return True


def make_ttlang_discrete_runtime(
    *, ttl: Any, ttnn: Any, device: Any, executor: "TTLangProgramPlanner"
) -> "TTLangDiscreteSweepRuntime":
    validate_ttlang_discrete_runtime(executor)
    return TTLangDiscreteSweepRuntime(ttl=ttl, ttnn=ttnn, device=device, executor=executor)


def make_primary_ttlang_executor(
    ttnn: Any,
    device: Any,
    program: BlockSamplingProgram,
    *,
    ttl_module: Any | None = None,
) -> "TTLangDiscreteSweepRuntime":
    """Build the primary TT-Lang executor for the currently proven program shape."""
    from .ttlang_backend import TTLangProgramPlanner

    executor = TTLangProgramPlanner(program)
    validate_ttlang_discrete_runtime(executor)
    if ttl_module is None:
        try:
            import ttl as ttl_module  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError("ttl is required for the primary TT-Lang executor") from exc
    return make_ttlang_discrete_runtime(ttl=ttl_module, ttnn=ttnn, device=device, executor=executor)


class TTLangDiscreteSweepRuntime:
    """Device-resident executor for the current supported TT-Lang sweep shape.

    This class intentionally wraps only the hardware-proven path: copy the
    pre-group state, then run independent spin and categorical block updates for
    each sampling group. It is a stable executor-shaped surface we can optimize
    behind without changing the runner contract.
    """

    dispatches_per_sweep = 6

    def __init__(self, *, ttl: Any, ttnn: Any, device: Any, executor: "TTLangProgramPlanner"):
        validate_ttlang_discrete_runtime(executor)
        self.ttl = ttl
        self.ttnn = ttnn
        self.device = device
        self.executor = executor

        spin_plans = {plan.block_index: plan for plan in executor.spin_categorical_plans}
        categorical_plans = {plan.block_index: plan for plan in executor.categorical_spin_plans}
        self.groups = _make_runtime_groups(executor, spin_plans, categorical_plans)
        self.operations = _define_operations(
            ttl, spin_plans, categorical_plans, total_lanes=executor.layout.total_lanes
        )
        self.constants = self._make_constants(spin_plans, categorical_plans)
        self._state_current: Any | None = None
        self._state_mid: Any | None = None
        self._state_next: Any | None = None

    def upload_state(self, lanes: Any) -> None:
        torch = _torch()
        self._state_current = self.from_torch(state_tiles(lanes))
        self._state_mid = self.from_torch(
            torch.zeros((self.executor.layout.total_lanes * TILE, TILE), dtype=torch.bfloat16)
        )
        self._state_next = self.from_torch(
            torch.zeros((self.executor.layout.total_lanes * TILE, TILE), dtype=torch.bfloat16)
        )

    def load_state(self, state_free: Sequence[Any], state_clamp: Sequence[Any] = ()) -> None:
        self.upload_state(self.executor.encode_state(list(state_free) + list(state_clamp)))

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

        spin_blocks = {plan.block_index for plan in self.executor.spin_categorical_plans}
        categorical_blocks = {plan.block_index for plan in self.executor.categorical_spin_plans}

        unsupported_spin_blocks = set(spin_threshold_logits) - spin_blocks
        if unsupported_spin_blocks:
            raise ValueError(f"unsupported TT-Lang spin threshold blocks: {tuple(sorted(unsupported_spin_blocks))}")
        unsupported_categorical_blocks = set(categorical_gumbel) - categorical_blocks
        if unsupported_categorical_blocks:
            raise ValueError(
                f"unsupported TT-Lang categorical gumbel blocks: {tuple(sorted(unsupported_categorical_blocks))}"
            )

        for block_index in sorted(spin_blocks):
            value = float(spin_threshold_logits.get(block_index, 0.0))
            self.constants[_spin_key(block_index, "threshold")] = self.from_torch(
                torch.full((TILE, TILE), value, dtype=torch.bfloat16)
            )

        for block_index in sorted(categorical_blocks):
            values = tuple(float(v) for v in categorical_gumbel.get(block_index, (0.0, 0.0, 0.0)))
            if len(values) != N_CATEGORIES:
                raise ValueError(f"TT-Lang categorical gumbel block {block_index} expects {N_CATEGORIES} values")
            self.constants[_categorical_key(block_index, "gumbel")] = self.from_torch(tile_planes(list(values)))

    def set_sweep_randomness_from_key(self, key: Any) -> None:
        """Upload one sweep of random inputs using THRML's JAX key schedule."""
        randomness = self.executor.sweep_randomness_from_key(key)
        self.set_sweep_randomness(
            spin_threshold_logits=randomness.spin_threshold_logits,
            categorical_gumbel=randomness.categorical_gumbel,
        )

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

    def read_state_lists(self) -> tuple[list[Any], list[Any]]:
        from .ttlang_backend import decode_state

        host_state = self.materialize_state().to(_torch().float32).cpu().tolist()
        lanes = [host_state[lane * TILE][0] for lane in range(self.executor.layout.total_lanes)]
        decoded = decode_state(self.executor.layout, lanes)
        n_free = len(self.executor.program.gibbs_spec.free_blocks)
        return list(decoded[:n_free]), list(decoded[n_free:])

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
        constants = {
            "one": self.from_torch(torch.ones((TILE, TILE), dtype=torch.bfloat16)),
            "half": self.from_torch(torch.full((TILE, TILE), 0.5, dtype=torch.bfloat16)),
        }
        for block_index, plan in sorted(spin_plans.items()):
            constants[_spin_key(block_index, "threshold")] = self.from_torch(
                torch.zeros((TILE, TILE), dtype=torch.bfloat16)
            )
            constants[_spin_key(block_index, "weights")] = self.from_torch(
                tile_planes(list(plan.categorical_weights[0]))
            )
            constants[_spin_key(block_index, "bias")] = self.from_torch(
                torch.full((TILE, TILE), plan.bias, dtype=torch.bfloat16)
            )
        for block_index, plan in sorted(categorical_plans.items()):
            constants[_categorical_key(block_index, "gumbel")] = self.from_torch(
                torch.zeros((N_CATEGORIES * TILE, TILE), dtype=torch.bfloat16)
            )
            constants[_categorical_key(block_index, "weights")] = self.from_torch(
                tile_planes(list(plan.spin_weights[0]))
            )
            constants[_categorical_key(block_index, "bias")] = self.from_torch(tile_planes(list(plan.bias)))
        return constants

    def _run_discrete_sweep(self, state_in: Any, state_mid: Any, state_out: Any) -> None:
        ops = self.operations
        constants = self.constants
        state_buffers = (state_in, state_mid, state_out)
        for group_index, group in enumerate(self.groups):
            group_in = state_buffers[group_index]
            group_out = state_buffers[group_index + 1]
            ops.copy_state(group_in, group_out)

            spin_block = group.spin_block
            if spin_block is not None:
                ops.spin_updates[spin_block](
                    group_in,
                    constants[_spin_key(spin_block, "weights")],
                    constants[_spin_key(spin_block, "bias")],
                    constants[_spin_key(spin_block, "threshold")],
                    constants["half"],
                    group_out,
                )

            categorical_block = group.categorical_block
            if categorical_block is not None:
                ops.categorical_updates[categorical_block](
                    group_in,
                    constants[_categorical_key(categorical_block, "weights")],
                    constants[_categorical_key(categorical_block, "bias")],
                    constants[_categorical_key(categorical_block, "gumbel")],
                    constants["one"],
                    constants["half"],
                    group_out,
                )

    def _require_state(self) -> tuple[Any, Any, Any]:
        if self._state_current is None or self._state_mid is None or self._state_next is None:
            raise RuntimeError("state must be uploaded before running TT-Lang sweeps")
        return self._state_current, self._state_mid, self._state_next


def _torch() -> Any:
    import torch

    return torch


@dataclass(frozen=True)
class _RuntimeGroup:
    block_indices: tuple[int, ...]
    spin_block: int | None
    categorical_block: int | None


class _Operations:
    def __init__(self, *, copy_state, spin_updates: dict[int, Any], categorical_updates: dict[int, Any]):
        self.copy_state = copy_state
        self.spin_updates = spin_updates
        self.categorical_updates = categorical_updates


def _spin_key(block_index: int, name: str) -> str:
    return f"spin:{block_index}:{name}"


def _categorical_key(block_index: int, name: str) -> str:
    return f"categorical:{block_index}:{name}"


def _make_runtime_groups(
    executor: "TTLangProgramPlanner", spin_plans: dict[int, Any], categorical_plans: dict[int, Any]
) -> tuple[_RuntimeGroup, ...]:
    groups = []
    for block_indices in executor.sampling_order:
        spin_blocks = tuple(block for block in block_indices if block in spin_plans)
        categorical_blocks = tuple(block for block in block_indices if block in categorical_plans)
        if len(spin_blocks) > 1 or len(categorical_blocks) > 1:
            raise ValueError(f"unsupported TT-Lang sampling order group shape: {block_indices}")
        groups.append(
            _RuntimeGroup(
                block_indices=tuple(block_indices),
                spin_block=spin_blocks[0] if spin_blocks else None,
                categorical_block=categorical_blocks[0] if categorical_blocks else None,
            )
        )
    return tuple(groups)


def _define_operations(
    ttl_module: Any, spin_plans: dict[int, Any], categorical_plans: dict[int, Any], *, total_lanes: int
) -> _Operations:
    global ttl
    ttl = ttl_module

    def define_copy_state(n_lanes: int):
        @ttl.operation(grid=(1, 1))
        def copy_state(inp, out):
            lane_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=n_lanes)

            @ttl.compute()
            def compute():
                pass

            @ttl.datamovement()
            def read():
                for lane in range(n_lanes):
                    with lane_dfb.reserve() as lane_blk:
                        tx_in = ttl.copy(inp[lane, 0], lane_blk)
                        tx_in.wait()

            @ttl.datamovement()
            def write():
                for lane in range(n_lanes):
                    with lane_dfb.wait() as lane_blk:
                        tx_out = ttl.copy(lane_blk, out[lane, 0])
                        tx_out.wait()

        return copy_state

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
        copy_state=define_copy_state(total_lanes),
        spin_updates={
            block_index: define_spin_update(
                source_start=plan.categorical_lane_groups[0][0],
                output_lane=plan.output_lane,
            )
            for block_index, plan in sorted(spin_plans.items())
        },
        categorical_updates={
            block_index: define_categorical_update(
                source_lane=plan.spin_lanes[0],
                output_start=plan.output_lanes[0],
            )
            for block_index, plan in sorted(categorical_plans.items())
        },
    )
