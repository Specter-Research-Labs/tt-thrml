"""TT-Lang-oriented state layout for THRML programs.

The existing TT-MLIR backend stores every THRML node in a flat scalar vector.
That shape is convenient for StableHLO gather/update lowering, but it is the
wrong device contract for TT-Lang categorical work. TT-Lang wants the data
already laid out as tiled arithmetic operands.

This module defines the backend-facing logical layout:

- spin blocks store one signed lane per node
- categorical blocks store one one-hot lane per category per node
- gaussian blocks store one value lane per node

The host-facing THRML API can keep using booleans, scalar category ids, and
floats. Only the device-resident backend state changes shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from thrml.block_sampling import BlockSamplingProgram

from .core import CompiledFusedBlock, Family, FusedBlockSpec


@dataclass(frozen=True)
class TTLangBlockLayout:
    block_index: int
    family: Family
    n_nodes: int
    n_categories: int | None
    scalar_start: int
    lane_start: int
    lanes_per_node: int

    @property
    def lane_count(self) -> int:
        return self.n_nodes * self.lanes_per_node


@dataclass(frozen=True)
class TTLangStateLayout:
    blocks: tuple[TTLangBlockLayout, ...]
    total_lanes: int
    scalar_to_lane_start: tuple[int, ...]

    def block(self, block_index: int) -> TTLangBlockLayout:
        return self.blocks[block_index]

    def node_lane_start(self, scalar_global_index: int) -> int:
        return self.scalar_to_lane_start[scalar_global_index]


@dataclass(frozen=True)
class TTLangSpinCategoricalPlan:
    """A first TT-Lang-lowerable spin block shape.

    The block has one spin node and zero or more categorical source
    contributions. Bias terms are pre-summed on the host as constants. Each
    categorical source contribution is represented as three category lanes and
    three corresponding weights for the target spin node.
    """

    block_index: int
    output_lane: int
    bias: float
    categorical_lane_groups: tuple[tuple[int, ...], ...]
    categorical_weights: tuple[tuple[float, ...], ...]

    @property
    def n_categorical_terms(self) -> int:
        return len(self.categorical_lane_groups)


@dataclass(frozen=True)
class TTLangSpinCategoricalRun:
    plan: TTLangSpinCategoricalPlan
    categorical_values: tuple[tuple[float, ...], ...]
    threshold_logit: float
    expected_spin: float


@dataclass(frozen=True)
class TTLangCategoricalSpinPlan:
    """A first TT-Lang-lowerable categorical block shape.

    The block has one categorical node and zero or more spin source
    contributions. Categorical state is represented as output category lanes;
    scores are computed per category and sampled with argmax(score + gumbel).
    """

    block_index: int
    output_lanes: tuple[int, ...]
    bias: tuple[float, ...]
    spin_lanes: tuple[int, ...]
    spin_weights: tuple[tuple[float, ...], ...]

    @property
    def n_categories(self) -> int:
        return len(self.output_lanes)


@dataclass(frozen=True)
class TTLangCategoricalSpinRun:
    plan: TTLangCategoricalSpinPlan
    spin_values: tuple[float, ...]
    gumbel: tuple[float, ...]
    expected_category: int
    expected_one_hot: tuple[float, ...]


class ExperimentalTTLangExecutor:
    """Planning shell for the first TT-Lang executor path.

    This class does not claim full THRML execution. It captures the shared
    pieces a real TT-Lang executor needs first: THRML block lowering,
    family-specific state encoding, and supported spin/categorical-source
    plan materialization.
    """

    def __init__(self, program: BlockSamplingProgram):
        self.program = program
        self.compiled_blocks = build_ttlang_compiled_blocks(program)
        self.layout = build_ttlang_state_layout(self.compiled_blocks)
        plans = []
        categorical_plans = []
        for block in self.compiled_blocks[: len(program.gibbs_spec.free_blocks)]:
            try:
                plans.append(build_spin_categorical_plan(self.layout, block.spec))
            except ValueError:
                pass
            try:
                categorical_plans.append(build_categorical_spin_plan(self.layout, block.spec))
            except ValueError:
                pass
        self.spin_categorical_plans = tuple(plans)
        self.categorical_spin_plans = tuple(categorical_plans)
        self._spin_plan_by_block = {plan.block_index: plan for plan in self.spin_categorical_plans}
        self._categorical_plan_by_block = {plan.block_index: plan for plan in self.categorical_spin_plans}
        self.sampling_order = tuple(tuple(int(index) for index in group) for group in program.gibbs_spec.sampling_order)

    def encode_state(self, state_free: Sequence[object], state_clamp: Sequence[object] = ()) -> np.ndarray:
        return encode_state(self.layout, tuple(state_free) + tuple(state_clamp))

    def materialize_spin_categorical_run(
        self,
        plan_index: int,
        state_free: Sequence[object],
        state_clamp: Sequence[object] = (),
        *,
        threshold_logit: float,
    ) -> TTLangSpinCategoricalRun:
        plan = self.spin_categorical_plans[plan_index]
        state_lanes = self.encode_state(state_free, state_clamp)
        categorical_values = []
        for lane_group in plan.categorical_lane_groups:
            categorical_values.append(tuple(float(state_lanes[lane]) for lane in lane_group))
        return TTLangSpinCategoricalRun(
            plan=plan,
            categorical_values=tuple(categorical_values),
            threshold_logit=float(threshold_logit),
            expected_spin=evaluate_spin_categorical_plan(plan, state_lanes, threshold_logit),
        )

    def materialize_categorical_spin_run(
        self,
        plan_index: int,
        state_free: Sequence[object],
        state_clamp: Sequence[object] = (),
        *,
        gumbel: Sequence[float],
    ) -> TTLangCategoricalSpinRun:
        plan = self.categorical_spin_plans[plan_index]
        state_lanes = self.encode_state(state_free, state_clamp)
        spin_values = tuple(float(state_lanes[lane]) for lane in plan.spin_lanes)
        expected_category = evaluate_categorical_spin_plan(plan, state_lanes, gumbel)
        expected_one_hot = tuple(1.0 if i == expected_category else 0.0 for i in range(plan.n_categories))
        return TTLangCategoricalSpinRun(
            plan=plan,
            spin_values=spin_values,
            gumbel=tuple(float(v) for v in gumbel),
            expected_category=expected_category,
            expected_one_hot=expected_one_hot,
        )

    def evaluate_discrete_sweep(
        self,
        state_lanes: np.ndarray,
        *,
        spin_threshold_logits: dict[int, float],
        categorical_gumbel: dict[int, Sequence[float]],
    ) -> np.ndarray:
        """Evaluate one supported TT-Lang discrete sweep on backend state lanes.

        Each sampling group reads a pre-group state snapshot and commits all
        supported spin/categorical updates together. Unsupported gaussian lanes
        are preserved by this narrow executor shell.
        """
        current = np.asarray(state_lanes, dtype=np.float32).reshape(-1).copy()
        if current.shape[0] != self.layout.total_lanes:
            raise ValueError(f"expected {self.layout.total_lanes} state lanes, got {current.shape[0]}")

        for group in self.sampling_order:
            group_input = current.copy()
            group_output = current.copy()
            for block_index in group:
                spin_plan = self._spin_plan_by_block.get(block_index)
                if spin_plan is not None:
                    threshold_logit = spin_threshold_logits[block_index]
                    group_output[spin_plan.output_lane] = evaluate_spin_categorical_plan(
                        spin_plan, group_input, threshold_logit
                    )
                    continue

                categorical_plan = self._categorical_plan_by_block.get(block_index)
                if categorical_plan is not None:
                    category = evaluate_categorical_spin_plan(
                        categorical_plan, group_input, categorical_gumbel[block_index]
                    )
                    group_output[np.asarray(categorical_plan.output_lanes, dtype=np.int32)] = 0.0
                    group_output[categorical_plan.output_lanes[category]] = 1.0

            current = group_output

        return current


def _as_spec(block_or_spec: CompiledFusedBlock | FusedBlockSpec) -> FusedBlockSpec:
    return block_or_spec.spec if isinstance(block_or_spec, CompiledFusedBlock) else block_or_spec


def build_ttlang_compiled_blocks(program: BlockSamplingProgram) -> tuple[CompiledFusedBlock, ...]:
    """Build THRML block specs for TT-Lang planning without compiling TT-MLIR."""
    from .compiler import _build_fused_block_spec, _build_global_state_layout, _infer_family

    scalar_layout = _build_global_state_layout(program.gibbs_spec)
    specs = []
    n_free = len(program.gibbs_spec.free_blocks)
    for block_index, block in enumerate(program.gibbs_spec.blocks):
        family = _infer_family(block, program.gibbs_spec)
        if block_index < n_free:
            spec = _build_fused_block_spec(
                program,
                block_index,
                block,
                family,
                scalar_layout.block_starts[block_index],
                scalar_layout.total_nodes,
                scalar_layout,
            )
        else:
            spec = FusedBlockSpec(
                block_index=block_index,
                family=family,
                n_nodes=len(block.nodes),
                n_categories=None,
                block_global_start=scalar_layout.block_starts[block_index],
                total_nodes=scalar_layout.total_nodes,
                interactions=(),
            )
        specs.append(CompiledFusedBlock(spec=spec, kernel_artifact=None))
    return tuple(specs)


def build_ttlang_state_layout(blocks: Sequence[CompiledFusedBlock | FusedBlockSpec]) -> TTLangStateLayout:
    """Build the family-specific logical TT-Lang state layout for compiled blocks."""
    layouts: list[TTLangBlockLayout] = []
    scalar_to_lane: list[int | None] = []
    lane_start = 0

    for block_or_spec in blocks:
        spec = _as_spec(block_or_spec)
        lanes_per_node = spec.n_categories if spec.family == Family.CATEGORICAL else 1
        if lanes_per_node is None:
            raise ValueError(f"categorical block {spec.block_index} is missing n_categories")

        block_layout = TTLangBlockLayout(
            block_index=spec.block_index,
            family=spec.family,
            n_nodes=spec.n_nodes,
            n_categories=spec.n_categories,
            scalar_start=spec.block_global_start,
            lane_start=lane_start,
            lanes_per_node=int(lanes_per_node),
        )
        layouts.append(block_layout)

        needed = spec.block_global_start + spec.n_nodes
        if len(scalar_to_lane) < needed:
            scalar_to_lane.extend([None] * (needed - len(scalar_to_lane)))
        for node_offset in range(spec.n_nodes):
            scalar_to_lane[spec.block_global_start + node_offset] = (
                lane_start + node_offset * block_layout.lanes_per_node
            )

        lane_start += block_layout.lane_count

    if any(value is None for value in scalar_to_lane):
        raise ValueError("compiled scalar state has holes; cannot build TT-Lang layout")

    return TTLangStateLayout(
        blocks=tuple(layouts),
        total_lanes=lane_start,
        scalar_to_lane_start=tuple(int(value) for value in scalar_to_lane if value is not None),
    )


def encode_block_state(layout: TTLangBlockLayout, state) -> np.ndarray:
    """Encode one THRML block state into backend lanes."""
    arr = np.asarray(state)
    if layout.family == Family.SPIN:
        return np.where(arr.astype(np.float32) > 0, 1.0, -1.0).reshape(layout.n_nodes)
    if layout.family == Family.CATEGORICAL:
        n_categories = layout.n_categories
        if n_categories is None:
            raise ValueError("categorical layout is missing n_categories")
        ids = arr.astype(np.int64).reshape(layout.n_nodes)
        if np.any((ids < 0) | (ids >= n_categories)):
            raise ValueError(f"categorical state contains ids outside [0, {n_categories})")
        return np.eye(n_categories, dtype=np.float32)[ids].reshape(layout.lane_count)
    return arr.astype(np.float32).reshape(layout.n_nodes)


def decode_block_state(layout: TTLangBlockLayout, lanes: np.ndarray) -> np.ndarray:
    """Decode one backend lane chunk back into THRML's host-facing state."""
    chunk = np.asarray(lanes, dtype=np.float32)
    if layout.family == Family.SPIN:
        return (chunk.reshape(layout.n_nodes) > 0).astype(bool)
    if layout.family == Family.CATEGORICAL:
        return chunk.reshape(layout.n_nodes, layout.lanes_per_node).argmax(axis=1).astype(np.int32)
    return chunk.reshape(layout.n_nodes).astype(np.float32)


def encode_state(layout: TTLangStateLayout, block_states: Sequence[object]) -> np.ndarray:
    """Encode all block states into one TT-Lang backend state vector."""
    if len(block_states) != len(layout.blocks):
        raise ValueError(f"expected {len(layout.blocks)} block states, got {len(block_states)}")
    lanes = np.empty(layout.total_lanes, dtype=np.float32)
    for block_layout, state in zip(layout.blocks, block_states, strict=True):
        start = block_layout.lane_start
        lanes[start : start + block_layout.lane_count] = encode_block_state(block_layout, state)
    return lanes


def decode_state(layout: TTLangStateLayout, lanes: np.ndarray) -> list[np.ndarray]:
    """Decode a complete TT-Lang backend state vector into THRML block states."""
    flat = np.asarray(lanes, dtype=np.float32).reshape(-1)
    if flat.shape[0] != layout.total_lanes:
        raise ValueError(f"expected {layout.total_lanes} lanes, got {flat.shape[0]}")
    result = []
    for block_layout in layout.blocks:
        start = block_layout.lane_start
        result.append(decode_block_state(block_layout, flat[start : start + block_layout.lane_count]))
    return result


def categorical_source_lanes(
    layout: TTLangStateLayout, scalar_global_indices: Iterable[int], n_categories: int
) -> np.ndarray:
    """Return lane indices for categorical source nodes as (n_sources, n_categories)."""
    starts = [layout.node_lane_start(int(index)) for index in scalar_global_indices]
    offsets = np.arange(n_categories, dtype=np.int32)
    return np.asarray(starts, dtype=np.int32)[:, None] + offsets[None, :]


def expand_categorical_gather_lanes(
    layout: TTLangStateLayout,
    scalar_gather_indices: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """Expand scalar categorical gather indices to one-hot category lane indices.

    The current TT-MLIR lowering gathers a scalar category id and then uses it
    to select a weight slice. TT-Lang kernels should instead gather all category
    lanes and compute `one_hot * weights` directly.
    """
    gather = np.asarray(scalar_gather_indices, dtype=np.int32)
    expanded = categorical_source_lanes(layout, gather.reshape(-1), n_categories)
    return expanded.reshape(gather.shape + (n_categories,))


def build_spin_categorical_plan(layout: TTLangStateLayout, spec: FusedBlockSpec) -> TTLangSpinCategoricalPlan:
    """Lower a simple spin block to a TT-Lang categorical-source plan.

    This intentionally accepts only the first production shape proven by the
    TT-Lang spikes: one spin node, scalar/bias terms, and categorical source
    terms represented as category planes. Unsupported interaction structure
    raises clearly so callers can fall back to the existing TT-MLIR backend.
    """
    if spec.family != Family.SPIN:
        raise ValueError(f"expected spin block, got {spec.family.value}")
    if spec.n_nodes != 1:
        raise ValueError("TT-Lang spin categorical plan currently supports one-node spin blocks")

    bias = 0.0
    lane_groups: list[tuple[int, ...]] = []
    weight_groups: list[tuple[float, ...]] = []

    for interaction in spec.interactions:
        weighted_mask = np.asarray(interaction.weighted_mask, dtype=np.float32)
        if interaction.n_spin:
            raise ValueError("spin source interactions are not yet supported by this TT-Lang plan")
        if interaction.n_categorical == 0:
            bias += float(weighted_mask.sum())
            continue
        if interaction.n_categorical != 1:
            raise ValueError("only one categorical source per interaction is currently supported")
        if weighted_mask.shape[0] != 1 or weighted_mask.shape[1] != interaction.n_terms:
            raise ValueError("unexpected spin categorical weight shape")
        if len(interaction.gather_indices) != 1:
            raise ValueError("categorical interaction must have exactly one gather index array")

        n_categories = int(weighted_mask.shape[2])
        gather_indices = np.asarray(interaction.gather_indices[0], dtype=np.int32)
        expanded_lanes = expand_categorical_gather_lanes(layout, gather_indices, n_categories)
        weights = weighted_mask.reshape(interaction.n_terms, n_categories)
        lanes = expanded_lanes.reshape(interaction.n_terms, n_categories)
        for term_lanes, term_weights in zip(lanes, weights, strict=True):
            lane_groups.append(tuple(int(v) for v in term_lanes))
            weight_groups.append(tuple(float(v) for v in term_weights))

    output_lane = layout.node_lane_start(spec.block_global_start)
    return TTLangSpinCategoricalPlan(
        block_index=spec.block_index,
        output_lane=output_lane,
        bias=bias,
        categorical_lane_groups=tuple(lane_groups),
        categorical_weights=tuple(weight_groups),
    )


def build_categorical_spin_plan(layout: TTLangStateLayout, spec: FusedBlockSpec) -> TTLangCategoricalSpinPlan:
    """Lower a simple categorical block with spin sources to a TT-Lang plan."""
    if spec.family != Family.CATEGORICAL:
        raise ValueError(f"expected categorical block, got {spec.family.value}")
    if spec.n_nodes != 1:
        raise ValueError("TT-Lang categorical spin plan currently supports one-node categorical blocks")
    n_categories = spec.n_categories
    if n_categories is None:
        raise ValueError("categorical block is missing n_categories")

    bias = np.zeros((n_categories,), dtype=np.float32)
    spin_lanes: list[int] = []
    spin_weights: list[tuple[float, ...]] = []

    for interaction in spec.interactions:
        weighted_mask = np.asarray(interaction.weighted_mask, dtype=np.float32)
        if interaction.n_categorical:
            raise ValueError("categorical source interactions are not yet supported by this TT-Lang plan")
        weights = weighted_mask.reshape(interaction.n_terms, n_categories)
        if interaction.n_spin == 0:
            bias += weights.sum(axis=0)
            continue
        if interaction.n_spin != 1:
            raise ValueError("only one spin source per interaction is currently supported")
        if len(interaction.gather_indices) != 1:
            raise ValueError("spin interaction must have exactly one gather index array")

        gather_indices = np.asarray(interaction.gather_indices[0], dtype=np.int32).reshape(interaction.n_terms)
        for scalar_index, term_weights in zip(gather_indices, weights, strict=True):
            spin_lanes.append(layout.node_lane_start(int(scalar_index)))
            spin_weights.append(tuple(float(v) for v in term_weights))

    output_start = layout.node_lane_start(spec.block_global_start)
    return TTLangCategoricalSpinPlan(
        block_index=spec.block_index,
        output_lanes=tuple(output_start + category for category in range(n_categories)),
        bias=tuple(float(v) for v in bias),
        spin_lanes=tuple(spin_lanes),
        spin_weights=tuple(spin_weights),
    )


def evaluate_spin_categorical_plan(
    plan: TTLangSpinCategoricalPlan, state_lanes: np.ndarray, threshold_logit: float
) -> float:
    """Reference evaluator for the first TT-Lang spin categorical plan."""
    lanes = np.asarray(state_lanes, dtype=np.float32).reshape(-1)
    gamma = plan.bias
    for lane_group, weight_group in zip(plan.categorical_lane_groups, plan.categorical_weights, strict=True):
        gamma += float(
            np.dot(lanes[np.asarray(lane_group, dtype=np.int32)], np.asarray(weight_group, dtype=np.float32))
        )
    return 1.0 if (2.0 * gamma) > float(threshold_logit) else -1.0


def evaluate_categorical_spin_plan(
    plan: TTLangCategoricalSpinPlan, state_lanes: np.ndarray, gumbel: Sequence[float]
) -> int:
    """Reference evaluator for the first TT-Lang categorical/spin plan."""
    lanes = np.asarray(state_lanes, dtype=np.float32).reshape(-1)
    scores = np.asarray(plan.bias, dtype=np.float32)
    for spin_lane, weight_group in zip(plan.spin_lanes, plan.spin_weights, strict=True):
        scores = scores + lanes[spin_lane] * np.asarray(weight_group, dtype=np.float32)
    perturbed = scores + np.asarray(gumbel, dtype=np.float32)
    return int(perturbed.argmax())
