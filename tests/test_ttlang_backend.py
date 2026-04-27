from __future__ import annotations

import numpy as np

from tests.parity._parity_runner import _make_mixed_case
from tt_thrml.compiler import _build_fused_block_spec, _build_global_state_layout, _infer_family
from tt_thrml.core import CompiledFusedBlock, Family, FusedBlockSpec
from tt_thrml.ttlang_backend import (
    build_spin_categorical_plan,
    build_ttlang_state_layout,
    categorical_source_lanes,
    decode_state,
    encode_state,
    evaluate_spin_categorical_plan,
    expand_categorical_gather_lanes,
)


def _compiled_specs_for(program):
    scalar_layout = _build_global_state_layout(program.gibbs_spec)
    specs = []
    for block_index, block in enumerate(program.gibbs_spec.blocks):
        family = _infer_family(block, program.gibbs_spec)
        if block_index < len(program.gibbs_spec.free_blocks):
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


def test_ttlang_state_layout_expands_categorical_blocks_to_one_hot_lanes():
    compiled_blocks = _compiled_specs_for(_make_mixed_case().program)
    layout = build_ttlang_state_layout(compiled_blocks)

    assert [block.family for block in layout.blocks] == [
        Family.SPIN,
        Family.CATEGORICAL,
        Family.GAUSSIAN,
        Family.SPIN,
        Family.CATEGORICAL,
        Family.GAUSSIAN,
    ]
    assert [block.lanes_per_node for block in layout.blocks] == [1, 3, 1, 1, 3, 1]
    assert layout.total_lanes == 10
    assert layout.scalar_to_lane_start == (0, 1, 4, 5, 6, 9)


def test_ttlang_state_round_trips_thrml_host_shapes():
    compiled_blocks = _compiled_specs_for(_make_mixed_case().program)
    layout = build_ttlang_state_layout(compiled_blocks)
    states = [
        np.asarray([True]),
        np.asarray([2], dtype=np.uint8),
        np.asarray([0.25], dtype=np.float32),
        np.asarray([False]),
        np.asarray([1], dtype=np.uint8),
        np.asarray([-0.75], dtype=np.float32),
    ]

    lanes = encode_state(layout, states)
    np.testing.assert_array_equal(
        lanes,
        np.asarray([1.0, 0.0, 0.0, 1.0, 0.25, -1.0, 0.0, 1.0, 0.0, -0.75], dtype=np.float32),
    )

    decoded = decode_state(layout, lanes)
    assert decoded[0].tolist() == [True]
    assert decoded[1].tolist() == [2]
    np.testing.assert_allclose(decoded[2], states[2])
    assert decoded[3].tolist() == [False]
    assert decoded[4].tolist() == [1]
    np.testing.assert_allclose(decoded[5], states[5])


def test_ttlang_categorical_source_lanes_address_all_category_planes():
    compiled_blocks = _compiled_specs_for(_make_mixed_case().program)
    layout = build_ttlang_state_layout(compiled_blocks)

    np.testing.assert_array_equal(categorical_source_lanes(layout, [1, 4], 3), np.asarray([[1, 2, 3], [6, 7, 8]]))


def test_ttlang_expands_mixed_program_categorical_gather_to_one_hot_lanes():
    compiled_blocks = _compiled_specs_for(_make_mixed_case().program)
    layout = build_ttlang_state_layout(compiled_blocks)

    first_spin = compiled_blocks[0].spec
    categorical_interaction = first_spin.interactions[1]

    assert categorical_interaction.n_categorical == 1
    np.testing.assert_array_equal(
        expand_categorical_gather_lanes(layout, categorical_interaction.gather_indices[0], 3),
        np.asarray([[[1, 2, 3]]], dtype=np.int32),
    )


def test_ttlang_builds_spin_categorical_plan_from_mixed_program_spec():
    compiled_blocks = _compiled_specs_for(_make_mixed_case().program)
    layout = build_ttlang_state_layout(compiled_blocks)
    plan = build_spin_categorical_plan(layout, compiled_blocks[0].spec)

    assert plan.block_index == 0
    assert plan.output_lane == 0
    assert plan.bias == 0.25
    assert plan.categorical_lane_groups == ((1, 2, 3),)
    np.testing.assert_allclose(plan.categorical_weights, ((0.55, -0.25, 0.15),))

    cat_zero = encode_state(
        layout,
        [
            np.asarray([True]),
            np.asarray([0], dtype=np.uint8),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            np.asarray([1], dtype=np.uint8),
            np.asarray([0.0], dtype=np.float32),
        ],
    )
    cat_one = encode_state(
        layout,
        [
            np.asarray([True]),
            np.asarray([1], dtype=np.uint8),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            np.asarray([1], dtype=np.uint8),
            np.asarray([0.0], dtype=np.float32),
        ],
    )

    assert evaluate_spin_categorical_plan(plan, cat_zero, threshold_logit=0.0) == 1.0
    assert evaluate_spin_categorical_plan(plan, cat_one, threshold_logit=0.0) == -1.0
