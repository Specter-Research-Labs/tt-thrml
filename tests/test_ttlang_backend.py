from __future__ import annotations

import numpy as np

from tt_thrml.core import Family
from tt_thrml.example_programs import make_mixed_spin_categorical_gaussian_program
from tt_thrml.ttlang_backend import (
    ExperimentalTTLangExecutor,
    build_categorical_spin_plan,
    build_spin_categorical_plan,
    build_ttlang_compiled_blocks,
    build_ttlang_state_layout,
    categorical_source_lanes,
    decode_state,
    encode_state,
    evaluate_categorical_spin_plan,
    evaluate_spin_categorical_plan,
    expand_categorical_gather_lanes,
)


def test_ttlang_state_layout_expands_categorical_blocks_to_one_hot_lanes():
    compiled_blocks = build_ttlang_compiled_blocks(make_mixed_spin_categorical_gaussian_program())
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
    compiled_blocks = build_ttlang_compiled_blocks(make_mixed_spin_categorical_gaussian_program())
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
    compiled_blocks = build_ttlang_compiled_blocks(make_mixed_spin_categorical_gaussian_program())
    layout = build_ttlang_state_layout(compiled_blocks)

    np.testing.assert_array_equal(categorical_source_lanes(layout, [1, 4], 3), np.asarray([[1, 2, 3], [6, 7, 8]]))


def test_ttlang_expands_mixed_program_categorical_gather_to_one_hot_lanes():
    compiled_blocks = build_ttlang_compiled_blocks(make_mixed_spin_categorical_gaussian_program())
    layout = build_ttlang_state_layout(compiled_blocks)

    first_spin = compiled_blocks[0].spec
    categorical_interaction = first_spin.interactions[1]

    assert categorical_interaction.n_categorical == 1
    np.testing.assert_array_equal(
        expand_categorical_gather_lanes(
            layout, np.asarray(categorical_interaction.gather_indices[0], dtype=np.int32), 3
        ),
        np.asarray([[[1, 2, 3]]], dtype=np.int32),
    )


def test_ttlang_builds_spin_categorical_plan_from_mixed_program_spec():
    compiled_blocks = build_ttlang_compiled_blocks(make_mixed_spin_categorical_gaussian_program())
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


def test_experimental_ttlang_executor_materializes_supported_spin_plan():
    executor = ExperimentalTTLangExecutor(make_mixed_spin_categorical_gaussian_program())

    assert len(executor.spin_categorical_plans) == 2

    run = executor.materialize_spin_categorical_run(
        0,
        [
            np.asarray([True]),
            np.asarray([1], dtype=np.uint8),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            np.asarray([1], dtype=np.uint8),
            np.asarray([0.0], dtype=np.float32),
        ],
        threshold_logit=0.0,
    )

    assert run.plan.block_index == 0
    assert run.categorical_values == ((0.0, 1.0, 0.0),)
    assert run.threshold_logit == 0.0
    assert run.expected_spin == -1.0


def test_ttlang_builds_categorical_spin_plan_from_mixed_program_spec():
    compiled_blocks = build_ttlang_compiled_blocks(make_mixed_spin_categorical_gaussian_program())
    layout = build_ttlang_state_layout(compiled_blocks)
    plan = build_categorical_spin_plan(layout, compiled_blocks[1].spec)

    assert plan.block_index == 1
    assert plan.output_lanes == (1, 2, 3)
    np.testing.assert_allclose(plan.bias, (0.2, -0.1, 0.0))
    assert plan.spin_lanes == (0,)
    np.testing.assert_allclose(plan.spin_weights, ((0.55, -0.25, 0.15),))

    state_lanes = encode_state(
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

    assert evaluate_categorical_spin_plan(plan, state_lanes, gumbel=(0.0, 0.0, 0.0)) == 0
    assert evaluate_categorical_spin_plan(plan, state_lanes, gumbel=(-2.0, 2.0, 0.0)) == 1


def test_experimental_ttlang_executor_materializes_supported_categorical_plan():
    executor = ExperimentalTTLangExecutor(make_mixed_spin_categorical_gaussian_program())

    assert len(executor.categorical_spin_plans) == 2

    run = executor.materialize_categorical_spin_run(
        0,
        [
            np.asarray([True]),
            np.asarray([0], dtype=np.uint8),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False]),
            np.asarray([1], dtype=np.uint8),
            np.asarray([0.0], dtype=np.float32),
        ],
        gumbel=(0.0, 0.0, 0.0),
    )

    assert run.plan.block_index == 1
    assert run.spin_values == (1.0,)
    assert run.expected_category == 0
    assert run.expected_one_hot == (1.0, 0.0, 0.0)


def test_experimental_ttlang_executor_evaluates_supported_discrete_sweep_by_group():
    executor = ExperimentalTTLangExecutor(make_mixed_spin_categorical_gaussian_program())
    state_lanes = executor.encode_state(
        [
            np.asarray([True]),
            np.asarray([1], dtype=np.uint8),
            np.asarray([0.25], dtype=np.float32),
            np.asarray([False]),
            np.asarray([1], dtype=np.uint8),
            np.asarray([-0.75], dtype=np.float32),
        ]
    )

    next_lanes = executor.evaluate_discrete_sweep(
        state_lanes,
        spin_threshold_logits={0: 0.0, 3: 0.0},
        categorical_gumbel={1: (0.0, 0.0, 0.0), 4: (0.0, 0.0, 0.0)},
    )

    np.testing.assert_array_equal(
        next_lanes,
        np.asarray([-1.0, 1.0, 0.0, 0.0, 0.25, 1.0, 1.0, 0.0, 0.0, -0.75], dtype=np.float32),
    )
    decoded = decode_state(executor.layout, next_lanes)
    assert decoded[0].tolist() == [False]
    assert decoded[1].tolist() == [0]
    np.testing.assert_allclose(decoded[2], [0.25])
    assert decoded[3].tolist() == [True]
    assert decoded[4].tolist() == [0]
    np.testing.assert_allclose(decoded[5], [-0.75])
