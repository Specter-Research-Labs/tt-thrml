from __future__ import annotations

import numpy as np
import pytest

from tt_thrml.core import Family
from tt_thrml.example_programs import make_mixed_spin_categorical_gaussian_program
from tt_thrml.ttlang_backend import (
    TTLangProgramPlanner,
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
    make_chain_sweep_randomness_from_key,
    make_sweep_randomness_from_key,
    make_sweep_randomness_window_from_key,
)
from tt_thrml.ttlang_runtime import (
    _can_fuse_groups,
    _make_runtime_groups,
    supports_ttlang_discrete_runtime,
    validate_ttlang_discrete_runtime,
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


def test_ttlang_program_planner_materializes_supported_spin_plan():
    executor = TTLangProgramPlanner(make_mixed_spin_categorical_gaussian_program())

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


def test_ttlang_program_planner_materializes_supported_categorical_plan():
    executor = TTLangProgramPlanner(make_mixed_spin_categorical_gaussian_program())

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


def test_ttlang_program_planner_evaluates_supported_discrete_sweep_by_group():
    executor = TTLangProgramPlanner(make_mixed_spin_categorical_gaussian_program())
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


def test_ttlang_program_planner_evaluates_repeated_supported_discrete_sweeps():
    executor = TTLangProgramPlanner(make_mixed_spin_categorical_gaussian_program())
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

    sweep_kwargs = {
        "spin_threshold_logits": {0: 0.0, 3: 0.0},
        "categorical_gumbel": {1: (0.0, 0.0, 0.0), 4: (0.0, 0.0, 0.0)},
    }
    expected = state_lanes
    for _ in range(4):
        expected = executor.evaluate_discrete_sweep(expected, **sweep_kwargs)

    np.testing.assert_array_equal(
        executor.evaluate_discrete_sweeps(state_lanes, 4, **sweep_kwargs),
        expected,
    )
    np.testing.assert_array_equal(
        executor.evaluate_discrete_sweeps(state_lanes, 0, **sweep_kwargs),
        state_lanes,
    )


def test_ttlang_sweep_randomness_follows_thrml_key_schedule():
    import jax
    import jax.numpy as jnp

    executor = TTLangProgramPlanner(make_mixed_spin_categorical_gaussian_program())
    key = jax.random.PRNGKey(17)

    randomness = make_sweep_randomness_from_key(executor, key)
    block_keys = jax.random.split(key, (len(executor.program.gibbs_spec.free_blocks),))

    spin_key = jax.random.split(block_keys[0], 2)[0]
    spin_uniform = jax.random.uniform(spin_key, (), dtype=jnp.float32)
    assert randomness.spin_threshold_logits[0] == pytest.approx(float(jnp.log(spin_uniform) - jnp.log1p(-spin_uniform)))

    categorical_key = jax.random.split(block_keys[1], 2)[0]
    categorical_gumbel = jax.random.gumbel(categorical_key, (3,), dtype=jnp.float32)
    np.testing.assert_allclose(randomness.categorical_gumbel[1], np.asarray(categorical_gumbel))

    chain_randomness = make_chain_sweep_randomness_from_key(executor, key, sweep_index=2, n_sweeps=5)
    direct_randomness = make_sweep_randomness_from_key(executor, jax.random.split(key, 5)[2])
    assert chain_randomness == direct_randomness


def test_ttlang_sweep_randomness_window_matches_thrml_chain_schedule():
    import jax

    executor = TTLangProgramPlanner(make_mixed_spin_categorical_gaussian_program())
    key = jax.random.PRNGKey(23)
    n_sweeps = 5

    window = make_sweep_randomness_window_from_key(executor, key, n_sweeps)

    assert window.n_sweeps == n_sweeps
    for sweep_index in range(n_sweeps):
        assert window.sweep(sweep_index) == make_chain_sweep_randomness_from_key(
            executor, key, sweep_index=sweep_index, n_sweeps=n_sweeps
        )


def test_ttlang_reference_sweeps_can_consume_randomness_window():
    import jax

    executor = TTLangProgramPlanner(make_mixed_spin_categorical_gaussian_program())
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
    window = executor.sweep_randomness_window_from_key(jax.random.PRNGKey(29), 4)

    expected = state_lanes
    for sweep_index in range(window.n_sweeps):
        randomness = window.sweep(sweep_index)
        expected = executor.evaluate_discrete_sweep(
            expected,
            spin_threshold_logits=dict(randomness.spin_threshold_logits),
            categorical_gumbel=dict(randomness.categorical_gumbel),
        )

    np.testing.assert_array_equal(
        executor.evaluate_discrete_sweeps_with_randomness_window(state_lanes, window),
        expected,
    )


def test_ttlang_runtime_support_boundary_accepts_only_proven_discrete_shape():
    program = make_mixed_spin_categorical_gaussian_program()
    executor = TTLangProgramPlanner(program)

    assert supports_ttlang_discrete_runtime(program)
    validate_ttlang_discrete_runtime(executor)

    executor.sampling_order = ((0, 1, 2, 3, 4, 5),)
    with pytest.raises(ValueError, match="sampling order"):
        validate_ttlang_discrete_runtime(executor)


def test_ttlang_runtime_supports_more_independent_sampling_groups():
    program = make_mixed_spin_categorical_gaussian_program(n_pairs=3)
    executor = TTLangProgramPlanner(program)

    assert executor.layout.total_lanes == 15
    assert executor.sampling_order == ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert [plan.block_index for plan in executor.spin_categorical_plans] == [0, 3, 6]
    assert [plan.block_index for plan in executor.categorical_spin_plans] == [1, 4, 7]
    assert supports_ttlang_discrete_runtime(program)
    validate_ttlang_discrete_runtime(executor)

    executor.sampling_order = executor.sampling_order[:2]
    with pytest.raises(ValueError, match="omits planned blocks"):
        validate_ttlang_discrete_runtime(executor)


def test_ttlang_runtime_can_fuse_regular_independent_groups():
    program = make_mixed_spin_categorical_gaussian_program(n_pairs=3)
    executor = TTLangProgramPlanner(program)
    spin_plans = {plan.block_index: plan for plan in executor.spin_categorical_plans}
    categorical_plans = {plan.block_index: plan for plan in executor.categorical_spin_plans}
    groups = _make_runtime_groups(executor, spin_plans, categorical_plans)

    assert _can_fuse_groups(groups, spin_plans, categorical_plans)


def test_ttlang_planner_combines_duplicate_discrete_terms():
    program = make_mixed_spin_categorical_gaussian_program(n_pairs=1, n_discrete_terms=2)
    executor = TTLangProgramPlanner(program)

    spin_plan = executor.spin_categorical_plans[0]
    categorical_plan = executor.categorical_spin_plans[0]

    assert spin_plan.categorical_lane_groups == ((1, 2, 3),)
    assert spin_plan.categorical_weights == ((1.100000023841858, -0.5, 0.30000001192092896),)
    assert categorical_plan.spin_lanes == (0,)
    assert categorical_plan.spin_weights == ((1.100000023841858, -0.5, 0.30000001192092896),)
    assert supports_ttlang_discrete_runtime(program)
    validate_ttlang_discrete_runtime(executor)
