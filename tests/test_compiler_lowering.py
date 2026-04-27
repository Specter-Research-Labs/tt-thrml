from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.pgm import SpinNode

from tests.parity._parity_runner import _make_mixed_case
from tt_thrml.compiler import (
    _build_fused_block_spec,
    _build_global_state_layout,
    _infer_family,
    _make_fused_kernel,
    _make_sampling_group_kernel,
)


def _spec_for(program, block_index):
    layout = _build_global_state_layout(program.gibbs_spec)
    block = program.gibbs_spec.blocks[block_index]
    return _build_fused_block_spec(
        program,
        block_index,
        block,
        _infer_family(block, program.gibbs_spec),
        layout.block_starts[block_index],
        layout.total_nodes,
        layout,
    )


def test_mixed_program_global_slices_lower_to_flat_offsets():
    program = _make_mixed_case().program

    assert np.asarray(_spec_for(program, 0).interactions[1].gather_indices[0]).tolist() == [[1]]
    assert np.asarray(_spec_for(program, 1).interactions[1].gather_indices[0]).tolist() == [[0]]
    assert np.asarray(_spec_for(program, 2).interactions[2].gather_indices[0]).tolist() == [[5]]
    assert np.asarray(_spec_for(program, 3).interactions[1].gather_indices[0]).tolist() == [[4]]
    assert np.asarray(_spec_for(program, 4).interactions[1].gather_indices[0]).tolist() == [[3]]
    assert np.asarray(_spec_for(program, 5).interactions[2].gather_indices[0]).tolist() == [[2]]


def test_spin_kernel_selects_categorical_source_axis():
    program = _make_mixed_case().program
    spec = _spec_for(program, 0)
    kernel = _make_fused_kernel(spec)
    rng = jnp.asarray([[0.0]], dtype=jnp.float32)

    cat_zero_state = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    cat_one_state = jnp.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    assert np.asarray(kernel(cat_zero_state, rng))[0] == 1.0
    assert np.asarray(kernel(cat_one_state, rng))[0] == -1.0


def test_sampling_group_kernel_updates_blocks_simultaneously():
    nodes = [SpinNode(), SpinNode()]
    free_blocks = (Block([nodes[0]]), Block([nodes[1]]))
    program = FactorSamplingProgram(
        BlockGibbsSpec([free_blocks], []),
        [SpinGibbsConditional(), SpinGibbsConditional()],
        [SpinEBMFactor([Block([nodes[0]]), Block([nodes[1]])], jnp.asarray([-1.0], dtype=jnp.float32))],
        [],
    )
    specs = (_spec_for(program, 0), _spec_for(program, 1))
    kernel = _make_sampling_group_kernel(specs)

    out = kernel(
        jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        jnp.asarray([[0.0]], dtype=jnp.float32),
        jnp.asarray([[0.0]], dtype=jnp.float32),
    )

    np.testing.assert_array_equal(np.asarray(out), np.asarray([1.0, 1.0], dtype=np.float32))
