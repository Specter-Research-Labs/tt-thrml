from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.pgm import SpinNode

from tests.ttnn_test_utils import FakeTTNN
from tt_thrml.compiler.spin_ops import dense_spin_gamma_op
from tt_thrml.runtime.program_executor import TTProgramExecutor
from tt_thrml.runtime_config import ParameterKernelBackend, SPIN_PARAMETER_FAMILY


pytest.importorskip("torch")


class CountingSpinSubmissionOp:
    def __init__(self):
        self.submissions = 0

    def __call__(self, *, ttnn, device, inputs):
        self.submissions += 1
        return dense_spin_gamma_op(ttnn=ttnn, device=device, inputs=inputs)


def _make_grouped_spin_program():
    nodes = [SpinNode() for _ in range(6)]
    free_blocks = [Block(nodes[0::2]), Block(nodes[1::2])]
    nearest_neighbor = SpinEBMFactor(
        [Block(nodes[:-1]), Block(nodes[1:])],
        jnp.linspace(0.25, -0.15, len(nodes) - 1, dtype=jnp.float32),
    )
    next_neighbor = SpinEBMFactor(
        [Block(nodes[:-2]), Block(nodes[2:])],
        jnp.linspace(-0.05, 0.2, len(nodes) - 2, dtype=jnp.float32),
    )
    program = FactorSamplingProgram(
        BlockGibbsSpec(free_blocks, []),
        [SpinGibbsConditional(), SpinGibbsConditional()],
        [nearest_neighbor, next_neighbor],
        [],
    )
    init_state_free = [
        jnp.asarray([True, False, True], dtype=jnp.bool_),
        jnp.asarray([False, True, False], dtype=jnp.bool_),
    ]
    return program, init_state_free


def test_spin_parameter_kernel_submits_once_per_block_for_grouped_interactions():
    counting_op = CountingSpinSubmissionOp()
    program, init_state_free = _make_grouped_spin_program()
    executor = TTProgramExecutor(
        ttnn=FakeTTNN(),
        device="fake:0",
        program=program,
        parameter_kernel_ops={SPIN_PARAMETER_FAMILY: counting_op},
        parameter_kernel_backends={SPIN_PARAMETER_FAMILY: ParameterKernelBackend.CUSTOM},
    )
    executor.load_state(init_state_free, [])
    executor.run_sweep(jax.random.key(0))

    assert counting_op.submissions == 2
