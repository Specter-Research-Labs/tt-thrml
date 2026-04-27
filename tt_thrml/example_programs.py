"""Small THRML programs used by backend smoke runners."""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    DiscreteEBMFactor,
    SpinEBMFactor,
    SpinGibbsConditional,
)
from thrml.pgm import AbstractNode, CategoricalNode, SpinNode

from tt_thrml.conditional_samplers import GaussianConditional


class ContinuousNode(AbstractNode):
    pass


class LinearInteraction(eqx.Module):
    weights: jax.Array


class QuadraticInteraction(eqx.Module):
    inverse_weights: jax.Array


class LinearFactor(AbstractFactor):
    weights: jax.Array

    def __init__(self, weights: jax.Array, block: Block):
        super().__init__([block])
        self.weights = weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[],
            )
        ]


class QuadraticFactor(AbstractFactor):
    inverse_weights: jax.Array

    def __init__(self, inverse_weights: jax.Array, block: Block):
        super().__init__([block])
        self.inverse_weights = inverse_weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=QuadraticInteraction(self.inverse_weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[],
            )
        ]


class CouplingFactor(AbstractFactor):
    weights: jax.Array

    def __init__(self, weights: jax.Array, blocks: tuple[Block, Block]):
        super().__init__(list(blocks))
        self.weights = weights

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[0],
                tail_nodes=[self.node_groups[1]],
            ),
            InteractionGroup(
                interaction=LinearInteraction(self.weights),
                head_nodes=self.node_groups[1],
                tail_nodes=[self.node_groups[0]],
            ),
        ]


def make_mixed_spin_categorical_gaussian_program(*, n_categories: int = 3) -> FactorSamplingProgram:
    """Build the mixed smoke program shared by parity tests and TT-Lang runners."""
    spin_nodes = [SpinNode() for _ in range(2)]
    categorical_nodes = [CategoricalNode() for _ in range(2)]
    continuous_nodes = [ContinuousNode() for _ in range(2)]
    free_super_blocks = [
        (
            Block([spin_nodes[0]]),
            Block([categorical_nodes[0]]),
            Block([continuous_nodes[0]]),
        ),
        (
            Block([spin_nodes[1]]),
            Block([categorical_nodes[1]]),
            Block([continuous_nodes[1]]),
        ),
    ]
    node_shape_dtypes = cast(
        Any,
        {
            SpinNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool_),
            CategoricalNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8),
            ContinuousNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        },
    )
    return FactorSamplingProgram(
        BlockGibbsSpec(free_super_blocks, [], node_shape_dtypes),
        [
            SpinGibbsConditional(),
            CategoricalGibbsConditional(n_categories),
            GaussianConditional(),
            SpinGibbsConditional(),
            CategoricalGibbsConditional(n_categories),
            GaussianConditional(),
        ],
        [
            SpinEBMFactor(
                [Block(spin_nodes)],
                jnp.asarray([0.25, -0.2], dtype=jnp.float32),
            ),
            CategoricalEBMFactor(
                [Block(categorical_nodes)],
                jnp.asarray(
                    [[0.2, -0.1, 0.0], [-0.15, 0.3, -0.05]],
                    dtype=jnp.float32,
                ),
            ),
            DiscreteEBMFactor(
                [Block(spin_nodes)],
                [Block(categorical_nodes)],
                jnp.asarray(
                    [[0.55, -0.25, 0.15], [-0.2, 0.4, 0.1]],
                    dtype=jnp.float32,
                ),
            ),
            LinearFactor(
                jnp.asarray([0.1, -0.2], dtype=jnp.float32),
                Block(continuous_nodes),
            ),
            QuadraticFactor(
                jnp.asarray([0.85, 0.75], dtype=jnp.float32),
                Block(continuous_nodes),
            ),
            CouplingFactor(
                jnp.asarray([0.11], dtype=jnp.float32),
                (Block([continuous_nodes[0]]), Block([continuous_nodes[1]])),
            ),
        ],
        [],
    )
