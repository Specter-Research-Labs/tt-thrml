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


def make_mixed_spin_categorical_gaussian_program(
    *, n_categories: int = 3, n_pairs: int = 2, n_discrete_terms: int = 1
) -> FactorSamplingProgram:
    """Build the mixed smoke program shared by parity tests and TT-Lang runners."""
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if n_discrete_terms <= 0:
        raise ValueError("n_discrete_terms must be positive")

    spin_nodes = [SpinNode() for _ in range(n_pairs)]
    categorical_nodes = [CategoricalNode() for _ in range(n_pairs)]
    continuous_nodes = [ContinuousNode() for _ in range(n_pairs)]
    free_super_blocks = [
        (
            Block([spin_nodes[i]]),
            Block([categorical_nodes[i]]),
            Block([continuous_nodes[i]]),
        )
        for i in range(n_pairs)
    ]
    node_shape_dtypes = cast(
        Any,
        {
            SpinNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool_),
            CategoricalNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8),
            ContinuousNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
        },
    )
    base_spin_bias = jnp.asarray([0.25, -0.2], dtype=jnp.float32)
    base_categorical_bias = jnp.asarray([[0.2, -0.1, 0.0], [-0.15, 0.3, -0.05]], dtype=jnp.float32)
    base_discrete_weights = jnp.asarray([[0.55, -0.25, 0.15], [-0.2, 0.4, 0.1]], dtype=jnp.float32)
    base_linear_weights = jnp.asarray([0.1, -0.2], dtype=jnp.float32)
    base_inverse_weights = jnp.asarray([0.85, 0.75], dtype=jnp.float32)
    repeats = (n_pairs + 1) // 2
    spin_bias = jnp.tile(base_spin_bias, repeats)[:n_pairs]
    categorical_bias = jnp.tile(base_categorical_bias, (repeats, 1))[:n_pairs, :n_categories]
    discrete_weights = jnp.tile(base_discrete_weights, (repeats, 1))[:n_pairs, :n_categories]
    linear_weights = jnp.tile(base_linear_weights, repeats)[:n_pairs]
    inverse_weights = jnp.tile(base_inverse_weights, repeats)[:n_pairs]
    factors: list[AbstractFactor] = [
        SpinEBMFactor(
            [Block(spin_nodes)],
            spin_bias,
        ),
        CategoricalEBMFactor(
            [Block(categorical_nodes)],
            categorical_bias,
        ),
        LinearFactor(
            linear_weights,
            Block(continuous_nodes),
        ),
        QuadraticFactor(
            inverse_weights,
            Block(continuous_nodes),
        ),
    ]
    for _ in range(n_discrete_terms):
        factors.append(
            DiscreteEBMFactor(
                [Block(spin_nodes)],
                [Block(categorical_nodes)],
                discrete_weights,
            )
        )
    if n_pairs >= 2:
        factors.append(
            CouplingFactor(
                jnp.asarray([0.11], dtype=jnp.float32),
                (Block([continuous_nodes[0]]), Block([continuous_nodes[1]])),
            )
        )
    return FactorSamplingProgram(
        BlockGibbsSpec(free_super_blocks, [], node_shape_dtypes),
        [
            conditional
            for _ in range(n_pairs)
            for conditional in (
                SpinGibbsConditional(),
                CategoricalGibbsConditional(n_categories),
                GaussianConditional(),
            )
        ],
        factors,
        [],
    )
