"""Shared helpers for upstream THRML notebook ports."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Hashable

import equinox as eqx
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec
from thrml.conditional_samplers import AbstractConditionalSampler, _SamplerState, _State
from thrml.factor import AbstractFactor
from thrml.interaction import InteractionGroup
from thrml.models.discrete_ebm import SpinGibbsConditional
from thrml.observers import MomentAccumulatorObserver
from thrml.pgm import AbstractNode

import tt_thrml


class ContinuousNode(AbstractNode):
    pass


class SpinNode(AbstractNode):
    pass


class LinearInteraction(eqx.Module):
    """An interaction of the form c_i x_i."""

    weights: jax.Array
    n_spin: int = 0


class QuadraticInteraction(eqx.Module):
    """An interaction of the form d_i x_i^2."""

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


class GaussianSampler(AbstractConditionalSampler):
    """Notebook-shaped Gaussian sampler with a tiny TT parameter-family hook."""

    def tt_parameter_family(self):
        return tt_thrml.ParameterFamily.GAUSSIAN

    def sample(
        self,
        key,
        interactions,
        active_flags,
        states,
        sampler_state: _SamplerState,
        output_sd,
    ):
        bias = jnp.zeros(shape=output_sd.shape, dtype=output_sd.dtype)
        var = jnp.zeros(shape=output_sd.shape, dtype=output_sd.dtype)

        for active, interaction, state in zip(
            active_flags,
            interactions,
            states,
            strict=True,
        ):
            if isinstance(interaction, LinearInteraction):
                state_prod = jnp.array(1.0, dtype=output_sd.dtype)
                if len(state) > 0:
                    state_prod = jnp.prod(jnp.stack(state, axis=-1), axis=-1)
                bias -= jnp.sum(interaction.weights * active * state_prod, axis=-1)

            if isinstance(interaction, QuadraticInteraction):
                var = active * interaction.inverse_weights
                var = var[..., 0]

        sample = (jnp.sqrt(var) * jax.random.normal(key, output_sd.shape)) + (bias * var)
        return sample, sampler_state

    def init(self) -> _SamplerState:
        return None


class ExtendedSpinGibbsSampler(SpinGibbsConditional):
    """Notebook-shaped custom spin sampler.

    TT execution still runs through the compiled spin family path; the custom
    linear interaction is expressed through the interaction metadata.
    """

    def compute_parameters(
        self,
        key,
        interactions,
        active_flags,
        states,
        sampler_state: _SamplerState,
        output_sd,
    ):
        field = jnp.zeros(output_sd.shape, dtype=float)

        unprocessed_interactions = []
        unprocessed_active = []
        unprocessed_states = []

        for interaction, active, state in zip(
            interactions,
            active_flags,
            states,
            strict=True,
        ):
            if isinstance(interaction, LinearInteraction):
                state_prod = jnp.prod(jnp.stack(state, axis=-1), axis=-1)
                field -= jnp.sum(interaction.weights * active * state_prod, axis=-1)
            else:
                unprocessed_interactions.append(interaction)
                unprocessed_active.append(active)
                unprocessed_states.append(state)

        field -= super().compute_parameters(
            key,
            unprocessed_interactions,
            unprocessed_active,
            unprocessed_states,
            sampler_state,
            output_sd,
        )[0]

        return field, sampler_state


def generate_continuous_grid_graph(
    *side_lengths: int,
) -> tuple[tuple[dict[Hashable, int], list[ContinuousNode], list[ContinuousNode]], tuple[list[ContinuousNode], list[ContinuousNode]], nx.Graph]:
    graph = nx.grid_graph(dim=side_lengths, periodic=False)

    coord_to_node = {coord: ContinuousNode() for coord in graph.nodes}
    nx.relabel_nodes(graph, coord_to_node, copy=False)

    for coord, node in coord_to_node.items():
        graph.nodes[node]["coords"] = coord

    bicol = nx.bipartite.color(graph)
    color0 = [node for node, color in bicol.items() if color == 0]
    color1 = [node for node, color in bicol.items() if color == 1]

    u_nodes, v_nodes = map(list, zip(*graph.edges()))
    return (bicol, color0, color1), (u_nodes, v_nodes), graph


def make_random_typed_grid(
    *,
    rows: int,
    cols: int,
    seed: int,
    p_cont: float = 0.5,
):
    rng = random.Random(seed)

    grid = [
        [ContinuousNode() if rng.random() < p_cont else SpinNode() for _ in range(cols)]
        for _ in range(rows)
    ]
    bicol = {grid[r][c]: ((r + c) & 1) for r in range(rows) for c in range(cols)}

    colors_by_type = {
        0: {SpinNode: [], ContinuousNode: []},
        1: {SpinNode: [], ContinuousNode: []},
    }
    for r in range(rows):
        for c in range(cols):
            node = grid[r][c]
            colors_by_type[bicol[node]][type(node)].append(node)

    return grid, colors_by_type


def build_skip_graph_from_grid(
    grid: list[list[AbstractNode]],
    skips: Sequence[int],
):
    rows, cols = len(grid), len(grid[0])
    graph = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            node = grid[r][c]
            graph.add_node(node, coords=(r, c))

    u_all: list[AbstractNode] = []
    v_all: list[AbstractNode] = []
    for skip in skips:
        for r in range(rows - skip):
            r2 = r + skip
            for c in range(cols):
                n1 = grid[r][c]
                n2 = grid[r2][c]
                u_all.append(n1)
                v_all.append(n2)
                graph.add_edge(n1, n2, skip=skip)

        for r in range(rows):
            for c in range(cols - skip):
                c2 = c + skip
                n1 = grid[r][c]
                n2 = grid[r][c2]
                u_all.append(n1)
                v_all.append(n2)
                graph.add_edge(n1, n2, skip=skip)

    return (u_all, v_all), graph


def partition_edges_by_type(
    edges: tuple[list[AbstractNode], list[AbstractNode]],
):
    spin_nodes: list[SpinNode] = []
    cont_nodes: list[ContinuousNode] = []
    for node in {*(edges[0]), *(edges[1])}:
        if isinstance(node, SpinNode):
            spin_nodes.append(node)
        else:
            cont_nodes.append(node)

    ss_edges = [[], []]
    cc_edges = [[], []]
    sc_edges = [[], []]
    for edge in zip(*edges, strict=True):
        lhs, rhs = edge
        if isinstance(lhs, SpinNode) and isinstance(rhs, SpinNode):
            ss_edges[0].append(lhs)
            ss_edges[1].append(rhs)
        elif isinstance(lhs, ContinuousNode) and isinstance(rhs, ContinuousNode):
            cc_edges[0].append(lhs)
            cc_edges[1].append(rhs)
        elif isinstance(lhs, SpinNode):
            sc_edges[0].append(lhs)
            sc_edges[1].append(rhs)
        else:
            sc_edges[0].append(rhs)
            sc_edges[1].append(lhs)

    return spin_nodes, cont_nodes, tuple(ss_edges), tuple(cc_edges), tuple(sc_edges)


def continuous_moment_observer(
    *,
    edges: tuple[list[ContinuousNode], list[ContinuousNode]],
):
    second_moments = [(e1, e2) for e1, e2 in zip(*edges, strict=True)]
    first_moments = [[(node,) for node in endpoint] for endpoint in edges]
    return MomentAccumulatorObserver(first_moments + [second_moments])


def make_mixed_block_spec(
    *,
    coloring_by_type: Mapping[int, Mapping[type[AbstractNode], Sequence[AbstractNode]]],
):
    free_super_blocks = [
        (
            Block(coloring_by_type[0][SpinNode]),
            Block(coloring_by_type[0][ContinuousNode]),
        ),
        (
            Block(coloring_by_type[1][SpinNode]),
            Block(coloring_by_type[1][ContinuousNode]),
        ),
    ]
    node_shape_dtypes = {
        SpinNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.bool),
        ContinuousNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    }
    return BlockGibbsSpec(free_super_blocks, [], node_shape_dtypes)


def make_continuous_block_spec(
    *,
    color0: Sequence[ContinuousNode],
    color1: Sequence[ContinuousNode],
):
    node_shape_dtypes = {ContinuousNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)}
    return BlockGibbsSpec([Block(color0), Block(color1)], [], node_shape_dtypes)


def construct_inv_cov(
    *,
    diag,
    all_edges: tuple[list[ContinuousNode], list[ContinuousNode]],
    off_diag,
    node_map: Mapping[ContinuousNode, int],
):
    inv_cov = np.diag(np.asarray(diag))
    for lhs, rhs, cov in zip(*all_edges, off_diag, strict=True):
        inv_cov[node_map[lhs], node_map[rhs]] = cov
        inv_cov[node_map[rhs], node_map[lhs]] = cov
    return inv_cov


__all__ = [
    "ContinuousNode",
    "CouplingFactor",
    "ExtendedSpinGibbsSampler",
    "GaussianSampler",
    "LinearFactor",
    "LinearInteraction",
    "QuadraticFactor",
    "QuadraticInteraction",
    "SpinNode",
    "build_skip_graph_from_grid",
    "construct_inv_cov",
    "continuous_moment_observer",
    "generate_continuous_grid_graph",
    "make_continuous_block_spec",
    "make_mixed_block_spec",
    "make_random_typed_grid",
    "partition_edges_by_type",
]
