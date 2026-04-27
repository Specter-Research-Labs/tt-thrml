"""Conditional samplers not yet in upstream THRML."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from thrml.conditional_samplers import AbstractParametricConditionalSampler


class GaussianConditional(AbstractParametricConditionalSampler):
    """Scalar Gaussian conditional using linear and precision contributions.

    Interaction duck-typing:
    - interaction.weights       -> linear contribution
    - interaction.inverse_weights -> precision contribution (stored as variance;
      precision = reciprocal(inverse_weights) following the convention in
      _lower_via_gaussian_shape from the original interaction lowering)
    """

    def init(self) -> None:
        return None

    def compute_parameters(
        self,
        key,
        interactions,
        active_flags,
        states,
        sampler_state,
        output_sd,
    ):
        del key
        dtype = output_sd.dtype
        linear = jnp.zeros(output_sd.shape, dtype=dtype)
        precision = jnp.zeros(output_sd.shape, dtype=dtype)

        for interaction, active, state_slots in zip(interactions, active_flags, states, strict=True):
            scale = active.astype(dtype)
            multiplicative = [
                jnp.asarray(s).astype(dtype)
                for s in state_slots
                if not jnp.issubdtype(jnp.asarray(s).dtype, jnp.integer)
            ]
            if multiplicative:
                scale = scale * jnp.prod(jnp.stack(multiplicative, axis=-1), axis=-1)

            if hasattr(interaction, "weights"):
                w = jnp.asarray(interaction.weights, dtype=dtype)
                weighted = w * scale
                partial = jnp.sum(weighted, axis=tuple(range(1, weighted.ndim)))
                linear = linear + partial
            elif hasattr(interaction, "inverse_weights"):
                w = jnp.reciprocal(jnp.asarray(interaction.inverse_weights, dtype=dtype))
                weighted = w * scale
                partial = jnp.sum(weighted, axis=tuple(range(1, weighted.ndim)))
                precision = precision + partial

        return (linear, precision), sampler_state

    def sample_given_parameters(
        self,
        key,
        parameters,
        sampler_state,
        output_sd,
    ):
        linear, precision = parameters
        variance = jnp.reciprocal(precision)
        mean = linear * variance
        noise = jax.random.normal(key, output_sd.shape, dtype=output_sd.dtype)
        return (jnp.sqrt(variance) * noise) + mean, sampler_state


__all__ = ["GaussianConditional"]
