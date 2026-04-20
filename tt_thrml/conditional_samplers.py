"""TT-local sampler helpers that upstream THRML does not provide."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from thrml.conditional_samplers import AbstractParametricConditionalSampler

from .compiler.gaussian_ops import gather_gaussian_tail_weights_jax
from .compiler.interaction_lowering import (
    lower_interaction_contribution,
)
from .runtime_config import GAUSSIAN_PARAMETER_FAMILY


class GaussianConditional(AbstractParametricConditionalSampler):
    """Scalar Gaussian conditional using linear and precision contributions."""

    def init(self) -> None:
        return None

    def tt_parameter_family(self):
        return GAUSSIAN_PARAMETER_FAMILY

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

        for interaction, active, state in zip(
            interactions,
            active_flags,
            states,
            strict=True,
        ):
            contribution = lower_interaction_contribution(
                interaction,
                parameter_family=GAUSSIAN_PARAMETER_FAMILY,
            )
            weights = jnp.asarray(contribution.weights, dtype=dtype)
            scale = active.astype(dtype)
            multiplicative_states = []
            categorical_states = []
            for slot in state:
                slot_array = jnp.asarray(slot)
                if jnp.issubdtype(slot_array.dtype, jnp.integer):
                    categorical_states.append(slot_array)
                else:
                    multiplicative_states.append(slot_array.astype(dtype))
            weights = gather_gaussian_tail_weights_jax(weights, tuple(categorical_states))
            if multiplicative_states:
                state_prod = jnp.prod(jnp.stack(multiplicative_states, axis=-1), axis=-1)
                scale = scale * state_prod
            weighted = weights * scale
            partial = jnp.sum(weighted, axis=tuple(range(1, weighted.ndim)))
            if contribution.contribution_kind == "linear":
                linear = linear + partial
            elif contribution.contribution_kind == "precision":
                precision = precision + partial
            else:
                raise TypeError(
                    "GaussianConditional only supports linear and precision "
                    f"interaction contributions, got {contribution.contribution_kind!r}."
                )

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
