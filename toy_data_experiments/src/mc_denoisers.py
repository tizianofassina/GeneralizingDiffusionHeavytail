### This file contains code for the "Monte Carlo" denoisers
from jax import numpy as jnp, nn as jnn, random as jrn, Array, vmap, jit
from typing import Union, Tuple, Dict
from numpyro import distributions as dists
from functools import partial
import blackjax
from jax.lax import scan


def log_joint_density(x_0, pi_ref, x_t, sigma):
    return pi_ref.log_prob(x_0) - (jnp.linalg.vector_norm(x_t - x_0, ord=2) ** 2) / (
        2 * sigma**2
    )


def student_is_denoiser(
    x_t,
    sigma,
    rng: Array,
    pi_ref,
    n_is: int,
    mixture_probability: float = 0.0,
    proposal_df: int = 3,
    debug: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    rng_student, rng_pi_ref, rng_component = jrn.split(rng, 3)
    dim = x_t.shape[-1]
    proposal_dist = dists.StudentT(df=proposal_df, validate_args=False)
    samples_proposal = jnp.where(
        jrn.uniform(rng_component, shape=(n_is, 1)) > mixture_probability,
        proposal_dist.sample(rng_student, sample_shape=(n_is, dim)) * sigma + x_t,
        pi_ref.sample(rng_pi_ref, sample_shape=(n_is,)),
    )

    log_proposal = jnn.logsumexp(
        jnp.stack(
            (
                proposal_dist.log_prob((samples_proposal - x_t[None]) / sigma).sum(
                    axis=-1
                ),
                pi_ref.log_prob(samples_proposal),
            ),
            axis=-1,
        ),
        axis=-1,
    )

    log_target = vmap(partial(log_joint_density, x_t=x_t, sigma=sigma, pi_ref=pi_ref))(
        samples_proposal
    )

    log_weights = log_target - log_proposal
    log_weights = log_weights - jnn.logsumexp(log_weights)

    pred_x0 = (jnp.exp(log_weights)[:, None] * samples_proposal).sum(axis=0)
    if debug:
        return pred_x0, {"log_weights": log_weights, "samples": samples_proposal}
    return pred_x0


def nuts_denoiser(
    x_t: jnp.ndarray,
    sigma: float,
    rng: Array,
    pi_ref: dists.Distribution,
    initial_dist: dists.Distribution = dists.Normal(scale=1e-2),
    adaptation="low_rank",
    num_warmup=500,
    num_samples=1000,
    max_rank=10,
    debug: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    dim = x_t.shape[-1]
    rng_init, rng_sampling = jrn.split(rng, 2)
    logdensity_fn = partial(log_joint_density, x_t=x_t, sigma=sigma, pi_ref=pi_ref)
    initial_samples = initial_dist.sample(rng_init, sample_shape=(dim,)) + x_t

    if adaptation == "low_rank":
        warmup = blackjax.low_rank_window_adaptation(
            blackjax.nuts, logdensity_fn, max_rank=max_rank
        )
    else:
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=(adaptation == "diag"),
        )

    def one_chain(initial_sample, rng_key):
        k_warm, k_sample = jrn.split(rng_key)
        (state, params), _ = warmup.run(k_warm, initial_sample, num_warmup)

        nuts = blackjax.nuts(logdensity_fn, **params)

        def step(carry, key):
            state, running_mean, count = carry
            state, info = nuts.step(key, state)
            count = count + 1
            running_mean = running_mean + (state.position - running_mean) / count
            return (state, running_mean, count), None

        keys = jrn.split(k_sample, num_samples)
        init_carry = (state, jnp.zeros(dim), 0)
        (_, mean, _), _ = scan(step, init_carry, keys)
        return mean

    pred_x0 = one_chain(initial_samples, rng_sampling)
    if debug:
        return pred_x0, {}
    return pred_x0


def barker_denoiser(
    x_t: jnp.ndarray,
    sigma: float,
    rng: Array,
    pi_ref: dists.Distribution,
    initial_dist: dists.Distribution = dists.Normal(scale=1e-2),
    adaptation="low_rank",
    max_rank=10,
    num_warmup=500,
    num_samples=1000,
    debug: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    dim = x_t.shape[-1]
    rng_init, rng_sampling = jrn.split(rng, 2)
    logdensity_fn = partial(log_joint_density, x_t=x_t, sigma=sigma, pi_ref=pi_ref)
    initial_samples = initial_dist.sample(rng_init, sample_shape=(dim,)) + x_t

    if adaptation == "low_rank":
        warmup = blackjax.low_rank_window_adaptation(
            blackjax.hmc, logdensity_fn, max_rank=max_rank, num_integration_steps=2
        )
    else:
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            logdensity_fn,
            is_mass_matrix_diagonal=(adaptation == "diag"),
            num_integration_steps=2
        )

    def one_chain(initial_sample, rng_key):
        k_warm, k_sample = jrn.split(rng_key)
        (state, params), _ = warmup.run(k_warm, initial_sample, num_warmup)

        mcmc = blackjax.barker(logdensity_fn, step_size=params["step_size"],
                               inverse_mass_matrix=params["inverse_mass_matrix"])
        state = mcmc.init(position=state.position)

        def step(carry, key):
            state, running_mean, count = carry
            state, info = mcmc.step(key, state)
            count = count + 1
            running_mean = running_mean + (state.position - running_mean) / count
            return (state, running_mean, count), None

        keys = jrn.split(k_sample, num_samples)
        init_carry = (state, jnp.zeros(dim), 0)
        (_, mean, _), _ = scan(step, init_carry, keys)
        return mean

    pred_x0 = one_chain(initial_samples, rng_sampling)
    if debug:
        return pred_x0, {}
    return pred_x0


def hmc_denoiser(
    x_t: jnp.ndarray,
    sigma: float,
    rng: Array,
    pi_ref: dists.Distribution,
    initial_dist: dists.Distribution = dists.Normal(scale=1e-2),
    adaptation="low_rank",
    max_rank=10,
    num_integration_steps=2,
    num_warmup=500,
    num_samples=1000,
    debug: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    dim = x_t.shape[-1]
    rng_init, rng_sampling = jrn.split(rng, 2)
    logdensity_fn = partial(log_joint_density, x_t=x_t, sigma=sigma, pi_ref=pi_ref)
    initial_samples = initial_dist.sample(rng_init, sample_shape=(dim,)) + x_t

    if adaptation == "low_rank":
        warmup = blackjax.low_rank_window_adaptation(
            blackjax.hmc, logdensity_fn, max_rank=max_rank, num_integration_steps=num_integration_steps
        )
    else:
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            logdensity_fn,
            is_mass_matrix_diagonal=(adaptation == "diag"),
            num_integration_steps=2
        )

    def one_chain(initial_sample, rng_key):
        k_warm, k_sample = jrn.split(rng_key)
        (state, params), _ = warmup.run(k_warm, initial_sample, num_warmup)

        mcmc = blackjax.hmc(logdensity_fn, **params)

        def step(carry, key):
            state, running_mean, count = carry
            state, info = mcmc.step(key, state)
            count = count + 1
            running_mean = running_mean + (state.position - running_mean) / count
            return (state, running_mean, count), None

        # Extra num_warmup steps as burn-in after adaptation (as in original)
        keys = jrn.split(k_sample, num_samples + num_warmup)
        init_carry = (state, jnp.zeros(dim), 0)
        (_, mean, _), _ = scan(step, init_carry, keys)
        return mean

    pred_x0 = one_chain(initial_samples, rng_sampling)
    if debug:
        return pred_x0, {}
    return pred_x0