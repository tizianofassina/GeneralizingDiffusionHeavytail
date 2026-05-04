from jax import random as jrn
from jax.lax import fori_loop
from tqdm import tqdm
from jax import jit, vmap, numpy as jnp
from functools import partial


def ddpm_one_step(
    x_t,
    rng,
    sigma_t,
    sigma_tm1,
    denoiser_fn,
):
    rng_gauss, rng_denoiser = jrn.split(rng, 2)
    pred_x0 = denoiser_fn(x_t, sigma_t, rng_denoiser)
    mean = (1 - (sigma_tm1**2 / sigma_t**2)) * pred_x0 + (
        sigma_tm1**2 / sigma_t**2
    ) * x_t
    std = (sigma_tm1 / sigma_t) * ((sigma_t**2 - sigma_tm1**2)) ** 0.5

    return mean + jrn.normal(key=rng_gauss, shape=(pred_x0.shape[-1],)) * std


def ddpm(initial_sample, rng, sigmas, denoiser_fn):

    n_steps = sigmas.shape[0] - 1
    rngs = jrn.split(rng, n_steps)

    def _body_fun(i, val):
        return ddpm_one_step(
            x_t=val,
            sigma_t=sigmas[-(i + 1)],
            sigma_tm1=sigmas[-(i + 2)],
            rng=rngs[i],
            denoiser_fn=denoiser_fn,
        )

    out = fori_loop(lower=0, upper=n_steps, body_fun=_body_fun, init_val=initial_sample)
    return out


def batch_ddpm(initial_samples, rng, sigmas, denoiser_fn, batch_size):
    pbar = tqdm(
        zip(reversed(sigmas[:-1]), reversed(sigmas[1:])), total=sigmas.shape[0] - 1
    )
    samples = jnp.array(initial_samples)
    dim = samples.shape[-1]
    for sigma_tm1, sigma_t in pbar:
        step_fn = jit(
            vmap(
                partial(
                    ddpm_one_step,
                    sigma_t=sigma_t,
                    sigma_tm1=sigma_tm1,
                    denoiser_fn=denoiser_fn,
                )
            )
        )

        new_samples = []
        for batch in tqdm(samples.reshape(-1, batch_size, dim)):
            rng, rng_batch = jrn.split(rng, 2)
            new_samples.append(step_fn(batch, jrn.split(rng_batch, batch_size)))
        samples = jnp.concatenate(new_samples)
        samples.block_until_ready()   
    return samples