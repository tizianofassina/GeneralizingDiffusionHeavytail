import jax.numpy as jnp
import jax.random as jrn
import numpyro.distributions as dist
from jax import vmap
import os
import sys
# ── GMM builder (mirrors build_heavy_tail_ref_dist, swap StudentT -> Normal) ─
def build_random_cholesky(key, dim):
    source      = jrn.normal(key, ((dim - 1) * dim // 2,))
    source_diag = jrn.uniform(key, (dim,), minval=0.1, maxval=1)
    A    = jnp.eye(dim).at[jnp.tril_indices(n=dim, k=-1)].set(source)
    diag = jnp.diag(source_diag)
    return A, diag


def build_gmm_ref_dist(rng, scale=0.1, dim=2, global_scale=4.,
                       concentration=None, n_grid=5):
    n_mixture = n_grid ** dim
    if concentration is None:
        concentration = jnp.ones((n_mixture,))

    rng_weights, rng_cholesky = jrn.split(rng, 2)

    # Fixed grid locs in {-2,-1,0,1,2}^2  (NOT multiplied by global_scale)
    axis = jnp.linspace(-(n_grid // 2), n_grid // 2, n_grid)
    grids = jnp.meshgrid(*([axis] * dim), indexing="ij")
    locs = jnp.stack([g.flatten() for g in grids], axis=-1)  # (n_mixture, dim)

    weights = dist.Dirichlet(concentration=concentration).sample(rng_weights)

    As, diags = vmap(lambda x: build_random_cholesky(x, dim))(
        jrn.split(rng_cholesky, n_mixture)
    )
    scale_trils = jnp.stack(
        [global_scale * A @ diag * scale for A, diag in zip(As, diags)],
        axis=0,
    )

    mixing_distribution = dist.Categorical(probs=weights)
    component_distributions = dist.MultivariateNormal(
        loc=locs, scale_tril=scale_trils, validate_args=False
    )
    pi_ref = dist.MixtureSameFamily(
        mixing_distribution, component_distributions, validate_args=False
    )

    # Normalize: subtract mean, divide by std (same as HT version)
    smps = pi_ref.sample(rng, (100_000,))
    mean, std = smps.mean(axis=0), smps.std()
    return dist.TransformedDistribution(
        base_distribution=pi_ref,
        transforms=[
            dist.transforms.AffineTransform(loc=-mean, scale=1),
            dist.transforms.AffineTransform(loc=jnp.zeros_like(mean), scale=1 / std),
        ],
    )

