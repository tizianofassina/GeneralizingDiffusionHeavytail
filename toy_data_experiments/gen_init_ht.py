"""
This script generates initial samples that are not p_theta.
  - p_inf (gaussian): N(0, sigma^2 I)
  - p_t: samples from pi_ref + N(0, sigma^2 I)

Iterates over a list of SIGMA values, using the SAME seed for every sigma
so that the only thing that changes between runs is the noise scale.
"""

import os
import jax.numpy as jnp
import jax.random as jrn

from hf_toy.create_data import build_heavy_tail_ref_dist


SIGMA        = [1.24404052, 1.05527906, 0.8012415]
DIM          = 2
TAIL_INDEX   = 3
SCALE        = 0.1
GLOBAL_SCALE = 4.0
N_MIXTURES   = 4
DIST_SEED    = 0
SEED_INIT    = 53
NUM_SAMPLES  = 10_000_000


os.makedirs(f"data/data_gen_noised/dim_{DIM}", exist_ok=True)

pi_ref = build_heavy_tail_ref_dist(
    rng=jrn.key(DIST_SEED),
    student_df=TAIL_INDEX,
    dim=DIM,
    scale=SCALE,
    global_scale=GLOBAL_SCALE,
    n_mixture=N_MIXTURES,
)

for sigma in SIGMA:
    # Same seed for every sigma: only the noise scale differs.
    KEY, SUBKEY = jrn.split(jrn.key(SEED_INIT))

    # p_inf: pure gaussian noise
    normal = sigma * jrn.normal(KEY, (NUM_SAMPLES, DIM))

    # p_t: pi_ref + gaussian noise
    p_t = pi_ref.sample(SUBKEY, (NUM_SAMPLES,)) + normal

    path_pt    = f"data/data_gen_noised/dim_{DIM}/samples_pt_sigma_{sigma}_size_{NUM_SAMPLES}.npy"
    path_p_inf = f"data/data_gen_noised/dim_{DIM}/samples_p_inf_sigma_{sigma}_size_{NUM_SAMPLES}.npy"

    jnp.save(path_pt,    p_t)
    jnp.save(path_p_inf, normal)

    print(f"[sigma={sigma}] Saved p_inf {normal.shape} -> {path_p_inf}")
    print(f"[sigma={sigma}] Saved p_t   {p_t.shape} -> {path_pt}")