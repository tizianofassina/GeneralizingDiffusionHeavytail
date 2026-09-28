from hf_toy.find_unimodality import worst_case_onset
from hf_toy.create_data import build_heavy_tail_ref_dist
import jax.numpy as jnp
import jax.random as jrn
import numpyro.distributions as dist
from jax import vmap
import os


"""
Unimodality test (dip test) on the dim 10, 20-mode mixture.
Imports of build_heavy_tail_ref_dist and worst_case_onset to be added by hand.
"""

DIM          = 10
N_MIXTURE    = 20
TAIL_INDEX   = 3
SCALE        = 0.1
GLOBAL_SCALE = 4.0
DIST_SEED    = 0

N_SAMPLES        = 100_000
N_SLICES         = 20_000
N_SLICES_VERIFY  = 20_000
BISECT_ITERS     = 12
Z_THRESHOLD      = 0.0
SIGMA_MIN        = 0.1
SIGMA_MAX        = 5.0
CALIB_SEED       = 0

pi_ref = build_heavy_tail_ref_dist(
    rng=jrn.key(DIST_SEED),
    student_df=float(TAIL_INDEX),
    dim=DIM,
    scale=SCALE,
    global_scale=GLOBAL_SCALE,
    n_mixture=N_MIXTURE,
)

result = worst_case_onset(
    pi_ref=pi_ref,
    dim=DIM,
    n_slices=N_SLICES,
    n_slices_verify=N_SLICES_VERIFY,
    n_samples=N_SAMPLES,
    z_threshold=Z_THRESHOLD,
    sigma_min=SIGMA_MIN,
    sigma_max=SIGMA_MAX,
    bisect_iters=BISECT_ITERS,
    rng_key=jrn.key(CALIB_SEED),
    verbose=True,
)

print("\n=== result ===")
print(f"worst_sigma (worst-case unimodality onset) = {result['worst_sigma']:.4f}")
print(f"verify z_dip range = [{result['verify_z_min']:.3f}, {result['verify_z_max']:.3f}]  "
      f"ok={result['verify_ok']}")