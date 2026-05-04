import os
import sys

import jax.numpy as jnp
import jax.random as jrn

from hf_toy.create_data_gmm import build_gmm_ref_dist


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIM          = 2
TAIL_INDEX   = 3      # kept for filename compatibility (unused for GMM)
N_GRID       = 5      # 5x5 = 25 modes
SCALE        = 0.1
GLOBAL_SCALE = 4.0

TEST_SIZE = 10_000_000

DISTRIBUTION_SEED = 0
SEED_TRAINING     = 34
SEED_TEST         = 35
SEED_TEST_2       = 39

if len(sys.argv) < 2:
    raise ValueError("Usage: python <script>.py <TRAIN_SIZE>  (e.g. 1000, 10000, 100000)")
TRAIN_SIZE = int(sys.argv[1])


# ---------------------------------------------------------------------------
# Build reference distribution and sample
# ---------------------------------------------------------------------------
pi_ref = build_gmm_ref_dist(
    jrn.key(DISTRIBUTION_SEED),
    scale=SCALE,
    dim=DIM,
    global_scale=GLOBAL_SCALE,
    concentration=None,
    n_grid=N_GRID,
)

train_set  = pi_ref.sample(jrn.key(SEED_TRAINING), (TRAIN_SIZE,))
test_set   = pi_ref.sample(jrn.key(SEED_TEST),     (TEST_SIZE,))
test_set_2 = pi_ref.sample(jrn.key(SEED_TEST_2),   (TEST_SIZE,))


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = f"data_gmm/dim_{DIM}"
os.makedirs(out_dir, exist_ok=True)

train_path  = f"{out_dir}/train_set_dim_{DIM}_size_{TRAIN_SIZE}_tail_{int(TAIL_INDEX)}_seed_{SEED_TRAINING}_dist_seed_{DISTRIBUTION_SEED}.npy"
test_path   = f"{out_dir}/test_set_dim_{DIM}_size_{TEST_SIZE}_tail_{int(TAIL_INDEX)}_seed_{SEED_TEST}_dist_seed_{DISTRIBUTION_SEED}.npy"
test_path_2 = f"{out_dir}/test_set_dim_{DIM}_size_{TEST_SIZE}_tail_{int(TAIL_INDEX)}_seed_{SEED_TEST_2}_dist_seed_{DISTRIBUTION_SEED}.npy"

jnp.save(train_path,  train_set)
jnp.save(test_path,   test_set)
jnp.save(test_path_2, test_set_2)

print(f"Train set shape: {train_set.shape}  ->  {train_path}")
print(f"Test set shape:  {test_set.shape}   ->  {test_path}")
print(f"Test set 2 shape:{test_set_2.shape} ->  {test_path_2}")