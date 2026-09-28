
import jax.numpy as jnp
import jax.random as jrn
import numpyro.distributions as dist
from jax import vmap
import os
import sys
from hf_toy.create_data import build_heavy_tail_ref_dist

DIM        = 10 #2 
TAIL_INDEX = 3
N_MIXTURES = 20 #4
SCALE = 0.1
GLOBAL_SCALE = 4.

TRAIN_SIZE = int(sys.argv[1]) 
TEST_SIZE  = 10_000_000
DISTRIBUTION_SEED = 0
SEED_TRAINING  = 34 
SEED_TEST = 35
SEED_TEST_2 = 39


os.makedirs(f"data/dim_{DIM}", exist_ok=True)
pi_ref = build_heavy_tail_ref_dist(jrn.key(DISTRIBUTION_SEED), student_df=TAIL_INDEX, scale=SCALE, dim=DIM, global_scale=GLOBAL_SCALE, concentration=None, n_mixture=N_MIXTURES)

train_set = pi_ref.sample(jrn.key(SEED_TRAINING), (TRAIN_SIZE,))
test_set  =pi_ref.sample(jrn.key(SEED_TEST), (TEST_SIZE,))
test_set_2  =pi_ref.sample(jrn.key(SEED_TEST_2), (TEST_SIZE,))
jnp.save(f"data/dim_{DIM}/train_set_dim_{DIM}_size_{TRAIN_SIZE}_tail_{int(TAIL_INDEX)}_seed_{SEED_TRAINING}_dist_seed_{DISTRIBUTION_SEED}.npy", train_set)
jnp.save(f"data/dim_{DIM}/test_set_dim_{DIM}_size_{TEST_SIZE}_tail_{int(TAIL_INDEX)}_seed_{SEED_TEST}_dist_seed_{DISTRIBUTION_SEED}.npy",  test_set)
jnp.save(f"data/dim_{DIM}/test_set_dim_{DIM}_size_{TEST_SIZE}_tail_{int(TAIL_INDEX)}_seed_{SEED_TEST_2}_dist_seed_{DISTRIBUTION_SEED}.npy",  test_set_2)


print(f"Train set shape: {train_set.shape}")
print(f"Test set shape:  {test_set.shape}")
