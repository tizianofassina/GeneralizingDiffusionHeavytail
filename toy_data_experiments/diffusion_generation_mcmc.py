"""
Generate denoised samples via DDPM with MCMC denoisers (HMC, Barker, or NUTS).

Usage:
    python diffusion_generation_mcmc.py <init> <sigma> <mcmc>

    init  : "gaussian", "p_t", or "p_theta"
    sigma : float, e.g. 1.24404052
    mcmc  : "hmc", "barker", or "nuts"

Example:
    python diffusion_generation_mcmc.py gaussian 1.24404052 hmc
    python diffusion_generation_mcmc.py p_theta  1.24404052 barker
    python diffusion_generation_mcmc.py p_t      1.24404052 nuts
"""

import os
import sys
from functools import partial

import jax.numpy as jnp
import jax.random as jrn

from hf_toy.create_data import build_heavy_tail_ref_dist
from hf_toy.diffusion_samplers import batch_ddpm
from hf_toy.mc_denoisers import hmc_denoiser, barker_denoiser, nuts_denoiser


# ── CLI args ────────────────────────────────────────────────────────────────
INIT  = sys.argv[1]          # "gaussian", "p_t", or "p_theta"
SIGMA = float(sys.argv[2])   # starting sigma
MCMC  = sys.argv[3]          # "hmc", "barker", or "nuts"

assert INIT in ("gaussian", "p_t", "p_theta"), f"Unknown init: {INIT}"
assert MCMC in ("hmc", "barker", "nuts"),       f"Unknown mcmc: {MCMC}"


# ── Config ──────────────────────────────────────────────────────────────────
NUM_SAMPLES      = 1_000_000
NAME_NUM_SAMPLES = 10_000_000
BATCH_SIZE       = 250_000

DIM          = 2
TAIL_INDEX   = 3
SCALE        = 0.1
GLOBAL_SCALE = 4.0
N_MIXTURES   = 4
DIST_SEED    = 0

# MCMC denoiser hyperparams
NUM_WARMUP       = 200
NUM_MCMC_SAMPLES = 1_000
MAX_RANK         = 10

# Sigma schedule (EDM, truncated)
SIGMA_MAX           = 3
SIGMA_MIN           = 0.0002
RHO                 = 2.0
NUM_STEPS_DENOISING = 40

# Flow checkpoint used as p_theta init source
TRAIN_MODALITY_FLOW = "dynamic"
N_TRAIN_FLOW        = 10000
N_EPOCHS_FLOW       = 3000
FLOW_LAYERS         = 5
FLOW_NN_WIDTH       = 50
FLOW_NN_DEPTH       = 3
FLOW_INF            = 1

init_path = None  
# ── Init samples path ───────────────────────────────────────────────────────
if INIT == "gaussian":
    init_path = (f"data/data_gen_noised/dim_{DIM}/"
                 f"samples_p_inf_sigma_{SIGMA}_size_{NAME_NUM_SAMPLES}.npy")
elif INIT == "p_t":
    init_path = (f"data/data_gen_noised/dim_{DIM}/"
                 f"samples_pt_sigma_{SIGMA}_size_{NAME_NUM_SAMPLES}.npy")
elif INIT == "p_theta":
    init_path = (f"data/data_gen_noised/dim_{DIM}/"
                 f"samples_flow_flow_dim{DIM}_sig{SIGMA}_tail{TAIL_INDEX}"
                 f"_layers{FLOW_LAYERS}_w{FLOW_NN_WIDTH}_d{FLOW_NN_DEPTH}"
                 f"_inf{FLOW_INF}_n{N_TRAIN_FLOW}_ep{N_EPOCHS_FLOW}"
                 f"_{TRAIN_MODALITY_FLOW}.npy")

assert os.path.isfile(init_path), f"Missing input file: {init_path}"

print(f"Loading initial samples from {init_path}")
initial_samples = jnp.load(init_path)[:NUM_SAMPLES]
print(f"  shape: {initial_samples.shape}")


# ── Sigma schedule ──────────────────────────────────────────────────────────
inv_rho      = 1.0 / RHO
step_indices = jnp.linspace(0, 1, NUM_STEPS_DENOISING)
sigmas       = (SIGMA_MAX**inv_rho
                + step_indices * (SIGMA_MIN**inv_rho - SIGMA_MAX**inv_rho)) ** RHO
sigmas       = jnp.sort(sigmas)
sigmas_truncated = sigmas[sigmas <= SIGMA]
print(f"Schedule: {len(sigmas_truncated)} steps, "
      f"from {float(sigmas_truncated[0]):.6f} to {float(sigmas_truncated[-1]):.6f}")


# ── Reference distribution (target for MCMC denoisers) ──────────────────────
pi_ref = build_heavy_tail_ref_dist(
    rng=jrn.key(DIST_SEED),
    student_df=float(TAIL_INDEX),
    dim=DIM,
    scale=SCALE,
    global_scale=GLOBAL_SCALE,
    n_mixture=N_MIXTURES,
)


# ── Build denoiser ──────────────────────────────────────────────────────────
DENOISER_MAP = {
    "hmc":    hmc_denoiser,
    "barker": barker_denoiser,
    "nuts":   nuts_denoiser,
}
denoiser_cls = DENOISER_MAP[MCMC]

denoiser_fn = partial(
    denoiser_cls,
    pi_ref=pi_ref,
    adaptation="low_rank",
    num_warmup=NUM_WARMUP,
    num_samples=NUM_MCMC_SAMPLES,
    max_rank=MAX_RANK,
    debug=False,
)


# ── Run DDPM ────────────────────────────────────────────────────────────────
print(f"Running DDPM with {MCMC} denoiser (n_mcmc={NUM_MCMC_SAMPLES}, warmup={NUM_WARMUP})...")
samples = batch_ddpm(
    initial_samples,
    rng=jrn.key(43),
    sigmas=sigmas_truncated,
    denoiser_fn=denoiser_fn,
    batch_size=BATCH_SIZE,
)


# ── Save ────────────────────────────────────────────────────────────────────
out_dir = f"data/data_gen_diffusion_mcmc/dim_{DIM}"
os.makedirs(out_dir, exist_ok=True)

init_tag = {"gaussian": "gaussian", "p_t": "pt", "p_theta": "p_theta"}[INIT]
out_path = (f"{out_dir}/samples_{init_tag}_{MCMC}"
            f"_sigma_{SIGMA}_size_{NUM_SAMPLES}.npy")

jnp.save(out_path, samples)
print(f"Saved {samples.shape} to {out_path}")