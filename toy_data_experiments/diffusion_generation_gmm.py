"""
Generate denoised samples via DDPM with the analytic GMM-convolved score.

For each (sigma, init) in SIGMAS_GEN x INITS:
  - init='gaussian'  : N(0, sigma^2 I)               (generated on the fly)
  - init='p_t'       : pi_ref + N(0, sigma^2 I)      (generated on the fly)
  - init='p_theta'   : samples from the flow trained at this sigma
                       (loaded from MODEL_DIR_FLOW)

Output: data_gen_diffusion_gmm/dim_2/samples_{init_tag}_analytic_sigma_{sigma}_size_{NUM_SAMPLES}.npy
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrn

from hf_toy.diffusion_samplers import batch_ddpm
from hf_toy.create_data_gmm import build_gmm_ref_dist
from gen_flow_gmm import GMMFlow


# ── Config ───────────────────────────────────────────────────────────────────
DIM          = 2
TAIL_INDEX   = 3
N_GRID       = 5
SCALE        = 0.1
GLOBAL_SCALE = 4.0
DIST_SEED    = 0

DATA_INPUT_DIR = f"data_gmm/dim_{DIM}"
MODEL_DIR_FLOW = "model_flow_gmm"
GEN_DIR        = f"data_gen_diffusion_gmm/dim_{DIM}"
os.makedirs(GEN_DIR, exist_ok=True)

SIGMAS_GEN = [1.24404052, 1.05527906, 0.8012415]
INITS      = ["gaussian", "p_t", "p_theta"]

NUM_SAMPLES    = 1_000_000
BATCH_SIZE_GEN = 250_000

SIGMA_MAX           = 3
SIGMA_MIN           = 0.0002
RHO                 = 2.0
NUM_STEPS_DENOISING = 40

# Flow checkpoint used as p_theta init source (same convention as gen_flow_gmm.py)
FLOW_TRAIN_SIZE = 10000
FLOW_EPOCHS     = 3000
FLOW_TRAIN_MODE = "dynamic"
FLOW_LAYERS     = 5
FLOW_NN_WIDTH   = 50
FLOW_NN_DEPTH   = 3
FLOW_INF        = 1

SEED_INIT = 7
SEED_DDPM = 43


def flow_path_for_sigma(sigma):
    """Path to the GMM flow checkpoint trained at this sigma."""
    name = (f"flow_gmm_dim{DIM}_sig{sigma}_tail{TAIL_INDEX}"
            f"_layers{FLOW_LAYERS}_w{FLOW_NN_WIDTH}_d{FLOW_NN_DEPTH}"
            f"_inf{FLOW_INF}_n{FLOW_TRAIN_SIZE}_ep{FLOW_EPOCHS}_{FLOW_TRAIN_MODE}")
    return os.path.join(MODEL_DIR_FLOW, f"{name}.eqx")


# ── Sigma schedule ────────────────────────────────────────────────────────────
def build_sigma_schedule(sigma_max, sigma_min, rho, n_steps, sigma_T):
    inv_rho = 1.0 / rho
    step_indices = jnp.linspace(0, 1, n_steps)
    sigmas = (sigma_max**inv_rho
              + step_indices * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho
    sigmas = jnp.sort(sigmas)
    return sigmas[sigmas <= sigma_T]


# ── Analytic GMM-convolved score ─────────────────────────────────────────────
def extract_gmm_components(pi_ref):
    base_mix = pi_ref.base_dist
    comp     = base_mix.component_distribution
    mix      = base_mix.mixing_distribution

    locs_x        = comp.loc
    scale_trils_x = comp.scale_tril
    log_weights   = jnp.log(mix.probs)

    t0, t1 = pi_ref.transforms
    mean_vec = -t0.loc
    inv_std  = t1.scale

    locs_y        = (locs_x - mean_vec) * inv_std
    scale_trils_y = scale_trils_x * inv_std
    return locs_y, scale_trils_y, log_weights


def make_analytic_denoiser(locs, scale_trils, log_weights):
    K, d = locs.shape
    Sigmas = jnp.einsum("kij,klj->kil", scale_trils, scale_trils)

    def log_prob_sigma(x, sigma):
        Sigmas_n = Sigmas + (sigma ** 2) * jnp.eye(d)
        diffs = x[None, :] - locs
        sol   = jnp.linalg.solve(Sigmas_n, diffs[..., None]).squeeze(-1)
        maha  = jnp.einsum("kd,kd->k", diffs, sol)
        sign, logdet = jnp.linalg.slogdet(Sigmas_n)
        log_norm = -0.5 * (d * jnp.log(2 * jnp.pi) + logdet)
        log_comp = log_norm - 0.5 * maha
        return jax.scipy.special.logsumexp(log_weights + log_comp)

    score_single = jax.grad(log_prob_sigma, argnums=0)

    def denoiser_fn(x_t, sigma_t, rng=None):
        if x_t.ndim == 1:
            score = score_single(x_t, sigma_t)
        else:
            score = jax.vmap(score_single, in_axes=(0, None))(x_t, sigma_t)
        return x_t + (sigma_t ** 2) * score

    return denoiser_fn


# ── Init builders ─────────────────────────────────────────────────────────────
def build_init_samples(init, sigma, n_samples, rng, pi_ref):
    if init == "gaussian":
        return sigma * jrn.normal(rng, (n_samples, DIM))

    elif init == "p_t":
        rng_data, rng_noise = jrn.split(rng)
        clean = pi_ref.sample(rng_data, (n_samples,))
        return clean + sigma * jrn.normal(rng_noise, clean.shape)

    elif init == "p_theta":
        # Load flow trained AT this sigma and sample directly.
        # The flow already models the noisy distribution at sigma.
        fpath = flow_path_for_sigma(sigma)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Flow for p_theta init not found at sigma={sigma}: {fpath}"
            )
        print(f"    Loading flow for p_theta init: {fpath}")
        flow_model, _ = GMMFlow.load(fpath)
        return flow_model.sample(rng, n_samples)

    else:
        raise ValueError(f"Unknown init {init}")


# ── Generation loop ───────────────────────────────────────────────────────────
def generate(sigma, init, denoiser_fn, pi_ref):
    init_tag = {"gaussian": "gaussian", "p_t": "pt", "p_theta": "p_theta"}[init]
    out_path = os.path.join(
        GEN_DIR,
        f"samples_{init_tag}_analytic_sigma_{sigma}_size_{NUM_SAMPLES}.npy"
    )
    if os.path.exists(out_path):
        print(f"  Already exists, skipping: {out_path}")
        return

    # Skip p_theta if its flow checkpoint is missing
    if init == "p_theta":
        fpath = flow_path_for_sigma(sigma)
        if not os.path.exists(fpath):
            print(f"  Skipping p_theta @ sigma={sigma}: missing flow {fpath}")
            return

    rng_init = jrn.fold_in(jrn.key(SEED_INIT), hash((init, str(sigma))) & 0xffffffff)
    initial_samples = build_init_samples(init, sigma, NUM_SAMPLES, rng_init, pi_ref)
    print(f"  init={init}  shape={initial_samples.shape}")

    sigmas_truncated = build_sigma_schedule(
        SIGMA_MAX, SIGMA_MIN, RHO, NUM_STEPS_DENOISING, sigma
    )
    print(f"  Schedule: {len(sigmas_truncated)} steps, "
          f"{float(sigmas_truncated[0]):.6f} → {float(sigmas_truncated[-1]):.6f}")

    samples = batch_ddpm(
        initial_samples,
        rng=jrn.key(SEED_DDPM),
        sigmas=sigmas_truncated,
        denoiser_fn=denoiser_fn,
        batch_size=BATCH_SIZE_GEN,
    )

    np.save(out_path, np.array(samples))
    print(f"  Saved {samples.shape} to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building reference GMM (analytic, no training)...")
    pi_ref = build_gmm_ref_dist(
        jrn.key(DIST_SEED),
        scale=SCALE, dim=DIM, global_scale=GLOBAL_SCALE,
        concentration=None, n_grid=N_GRID,
    )
    locs, scale_trils, log_weights = extract_gmm_components(pi_ref)
    print(f"  K={locs.shape[0]} components")

    # ── Build denoiser ───────────────────────────────────────────────────
    denoiser_fn = make_analytic_denoiser(locs, scale_trils, log_weights)

    # ── Generation loop ──────────────────────────────────────────────────
    for sigma in SIGMAS_GEN:
        for init in INITS:
            print(f"\n--- sigma={sigma}  init={init} ---")
            generate(sigma, init, denoiser_fn, pi_ref)

    print("\nDone.")