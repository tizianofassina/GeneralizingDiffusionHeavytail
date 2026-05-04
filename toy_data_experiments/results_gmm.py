"""
Evaluation script for the GMM toy.

PART A — Global Max-Sliced Wasserstein
    Compares against the reference (clean GMM test set):
      - Analytic-score diffusion samples for 3 inits x 3 sigmas
      - Reference vs reference baseline (Monte Carlo floor)
    Computes 10 chunks of 1M samples each, reports mean ± std.
    Output: results_global_gmm.csv

PART B — Monte Carlo KL estimation
    For each (n_train, sigma), loads the flow checkpoint and estimates
        KL(p_t || p_theta) ≈ E_{x ~ test_set + sigma·N(0,I)} [log p_t(x) - log p_theta(x)]
    where:
        - log p_t   = analytic log-density of the GMM convolved with N(0, sigma^2 I)
        - log p_θ   = flow log-density
    Output: results_kl_gmm_{TRAIN_MODALITY}.csv (rows = n_train, columns = sigma)

Usage:
    python results_gmm.py <fix|dynamic>
"""
import os
import sys
import csv
import numpy as np
import pandas as pd
import jax
import jax.scipy.special as jsp
from jax import random as jrn, numpy as jnp

from hf_toy.bulk_metrics import max_sliced_wasserstein
from hf_toy.create_data_gmm import build_gmm_ref_dist
from gen_flow_gmm import GMMFlow


# ── CLI ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    raise ValueError("Usage: python results_gmm.py <fix|dynamic>")
TRAIN_MODALITY = sys.argv[1].lower()
if TRAIN_MODALITY not in ("fix", "dynamic"):
    raise ValueError(f"Unknown TRAIN_MODALITY: {TRAIN_MODALITY!r}. Use 'fix' or 'dynamic'.")


# ── Config ──────────────────────────────────────────────────────────────────
DIM          = 2
TAIL_INDEX   = 3
N_GRID       = 5
SCALE        = 0.1
GLOBAL_SCALE = 4.0
DIST_SEED    = 0

DATA_DIR_GMM   = f"data_gmm/dim_{DIM}"
DIFFUSION_DIR  = f"data_gen_diffusion_gmm/dim_{DIM}"
MODEL_DIR_FLOW = "model_flow_gmm"

SEED_TEST = 35
TEST_FILE = os.path.join(
    DATA_DIR_GMM,
    f"test_set_dim_{DIM}_size_10000000_tail_{TAIL_INDEX}_seed_{SEED_TEST}_dist_seed_{DIST_SEED}.npy"
)

# ── PART A: MSW config ─────────────────────────────────────────────────────
SIGMAS_DIFF = ["0.8012415", "1.05527906", "1.24404052"]
INITS_DIFF  = ["gaussian", "pt", "p_theta"]
DIFF_SAMPLES_PER_FILE = 1_000_000

N_CHUNKS   = 10
CHUNK_SIZE = 1_000_000
SEED_MSW   = 42

OUTPUT_CSV_MSW = "results_global_gmm.csv"

# ── PART B: KL config ──────────────────────────────────────────────────────
# Same lists as gen_flow_gmm.py (and gen_flow_ht.py)
SIGMAS_KL = [2.55969277, 2.28555770, 2.02694511, 1.78385499, 1.24404052, 1.05527906,
             0.801241500, 0.582129509, 0.397943082, 0.248682220,
             0.134346923, 0.0549371894, 0.0104530209]
N_TRAIN_KL = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000,
              3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500,
              8000, 8500, 9500, 10000, 100000]

# Flow filename hyperparameters (must match gen_flow_gmm.py)
FLOW_LAYERS   = 5
FLOW_NN_WIDTH = 50
FLOW_NN_DEPTH = 3
FLOW_INF      = 1
FLOW_EPOCHS   = 3000

SEED_KL_NOISE = 1
OUTPUT_CSV_KL = f"results_kl_gmm_{TRAIN_MODALITY}.csv"


# ────────────────────────────────────────────────────────────────────────────
# PART A — MSW HELPERS
# ────────────────────────────────────────────────────────────────────────────
def compute_global_msw(gen_samples, ref_pool, rng,
                       n_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE):
    """
    For each chunk: draw a fresh subset of `chunk_size` from ref_pool,
    take the first `chunk_size` (or all) of gen_samples, compute MSW.
    Returns a list of n_chunks MSW values.
    """
    print(f"    gen size: {len(gen_samples)}")
    msw_values = []
    for i in range(n_chunks):
        rng, rng_ref, rng_msw = jrn.split(rng, 3)
        idx_ref = jrn.choice(rng_ref, len(ref_pool),
                             shape=(chunk_size,), replace=False)
        ref_draw = jnp.array(ref_pool[idx_ref])

        n = min(len(gen_samples), chunk_size)
        msw, _ = max_sliced_wasserstein(
            jnp.array(gen_samples[:n]),
            ref_draw[:n],
            key=jrn.fold_in(rng_msw, i),
            tol=1e-7, lr=1e-3, max_iter=10_000, disable_pbar=True,
        )
        msw_values.append(float(msw))
        print(f"    chunk {i}: MSW={float(msw):.6f}  (n={n})")
    return msw_values


def run_msw_evaluation(ref_pool, rng):
    rows = []

    # Diffusion (analytic score) results
    for init in INITS_DIFF:
        for sigma in SIGMAS_DIFF:
            fname = f"samples_{init}_analytic_sigma_{sigma}_size_{DIFF_SAMPLES_PER_FILE}.npy"
            fpath = os.path.join(DIFFUSION_DIR, fname)

            if not os.path.exists(fpath):
                print(f"\n!! Missing diffusion file: {fpath} -- skipping.")
                continue

            print(f"\n{'='*70}")
            print(f"Diffusion (analytic): init={init}  sigma={sigma}")
            print(f"  Loading {fpath}")
            gen_samples = np.load(fpath)
            print(f"  shape: {gen_samples.shape}")

            rng, rng_msw = jrn.split(rng)
            vals = compute_global_msw(gen_samples, ref_pool, rng_msw)
            rows.append({
                "method":   "diffusion_analytic",
                "init":     init,
                "sigma":    sigma,
                "mean_msw": float(np.mean(vals)),
                "std_msw":  float(np.std(vals)),
            })
            print(f"  => mean={np.mean(vals):.6f}  std={np.std(vals):.6f}")

    # Reference vs reference baseline
    print(f"\n{'='*70}")
    print("Reference vs reference (Monte Carlo floor)...")
    rng, rng_ref_draw, rng_ref_msw = jrn.split(rng, 3)
    idx_ref_gen = jrn.choice(rng_ref_draw, len(ref_pool),
                             shape=(CHUNK_SIZE,), replace=False)
    ref_as_gen = np.array(ref_pool[np.array(idx_ref_gen)])
    vals = compute_global_msw(ref_as_gen, ref_pool, rng_ref_msw)
    rows.append({
        "method":   "reference",
        "init":     "-",
        "sigma":    "-",
        "mean_msw": float(np.mean(vals)),
        "std_msw":  float(np.std(vals)),
    })
    print(f"  => ref mean={np.mean(vals):.6f}  std={np.std(vals):.6f}")

    # Write CSV
    with open(OUTPUT_CSV_MSW, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "init", "sigma", "mean_msw", "std_msw"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n>> MSW results saved to {OUTPUT_CSV_MSW}")
    return rng


# ────────────────────────────────────────────────────────────────────────────
# PART B — KL HELPERS (extracted from gen_flow_gmm.py original)
# ────────────────────────────────────────────────────────────────────────────
def extract_gmm_components(pi_ref):
    """Extract GMM params accounting for the centering+scaling transforms."""
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


def make_gmm_log_prob_fn(locs, scale_trils, log_weights):
    """
    Build a JIT-compiled batched log_prob_sigma(x, sigma) for the GMM
    convolved with N(0, sigma^2 I).
    """
    K, d = locs.shape
    Sigmas = jnp.einsum("kij,klj->kil", scale_trils, scale_trils)

    @jax.jit
    def log_prob_sigma(x, sigma):
        Sigmas_n = Sigmas + (sigma ** 2) * jnp.eye(d)
        Ls       = jnp.linalg.cholesky(Sigmas_n)
        logdets  = 2.0 * jnp.sum(jnp.log(jnp.diagonal(Ls, axis1=-2, axis2=-1)), axis=-1)
        log_norm = -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdets)

        def single_log_prob(point):
            diffs = point[None, :] - locs
            ys = jax.vmap(jax.scipy.linalg.solve_triangular,
                          in_axes=(0, 0, None))(Ls, diffs, True)
            maha = jnp.sum(ys ** 2, axis=-1)
            log_comp = log_norm - 0.5 * maha
            return jsp.logsumexp(log_weights + log_comp)

        return jax.vmap(single_log_prob)(x)

    return log_prob_sigma


def flow_checkpoint_path(n_train, sigma):
    """Path to the flow checkpoint trained at (n_train, sigma)."""
    base_name = (f"flow_gmm_dim{DIM}_sig{sigma}_tail{TAIL_INDEX}"
                 f"_layers{FLOW_LAYERS}_w{FLOW_NN_WIDTH}_d{FLOW_NN_DEPTH}"
                 f"_inf{FLOW_INF}_n{n_train}_ep{FLOW_EPOCHS}_{TRAIN_MODALITY}")
    return os.path.join(MODEL_DIR_FLOW, f"{base_name}.eqx")


def run_kl_evaluation(test_data, log_prob_sigma_fn):
    # Fixed noise base, identical across all (n_train, sigma) pairs
    test_noise_base = jrn.normal(jrn.key(SEED_KL_NOISE), test_data.shape)

    # results[sigma][n_train] = KL value
    results_kl = {s: {} for s in SIGMAS_KL}

    for n_train in N_TRAIN_KL:
        for sigma in SIGMAS_KL:
            fpath = flow_checkpoint_path(n_train, sigma)
            if not os.path.exists(fpath):
                print(f"  [n_train={n_train}, sigma={sigma}] missing checkpoint: {fpath} -- NaN")
                results_kl[sigma][n_train] = float("nan")
                continue

            print(f"  [n_train={n_train}, sigma={sigma}] loading flow...")
            model, _ = GMMFlow.load(fpath)

            noisy_test = test_data + sigma * test_noise_base
            log_p = log_prob_sigma_fn(noisy_test, sigma)
            log_q = model.log_prob(noisy_test)

            kl_raw = float(jnp.mean(log_p - log_q))
            # MC estimator can be negative for small n_train / well-fit q;
            # floor at 0 since KL >= 0 by definition.
            kl_div = max(0.0, kl_raw)
            results_kl[sigma][n_train] = kl_div
            if kl_raw < 0:
                print(f"    KL = {kl_div:.6f}  (raw MC estimate was {kl_raw:.6f}, floored to 0)")
            else:
                print(f"    KL = {kl_div:.6f}")
                
                
    # Build DataFrame: rows = n_train, columns = sigma
    df = pd.DataFrame(results_kl)
    df.index.name = "n_train"
    # Order rows and columns explicitly for stability
    df = df.reindex(index=N_TRAIN_KL, columns=SIGMAS_KL)
    df.to_csv(OUTPUT_CSV_KL)
    print(f"\n>> KL results saved to {OUTPUT_CSV_KL}")


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Shared: load test set
    print(f"Loading reference test set from {TEST_FILE} ...")
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    ref_pool = np.load(TEST_FILE)
    print(f"  Reference pool shape: {ref_pool.shape}")

    # Shared: build pi_ref (for KL analytic density)
    print("\nBuilding reference GMM (for analytic density)...")
    pi_ref = build_gmm_ref_dist(
        jrn.key(DIST_SEED),
        scale=SCALE, dim=DIM, global_scale=GLOBAL_SCALE,
        concentration=None, n_grid=N_GRID,
    )
    locs, scale_trils, log_weights = extract_gmm_components(pi_ref)
    log_prob_sigma_fn = make_gmm_log_prob_fn(locs, scale_trils, log_weights)
    print(f"  K={locs.shape[0]} components")

    # ── PART A: MSW ──────────────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("# PART A — Global MSW evaluation")
    print("#" * 70)
    rng = jrn.key(SEED_MSW)
    rng = run_msw_evaluation(ref_pool, rng)

    # ── PART B: KL ───────────────────────────────────────────────────────
    print("\n" + "#" * 70)
    print(f"# PART B — Monte Carlo KL estimation  (modality: {TRAIN_MODALITY})")
    print("#" * 70)
    test_data = jnp.array(ref_pool)  # reuse already-loaded reference test set
    run_kl_evaluation(test_data, log_prob_sigma_fn)

    print("\nDone.")