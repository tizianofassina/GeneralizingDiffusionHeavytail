import os
import csv
import numpy as np
from jax import random as jrn, numpy as jnp

from hf_toy.bulk_metrics import max_sliced_wasserstein
from gen_flow_ht import HeavyTailFlow

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR_NN   = "data/data_gen_diffusion_nn/dim_2"
DATA_DIR_MCMC = "data/data_gen_diffusion_mcmc/dim_2"
TEST_FILE     = "data/dim_2/test_set_dim_2_size_10000000_tail_3_seed_35_dist_seed_0.npy"
TRAIN_FILE    = "data/dim_2/train_set_dim_2_size_10000_tail_3_seed_34_dist_seed_0.npy"

RESULTS_DIR = "results"
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "results_diffusion_ht.csv")

N_DRAWS                  = 10
DRAW_SIZE                = 1_000_000
QUANTILE_LO, QUANTILE_HI = 0.1, 0.9
TAIL_QUANTILES           = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
SEED                     = 42

MCMC_METHODS = ["barker", "hmc", "nuts"]
SIGMAS       = ["0.801241500", "1.05527906", "1.24404052"]
INITS        = ["gaussian", "pt", "p_theta"]
NN_TRAIN_SIZES = [1000, 10000, 100000]

# Flow sigma=0 hyperparameters (same as gen_flow_ht.py)
FLOW_DIM           = 2
FLOW_TAIL_INDEX    = 3
FLOW_LAYERS        = 5
FLOW_NN_ACTIVATION = "relu6"
FLOW_NN_WIDTH      = 50
FLOW_NN_DEPTH      = 3
FLOW_KEY           = 42
FLOW_TRAIN_SEED    = 42
FLOW_SAMPLE_SEED   = 999
FLOW_INFLATION     = 1
FLOW_N_EPOCHS      = 3000
FLOW_LR            = 0.001
FLOW_BATCH_SIZE    = 1000
FLOW_TRAIN_SIZE    = 10000
FLOW_N_SAMPLES     = 1_000_000


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_bulk_quantile_thresholds(ref_data, lo=QUANTILE_LO, hi=QUANTILE_HI):
    thresholds = []
    for d in range(ref_data.shape[1]):
        q_lo = np.quantile(ref_data[:, d], lo)
        q_hi = np.quantile(ref_data[:, d], hi)
        thresholds.append((q_lo, q_hi))
    return thresholds


def apply_bulk_mask(samples, thresholds):
    mask = np.ones(len(samples), dtype=bool)
    for d, (q_lo, q_hi) in enumerate(thresholds):
        mask &= (samples[:, d] >= q_lo) & (samples[:, d] <= q_hi)
    return samples[mask]


def compute_bulk_msw(gen_samples, ref_pool, bulk_thresholds, rng,
                     n_draws=N_DRAWS, draw_size=DRAW_SIZE):
    gen_bulk = apply_bulk_mask(gen_samples, bulk_thresholds)
    print(f"    gen bulk size: {len(gen_bulk)}")

    msw_values = []
    for i in range(n_draws):
        rng, rng_ref, rng_msw = jrn.split(rng, 3)
        idx_ref = jrn.choice(rng_ref, len(ref_pool), shape=(draw_size,), replace=False)
        ref_draw = apply_bulk_mask(np.array(ref_pool[idx_ref]), bulk_thresholds)
        n = min(len(gen_bulk), len(ref_draw))
        msw, _ = max_sliced_wasserstein(
            jnp.array(gen_bulk[:n]),
            jnp.array(ref_draw[:n]),
            key=jrn.fold_in(rng_msw, i),
            tol=1e-7, lr=1e-3, max_iter=10_000, disable_pbar=True,
        )
        msw_values.append(float(msw))
        print(f"    bulk draw {i}: MSW={float(msw):.6f}  (n={n})")
    return msw_values


def find_worst_direction(pt_samples, ref_pool, rng,
                         n_draws=N_DRAWS, draw_size=DRAW_SIZE):
    """Find worst direction using pt_samples vs ref_pool."""
    best_msw = -np.inf
    worst_dir = None
    for i in range(n_draws):
        rng, rng_ref, rng_msw = jrn.split(rng, 3)
        idx_ref = jrn.choice(rng_ref, len(ref_pool), shape=(draw_size,), replace=False)
        ref_draw = jnp.array(ref_pool[idx_ref])
        n = min(len(pt_samples), draw_size)
        msw, direction = max_sliced_wasserstein(
            jnp.array(pt_samples[:n]),
            ref_draw[:n],
            key=jrn.fold_in(rng_msw, i),
            tol=1e-7, lr=1e-3, max_iter=10_000, disable_pbar=True,
        )
        print(f"      dir search draw {i}: MSW={float(msw):.6f}")
        if float(msw) > best_msw:
            best_msw = float(msw)
            worst_dir = np.array(direction)
    print(f"    Worst direction found: MSW={best_msw:.6f}  dir={worst_dir}")
    return worst_dir


def compute_tail_quantiles(gen_samples, ref_pool, worst_dir, rng,
                           n_draws=N_DRAWS, draw_size=DRAW_SIZE):
    """Compute tail quantiles along fixed worst_dir for gen and ref."""
    gen_proj = gen_samples @ worst_dir
    gen_quantiles = np.quantile(gen_proj, TAIL_QUANTILES)

    ref_quantiles_draws = []
    for i in range(n_draws):
        rng, rng_ref = jrn.split(rng)
        idx_ref = jrn.choice(rng_ref, len(ref_pool), shape=(draw_size,), replace=False)
        ref_proj = ref_pool[idx_ref] @ worst_dir
        ref_quantiles_draws.append(np.quantile(ref_proj, TAIL_QUANTILES))
    ref_quantiles_draws = np.array(ref_quantiles_draws)
    return gen_quantiles, ref_quantiles_draws.mean(axis=0), ref_quantiles_draws.std(axis=0)


def empty_tail_dict():
    """Return a dict with all tail quantile fields set to empty (for bulk-only rows)."""
    d = {}
    for q in TAIL_QUANTILES:
        d[f"gen_q{q}"] = ""
        d[f"ref_mean_q{q}"] = ""
        d[f"ref_std_q{q}"] = ""
    return d


def fill_tail_dict(gen_q, ref_q_mean, ref_q_std):
    d = {}
    for i, q in enumerate(TAIL_QUANTILES):
        d[f"gen_q{q}"] = gen_q[i]
        d[f"ref_mean_q{q}"] = ref_q_mean[i]
        d[f"ref_std_q{q}"] = ref_q_std[i]
    return d


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = jrn.key(SEED)

    print("Loading test set...")
    ref_pool = np.load(TEST_FILE)
    print(f"  Reference pool shape: {ref_pool.shape}")

    print("Computing bulk thresholds from full reference...")
    bulk_thresholds = compute_bulk_quantile_thresholds(ref_pool)
    print(f"  Thresholds: {bulk_thresholds}")

    rows = []  # accumulated rows for the unique CSV

    # ─────────────────────────────────────────────────────────────────────
    # PART 1 — MCMC methods: for each (mcmc, sigma) find worst direction
    #          on pt, then evaluate all inits along that direction.
    # ─────────────────────────────────────────────────────────────────────
    for mcmc in MCMC_METHODS:
        for sigma in SIGMAS:
            # 1. Worst direction from pt
            pt_fname = f"samples_pt_{mcmc}_sigma_{sigma}_size_1000000.npy"
            pt_path  = os.path.join(DATA_DIR_MCMC, pt_fname)
            assert os.path.isfile(pt_path), f"Missing pt file: {pt_path}"

            print(f"\n{'='*70}")
            print(f"Worst dir search:  mcmc={mcmc}  sigma={sigma}")
            pt_samples = np.load(pt_path)
            rng, rng_dir = jrn.split(rng)
            worst_dir = find_worst_direction(pt_samples, ref_pool, rng_dir)

            # 2. Evaluate each init using this worst direction
            for init in INITS:
                fname = f"samples_{init}_{mcmc}_sigma_{sigma}_size_1000000.npy"
                fpath = os.path.join(DATA_DIR_MCMC, fname)
                assert os.path.isfile(fpath), f"Missing file: {fpath}"

                print(f"\n--- method={mcmc}  sigma={sigma}  init={init} ---")
                gen_samples = np.load(fpath)
                print(f"  gen shape: {gen_samples.shape}")

                # Bulk MSW
                rng, rng_bulk = jrn.split(rng)
                bulk_vals = compute_bulk_msw(gen_samples, ref_pool, bulk_thresholds, rng_bulk)

                # Tail quantiles
                rng, rng_tail = jrn.split(rng)
                gen_q, ref_q_mean, ref_q_std = compute_tail_quantiles(
                    gen_samples, ref_pool, worst_dir, rng_tail
                )

                row = {
                    "method":     mcmc,
                    "sigma":      sigma,
                    "init":       init,
                    "train_size": "",
                    "mean_msw":   float(np.mean(bulk_vals)),
                    "std_msw":    float(np.std(bulk_vals)),
                    **fill_tail_dict(gen_q, ref_q_mean, ref_q_std),
                }
                rows.append(row)
                print(f"  => bulk mean={np.mean(bulk_vals):.6f}  std={np.std(bulk_vals):.6f}")

    # ─────────────────────────────────────────────────────────────────────
    # PART 2 — NN method: for each (sigma, train_size) find worst direction
    #          on pt, then evaluate all inits along that direction.
    # ─────────────────────────────────────────────────────────────────────
    for train_size in NN_TRAIN_SIZES:
        for sigma in SIGMAS:
            # 1. Worst direction from pt
            pt_fname = f"samples_pt_nn_sigma_{sigma}_size_{train_size}_1000000.npy"
            pt_path  = os.path.join(DATA_DIR_NN, pt_fname)
            assert os.path.isfile(pt_path), f"Missing pt file: {pt_path}"

            print(f"\n{'='*70}")
            print(f"Worst dir search:  method=nn  sigma={sigma}  train_size={train_size}")
            pt_samples = np.load(pt_path)
            rng, rng_dir = jrn.split(rng)
            worst_dir = find_worst_direction(pt_samples, ref_pool, rng_dir)

            # 2. Evaluate each init using this worst direction
            for init in INITS:
                fname = f"samples_{init}_nn_sigma_{sigma}_size_{train_size}_1000000.npy"
                fpath = os.path.join(DATA_DIR_NN, fname)
                assert os.path.isfile(fpath), f"Missing file: {fpath}"

                print(f"\n--- method=nn  sigma={sigma}  train_size={train_size}  init={init} ---")
                gen_samples = np.load(fpath)
                print(f"  gen shape: {gen_samples.shape}")

                # Bulk MSW
                rng, rng_bulk = jrn.split(rng)
                bulk_vals = compute_bulk_msw(gen_samples, ref_pool, bulk_thresholds, rng_bulk)

                # Tail quantiles
                rng, rng_tail = jrn.split(rng)
                gen_q, ref_q_mean, ref_q_std = compute_tail_quantiles(
                    gen_samples, ref_pool, worst_dir, rng_tail
                )

                row = {
                    "method":     "nn",
                    "sigma":      sigma,
                    "init":       init,
                    "train_size": train_size,
                    "mean_msw":   float(np.mean(bulk_vals)),
                    "std_msw":    float(np.std(bulk_vals)),
                    **fill_tail_dict(gen_q, ref_q_mean, ref_q_std),
                }
                rows.append(row)
                print(f"  => bulk mean={np.mean(bulk_vals):.6f}  std={np.std(bulk_vals):.6f}")

    # ─────────────────────────────────────────────────────────────────────
    # PART 3 — Reference vs reference baseline (single row, bulk only)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Computing reference bulk MSW (ref vs ref)...")
    rng, rng_ref_draw, rng_ref_bulk = jrn.split(rng, 3)
    idx_ref_gen = jrn.choice(rng_ref_draw, len(ref_pool), shape=(DRAW_SIZE,), replace=False)
    ref_as_gen  = ref_pool[np.array(idx_ref_gen)]
    bulk_ref    = compute_bulk_msw(ref_as_gen, ref_pool, bulk_thresholds, rng_ref_bulk)
    rows.append({
        "method":     "reference",
        "sigma":      "-",
        "init":       "-",
        "train_size": "",
        "mean_msw":   float(np.mean(bulk_ref)),
        "std_msw":    float(np.std(bulk_ref)),
        **empty_tail_dict(),
    })
    print(f"  => ref bulk mean={np.mean(bulk_ref):.6f}  std={np.std(bulk_ref):.6f}")

    # ─────────────────────────────────────────────────────────────────────
    # PART 4 — Flow with sigma=0: train on the fly, sample 1M, bulk MSW only.
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Training flow (sigma=0) on the fly...")
    print(f"  Loading train data from {TRAIN_FILE}")
    train_data = jnp.array(np.load(TRAIN_FILE))[:FLOW_TRAIN_SIZE]
    print(f"  Train data shape: {train_data.shape}")

    flow_model = HeavyTailFlow(
        dim=FLOW_DIM,
        sigma=0.0,
        tail_index=FLOW_TAIL_INDEX,
        flow_layers=FLOW_LAYERS,
        nn_activation=FLOW_NN_ACTIVATION,
        nn_width=FLOW_NN_WIDTH,
        nn_depth=FLOW_NN_DEPTH,
        key=FLOW_KEY,
    )
    print("  Training (dynamic mode)...")
    flow_model, _ = flow_model.train_dynamic(
        rng=jrn.key(FLOW_TRAIN_SEED),
        train_data=train_data,
        inflation=FLOW_INFLATION,
        n_epochs=FLOW_N_EPOCHS,
        lr=FLOW_LR,
        batch_size_general=FLOW_BATCH_SIZE,
    )

    print(f"  Sampling {FLOW_N_SAMPLES} points from flow...")
    flow_samples = np.asarray(flow_model.sample(jrn.key(FLOW_SAMPLE_SEED), FLOW_N_SAMPLES))
    print(f"  Flow samples shape: {flow_samples.shape}")

    print("  Computing bulk MSW for flow (sigma=0)...")
    rng, rng_flow_bulk = jrn.split(rng)
    bulk_flow = compute_bulk_msw(flow_samples, ref_pool, bulk_thresholds, rng_flow_bulk)
    rows.append({
        "method":     "flow_sigma0",
        "sigma":      "0",
        "init":       "-",
        "train_size": FLOW_TRAIN_SIZE,
        "mean_msw":   float(np.mean(bulk_flow)),
        "std_msw":    float(np.std(bulk_flow)),
        **empty_tail_dict(),
    })
    print(f"  => flow_sigma0 bulk mean={np.mean(bulk_flow):.6f}  std={np.std(bulk_flow):.6f}")

    # ─────────────────────────────────────────────────────────────────────
    # Write CSV
    # ─────────────────────────────────────────────────────────────────────
    fieldnames = ["method", "sigma", "init", "train_size", "mean_msw", "std_msw"] \
                 + [f"gen_q{q}"      for q in TAIL_QUANTILES] \
                 + [f"ref_mean_q{q}" for q in TAIL_QUANTILES] \
                 + [f"ref_std_q{q}"  for q in TAIL_QUANTILES]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {OUTPUT_CSV}")