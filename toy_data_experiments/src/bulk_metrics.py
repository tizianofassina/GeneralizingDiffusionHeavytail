"""
Compute Max Sliced Wasserstein distance (truncated to [-4.5, 4.5])
for all generated samples vs test set.
Results saved to CSV with columns:
  init_diffusion, sigma, train_size, inflation, denoising, mean_max_sw, std_max_sw
"""


import os
import re
import numpy as np
import csv
from jax import random as jrn, numpy as jnp
from jax import grad, vmap, disable_jit, value_and_grad, jit
import matplotlib.pyplot as plt
from typing import Tuple
from optax import adam, apply_updates
from tqdm import tqdm

def _from_param_to_angle(param: jnp.ndarray) -> jnp.ndarray:
    """
    Stereographic projection (https://en.wikipedia.org/wiki/Stereographic_projection)
    Args:
        param (torch.Tensor): d-1 tensor

    Returns:
        torch.Tensor: d tensor with unit norm
    """
    s2 = (param**2).sum()
    x_0 = (s2 - 1) / (s2 + 1)
    other_coords = 2 * param / (s2 + 1)
    angle = jnp.concat((x_0 * jnp.ones((1,)), other_coords))
    return angle


def _1D_wasserstein(
    samples_1: jnp.ndarray, samples_2: jnp.ndarray, p: int = 2
) -> jnp.ndarray:
    """
    Calculate 1d wasserstein distance
    Args:
        samples_1 (torch.Tensor): Samples 1
        samples_2 (torch.Tensor): Samples 2
        p (int): p

    Returns:
        torch.Tensor: Wasserstein
    """
    diffs = jnp.sort(samples_1) - jnp.sort(samples_2)

    wasserstein_distance = jnp.pow(jnp.abs(diffs), p).mean() ** (1 / p)
    return wasserstein_distance


def calc_err(param, samples_1, samples_2, p):
    angle = _from_param_to_angle(param)
    err = _1D_wasserstein(samples_1 @ angle, samples_2 @ angle, p=p)
    return err


def max_sliced_wasserstein(
    samples_1: jnp.ndarray,
    samples_2: jnp.ndarray,
    key: jrn.PRNGKey,
    tol: float = 1e-7,
    lr: float = 1e-3,
    p: int = 2,
    max_iter: int = 100_000,
    disable_pbar=False,
) -> Tuple[float, jnp.ndarray]:
    """
    Calculate maximum Wasserstein distance by solving optimization problem.
    Args:
        samples_1 (torch.Tensor): First set of samples (n_samples, dim)
        samples_2 (torch.Tensor): Second set of samples (n_samples, dim)
        tol (float, optional): Tolerance of the optimization procedure. Defaults to 1e-4.
        p (int, optional): Wassertein parameter. Defaults to 2.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10_000.
        disable_pbar (bool, optional): Disable tqdm pbar. Defaults to False.

    Returns:
        Tuple[float, Torch.Tensor]: Returns the value of the wasserstein and the optimal direction
    """
    device = samples_1.device
    assert device == samples_2.device
    n_1, d_1 = samples_1.shape
    n_2, d_2 = samples_2.shape
    assert d_1 == d_2
    d_x = d_1

    # initialization
    latent_param = jrn.normal(key=key, shape=(d_x - 1,))
    solver = adam(learning_rate=lr)
    opt_state = solver.init(latent_param)

    grad_fn = value_and_grad(lambda x: calc_err(x, samples_1, samples_2, p))

    @jit
    def calc_and_grad(latent_param, opt_state):
        err, grad = grad_fn(latent_param)
        updates, opt_state = solver.update(-grad, opt_state, latent_param)
        latent_param = apply_updates(latent_param, updates)
        return err, latent_param, opt_state

    err = 0 * jnp.ones((1,))
    old_err = -10 * jnp.ones((1))
    pbar = tqdm(range(max_iter), disable=disable_pbar)
    for i in pbar:

        err, latent_param, opt_state = calc_and_grad(latent_param, opt_state)
        if jnp.abs(err - old_err) < tol:
            break
        else:
            old_err = err
        if (i%100== 99):
            pbar.set_postfix({"err": err})
        if err.item() != err.item():
            raise ValueError(f"{latent_param}, {err}")

    return err, _from_param_to_angle(latent_param)

# ── Config ──────────────────────────────────────────────────────────────────
DIM = 2
BATCH_SIZE = 1_000_000
N_BATCHES = 10  # 10M / 1M
CLIP = 4.5
SEED = 123
DATA_DIR = "data/data_gen/dim_2"
TEST_FILE_70 = "data/dim_2/test_set_dim_2_size_10000000_tail_3_seed_70_dist_seed_0.npy"
TEST_FILE_44 = "data/dim_2/test_set_dim_2_size_10000000_tail_3_seed_44_dist_seed_0.npy"
OUTPUT_CSV = "results_max_sw.csv"


# ── Helpers ─────────────────────────────────────────────────────────────────
def clip_samples(x, clip=CLIP):
    mask = (jnp.abs(x) <= clip).all(axis=-1)
    return x[mask]


def compute_max_sw_batches(gen_data, ref_data, rng, n_batches=N_BATCHES,
                           batch_size=BATCH_SIZE):
    """Shuffle both arrays, split into batches, compute max-SW per batch."""
    rng_shuf_gen, rng_shuf_ref = jrn.split(rng)

    # Reproducible shuffle via permutation
    perm_gen = jrn.permutation(rng_shuf_gen, len(gen_data))
    perm_ref = jrn.permutation(rng_shuf_ref, len(ref_data))
    gen_data = gen_data[perm_gen]
    ref_data = ref_data[perm_ref]

    results = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_gen = jnp.array(gen_data[start:end])
        batch_ref = jnp.array(ref_data[start:end])

        # Clip
        batch_gen_c = clip_samples(batch_gen)
        batch_ref_c = clip_samples(batch_ref)
        n = min(len(batch_gen_c), len(batch_ref_c))
        if n < 1000:
            print(f"  [warn] batch {i}: only {n} samples after clipping, skipping")
            continue

        rng_sw = jrn.fold_in(rng, i)
        msw, _ = max_sliced_wasserstein(
            samples_1=batch_gen_c[:n],
            samples_2=batch_ref_c[:n],
            key=rng_sw,
            tol=1e-9,
            lr=1e-4,
            disable_pbar=True,
        )
        msw_norm = msw / (DIM ** 0.5)
        results.append(float(msw_norm))
        print(f"    batch {i}: max-SW = {msw_norm:.6f}  (n={n})")

    return results


def parse_filename(fname):
    """Extract metadata from a generated-data filename.

    Returns dict with keys:
        init_diffusion, sigma, train_size, inflation, denoising
    or None if the file doesn't match expected patterns.
    """
    # Determine init_diffusion from prefix
    if fname.startswith("samples_pt_flow_"):
        init_diffusion = "p_t"
        rest = fname[len("samples_pt_flow_"):]
    elif fname.startswith("samples_normal_flow_"):
        init_diffusion = "normal"
        rest = fname[len("samples_normal_flow_"):]
    elif fname.startswith("samples_flow_"):
        init_diffusion = "flow"
        rest = fname[len("samples_flow_"):]
    else:
        return None

    # Extract parameters with regex
    m_sigma = re.search(r'_sigma_([\d.]+)', rest)
    m_train = re.search(r'_train_size_(\d+)', rest)
    m_infl = re.search(r'_inflation_(\d+)', rest)

    if not (m_sigma and m_train and m_infl):
        return None

    sigma = m_sigma.group(1)
    train_size = m_train.group(1)
    inflation = m_infl.group(1)

    # Determine denoising method
    if '_denoised_nn' in rest:
        denoising = "NN"
    elif '_denoised_n_mcmc_' in rest:
        denoising = "MCMC"
    else:
        denoising = "none"

    return {
        "init_diffusion": init_diffusion,
        "sigma": sigma,
        "train_size": train_size,
        "inflation": inflation,
        "denoising": denoising,
    }


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading test set (seed 43)...")
    test_43 = np.load(TEST_FILE_70)
    print(f"  shape: {test_43.shape}")

    rng = jrn.key(SEED)

    # Collect all generated files
    gen_files = sorted(os.listdir(DATA_DIR))
    rows = []

    for fname in gen_files:
        if not fname.endswith(".npy"):
            continue
        meta = parse_filename(fname)
        if meta is None:
            print(f"Skipping unrecognized file: {fname}")
            continue

        print(f"\n{'='*80}")
        print(f"init={meta['init_diffusion']}  sigma={meta['sigma']}  "
              f"train_size={meta['train_size']}  inflation={meta['inflation']}  "
              f"denoising={meta['denoising']}")

        filepath = os.path.join(DATA_DIR, fname)
        gen_data = np.load(filepath)
        print(f"  gen shape: {gen_data.shape}")

        rng, rng_batch = jrn.split(rng)
        sw_values = compute_max_sw_batches(gen_data, test_43, rng_batch)

        if len(sw_values) > 0:
            mean_sw = np.mean(sw_values)
            std_sw = np.std(sw_values)
        else:
            mean_sw = float("nan")
            std_sw = float("nan")

        print(f"  => mean={mean_sw:.6f}  std={std_sw:.6f}")
        rows.append({**meta, "mean_max_sw": mean_sw, "std_max_sw": std_sw})

    # ── Reference row: test_43 vs test_44 ───────────────────────────────────
    print(f"\n{'='*80}")
    print("Computing reference: test_43 vs test_44")
    test_44 = np.load(TEST_FILE_44)
    print(f"  test_44 shape: {test_44.shape}")

    rng, rng_ref = jrn.split(rng)
    ref_values = compute_max_sw_batches(test_43, test_44, rng_ref)
    mean_ref = np.mean(ref_values)
    std_ref = np.std(ref_values)
    print(f"  => ref mean={mean_ref:.6f}  std={std_ref:.6f}")

    rows.append({
        "init_diffusion": "reference",
        "sigma": "-",
        "train_size": "-",
        "inflation": "-",
        "denoising": "-",
        "mean_max_sw": mean_ref,
        "std_max_sw": std_ref,
    })

    # ── Worst direction: full 10M test_43 vs test_44 (NO clip) ─────────
    print(f"\n{'='*80}")
    print("Computing worst direction on full test sets (NO clip)...")
    rng, rng_wd = jrn.split(rng)
    _, worst_direction = max_sliced_wasserstein(
        samples_1=jnp.array(test_43),
        samples_2=jnp.array(test_44),
        key=rng_wd,
        tol=1e-9,
        lr=1e-4,
        disable_pbar=False,
    )
    np.save("worst_direction_ref.npy", np.array(worst_direction))
    print(f"  Worst direction saved to worst_direction_ref.npy")
    print(f"  Direction: {worst_direction}")

    # ── Write CSV ───────────────────────────────────────────────────────────
    fieldnames = ["init_diffusion", "sigma", "train_size", "inflation",
                  "denoising", "mean_max_sw", "std_max_sw"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {OUTPUT_CSV}")

    # ── QQ tail plots along worst direction ─────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR = "plot_results_dim_2"
    os.makedirs(PLOT_DIR, exist_ok=True)

    QUANTILES = [0.99, 0.995, 0.999, 0.9996, 0.9999]
    Q_LABELS = ["0.99", "0.995", "0.999", "0.9996", "0.9999"]
    x_positions = np.arange(len(QUANTILES))  # equidistant on x-axis

    # Precompute test quantiles per batch (for reference band)
    print("\nPrecomputing test quantile batches along worst direction...")
    rng, rng_shuf_test = jrn.split(rng)
    perm_test = jrn.permutation(rng_shuf_test, len(test_43))
    test_shuffled = test_43[perm_test]

    test_q_per_batch = []  # list of arrays, one per batch
    for i in range(N_BATCHES):
        batch = jnp.array(test_shuffled[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        proj = (batch @ worst_direction).ravel()
        qs = [float(jnp.percentile(proj, q * 100)) for q in QUANTILES]
        test_q_per_batch.append(qs)
    test_q_per_batch = np.array(test_q_per_batch)  # (N_BATCHES, n_quantiles)
    test_q_mean = test_q_per_batch.mean(axis=0)
    test_q_std = test_q_per_batch.std(axis=0)

    # Plot for each generated file
    for fname in gen_files:
        if not fname.endswith(".npy"):
            continue
        meta = parse_filename(fname)
        if meta is None:
            continue

        print(f"\nQQ plot: init={meta['init_diffusion']}  sigma={meta['sigma']}  "
              f"train={meta['train_size']}  infl={meta['inflation']}  "
              f"denois={meta['denoising']}")

        filepath = os.path.join(DATA_DIR, fname)
        gen_data = np.load(filepath)

        rng, rng_shuf_gen = jrn.split(rng)
        perm_gen = jrn.permutation(rng_shuf_gen, len(gen_data))
        gen_shuffled = gen_data[perm_gen]

        gen_q_per_batch = []
        for i in range(N_BATCHES):
            batch = jnp.array(gen_shuffled[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
            proj = (batch @ worst_direction).ravel()
            qs = [float(jnp.percentile(proj, q * 100)) for q in QUANTILES]
            gen_q_per_batch.append(qs)
        gen_q_per_batch = np.array(gen_q_per_batch)
        gen_q_mean = gen_q_per_batch.mean(axis=0)
        gen_q_std = gen_q_per_batch.std(axis=0)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        # Test reference
        ax.plot(x_positions, test_q_mean, "o-", color="#6baed6", label="Test",
                markersize=5, linewidth=1.5)
        ax.fill_between(x_positions,
                         test_q_mean - test_q_std,
                         test_q_mean + test_q_std,
                         color="#6baed6", alpha=0.15)

        # Generated
        ax.plot(x_positions, gen_q_mean, "s-", color="#fd8d3c", label="Generated",
                markersize=5, linewidth=1.5)
        ax.fill_between(x_positions,
                         gen_q_mean - gen_q_std,
                         gen_q_mean + gen_q_std,
                         color="#fd8d3c", alpha=0.15)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(Q_LABELS)
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Value (projected on worst direction)")
        ax.set_title(f"init={meta['init_diffusion']}  σ={meta['sigma']}  "
                     f"train={meta['train_size']}  infl={meta['inflation']}  "
                     f"denois={meta['denoising']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_name = (f"qq_tail_{meta['init_diffusion']}_sigma_{meta['sigma']}"
                     f"_train_{meta['train_size']}_infl_{meta['inflation']}"
                     f"_denois_{meta['denoising']}.png")
        fig.savefig(os.path.join(PLOT_DIR, plot_name), dpi=150)
        plt.close(fig)
        print(f"  Saved {plot_name}")

    print(f"\nAll QQ tail plots saved to {PLOT_DIR}/")
