import equinox as eqx
import json
from jax import random as jrn
from jax import numpy as jnp
from jax import nn as jnn
from flowjax.flows import coupling_flow
from flowjax import distributions as flow_dist
from flowjax.bijections import TriangularAffine
import optax
import jax
import numpy as np
from hf_toy.bulk_metrics import max_sliced_wasserstein
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

class HeavyTailFlow(eqx.Module):
    flow: eqx.Module
    dim: int
    sigma: float
    tail_index: int
    flow_layers: int
    nn_activation: str
    nn_width: int
    nn_depth: int

    def __init__(self, dim: int, sigma: float, tail_index: int, flow_layers: int,
             nn_activation: str, key: int, nn_width: int, nn_depth: int):
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        if tail_index <= 0:
            raise ValueError(f"tail_index must be positive, got {tail_index}")
        if flow_layers <= 0:
            raise ValueError(f"flow_layers must be positive, got {flow_layers}")
        if nn_width <= 0:
            raise ValueError(f"nn_width must be positive, got {nn_width}")
        if nn_depth <= 0:
            raise ValueError(f"nn_depth must be positive, got {nn_depth}")

        self.dim = dim
        self.sigma = sigma
        self.tail_index = tail_index
        self.flow_layers = flow_layers
        self.nn_activation = nn_activation
        self.nn_width = nn_width
        self.nn_depth = nn_depth

        if self.nn_activation == "relu6":
            nn_activation = jnn.relu6
        else:
            raise NotImplementedError(f"Activation {self.nn_activation} not supported")

        base_dist = flow_dist.Transformed(
            base_dist=flow_dist.StudentT(
                df=tail_index,
                loc=jnp.zeros((dim,)),
                scale=sigma if sigma != 0. else 1.0
            ),
            bijection=TriangularAffine(loc=jnp.zeros((dim,)), arr=jnp.eye(dim)),
        )
        self.flow = coupling_flow(
            key=jrn.key(key),
            base_dist=base_dist,
            flow_layers=flow_layers,
            nn_activation=nn_activation,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

    def sample(self, rng, n_samples: int):
        return self.flow.sample(rng, (n_samples,))

    def log_prob(self, x):
        return self.flow.log_prob(x)

    def save(self, filename: str, losses=None):
        if isinstance(losses, dict):
            losses_serializable = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in losses.items()}
        elif losses is not None:
            losses_serializable = losses.tolist() if hasattr(losses, 'tolist') else losses
        else:
            losses_serializable = None
        hyperparams = {
            "dim": self.dim,
            "sigma": self.sigma,
            "tail_index": self.tail_index,
            "flow_layers": self.flow_layers,
            "nn_activation": self.nn_activation,
            "nn_width": self.nn_width,
            "nn_depth": self.nn_depth,
            "losses": losses_serializable,
        }
        with open(filename, "wb") as f:
            f.write((json.dumps(hyperparams) + "\n").encode())
            eqx.tree_serialise_leaves(f, self.flow)

    @classmethod
    def load(cls, filename: str):
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            obj = cls(
                dim=hyperparams["dim"],
                sigma=hyperparams["sigma"],
                tail_index=hyperparams["tail_index"],
                flow_layers=hyperparams["flow_layers"],
                nn_activation=hyperparams["nn_activation"],
                nn_width=hyperparams["nn_width"],
                nn_depth=hyperparams["nn_depth"],
                key=0,
            )
            obj = eqx.tree_at(
                lambda m: m.flow,
                obj,
                eqx.tree_deserialise_leaves(f, obj.flow)
            )
        return obj, hyperparams

    def train_dynamic(self, rng, train_data, inflation, n_epochs, lr, batch_size_general):
        rng_noise, rng_train = jrn.split(rng, 2)


        n_originals = len(train_data)
        total_inflated_samples = n_originals * inflation
        batch_size  =  min(batch_size_general, total_inflated_samples)
        n_batches = total_inflated_samples // batch_size


        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs * n_batches)
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(eqx.filter(self.flow, eqx.is_inexact_array))

        flow = self.flow
        sigma = self.sigma


        @eqx.filter_jit
        def run_epoch(flow, opt_state, train_data, rng_epoch):
            rng_perm, rng_noise_epoch = jrn.split(rng_epoch, 2)

            indices = jrn.permutation(rng_perm, jnp.arange(total_inflated_samples))
            indices = indices[:n_batches * batch_size]
            indices = indices.reshape(n_batches, batch_size)
            original_indices = indices % n_originals
            batches = train_data[original_indices]

            noise_keys = jrn.split(rng_noise_epoch, n_batches)

            flow_dyn, flow_static = eqx.partition(flow, eqx.is_inexact_array)

            def train_step(carry, batch_inputs):
                flow_dyn, opt_state = carry
                batch_data, rng_noise_batch = batch_inputs
                flow = eqx.combine(flow_dyn, flow_static)

                def loss_fn(f):
                    noise = sigma * jrn.normal(rng_noise_batch, batch_data.shape)
                    noisy_batch = batch_data + noise
                    return -jnp.mean(f.log_prob(noisy_batch))

                loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
                updates, opt_state = optimizer.update(grads, opt_state)
                flow = eqx.apply_updates(flow, updates)

                flow_dyn, _ = eqx.partition(flow, eqx.is_inexact_array)
                return (flow_dyn, opt_state), loss

            (flow_dyn, opt_state), losses = jax.lax.scan(
                train_step, (flow_dyn, opt_state), (batches, noise_keys)
            )
            flow = eqx.combine(flow_dyn, flow_static)
            return flow, opt_state, losses

        losses_all = []
        for epoch in range(n_epochs):
            rng_epoch = jrn.fold_in(rng_train, epoch)
            flow, opt_state, epoch_losses = run_epoch(flow, opt_state, train_data, rng_epoch)
            avg_loss = float(jnp.mean(epoch_losses))
            losses_all.append(avg_loss)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")

        print()
        model = eqx.tree_at(lambda m: m.flow, self, flow)
        return model, losses_all

    def train_fix(self, rng, train_data, inflation, n_epochs, lr, batch_size_general):
        rng_noise_fixed, rng_train = jrn.split(rng, 2)

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self.flow, eqx.is_inexact_array))

        flow = self.flow
        sigma = self.sigma

        n_originals = len(train_data)
        total_inflated_samples = n_originals * inflation
        batch_size  =  min(batch_size_general, total_inflated_samples)
        n_batches = total_inflated_samples // batch_size

        fixed_noise = sigma * jrn.normal(
            rng_noise_fixed, (total_inflated_samples,) + train_data.shape[1:]
        )

        @eqx.filter_jit
        def run_epoch(flow, opt_state, train_data, fixed_noise, rng_perm):
            indices = jrn.permutation(rng_perm, jnp.arange(total_inflated_samples))
            indices = indices[:n_batches * batch_size]
            indices = indices.reshape(n_batches, batch_size)
            original_indices = indices % n_originals
            batches = train_data[original_indices]
            batches_noise = fixed_noise[indices]

            flow_dyn, flow_static = eqx.partition(flow, eqx.is_inexact_array)

            def train_step(carry, batch_inputs):
                flow_dyn, opt_state = carry
                batch_data, batch_noise = batch_inputs
                flow = eqx.combine(flow_dyn, flow_static)

                def loss_fn(f):
                    noisy_batch = batch_data + batch_noise
                    return -jnp.mean(f.log_prob(noisy_batch))

                loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
                updates, opt_state = optimizer.update(grads, opt_state)
                flow = eqx.apply_updates(flow, updates)

                flow_dyn, _ = eqx.partition(flow, eqx.is_inexact_array)
                return (flow_dyn, opt_state), loss

            (flow_dyn, opt_state), losses = jax.lax.scan(
                train_step, (flow_dyn, opt_state), (batches, batches_noise)
            )
            flow = eqx.combine(flow_dyn, flow_static)
            return flow, opt_state, losses

        losses_all = []
        for epoch in range(n_epochs):
            rng_perm = jrn.fold_in(rng_train, epoch)
            flow, opt_state, epoch_losses = run_epoch(flow, opt_state, train_data, fixed_noise, rng_perm)
            avg_loss = float(jnp.mean(epoch_losses))
            losses_all.append(avg_loss)

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")

        print()
        model = eqx.tree_at(lambda m: m.flow, self, flow)
        return model, losses_all


# ============================================================================
# QUANTILE, REPORT & PLOT UTILITIES
# ============================================================================

QUANTILE_ORDERS = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
QUANTILE_BULK = np.linspace(0.1, 0.9, 500)
N_SAMPLES_QUANTILE = 10_000_000

# Seeds for max-SW direction search (5 different initializations)
MSW_SEEDS = [79, 113, 257, 401, 911]

PLOT_DIR = "plots"
MODEL_DIR = "model_flow"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

COLORS = {
    "Reference": "black",
    "Flow":      "steelblue",
}

def compute_quantiles(samples, orders):
    return np.quantile(np.array(samples[:, 0]), orders)

def format_quantile_table(ref_quantiles, model_quantiles_dict, orders):
    header = f"{'Method':<25s}" + "".join(f"{'q=' + str(q):>12s}" for q in orders)
    lines = [header, "-" * len(header)]
    row = f"{'Reference (test data)':<25s}" + "".join(f"{v:>12.4f}" for v in ref_quantiles)
    lines.append(row)
    for name, quantiles in model_quantiles_dict.items():
        row = f"{name:<25s}" + "".join(f"{v:>12.4f}" for v in quantiles)
        lines.append(row)
    return "\n".join(lines)

def make_quantile_plot_1d(out_path, config_key, ref_1d, flow_1d, label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{config_key} | {label}\nBulk (q 0.1–0.9) and Right Tail: Flow vs Reference",
        fontsize=12,
    )
    ref_arr = np.asarray(ref_1d)
    flow_arr = np.asarray(flow_1d)

    q_bulk_ref  = np.quantile(ref_arr,  QUANTILE_BULK)
    q_bulk_flow = np.quantile(flow_arr, QUANTILE_BULK)
    q_tail_ref  = np.quantile(ref_arr,  QUANTILE_ORDERS)
    q_tail_flow = np.quantile(flow_arr, QUANTILE_ORDERS)

    x_bulk = np.arange(len(QUANTILE_BULK))
    x_tail = np.arange(len(QUANTILE_ORDERS))

    axes[0].plot(x_bulk, q_bulk_ref,  color=COLORS["Reference"], lw=2,   alpha=0.6, label="Reference")
    axes[0].plot(x_bulk, q_bulk_flow, color=COLORS["Flow"],      lw=1.8, alpha=0.85, label="Flow")
    ax = axes[0]
    n_ticks = 10
    tick_idx = np.linspace(0, len(QUANTILE_BULK) - 1, n_ticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{QUANTILE_BULK[i]:.2f}" for i in tick_idx], rotation=45, ha="right")
    ax.set_title("Bulk Quantiles (projected)", fontsize=11)
    ax.set_xlabel("Quantile level"); ax.set_ylabel("Value")
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(x_tail, q_tail_ref,  color=COLORS["Reference"], lw=2,   alpha=0.6, marker='o', ms=5, label="Reference")
    axes[1].plot(x_tail, q_tail_flow, color=COLORS["Flow"],      lw=1.8, alpha=0.85, marker='o', ms=5, label="Flow")
    ax = axes[1]
    ax.set_xticks(np.arange(len(QUANTILE_ORDERS)))
    ax.set_xticklabels([f"{q:.4g}" for q in QUANTILE_ORDERS], rotation=45, ha="right")
    ax.set_title("Right Tail Quantiles (projected)", fontsize=11)
    ax.set_xlabel("Quantile level"); ax.set_ylabel("Value")
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {out_path}")

def make_quantile_plot(out_path, config_key, ref_samples, flow_samples, dim_idx, label_dim):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{config_key} | {label_dim}\nBulk (q 0.1–0.9) and Right Tail: Flow vs Reference",
        fontsize=12,
    )
    ref_arr = np.array(ref_samples[:, dim_idx])
    flow_arr = np.array(flow_samples[:, dim_idx])

    q_bulk_ref  = np.quantile(ref_arr,  QUANTILE_BULK)
    q_bulk_flow = np.quantile(flow_arr, QUANTILE_BULK)
    q_tail_ref  = np.quantile(ref_arr,  QUANTILE_ORDERS)
    q_tail_flow = np.quantile(flow_arr, QUANTILE_ORDERS)

    x_bulk = np.arange(len(QUANTILE_BULK))
    x_tail = np.arange(len(QUANTILE_ORDERS))

    axes[0].plot(x_bulk, q_bulk_ref,  color=COLORS["Reference"], lw=2,   alpha=0.6, label="Reference")
    axes[0].plot(x_bulk, q_bulk_flow, color=COLORS["Flow"],      lw=1.8, alpha=0.85, label="Flow")
    ax = axes[0]
    n_ticks = 10
    tick_idx = np.linspace(0, len(QUANTILE_BULK) - 1, n_ticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{QUANTILE_BULK[i]:.2f}" for i in tick_idx], rotation=45, ha="right")
    ax.set_title(f"Bulk Quantiles (dim {dim_idx})", fontsize=11)
    ax.set_xlabel("Quantile level"); ax.set_ylabel("Value")
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(x_tail, q_tail_ref,  color=COLORS["Reference"], lw=2,   alpha=0.6, marker='o', ms=5, label="Reference")
    axes[1].plot(x_tail, q_tail_flow, color=COLORS["Flow"],      lw=1.8, alpha=0.85, marker='o', ms=5, label="Flow")
    ax = axes[1]
    ax.set_xticks(np.arange(len(QUANTILE_ORDERS)))
    ax.set_xticklabels([f"{q:.4g}" for q in QUANTILE_ORDERS], rotation=45, ha="right")
    ax.set_title(f"Right Tail Quantiles (dim {dim_idx})", fontsize=11)
    ax.set_xlabel("Quantile level"); ax.set_ylabel("Value")
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {out_path}")


def compute_reference_worst_directions(noisy_test_data, seeds):
    """
    Split noisy_test_data into two disjoint halves and find 5 worst directions
    between them with different seeds. Returns a list of (msw_value, direction) tuples.
    This is the reference 'noise floor' of direction search.
    """
    n = len(noisy_test_data)
    half = n // 2
    a = noisy_test_data[:half]
    b = noisy_test_data[half:2*half]

    print(f"\nComputing reference worst directions (ref-vs-ref, {half} vs {half} samples)...")
    results = []
    for seed in seeds:
        msw, direction = max_sliced_wasserstein(
            a, b, jrn.key(seed),
            tol=1e-7, lr=1e-3, max_iter=10_000, disable_pbar=True,
        )
        dir_np = np.asarray(direction)
        print(f"  seed={seed:4d}  max-SW={float(msw):.6f}  dir={dir_np}")
        results.append((float(msw), dir_np))
    return results


def compute_model_worst_directions(ref_data, flow_samples, seeds):
    """Find 5 worst directions between ref and flow with different seeds."""
    print(f"  Computing 5 worst directions for model (5 seeds)...")
    results = []
    for seed in seeds:
        msw, direction = max_sliced_wasserstein(
            ref_data, flow_samples, jrn.key(seed),
            tol=1e-7, lr=1e-3, max_iter=10_000, disable_pbar=True,
        )
        dir_np = np.asarray(direction)
        print(f"    seed={seed:4d}  max-SW={float(msw):.6f}  dir={dir_np}")
        results.append((float(msw), dir_np))
    return results


def plot_all_directions(base_name, config_key, ref_data, flow_samples,
                        model_dirs, ref_dirs):
    """Generate a plot for each direction (5 model + 5 reference = 10 plots)."""
    ref_np = np.asarray(ref_data)
    flow_np = np.asarray(flow_samples)

    for i, (msw, direction) in enumerate(model_dirs):
        ref_proj  = ref_np @ direction
        flow_proj = flow_np @ direction
        label = (f"MODEL worst_dir seed{MSW_SEEDS[i]} "
                 f"{np.round(direction, 3).tolist()} (max-SW={msw:.4f})")
        out_path = os.path.join(PLOT_DIR, f"{base_name}_worstdir_model_seed{MSW_SEEDS[i]}.png")
        make_quantile_plot_1d(out_path, config_key, ref_proj, flow_proj, label)

    for i, (msw, direction) in enumerate(ref_dirs):
        ref_proj  = ref_np @ direction
        flow_proj = flow_np @ direction
        label = (f"REFERENCE worst_dir seed{MSW_SEEDS[i]} "
                 f"{np.round(direction, 3).tolist()} (ref-vs-ref max-SW={msw:.4f})")
        out_path = os.path.join(PLOT_DIR, f"{base_name}_worstdir_ref_seed{MSW_SEEDS[i]}.png")
        make_quantile_plot_1d(out_path, config_key, ref_proj, flow_proj, label)


# ============================================================================
# CONFIGURATION
# ============================================================================
if __name__ == "__main__":
    TRAIN_MODALITY = sys.argv[1].lower()
    if TRAIN_MODALITY not in ("fix", "dynamic"):
        raise ValueError(f"Unknown TRAIN_MODALITY: {TRAIN_MODALITY!r}. Use 'fix' or 'dynamic'.")

    dim           = 10
    sigma         = [2.3]
    tail_index    = 3
    flow_layers   = 5
    nn_activation = "relu6"
    nn_width      = 50
    nn_depth      = 3
    key_flow      = 42
    inflation     = [1]
    n_epochs      = 20_000
    lr            = 0.001
    train_seed    = 42
    n_train       = [10_000]
    batch_size_general = 1000

    DISTRIBUTION_SEED = 0
    SEED_TRAINING = 34
    SEED_TEST = 35

    AVAILABLE_TRAIN_SIZES = [10_000]

    DATA_INPUT_DIR = "data/dim_10"

    print("\n" + "=" * 80)
    print("Heavy Tail Flow Training")
    print("=" * 80)
    print(f"\n  Optimizer: optax.adam(lr={lr})")
    print(f"  Epochs: {n_epochs}")
    print(f"  MSW seeds: {MSW_SEEDS}")
    print(f"  Data loaded from: {DATA_INPUT_DIR}/\n")
    os.makedirs("data/data_gen_noised", exist_ok=True)
    test_file = os.path.join(
        DATA_INPUT_DIR,
        f"test_set_dim_{dim}_size_10000000_tail_{int(tail_index)}_seed_{SEED_TEST}_dist_seed_{DISTRIBUTION_SEED}.npy"
    )
    print(f"Loading test data from: {test_file}")
    test_data = jnp.load(test_file)
    print(f"  Test data shape: {test_data.shape}")

    rng_test_noise = jrn.key(1)
    noise = jrn.normal(rng_test_noise, test_data.shape)

    configs = []
    for train_size in n_train:
        for inf in inflation:
            for sig in sigma:
                configs.append((train_size, inf, sig))

    def config_sort_key(cfg):
        train_size, inf, sig = cfg
        if train_size == 10_000 and inf == 10:
            return (1, train_size, inf, sig)
        return (0, train_size, inf, sig)

    configs.sort(key=config_sort_key)

    n_configs = len(configs)
    rngs = jrn.split(jrn.key(train_seed), n_configs)
    rng_sample = jrn.key(999)

    ref_dirs_cache = {}

    train_data_cache = {}
    test_losses = {}
    report_sections = []

    for rng_idx, (train_size, inf, sig) in enumerate(configs):
        # Find the smallest available size >= train_size
        available = [s for s in AVAILABLE_TRAIN_SIZES if s >= train_size]
        if not available:
            raise FileNotFoundError(
                f"No train set with at least {train_size} samples available. "
                f"Available sizes: {AVAILABLE_TRAIN_SIZES}"
            )
        source_size = available[0]

        if source_size not in train_data_cache:
            train_file = os.path.join(
                DATA_INPUT_DIR,
                f"train_set_dim_{dim}_size_{source_size}_tail_{int(tail_index)}_seed_{SEED_TRAINING}_dist_seed_{DISTRIBUTION_SEED}.npy"
            )
            print(f"\nLoading train data from: {train_file}")
            train_data_cache[source_size] = jnp.load(train_file)
            print(f"  Train data shape: {train_data_cache[source_size].shape}")

        # Take the first `train_size` samples (subset of the source)
        train_data = train_data_cache[source_size][:train_size]
        print(f"  Using {train_size} samples (from source of size {source_size})")
        rng = rngs[rng_idx]
        noisy_test_data = test_data + sig * noise

        config_key = f"sigma={sig}_inflation={inf}_train_size={train_size}"
        print("\n" + "=" * 80)
        print(f"Configuration: {config_key}")
        print("=" * 80)

        print("\nTraining flow...")
        model = HeavyTailFlow(
            dim=dim, sigma=sig, tail_index=tail_index, flow_layers=flow_layers,
            nn_activation=nn_activation, nn_width=nn_width, nn_depth=nn_depth,
            key=key_flow,
        )
        if TRAIN_MODALITY=="fix":
            model, losses = model.train_fix(
                rng=rng, train_data=train_data, inflation=inf,
                n_epochs=n_epochs, lr=lr, batch_size_general=batch_size_general,
            )
        elif TRAIN_MODALITY=="dynamic":
            model, losses = model.train_dynamic(
                rng=rng, train_data=train_data, inflation=inf,
                n_epochs=n_epochs, lr=lr, batch_size_general=batch_size_general,
            )

        test_losses[config_key] = float(model.log_prob(noisy_test_data).mean())
        print(f"  Test log-prob: {test_losses[config_key]:.6f}")

        print(f"\n  Sampling {N_SAMPLES_QUANTILE} points from flow...")
        rng_s = jrn.fold_in(rng_sample, rng_idx)
        flow_samples = model.sample(rng_s, N_SAMPLES_QUANTILE)

        base_name = (f"flow_dim{dim}_sig{sig}_tail{tail_index}_layers{flow_layers}"
                     f"_w{nn_width}_d{nn_depth}_inf{inf}_n{train_size}_ep{n_epochs}_{TRAIN_MODALITY}")
        np.save(f"data/data_gen_noised/samples_flow_{base_name}.npy", np.asarray(flow_samples))

        # Canonical marginals
        make_quantile_plot(
            os.path.join(PLOT_DIR, f"{base_name}_dim0.png"),
            config_key, noisy_test_data, flow_samples, dim_idx=0, label_dim="dim_0",
        )
        make_quantile_plot(
            os.path.join(PLOT_DIR, f"{base_name}_dim1.png"),
            config_key, noisy_test_data, flow_samples, dim_idx=1, label_dim="dim_1",
        )

        # MODEL worst directions (5 seeds) and REFERENCE worst directions (5 seeds)
        model_dirs = compute_model_worst_directions(noisy_test_data, flow_samples, MSW_SEEDS)

        if sig not in ref_dirs_cache:
            ref_dirs_cache[sig] = compute_reference_worst_directions(noisy_test_data, MSW_SEEDS)
        ref_dirs = ref_dirs_cache[sig]

        plot_all_directions(base_name, config_key, noisy_test_data, flow_samples,
                            model_dirs, ref_dirs)

        # Quantile table (on x_0)
        ref_quantiles  = compute_quantiles(noisy_test_data, QUANTILE_ORDERS)
        flow_quantiles = compute_quantiles(flow_samples,    QUANTILE_ORDERS)
        table_str = format_quantile_table(ref_quantiles, {"Flow": flow_quantiles}, QUANTILE_ORDERS)

        print(f"\n  Saving model to {MODEL_DIR}/...")
        model.save(
            os.path.join(MODEL_DIR, f"{base_name}.eqx"),
            losses=losses,
        )

        # Report
        section = []
        section.append("=" * 80)
        section.append(f"Configuration: {config_key}")
        section.append("=" * 80)
        section.append("")
        section.append("--- Test Log-Probabilities ---")
        section.append(f"  Flow: {test_losses[config_key]:>12.6f}")
        section.append("")
        section.append("--- Max-Sliced Wasserstein (MODEL vs REFERENCE, 5 seeds) ---")
        for i, (msw, d) in enumerate(model_dirs):
            section.append(f"  seed={MSW_SEEDS[i]:4d}  msw={msw:.6f}  dir={d.tolist()}")
        msw_model_arr = np.array([m for m, _ in model_dirs])
        section.append(f"  mean={msw_model_arr.mean():.6f}  std={msw_model_arr.std():.6f}")
        section.append("")
        section.append("--- Max-Sliced Wasserstein (REFERENCE vs REFERENCE, 5 seeds) ---")
        for i, (msw, d) in enumerate(ref_dirs):
            section.append(f"  seed={MSW_SEEDS[i]:4d}  msw={msw:.6f}  dir={d.tolist()}")
        msw_ref_arr = np.array([m for m, _ in ref_dirs])
        section.append(f"  mean={msw_ref_arr.mean():.6f}  std={msw_ref_arr.std():.6f}")
        section.append("")
        section.append("--- Right-Tail Quantiles of x_0 ---")
        section.append(table_str)
        section.append("")
        report_sections.append("\n".join(section))

    # ========================================================================
    # WRITE REPORT
    # ========================================================================
    report_file = f"results_report_{base_name}.txt"
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HEAVY TAIL FLOW - FULL RESULTS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"  dim            = {dim}\n")
        f.write(f"  tail_index     = {tail_index}\n")
        f.write(f"  flow_layers    = {flow_layers}\n")
        f.write(f"  nn_activation  = {nn_activation}\n")
        f.write(f"  nn_width       = {nn_width}\n")
        f.write(f"  nn_depth       = {nn_depth}\n")
        f.write(f"  n_epochs       = {n_epochs}\n")
        f.write(f"  lr             = {lr}\n")
        f.write(f"  batch_size     = {batch_size_general}\n")
        f.write(f"  MSW seeds      = {MSW_SEEDS}\n")
        f.write(f"  quantile_samples = {N_SAMPLES_QUANTILE}\n")
        f.write("\n\n")
        for section in report_sections:
            f.write(section + "\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")
        header = f"{'Config':<45s} {'Flow':>12s}"
        f.write("Test Log-Probabilities:\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for config_key in sorted(test_losses.keys()):
            f.write(f"{config_key:<45s} {test_losses[config_key]:>12.6f}\n")
        f.write("\n")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nFull report written to: {report_file}")
    print(f"Plots saved in: {PLOT_DIR}/")
