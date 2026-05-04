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

    def __init__(self, dim: int, sigma: float, tail_index: int, flow_layers: int, nn_activation: str, key: int, nn_width: int, nn_depth: int):
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

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self.flow, eqx.is_array))

        flow = self.flow
        sigma = self.sigma

        n_originals = len(train_data)
        total_inflated_samples = n_originals * inflation
        batch_size = min(batch_size_general, total_inflated_samples)
        n_batches = total_inflated_samples // batch_size

        @eqx.filter_jit
        def run_epoch(flow, opt_state, train_data, rng_epoch):
            rng_perm, rng_noise_epoch = jrn.split(rng_epoch, 2)

            indices = jrn.permutation(rng_perm, jnp.arange(total_inflated_samples))
            indices = indices[:n_batches * batch_size]
            indices = indices.reshape(n_batches, batch_size)
            original_indices = indices % n_originals
            batches = train_data[original_indices]

            noise_keys = jrn.split(rng_noise_epoch, n_batches)

            flow_dyn, flow_static = eqx.partition(flow, eqx.is_array)

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

                flow_dyn, _ = eqx.partition(flow, eqx.is_array)
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
        opt_state = optimizer.init(eqx.filter(self.flow, eqx.is_array))

        flow = self.flow
        sigma = self.sigma

        n_originals = len(train_data)
        total_inflated_samples = n_originals * inflation
        batch_size = min(batch_size_general, total_inflated_samples)
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

            flow_dyn, flow_static = eqx.partition(flow, eqx.is_array)

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

                flow_dyn, _ = eqx.partition(flow, eqx.is_array)
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
# CONFIGURATION
# ============================================================================
N_SAMPLES_GENERATE = 10_000_000

MODEL_DIR = "model_flow"
os.makedirs(MODEL_DIR, exist_ok=True)


if __name__ == "__main__":
    TRAIN_MODALITY = sys.argv[1].lower()
    if TRAIN_MODALITY not in ("fix", "dynamic"):
        raise ValueError(f"Unknown TRAIN_MODALITY: {TRAIN_MODALITY!r}. Use 'fix' or 'dynamic'.")

    dim           = 2
    sigma         = [2.55969277, 2.28555770, 2.02694511, 1.78385499, 1.24404052, 1.05527906,
                     0.801241500, 0.582129509, 0.397943082, 0.248682220,
                     0.134346923, 0.0549371894, 0.0104530209]
    tail_index    = 3
    flow_layers   = 5
    nn_activation = "relu6"
    nn_width      = 50
    nn_depth      = 3
    key_flow      = 42
    inflation     = [1]
    n_epochs      = 3000
    lr            = 0.001
    train_seed    = 42
    n_train       = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9500, 10000, 100000]
    batch_size_general = 1000

    DISTRIBUTION_SEED = 0
    SEED_TRAINING = 34
    SEED_TEST = 35

    AVAILABLE_TRAIN_SIZES = [1000, 10_000, 100_000]

    DATA_INPUT_DIR = "data/dim_2"

    print("\n" + "=" * 80)
    print("Heavy Tail Flow Training")
    print("=" * 80)
    print(f"\n  Train modality: {TRAIN_MODALITY}")
    print(f"  Optimizer: optax.adam(lr={lr})")
    print(f"  Epochs: {n_epochs}")
    print(f"  Data loaded from: {DATA_INPUT_DIR}/\n")

    os.makedirs("data/data_gen_noised", exist_ok=True)

    configs = []
    for train_size in n_train:
        for inf in inflation:
            for sig in sigma:
                configs.append((train_size, inf, sig))

    def config_sort_key(cfg):
        train_size, inf, sig = cfg
        if train_size == 100_000 and inf == 10:
            return (1, train_size, inf, sig)
        return (0, train_size, inf, sig)

    configs.sort(key=config_sort_key)

    n_configs = len(configs)
    rngs = jrn.split(jrn.key(train_seed), n_configs)
    rng_sample = jrn.key(999)

    train_data_cache = {}

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
        losses = None
        if TRAIN_MODALITY == "fix":
            model, losses = model.train_fix(
                rng=rng, train_data=train_data, inflation=inf,
                n_epochs=n_epochs, lr=lr, batch_size_general=batch_size_general,
            )
        elif TRAIN_MODALITY == "dynamic":
            model, losses = model.train_dynamic(
                rng=rng, train_data=train_data, inflation=inf,
                n_epochs=n_epochs, lr=lr, batch_size_general=batch_size_general,
            )

        print(f"\n  Sampling {N_SAMPLES_GENERATE} points from flow...")
        rng_s = jrn.fold_in(rng_sample, rng_idx)
        flow_samples = model.sample(rng_s, N_SAMPLES_GENERATE)

        base_name = (f"flow_dim{dim}_sig{sig}_tail{tail_index}_layers{flow_layers}"
                     f"_w{nn_width}_d{nn_depth}_inf{inf}_n{train_size}_ep{n_epochs}_{TRAIN_MODALITY}")
        os.makedirs(f"data/data_gen_noised/dim_{dim}", exist_ok=True)
        np.save(f"data/data_gen_noised/dim_{dim}/samples_flow_{base_name}.npy", np.asarray(flow_samples))

        print(f"\n  Saving model to {MODEL_DIR}/...")
        model.save(
        os.path.join(MODEL_DIR, f"{base_name}.eqx"),
            losses=losses,
        )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)