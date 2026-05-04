import os
import yaml
import torch
import numpy as np
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset
import jax.numpy as jnp
import jax.random as jrn

from VE_SGM.architecture import AbstractDiffusion
from VE_SGM.data_loader import VEDataModule
from hf_toy.diffusion_samplers import batch_ddpm

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH    = "config_diffusion.yaml"
LOG_DIR        = "logs"
MODEL_DIR      = "model_diffusion"
DIM            = 2
GEN_DIR        = f"data/data_gen_diffusion_nn/dim_{DIM}"

TAIL_INDEX     = 3
DATA_TRAIN_SEED = 34   # seed used in create_data.py for the saved train sets
DIST_SEED      = 0
DATA_INPUT_DIR = f"data/dim_{DIM}"

TRAIN_SIZES           = [ 1000, 10000, 100_000]
AVAILABLE_TRAIN_SIZES = [1000, 10000, 100000]

SIGMAS_GEN = [1.24404052, 1.05527906, 0.8012415]
INITS      = ["gaussian", "p_t", "p_theta"]

# Flow checkpoint used as p_theta init source
TRAIN_MODALITY_FLOW = "dynamic"
N_TRAIN_FLOW        = 10000
N_EPOCHS_FLOW       = 3000
FLOW_LAYERS         = 5
FLOW_NN_WIDTH       = 50
FLOW_NN_DEPTH       = 3
FLOW_INF            = 1

NUM_SAMPLES      = 1_000_000
NAME_NUM_SAMPLES = 10_000_000
BATCH_SIZE_GEN   = 250_000

SIGMA_MAX           = 3
SIGMA_MIN           = 0.0002
RHO                 = 2.0
NUM_STEPS_DENOISING = 40

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GEN_DIR,   exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_train_file(train_size):
    """Find smallest available file >= train_size and return path + source_size."""
    available = [s for s in AVAILABLE_TRAIN_SIZES if s >= train_size]
    if not available:
        raise FileNotFoundError(
            f"No train set with at least {train_size} samples. "
            f"Available: {AVAILABLE_TRAIN_SIZES}"
        )
    source_size = available[0]
    path = os.path.join(
        DATA_INPUT_DIR,
        f"train_set_dim_{DIM}_size_{source_size}_tail_{TAIL_INDEX}"
        f"_seed_{DATA_TRAIN_SEED}_dist_seed_{DIST_SEED}.npy"
    )
    return path, source_size


def build_sigma_schedule(sigma_max, sigma_min, rho, n_steps, sigma_T):
    """Build EDM schedule truncated at sigma_T."""
    inv_rho      = 1.0 / rho
    step_indices = jnp.linspace(0, 1, n_steps)
    sigmas = (sigma_max**inv_rho
              + step_indices * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho
    sigmas = jnp.sort(sigmas)
    return sigmas[sigmas <= sigma_T]


def get_init_path(init, sigma):
    if init == "gaussian":
        return (f"data/data_gen_noised/dim_{DIM}/"
                f"samples_p_inf_sigma_{sigma}_size_{NAME_NUM_SAMPLES}.npy")
    elif init == "p_t":
        return (f"data/data_gen_noised/dim_{DIM}/"
                f"samples_pt_sigma_{sigma}_size_{NAME_NUM_SAMPLES}.npy")
    elif init == "p_theta":
        return (f"data/data_gen_noised/dim_{DIM}/"
                f"samples_flow_flow_dim{DIM}_sig{sigma}_tail{TAIL_INDEX}"
                f"_layers{FLOW_LAYERS}_w{FLOW_NN_WIDTH}_d{FLOW_NN_DEPTH}"
                f"_inf{FLOW_INF}_n{N_TRAIN_FLOW}_ep{N_EPOCHS_FLOW}_{TRAIN_MODALITY_FLOW}.npy")


def make_nn_denoiser(diffusion_model, device):
    """Wrap AbstractDiffusion.denoiser into a function compatible with batch_ddpm."""
    denoiser_net = diffusion_model.denoiser.eval().to(device)

    def denoiser_fn(x_t, sigma_t, rng=None):
        # x_t: jax array (d,) — single sample
        x_np = np.array(x_t)[None]  # (1, d)
        x_torch = torch.tensor(x_np, dtype=torch.float32, device=device)
        sigma_torch = torch.tensor([float(sigma_t)], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = denoiser_net(x_torch, sigma_torch)  # (1, d)
        return jnp.array(pred.cpu().numpy()[0])        # (d,)

    return denoiser_fn


# ── Training ──────────────────────────────────────────────────────────────────
def train_diffusion(train_size, config_path, train_data_cache):
    run_name        = f"diffusion_dim_{DIM}_size_{train_size}_tail_{TAIL_INDEX}"
    checkpoint_path = os.path.join(MODEL_DIR, f"{run_name}_model.ckpt")
    meta_path       = os.path.join(MODEL_DIR, f"{run_name}_meta.pt")

    if os.path.exists(checkpoint_path):
        print(f"  Checkpoint already exists, skipping training: {checkpoint_path}")
        return

    # Load data
    fpath, source_size = get_train_file(train_size)
    if source_size not in train_data_cache:
        print(f"  Loading train data from {fpath}")
        train_data_cache[source_size] = np.load(fpath)
    data_np = train_data_cache[source_size][:train_size]
    data    = torch.tensor(data_np, dtype=torch.float32)
    print(f"  Using {train_size} samples (source size {source_size})")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    trainer_cfg   = config["trainer_config"]
    diffusion_cfg = config["diffusion_config"]
    denoiser_cfg  = config["denoiser_config"]
    optim_cfg     = config["optim_config"]

    sigma_data = data.std().item()
    denoiser_cfg["sigma_data"] = sigma_data
    print(f"  sigma_data = {sigma_data:.4f}")

    dataset     = TensorDataset(data)
    data_module = VEDataModule(
        base_dataset=dataset,
        batch_size=trainer_cfg["batch_size"],
        diffusion_cfg=diffusion_cfg,
        num_workers=trainer_cfg["num_workers"],
        val_ptg=0.0,
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices     = trainer_cfg["devices"]
    else:
        accelerator = "cpu"
        devices     = 1

    trainer = L.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=accelerator,
        devices=devices,
        logger=TensorBoardLogger(save_dir=LOG_DIR, name=f"{run_name}_logger"),
        strategy=trainer_cfg["strategy"],
    )

    with trainer.init_module():
        diffusion_model = AbstractDiffusion(
            diffusion_config=diffusion_cfg,
            optim_config=optim_cfg,
            denoiser_config=denoiser_cfg,
            validation_sigmas=range(1, int(diffusion_cfg["sigma_max"]), 1),
            batch_size=trainer_cfg["batch_size"],
        ).to(dtype=torch.float32)

    print(f"  Training {run_name}...")
    trainer.fit(model=diffusion_model, datamodule=data_module)

    trainer.save_checkpoint(checkpoint_path)
    torch.save({"sigma_data": sigma_data}, meta_path)
    print(f"  Saved checkpoint: {checkpoint_path}")
    print(f"  Saved meta:       {meta_path}")


# ── Generation ────────────────────────────────────────────────────────────────
def generate(train_size, sigma, init, config_path):
    run_name        = f"diffusion_dim_{DIM}_size_{train_size}_tail_{TAIL_INDEX}"
    checkpoint_path = os.path.join(MODEL_DIR, f"{run_name}_model.ckpt")
    meta_path       = os.path.join(MODEL_DIR, f"{run_name}_meta.pt")

    init_tag = {"gaussian": "gaussian", "p_t": "pt", "p_theta": "p_theta"}[init]
    out_path = os.path.join(
        GEN_DIR,
        f"samples_{init_tag}_nn_sigma_{sigma}_size_{train_size}_{NUM_SAMPLES}.npy"
    )

    if os.path.exists(out_path):
        print(f"  Already exists, skipping: {out_path}")
        return

    # Init samples
    init_path = get_init_path(init, sigma)
    if not os.path.isfile(init_path):
        print(f"  Missing init file, skipping: {init_path}")
        return
    print(f"  Loading init samples from {init_path}")
    initial_samples = jnp.array(np.load(init_path)[:NUM_SAMPLES])
    print(f"    shape: {initial_samples.shape}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    diffusion_cfg = config["diffusion_config"]
    denoiser_cfg  = config["denoiser_config"]
    optim_cfg     = config["optim_config"]
    trainer_cfg   = config["trainer_config"]

    # Load model
    if not os.path.exists(checkpoint_path):
        print(f"  Missing checkpoint, skipping: {checkpoint_path}")
        return
    meta       = torch.load(meta_path)
    sigma_data = meta["sigma_data"]
    denoiser_cfg["sigma_data"] = sigma_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_model = AbstractDiffusion.load_from_checkpoint(
        checkpoint_path,
        diffusion_config=diffusion_cfg,
        optim_config=optim_cfg,
        denoiser_config=denoiser_cfg,
        validation_sigmas=range(1, int(diffusion_cfg["sigma_max"]), 1),
        batch_size=trainer_cfg["batch_size"],
    ).to(device=device, dtype=torch.float32)
    diffusion_model.eval()

    denoiser_fn = make_nn_denoiser(diffusion_model, device)

    # Sigma schedule
    sigmas_truncated = build_sigma_schedule(SIGMA_MAX, SIGMA_MIN, RHO,
                                            NUM_STEPS_DENOISING, sigma)
    print(f"  Schedule: {len(sigmas_truncated)} steps, "
          f"{float(sigmas_truncated[0]):.6f} → {float(sigmas_truncated[-1]):.6f}")

    print(f"  Running DDPM with NN denoiser...")
    samples = batch_ddpm(
        initial_samples,
        rng=jrn.key(43),
        sigmas=sigmas_truncated,
        denoiser_fn=denoiser_fn,
        batch_size=BATCH_SIZE_GEN,
    )

    np.save(out_path, np.array(samples))
    print(f"  Saved {samples.shape} to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_data_cache = {}

    # ── Training loop ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    for train_size in TRAIN_SIZES:
        print(f"\n--- train_size={train_size} ---")
        train_diffusion(train_size, CONFIG_PATH, train_data_cache)

    # ── Generation loop ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GENERATION")
    print("=" * 80)
    for train_size in TRAIN_SIZES:
        for sigma in SIGMAS_GEN:
            for init in INITS:
                print(f"\n--- train_size={train_size}  sigma={sigma}  init={init} ---")
                generate(train_size, sigma, init, CONFIG_PATH)

    print("\nDone.")