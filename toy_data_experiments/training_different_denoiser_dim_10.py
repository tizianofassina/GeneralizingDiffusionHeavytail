"""
Train a NN (VE-SGM) denoiser on ONE train-set size (dim-10 samples).

The train-set size is a command-line parameter, so several sizes can be run
in parallel (e.g. as a SLURM job array), one process per size.

    python -u nn_diffusion_dim_10_train_only.py --train-size 10000

Idempotent:
  - checkpoint already present -> training skipped

Self-contained: config is embedded.
Project imports needed for the architecture itself:
    VE_SGM.architecture.AbstractDiffusion
    VE_SGM.data_loader.VEDataModule
"""

import os
import sys
import gc
import math
import time
import argparse

import numpy as np
import torch
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset

from VE_SGM.architecture import AbstractDiffusion
from VE_SGM.data_loader import VEDataModule


# ── Config (embedded; input_dim raised 2 -> 10) ──────────────────────────────
CONFIG = {
    "diffusion_config": {
        "sigma_min": 0.002,
        "sigma_max": 80.0,
        "log_mean": -1.0,
        "log_std": 1.2,
        "num_workers": 6,
    },
    "optim_config": {"lr": 1e-4},
    "denoiser_config": {
        "sigma_data": 1,
        "sigma_min": 0.002,        # same as t-EDM
        "sigma_max": 80.0,
        "sigma_disc": 1000,
        "input_dim": 10,
        "embed_dim": 256,
        "channel_mult": [2, 4, 4, 2],
    },
    "trainer_config": {
        "max_epochs": 20_000,
        "devices": 1,
        "batch_size": 1000,
        "strategy": "auto",
        "num_workers": 6,
    },
}


# ── Problem / data ───────────────────────────────────────────────────────────
DIM         = 10
TAIL_INDEX  = 3
TRAIN_SEED  = 34
DIST_SEED   = 0

DATA_INPUT_DIR = f"data/dim_{DIM}"
MODEL_DIR      = "model_diffusion"
LOG_DIR        = "logs"


def fmt(seconds):
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── Paths ────────────────────────────────────────────────────────────────────
def run_name(train_size):
    return f"vesgm_dim_{DIM}_size_{train_size}_tail_{TAIL_INDEX}"


def checkpoint_path(train_size):
    return os.path.join(MODEL_DIR, f"{run_name(train_size)}_model.ckpt")


def meta_path(train_size):
    return os.path.join(MODEL_DIR, f"{run_name(train_size)}_meta.pt")


def get_train_path(train_size):
    """One file per size; no slicing of a larger file."""
    return os.path.join(
        DATA_INPUT_DIR,
        f"train_set_dim_{DIM}_size_{train_size}_tail_{TAIL_INDEX}"
        f"_seed_{TRAIN_SEED}_dist_seed_{DIST_SEED}.npy")


# ── Training ─────────────────────────────────────────────────────────────────
def train(train_size, max_epochs=None):
    """Train the denoiser for one size. Returns True if a checkpoint is available."""
    ckpt = checkpoint_path(train_size)
    meta = meta_path(train_size)

    if os.path.exists(ckpt):
        print(f"Checkpoint exists, skipping training: {ckpt}", flush=True)
        if not os.path.exists(meta):
            torch.save({"sigma_data": CONFIG["denoiser_config"]["sigma_data"]}, meta)
            print(f"  (meta rebuilt: {meta})", flush=True)
        return True

    path = get_train_path(train_size)
    if not os.path.isfile(path):
        print(f"  MISSING train file for size {train_size}: {path}", flush=True)
        return False

    print(f"Loading train data from {path}", flush=True)
    data = torch.tensor(np.asarray(np.load(path)), dtype=torch.float32)
    assert data.shape == (train_size, DIM), (data.shape, train_size, DIM)
    print(f"  shape {tuple(data.shape)}")

    diffusion_cfg = dict(CONFIG["diffusion_config"])
    denoiser_cfg  = dict(CONFIG["denoiser_config"])
    optim_cfg     = dict(CONFIG["optim_config"])
    trainer_cfg   = dict(CONFIG["trainer_config"])

    sigma_data = 1
    denoiser_cfg["sigma_data"] = sigma_data
    print(f"  sigma_data = {sigma_data:.4f}", flush=True)

    if max_epochs is None:
        max_epochs = trainer_cfg["max_epochs"]
    effective_batch_size = min(trainer_cfg["batch_size"], train_size)
    steps_per_epoch = math.ceil(train_size / effective_batch_size)
    print(f"  batch_size = {effective_batch_size}   max_epochs = {max_epochs}   "
          f"steps/epoch = {steps_per_epoch}   total steps = {max_epochs * steps_per_epoch}",
          flush=True)

    data_module = VEDataModule(
        base_dataset=TensorDataset(data),
        batch_size=effective_batch_size,
        diffusion_cfg=diffusion_cfg,
        num_workers=trainer_cfg["num_workers"],
        val_ptg=0.0,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = trainer_cfg["devices"] if accelerator == "gpu" else 1

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=TensorBoardLogger(save_dir=LOG_DIR, name=f"{run_name(train_size)}_logger"),
        strategy=trainer_cfg["strategy"],
    )

    with trainer.init_module():
        model = AbstractDiffusion(
            diffusion_config=diffusion_cfg,
            optim_config=optim_cfg,
            denoiser_config=denoiser_cfg,
            validation_sigmas=range(1, int(diffusion_cfg["sigma_max"]), 1),
            batch_size=effective_batch_size,
        ).to(dtype=torch.float32)

    print(f"Training {run_name(train_size)} ({max_epochs} epochs)...", flush=True)
    t0 = time.time()
    trainer.fit(model=model, datamodule=data_module)
    print(f"Training done in {fmt(time.time() - t0)}", flush=True)

    trainer.save_checkpoint(ckpt)
    torch.save({"sigma_data": sigma_data}, meta)
    print(f"Saved {ckpt}", flush=True)

    # Free GPU/host memory.
    del model, trainer, data_module, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="VE-SGM denoiser training for a single train-set size.")
    p.add_argument("--train-size", type=int, required=True,
                   help="Number of training samples (one .npy file per size).")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override CONFIG trainer_config.max_epochs for this run.")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"NN DIFFUSION TRAINING  dim={DIM}  tail={TAIL_INDEX}")
    print(f"train size: {args.train_size}   device: {device}")
    print("=" * 80, flush=True)

    t_all = time.time()

    ok = train(args.train_size, max_epochs=args.max_epochs)

    if not ok:
        print("Training could not run (missing train file). Exiting with status 1.",
              flush=True)
        sys.exit(1)

    print(f"\nDone. Total {fmt(time.time() - t_all)}")