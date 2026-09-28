"""Train DLPM (Shariatian et al.) on 100k dim-10 heavy-tailed samples and generate
1M points with the deterministic DLIM sampler. Architecture / lr / lr-schedule from
the authors' 2D config, adapted to dim 10. No validation split: the full 100k train
set is used, matching the t-EDM and VE-SGM runs.
"""

import os
import time

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset

from VE_SGM.architecture_sampler_levy import (
    AbstractDiffusionLevy,
    EMACallbackLevy,
    generate_levy,
)


ALPHA      = 1.7
T_TRAIN    = 4000
K_SAMPLE   = 25
ISOTROPIC  = True
USE_EMA    = True
GROUP_NORM = True
NUNITS     = 464
SCHED_INT  = "epoch"        # "step" = letterale repo (LR muore) | "epoch" = adattato
GRAD_CLIP  = None           # loro: null

CONFIG = {
    "model_config": {
        "nfeatures": 10,
        "nunits": NUNITS,
        "nblocks": 4,
        "time_emb_size": 32,
        "skip_connection": True,
        "group_norm": GROUP_NORM,
        "dropout_rate": 0.0,
    },
    "loss_config": {
        "alpha": ALPHA,
        "T": T_TRAIN,
        "isotropic": ISOTROPIC,
        "rescale_timesteps": True,
    },
    "optim_config": {
        "lr": 5e-3,
        "lr_step_size": 400,
        "lr_gamma": 0.99,
        "sched_interval": SCHED_INT,
        "weight_decay": 1e-4,
    },
    "trainer_config": {
        "max_epochs": 20_000,
        "batch_size": 1000,
        "devices": 1,
        "num_workers": 6,
        "strategy": "auto",
    },
}

DIM        = 10
TAIL_INDEX = 3
TRAIN_SEED = 34
DIST_SEED  = 0
TRAIN_SIZE = 10_000

NUM_SAMPLES = 1_000_000
BATCH_GEN   = 100_000
GEN_SEED    = 43

CKPT_EVERY_N_EPOCHS = 250

DATA_INPUT_DIR = f"data/dim_{DIM}"
GEN_DIR        = f"data/data_gen_diffusion/dim_{DIM}"
MODEL_DIR      = "model_diffusion"
LOG_DIR        = "logs"

os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RUN_NAME = (f"dlpm_dim_{DIM}_size_{TRAIN_SIZE}_tail_{TAIL_INDEX}_alpha_{ALPHA}"
            f"_T_{T_TRAIN}_iso_{int(ISOTROPIC)}_ema_{int(USE_EMA)}"
            f"_gn_{int(GROUP_NORM)}_nu_{NUNITS}_si_{SCHED_INT}_bs_"
            f"{CONFIG['trainer_config']['batch_size']}")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"{RUN_NAME}_model.ckpt")
PERIODIC_PATH   = os.path.join(MODEL_DIR, f"{RUN_NAME}_periodic.ckpt")
META_PATH       = os.path.join(MODEL_DIR, f"{RUN_NAME}_meta.pt")
OUT_PATH = os.path.join(
    GEN_DIR,
    f"samples_dlpm_alpha_{ALPHA}_K_{K_SAMPLE}_nu_{NUNITS}"
    f"_gn_{int(GROUP_NORM)}_size_{TRAIN_SIZE}_{NUM_SAMPLES}_tail_{TAIL_INDEX}.npy",
)


def fmt(seconds):
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load_train():
    path = os.path.join(
        DATA_INPUT_DIR,
        f"train_set_dim_{DIM}_size_{TRAIN_SIZE}_tail_{TAIL_INDEX}"
        f"_seed_{TRAIN_SEED}_dist_seed_{DIST_SEED}.npy",
    )
    print(f"Loading train data from {path}", flush=True)
    arr = np.asarray(np.load(path, mmap_mode="r")[:TRAIN_SIZE])
    assert arr.shape == (TRAIN_SIZE, DIM), arr.shape
    data = torch.tensor(arr, dtype=torch.float32)
    print(f"  std = {data.std().item():.4f}   |x|max = {data.abs().max().item():.1f}",
          flush=True)
    return data


def build_model(batch_size):
    return AbstractDiffusionLevy(
        model_config=CONFIG["model_config"],
        loss_config=CONFIG["loss_config"],
        optim_config=CONFIG["optim_config"],
        batch_size=batch_size,
    )


def train():
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint exists, skipping training: {CHECKPOINT_PATH}", flush=True)
        return

    data = load_train()
    trainer_cfg = CONFIG["trainer_config"]
    bs = min(trainer_cfg["batch_size"], TRAIN_SIZE)

    loader = DataLoader(TensorDataset(data), batch_size=bs, shuffle=True,
                        drop_last=True, num_workers=trainer_cfg["num_workers"],
                        pin_memory=torch.cuda.is_available())
    print(f"  train on all {len(data)} samples | {len(loader)} batches/epoch",
          flush=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = trainer_cfg["devices"] if accelerator == "gpu" else 1

    callbacks = [EMACallbackLevy(rate=0.9999)] if USE_EMA else []
    callbacks.append(ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename=f"{RUN_NAME}_periodic",
        every_n_epochs=CKPT_EVERY_N_EPOCHS,
        save_top_k=1,
        monitor=None,
        enable_version_counter=False,
    ))

    trainer = L.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=accelerator,
        devices=devices,
        strategy=trainer_cfg["strategy"],
        logger=TensorBoardLogger(save_dir=LOG_DIR, name=f"{RUN_NAME}_logger"),
        callbacks=callbacks,
        gradient_clip_val=GRAD_CLIP,
        gradient_clip_algorithm="norm" if GRAD_CLIP else None,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
    )

    with trainer.init_module():
        model = build_model(bs).to(dtype=torch.float32)

    n_par = sum(p.numel() for p in model.model.parameters())
    print(f"Training {RUN_NAME}", flush=True)
    print(f"  {n_par:,} parameters | {trainer_cfg['max_epochs']} epochs "
          f"| lr {CONFIG['optim_config']['lr']} StepLR/{SCHED_INT}", flush=True)

    resume = PERIODIC_PATH if os.path.exists(PERIODIC_PATH) else None
    if resume:
        print(f"  resuming from {resume}", flush=True)

    t0 = time.time()
    trainer.fit(model=model, train_dataloaders=loader, ckpt_path=resume)
    print(f"Training done in {fmt(time.time() - t0)}", flush=True)

    trainer.save_checkpoint(CHECKPOINT_PATH)
    torch.save({"alpha": ALPHA, "T": T_TRAIN, "isotropic": ISOTROPIC,
                "use_ema": USE_EMA, "group_norm": GROUP_NORM, "nunits": NUNITS,
                "batch_size": bs, "n_par": n_par}, META_PATH)
    print(f"Saved {CHECKPOINT_PATH}", flush=True)


def load_for_generation(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = build_model(CONFIG["trainer_config"]["batch_size"])

    if USE_EMA and "ema_shadow_weights" in ckpt:
        model.model.load_state_dict(ckpt["ema_shadow_weights"])
        print("Loaded EMA weights.")
    else:
        if USE_EMA:
            print("WARNING: USE_EMA=True but no 'ema_shadow_weights' in ckpt; "
                  "falling back to online weights.")
        online = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
        model.model.load_state_dict(online)
        print("Loaded online weights.")

    model = model.to(device=device, dtype=torch.float32).eval()
    return model, model.model.eval().to(device)


@torch.no_grad()
def probe_tail_calibration(module, device, n=100_000, t_val=2000):
    """Predicted vs true |eps| by magnitude bucket: tests whether the net can
    represent the tail of the target at all."""
    data = load_train()[:n].to(device)
    eps_t = module._noise_batch(data)
    t = torch.full((data.shape[0],), t_val, device=device, dtype=torch.long)
    x_t = module.bargammas[t][:, None] * data + module.barsigmas[t][:, None] * eps_t
    pred = module.model(x_t, t.float() / module.T)

    nt, npd = eps_t.norm(dim=1), pred.norm(dim=1)
    print(f"\n  tail calibration at t={t_val}:")
    for lo, hi in [(0, 1), (1, 10), (10, 100), (100, 1e3), (1e3, 1e12)]:
        m = (nt >= lo) & (nt < hi)
        if m.sum():
            print(f"    |eps| in [{lo:g},{hi:g}): n={int(m.sum()):>6}  "
                  f"true={nt[m].mean():>10.2f}  pred={npd[m].mean():>10.2f}  "
                  f"ratio={float(npd[m].mean() / nt[m].mean()):.3f}")


if __name__ == "__main__":
    print("=" * 80)
    print(f"DLPM  dim={DIM}  alpha={ALPHA}  T={T_TRAIN}  K={K_SAMPLE}  "
          f"iso={ISOTROPIC}  ema={USE_EMA}  gn={GROUP_NORM}  nunits={NUNITS}  "
          f"bs={CONFIG['trainer_config']['batch_size']}")
    print("=" * 80, flush=True)

    train()

    assert os.path.exists(CHECKPOINT_PATH), f"missing checkpoint {CHECKPOINT_PATH}"
    meta = torch.load(META_PATH, map_location="cpu")
    assert (meta["alpha"] == ALPHA and meta["T"] == T_TRAIN
            and meta["isotropic"] == ISOTROPIC
            and meta["group_norm"] == GROUP_NORM
            and meta["nunits"] == NUNITS), "config di generazione != training"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module, denoiser = load_for_generation(device)

    probe_tail_calibration(module, device)

    if os.path.exists(OUT_PATH):
        print(f"\n  already exists, skipping: {OUT_PATH}", flush=True)
    else:
        print(f"\n--- generating: alpha-stable init, DLIM K={K_SAMPLE} "
              f"({K_SAMPLE - 1} NFE) ---", flush=True)
        torch.manual_seed(GEN_SEED)
        torch.cuda.manual_seed_all(GEN_SEED)

        t0 = time.time()
        samples = generate_levy(
            denoiser, n_samples=NUM_SAMPLES, dim=DIM, alpha=ALPHA,
            n_steps=K_SAMPLE, device=device, isotropic=ISOTROPIC,
            batch_size=BATCH_GEN,
            log_fn=lambda d, n: print(f"    [{d}/{n}]", flush=True),
        )
        n_bad = (~np.isfinite(samples)).any(axis=1).sum()
        np.save(OUT_PATH, samples)
        print(f"  saved {samples.shape} in {fmt(time.time() - t0)}  "
              f"non-finite rows: {n_bad}\n  -> {OUT_PATH}", flush=True)

    print("\nDone.")
