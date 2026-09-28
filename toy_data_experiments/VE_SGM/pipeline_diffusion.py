import numpy as np
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset
import os
import torch
from VE_SGM.architecture import AbstractDiffusion
from VE_SGM.sampling import edm_sampling as sampling
from VE_SGM.data_loader import VEDataModule

seed_value = 1000


def check_non_finite(arr: np.ndarray, name: str = ""):
    """
    Check if a numpy array contains NaN or Inf values and print a summary.

    Args:
        arr (np.ndarray): Array to check.
        name (str): Optional name for the array, used in print messages.
    """
    num_nan = np.isnan(arr).sum()
    num_pos_inf = np.isposinf(arr).sum()
    num_neg_inf = np.isneginf(arr).sum()
    num_inf = num_pos_inf + num_neg_inf

    if num_nan or num_inf:
        print(f"⚠️ Non-finite values detected in {name or 'array'}")
        print(
            f"NaNs: {num_nan}, +Inf: {num_pos_inf}, -Inf: {num_neg_inf}, Total Inf: {num_inf}"
        )
    else:
        print(f"✅ No NaN/Inf found in {name or 'array'}")


def pipeline_diffusion(
    data: torch.Tensor,
    config_diffusion: dict,
    log_dir: str,
    model_dir: str,
    name: str,
    sample: torch.Tensor = None,
) -> np.ndarray:
    """
    Training and generation pipeline for a diffusion model using a unified configuration dictionary.
    Data and sample tensors must be on CPU.

    Args:
        data (torch.Tensor): Input training data (N x d).
        config_diffusion (dict): Unified configuration dictionary containing:
            - diffusion_config
            - denoiser_config
            - optim_config
            - trainer_config
        log_dir (str): Directory for TensorBoard logs.
        model_dir (str): Directory to save model checkpoints.
        name (str): Base name for logs and checkpoints.
        sample (torch.Tensor): Initial samples to feed the diffusion sampler.

    Returns:
        np.ndarray: Generated samples as a NumPy array.
    """
    device = data.device
    dtype = torch.float32
    assert data.device == sample.device
    assert torch.device("cpu") == data.device
    trainer_cfg = config_diffusion["trainer_config"]
    diffusion_cfg = config_diffusion["diffusion_config"]
    denoiser_cfg = config_diffusion["denoiser_config"]
    optim_cfg = config_diffusion["optim_config"]

    batch_size_gen = trainer_cfg["batch_gen"]
    total_samples = sample.shape[0]
    n_batches = sample.shape[0] // batch_size_gen

    # Normalize data
    if config_diffusion["normalization"]:
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0)
        data = (data - data_mean) / data_std
        print("Data mean:", data_mean)
        print("Data std:", data_std)

    # Dataset and DataModule
    dataset = TensorDataset(data)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = trainer_cfg["devices"]  # -1 is okay usually for multi gpu
    else:
        accelerator = "cpu"
        devices = 1
    data_module = VEDataModule(
        base_dataset=dataset,
        batch_size=trainer_cfg["batch_size"],
        diffusion_cfg=diffusion_cfg,
        num_workers=trainer_cfg["num_workers"],
    )

    # Trainer setup
    trainer = L.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=accelerator,
        devices=devices,
        logger=TensorBoardLogger(save_dir=log_dir, name=f"{name}_logger"),
        strategy=trainer_cfg["strategy"],
    )

    # Initialize diffusion model
    with trainer.init_module():
        diffusion_model = AbstractDiffusion(
            diffusion_config=diffusion_cfg,
            optim_config=optim_cfg,
            denoiser_config=denoiser_cfg,
            validation_sigmas=range(0, int(diffusion_cfg["sigma_max"]), 1),
            batch_size=trainer_cfg["batch_size"],
        ).to(dtype=dtype)

    # Train model
    trainer.fit(model=diffusion_model, datamodule=data_module)

    # Save checkpoint
    checkpoint_path = os.path.join(model_dir, f"{name}_model.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"{name}_model saved.")

    # GENERATION

    # WE ENSURE A CORRECT DEVICE AND DTYPE
    sigma_min = diffusion_cfg["sigma_min"]
    sigma_max = diffusion_cfg["sigma_max"]
    n_steps = trainer_cfg["n_steps"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion_model.eval()
    if sample == None:
        print("No generation will be performed. Pipeline finished")
        return
    denoiser_fn = diffusion_model.denoiser.eval().requires_grad_(False).to(device)
    generated_samples = []
    sample = sample.to(device=device, dtype=dtype)
    if config_diffusion["normalization"]:
        data_mean = data_mean.to(device)
        data_std = data_std.to(device)

    print(f"Starting generation on {device}...")
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {batch_size_gen}")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size_gen
            end_idx = (
                (batch_idx + 1) * batch_size_gen
                if batch_idx < n_batches - 1
                else total_samples
            )

            batch_samples = sampling(
                sample[start_idx:end_idx, :],
                denoiser_fn,
                n_steps=n_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=7.0,
                device=device,
            )
            if config_diffusion["normalization"]:
                batch_samples = batch_samples * data_std + data_mean

            generated_samples.append(batch_samples)

            percent = (batch_idx + 1) / n_batches * 100
            print(f"\r{name} : Generation progress {percent:.5f}% ", end="", flush=True)

    generation = torch.cat(generated_samples, dim=0).cpu().numpy()
    check_non_finite(generation, "Generated data " + name)
    return generation
