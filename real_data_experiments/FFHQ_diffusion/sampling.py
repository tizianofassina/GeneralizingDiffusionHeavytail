import torch
from torch import distributions as dist
from tqdm import tqdm
import matplotlib.pyplot as plt
import lightning as L
from functools import partial
import math
from typing import Dict, Any
import os
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from lightning import LightningDataModule
from torch import Generator, float as torch_float, randn, randn_like
from torchvision import transforms
import datetime
import torch.nn as nn

# Alias per i tipi di tensori e funzioni random
torch_float = torch.float
randn = torch.randn
randn_like = torch.randn_like


def ddpm_sampling(
    samples: torch.Tensor, sigmas: torch.Tensor, denoiser_fn
) -> torch.Tensor:
    """
    Classic DDPM-style sampling loop using given noise schedule.

    Args:
        samples (torch.Tensor): Initial samples (usually Gaussian noise).
        sigmas (torch.Tensor): Noise schedule for the diffusion process.
        denoiser_fn (callable): Function that predicts the denoised data x0 given samples and sigma.

    Returns:
        torch.Tensor: Generated samples after running the DDPM reverse process.
    """
    for sigma_tm1, sigma_t in zip(reversed(sigmas[:-1]), reversed(sigmas[1:])):
        pred_x0 = denoiser_fn(samples, sigma_t[None])
        mean = pred_x0 + (sigma_tm1**2 / sigma_t**2) * (samples - pred_x0)
        std = ((sigma_tm1**2 / sigma_t**2) * (sigma_t**2 - sigma_tm1**2)) ** 0.5
        samples = mean + std * torch.randn_like(samples)
    return samples


def edm_sampling(
    x: torch.Tensor,
    denoiser_fn,
    n_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sampling using the EDM (Elucidated Diffusion Models) second-order scheme.

    Args:
        x (torch.Tensor): Initial noise samples.
        denoiser_fn (callable): Denoiser function.
        n_steps (int): Number of sampling steps.
        sigma_min (float): Minimum noise level.
        sigma_max (float): Maximum noise level.
        rho (float): Exponent for geometric noise schedule.
        device (str): Device to run computation on.

    Returns:
        torch.Tensor: Generated samples.
    """
    if x.device != device:
        x = x.to(device)

    t = torch.linspace(0, 1, n_steps, device=device)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max**inv_rho + t * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho

    for i in range(len(sigmas) - 1):
        sigma_t = sigmas[i].expand(x.shape[0]).to(device)
        sigma_s = sigmas[i + 1].expand(x.shape[0]).to(device)
        # Compute first slope (k1)
        x0_pred = denoiser_fn(x, sigma_t)
        k1 = (x - x0_pred) / sigma_t[:, None]

        # Euler step
        x_pred = x + (sigma_s - sigma_t)[:, None] * k1

        # Compute second slope (k2)
        x0_pred_pred = denoiser_fn(x_pred, sigma_s)
        k2 = (x_pred - x0_pred_pred) / sigma_s[:, None]

        # Update samples
        x = x + (sigma_s - sigma_t)[:, None] * 0.5 * (k1 + k2)
    return x


import torch


import torch


def edm_sampling_images(
    x: torch.Tensor,
    denoiser_fn,
    n_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    EDM (Elucidated Diffusion Models) 2nd-order (Heun) sampler for images.

    Args:
        x (torch.Tensor): Initial noise tensor [B, C, H, W].
        denoiser_fn (callable): Function f(x, sigma) returning denoised image.
        n_steps (int): Number of diffusion steps.
        sigma_min (float): Minimum noise level.
        sigma_max (float): Maximum noise level.
        rho (float): Rho for geometric noise schedule.
        device (str): Device ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Final denoised images.
    """
    x = x.to(device)

    # Time schedule
    t = torch.linspace(0, 1, n_steps, device=device)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max**inv_rho + t * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho
    sigmas = sigmas.to(device)

    for i in range(n_steps - 1):
        # Batch-wise sigma for broadcasting
        sigma_t = sigmas[i].expand(x.shape[0]).view(-1, *[1] * (x.ndim - 1))
        sigma_s = sigmas[i + 1].expand(x.shape[0]).view(-1, *[1] * (x.ndim - 1))

        # --- Heun step ---
        # 1. Denoise at current sigma
        x0_pred = denoiser_fn(x, sigma_t)

        # 2. Compute slope
        d = (x - x0_pred) / sigma_t

        # 3. Euler prediction
        x_euler = x + (sigma_s - sigma_t) * d

        # 4. Denoise again at next sigma
        x0_pred_2 = denoiser_fn(x_euler, sigma_s)
        d_2 = (x_euler - x0_pred_2) / sigma_s

        # 5. Heun correction
        x = x + (sigma_s - sigma_t) * 0.5 * (d + d_2)

    return x


def edm_sampling_images_vector(
    x: torch.Tensor,
    denoiser_fn,
    sigmas: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    EDM (Elucidated Diffusion Models) 2nd-order (Heun) sampler for images.

    Args:
        x (torch.Tensor): Initial noise tensor [B, C, H, W].
        denoiser_fn (callable): Function f(x, sigma) returning denoised image.
        n_steps (int): Number of diffusion steps.
        rho (float): Rho for geometric noise schedule.
        device (str): Device ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Final denoised images.
    """
    x = x.to(device)

    # Time schedule
    n_steps = sigmas.shape[0]
    sigmas = sigmas.to(device)

    for i in range(n_steps - 1):
        # Batch-wise sigma for broadcasting
        sigma_t = sigmas[i].expand(x.shape[0]).view(-1, *[1] * (x.ndim - 1))
        sigma_s = sigmas[i + 1].expand(x.shape[0]).view(-1, *[1] * (x.ndim - 1))

        # --- Heun step ---
        # 1. Denoise at current sigma
        x0_pred = denoiser_fn(x, sigma_t)

        # 2. Compute slope
        d = (x - x0_pred) / sigma_t

        # 3. Euler prediction
        x_euler = x + (sigma_s - sigma_t) * d

        # 4. Denoise again at next sigma
        x0_pred_2 = denoiser_fn(x_euler, sigma_s)
        d_2 = (x_euler - x0_pred_2) / sigma_s

        # 5. Heun correction
        x = x + (sigma_s - sigma_t) * 0.5 * (d + d_2)

    return x


def edm_sampling_images_karras(
    x: torch.Tensor,
    denoiser_fn,
    n_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: str = "cuda",
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
    round_sigma_fn=None,  # funzione opzionale per allineare sigma al modello
) -> torch.Tensor:
    """
    Adjusted EDM sampler (Heun 2nd-order + optional stochasticity).

    Args:
        x: initial noise [B, C, H, W]
        denoiser_fn: function f(x, sigma)
        n_steps: number of steps
        sigma_min, sigma_max, rho: noise schedule
        device: cuda or cpu
        S_churn: stochasticity factor
        S_min, S_max: range for stochasticity
        S_noise: scale of added noise
        round_sigma_fn: optional rounding of sigma (if model requires)

    Returns:
        Generated images [B, C, H, W]
    """
    x = x.to(device)
    batch_size = x.shape[0]

    # Geometric noise schedule
    t = torch.linspace(0, 1, n_steps, device=device)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max**inv_rho + t * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho

    for i in range(n_steps - 1):
        sigma_t = sigmas[i].expand(batch_size, *[1] * (x.ndim - 1))
        sigma_s = sigmas[i + 1].expand(batch_size, *[1] * (x.ndim - 1))

        # --- Optional stochasticity ---
        gamma = torch.zeros_like(sigma_t)
        mask = (sigmas[i] >= S_min) & (sigmas[i] <= S_max)
        if mask.any() and S_churn > 0:
            gamma_val = min(S_churn / n_steps, np.sqrt(2) - 1)
            gamma[mask] = gamma_val
            sigma_hat = sigma_t * (1 + gamma)
            x_hat = x + torch.sqrt(
                (sigma_hat**2 - sigma_t**2)
            ) * S_noise * torch.randn_like(x)
        else:
            sigma_hat = sigma_t
            x_hat = x

        # --- Heun step ---
        # 1. Denoise at current sigma_hat
        x0_pred = denoiser_fn(
            x_hat, sigma_hat if round_sigma_fn is None else round_sigma_fn(sigma_hat)
        )

        # 2. Compute slope
        d = (x_hat - x0_pred) / sigma_hat

        # 3. Euler prediction
        x_euler = x_hat + (sigma_s - sigma_hat) * d

        # 4. Denoise at next sigma
        x0_pred_2 = denoiser_fn(
            x_euler, sigma_s if round_sigma_fn is None else round_sigma_fn(sigma_s)
        )
        d_2 = (x_euler - x0_pred_2) / sigma_s

        # 5. Heun correction
        x = x_hat + (sigma_s - sigma_hat) * 0.5 * (d + d_2)

    return x


def euler_sampling(
    x: torch.Tensor,
    denoiser_fn,
    n_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sampling using a first-order Euler-Maruyama scheme for stochastic diffusion.

    Args:
        x (torch.Tensor): Initial noise samples.
        denoiser_fn (callable): Denoiser function.
        n_steps (int): Number of sampling steps.
        sigma_min (float): Minimum noise level.
        sigma_max (float): Maximum noise level.
        rho (float): Exponent for geometric noise schedule.
        device (str): Device to run computation on.

    Returns:
        torch.Tensor: Generated samples.
    """
    t = torch.linspace(0, 1, n_steps, device=device)
    inv_rho = 1.0 / rho
    sigmas_base = sigma_max**inv_rho + t * (sigma_min**inv_rho - sigma_max**inv_rho)
    sigmas = sigmas_base**rho
    sigmas = torch.cat(
        [sigmas, torch.zeros(1, device=device)]
    )  # Append zero for last step

    for i in range(n_steps):
        z = torch.randn_like(x)
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        # Compute score
        score = (denoiser_fn(x, sigma_i.expand(x.shape[0])) - x) / (sigma_i**2)

        # Compute drift and diffusion
        delta_var = sigma_i**2 - sigma_ip1**2
        drift = delta_var * score
        diffusion = torch.sqrt(delta_var) * z

        # Update samples
        x = x + drift + diffusion

    return x


import torch


def euler_sampling_images(
    x: torch.Tensor,
    denoiser_fn,
    n_steps: int = 20,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Stochastic Euler-Maruyama sampling for images (batch [B, C, H, W]).

    Args:
        x (torch.Tensor): Initial noise samples [B, C, H, W].
        denoiser_fn (callable): Function f(x, sigma) returning denoised image.
        n_steps (int): Number of diffusion steps.
        sigma_min (float): Minimum noise level.
        sigma_max (float): Maximum noise level.
        rho (float): Exponent for geometric noise schedule.
        device (str): Device to run computation on.

    Returns:
        torch.Tensor: Generated images [B, C, H, W].
    """
    x = x.to(device)

    # Noise schedule
    t = torch.linspace(0, 1, n_steps, device=device)
    inv_rho = 1.0 / rho
    sigmas_base = sigma_max**inv_rho + t * (sigma_min**inv_rho - sigma_max**inv_rho)
    sigmas = sigmas_base**rho
    sigmas = torch.cat(
        [sigmas, torch.zeros(1, device=device)]
    )  # Append 0 for last step

    for i in range(n_steps):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]

        # Broadcast sigma for batch and image dimensions
        sigma_i_exp = sigma_i.view(1, 1, 1, 1).expand_as(x)
        sigma_ip1_exp = sigma_ip1.view(1, 1, 1, 1).expand_as(x)

        # Compute score (denoiser output)
        score = (denoiser_fn(x, sigma_i_exp) - x) / (sigma_i_exp**2)

        # Drift and diffusion
        delta_var = sigma_i_exp**2 - sigma_ip1_exp**2
        drift = delta_var * score
        diffusion = torch.sqrt(delta_var) * torch.randn_like(x)

        # Update samples
        x = x + drift + diffusion

    return x
