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



@torch.no_grad()
def heun_sampling(x, denoiser_fn, sigmas):
    """2nd-order Heun on the PF-ODE; len(sigmas)-1 steps, 2 NFE each."""
    for i in range(len(sigmas) - 1):
        if not torch.isfinite(x).all():
            raise FloatingPointError(
                f"non-finite x at step {i} (sigma={float(sigmas[i]):.4g}): "
                f"{(~torch.isfinite(x)).sum().item()} bad entries"
            )
        s_t = sigmas[i].expand(x.shape[0])
        s_s = sigmas[i + 1].expand(x.shape[0])

        d = (x - denoiser_fn(x, s_t)) / s_t[:, None]
        x_e = x + (s_s - s_t)[:, None] * d
        d2 = (x_e - denoiser_fn(x_e, s_s)) / s_s[:, None]
        x = x + (s_s - s_t)[:, None] * 0.5 * (d + d2)
    return x

