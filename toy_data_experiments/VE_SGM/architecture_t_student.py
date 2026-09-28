"""t-EDM (Pandey et al.) denoiser for VE-SGM on vector data.

Same backbone as VE_SGM.architecture; changes: Student-t forward noise,
sigma_eff = sigma * sqrt(nu/(nu-2)) in the preconditioning, log-spaced sigma
embedding (linear binning is degenerate at sigma_max=80), Heun PF-ODE sampler
started from a Student-t prior.
"""

import math

import torch
import torch.nn as nn
import lightning as L
import numpy as np
import time
from VE_SGM.architecture import FNetGroupNorm, AbstractDiffusion


def t_scale(nu):
    return math.sqrt(nu / (nu - 2.0))


def sample_t(shape, nu, device, iid=False, dtype=torch.float32):
    """t_d(0, I_d, nu) = eps / sqrt(kappa), kappa ~ Gamma(nu/2, nu/2).
    iid=False -> one scalar kappa per sample (multivariate t, paper's version).
    iid=True  -> one kappa per coordinate (product of univariate t marginals)."""
    shape = tuple(shape)
    eps = torch.randn(shape, device=device, dtype=dtype)
    conc = torch.tensor(nu / 2.0, device=device, dtype=dtype)
    kappa_shape = shape if iid else (shape[0],)
    kappa = torch.distributions.Gamma(conc, conc).sample(kappa_shape)
    if not iid:
        kappa = kappa.reshape(shape[0], *([1] * (len(shape) - 1)))
    return eps / (kappa.sqrt() + 1e-8)


class TFNet(FNetGroupNorm):
    def __init__(self, log_time=True, **kwargs):
        super().__init__(**kwargs)
        self.log_time = log_time

    def forward(self, x, sigma):
        if not self.log_time:
            return super().forward(x, sigma)
        lmin, lmax = math.log(self.sigma_min), math.log(self.sigma_max)
        u = ((sigma.log() - lmin) / (lmax - lmin)).clamp(0, 1)
        t_sigma = torch.round(u * self.sigma_disc).int()
        x_emb = self.input_embedding(x)
        x_emb = self.time_embedding(x_emb, t_sigma)
        return self.net(x_emb)


class TDenoiser(nn.Module):
    def __init__(self, sigma_data, sigma_max, sigma_min, nu, log_time=True, **kwargs):
        super().__init__()
        assert nu > 2, f"t-EDM requires nu > 2 (finite variance); got {nu}"
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.nu = nu
        self.t_scale = t_scale(nu)
        self.fnet = TFNet(
            log_time=log_time, sigma_min=sigma_min, sigma_max=sigma_max, **kwargs
        )

    def forward(self, x, sigma):
        s = sigma * self.t_scale
        total = s**2 + self.sigma_data**2
        c_skip = self.sigma_data**2 / total
        c_out = s * self.sigma_data / total.sqrt()
        c_in = 1 / total.sqrt()

        x_in = c_in[:, None] * x
        F_x = self.fnet(x_in, sigma)          # raw sigma feeds the embedding
        return c_skip[:, None] * x + c_out[:, None] * F_x.to(torch.float32)


class AbstractDiffusionStudentT(AbstractDiffusion):
    def __init__(
        self,
        optim_config,
        denoiser_config,
        diffusion_config,
        batch_size,
        validation_sigmas=(0.1, 1.0, 5.0, 20.0, 80.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        **kwargs,
    ):
        L.LightningModule.__init__(self)
        self.optim_config = optim_config
        self.diffusion_config = diffusion_config
        self.denoiser_config = denoiser_config
        self.validation_sigmas = list(validation_sigmas)
        self.device_ = device
        self.dtype_ = dtype
        self.batch_size = batch_size
        self.automatic_optimization = True

        self.nu = float(denoiser_config["nu"])
        self.iid = bool(diffusion_config.get("iid", False))
        self.log_mean = float(diffusion_config["log_mean"])
        self.log_std = float(diffusion_config["log_std"])
        self.sigma_min = float(diffusion_config["sigma_min"])
        self.sigma_max = float(diffusion_config["sigma_max"])

        self.denoiser = TDenoiser(**denoiser_config).to(device=device, dtype=dtype)

    def _draw_sigma(self, n, device):
        return (
            (torch.randn(n, device=device) * self.log_std + self.log_mean)
            .exp()
            .clamp(self.sigma_min, self.sigma_max)
        )

    def _weighted_mse(self, pred, target, sigma):
        s = sigma * self.denoiser.t_scale
        c_out = self.denoiser.sigma_data * s / (self.denoiser.sigma_data**2 + s**2).sqrt()
        umse = (pred - target) ** 2 / target.shape[1]
        return (umse * (1.0 / c_out**2)[:, None]).mean(dim=0).sum()

    def training_step(self, batch, batch_idx):
        # noisy_sample / noise_level from VEDataModule are ignored: they are Gaussian
        x = batch["data_sample"].float()
        sigma = self._draw_sigma(x.shape[0], x.device)
        noise = sample_t(x.shape, self.nu, x.device, iid=self.iid)
        pred = self.denoiser(x + sigma[:, None] * noise, sigma)
        loss = self._weighted_mse(pred, x, sigma)
        self.log("train/mse", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["data_sample"].float()
        for s_val in self.validation_sigmas:
            sigma = torch.full((x.shape[0],), float(s_val), device=x.device)
            noise = sample_t(x.shape, self.nu, x.device, iid=self.iid)
            with torch.no_grad():
                pred = self.denoiser(x + sigma[:, None] * noise, sigma)
            self.log(
                f"val/mse/{s_val}",
                ((pred - x) ** 2).mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def on_validation_epoch_end(self):
        pass


def karras_schedule(sigma_max, sigma_min, rho, n_steps, device):
    t = torch.linspace(0, 1, n_steps, device=device)
    inv_rho = 1.0 / rho
    return (sigma_max**inv_rho + t * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho


@torch.no_grad()
def sample_t_prior(n, dim, nu, sigma, device, iid=False):
    return sigma * sample_t((n, dim), nu, device, iid=iid)


