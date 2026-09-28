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
import io
import torchvision.transforms as T
from PIL import Image

torch_float = torch.float
randn = torch.randn
randn_like = torch.randn_like


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe.float())

    def forward(self, x, t_ind):
        x = x + self.pe[t_ind, :]
        return x


class FNetGroupNorm(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=128,
        sigma_max=80,
        sigma_min=0.02,
        sigma_disc=1000,
        channel_mult=[1, 2, 3],
    ):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_disc = sigma_disc

        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.time_embedding = PositionalEncoding(
            max_len=sigma_disc + 1, d_model=embed_dim
        )

        layers = []
        in_dim = embed_dim
        for mult in channel_mult:
            out_dim = embed_dim * mult
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.GroupNorm(num_groups=8, num_channels=out_dim),
                nn.SiLU(),
            ]
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigma):
        # Normalize sigma to discrete index
        t_sigma = torch.round(
            ((sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)).clamp(0, 1)
            * self.sigma_disc
        ).int()

        x_emb = self.input_embedding(x)
        x_emb = self.time_embedding(x_emb, t_sigma)
        return self.net(x_emb)


class Denoiser(torch.nn.Module):
    def __init__(self, sigma_data, sigma_max, sigma_min, **kwargs):
        super().__init__()
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.fnet = FNetGroupNorm(sigma_min=sigma_min, sigma_max=sigma_max, **kwargs)

    def forward(self, x, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()

        # Run the model.
        x_in = c_in[:, None] * x
        F_x = self.fnet(x_in, sigma)
        D_x = c_skip[:, None] * x + c_out[:, None] * F_x.to(torch.float32)
        return D_x


class AbstractDiffusion(L.LightningModule):
    def __init__(
        self,
        optim_config,
        denoiser_config,
        diffusion_config,
        batch_size,
        validation_sigmas=[0.1, 0.3, 0.5, 1, 2, 4, 8, 10],
        device=torch.device("cpu"),
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.optim_config = optim_config
        self.diffusion_config = diffusion_config
        self.validation_sigmas = validation_sigmas
        self.denoiser_config = denoiser_config
        self.device_ = device
        self.dtype_ = dtype
        self.denoiser = Denoiser(**denoiser_config).to(device=device, dtype=dtype)
        self.automatic_optimization = True
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
        target = batch["data_sample"].float()
        sigma = batch["noise_level"].float()
        prediction = self.denoiser(batch["noisy_sample"], sigma)
        cout = (
            self.denoiser.sigma_data
            * sigma
            / (self.denoiser.sigma_data**2 + sigma**2) ** 0.5
        )
        mse_loss = (prediction - target) ** 2
        umse_loss = mse_loss / prediction[0].numel()
        weights = 1 / (cout**2)
        loss = (umse_loss * weights[:, None]).mean(dim=0).sum()
        self.log("train/mse", loss.item(), prog_bar=True)
        return loss

    def log_sigma_mse_curve(self, batch, step_or_epoch, mode="train"):
        device = self.device
        dtype = torch.float32

        num_sigmas = 50
        sigmas = torch.logspace(
            math.log10(self.denoiser.sigma_min),
            math.log10(self.denoiser.sigma_max),
            steps=num_sigmas,
            device=device,
            dtype=dtype,
        )
        data = batch["data_sample"].to(device, dtype=dtype)
        batch_shape = data.shape
        B = batch_shape[0]
        data_rep = data.unsqueeze(0).expand(num_sigmas, *batch_shape)
        noise = torch.randn_like(data_rep[0], device=device)
        sigmas_rep = sigmas[:, None].view(num_sigmas, *([1] * (data.ndim))).to(device)

        noisy = data_rep + sigmas_rep * noise[None, :]

        sigma_input = sigmas.repeat_interleave(B)
        noisy_flat = noisy.reshape(num_sigmas * B, *batch_shape[1:])

        # Forward
        with torch.no_grad():
            out_flat = self.denoiser(noisy_flat, sigma_input.to(device))
            out = out_flat.reshape(num_sigmas, B, *batch_shape[1:])

        mse = (out - data_rep) ** 2

        cout = (
            self.denoiser.sigma_data
            * sigmas[:, None]
            / (self.denoiser.sigma_data**2 + sigmas[:, None] ** 2).sqrt()
        )
        weights = 1 / (cout**2)
        mse_weighted = (
            (mse * weights[:, None]).mean(dim=list(range(1, mse.ndim))).cpu().numpy()
        )

        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(torch.log(sigmas).cpu().numpy(), mse_weighted)
        plt.xlabel("log(sigma)")
        plt.ylabel("Weighted MSE")
        plt.title(f"{mode} log(sigma) → Weighted MSE")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        img_t = T.ToTensor()(img)

        self.logger.experiment.add_image(
            f"{mode}_log_sigma_mse_curve", img_t, step_or_epoch
        )

    def validation_step(self, batch, batch_idx):
        data_shape = tuple(batch["data_sample"].shape)
        noise = torch.randn(
            size=(len(self.validation_sigmas), *data_shape), device=self.device
        )
        corrupt_data = batch["data_sample"][None] + torch.stack(
            [n * s for n, s in zip(noise, self.validation_sigmas)]
        )
        predicted_clean = self.denoiser(
            corrupt_data.reshape(
                data_shape[0] * len(self.validation_sigmas), *data_shape[1:]
            ),
            torch.stack(
                [
                    s * torch.ones((data_shape[0],), device=self.device)
                    for s in self.validation_sigmas
                ]
            ).reshape(
                data_shape[0] * len(self.validation_sigmas),
            ),
        ).reshape(len(self.validation_sigmas), *data_shape)

        batch_errors = (
            torch.nn.functional.mse_loss(
                predicted_clean,
                batch["data_sample"].repeat(
                    len(self.validation_sigmas), *((1,) * len(data_shape))
                ),
                reduction="none",
            )
            / batch["data_sample"][0].numel()
        ).mean(dim=-1)

        for s, e in zip(self.validation_sigmas, batch_errors):
            self.log(
                f"val/mse/{s}",
                e.mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def on_validation_epoch_end(self):
        if self.current_epoch % 50 == 0:
            try:
                batch = next(iter(self.trainer.datamodule.val_dataloader()))
            except Exception:
                batch = None
            if batch is not None:
                self.log_sigma_mse_curve(batch, self.current_epoch, mode="val")

    def on_before_optimizer_step(self, optimizer):
        norms = {
            **{
                f"grad/{k}": v.item()
                for k, v in L.pytorch.utilities.grad_norm(
                    self.denoiser, norm_type=2
                ).items()
            },
        }
        self.log_dict(norms)

    def configure_optimizers(self):
        lr = float(self.optim_config["lr"])
        optimizer = torch.optim.AdamW(
            self.denoiser.parameters(),
            lr=lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
