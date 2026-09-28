"""DLPM / DLIM (Shariatian et al.) alpha-stable diffusion for vector data.
Forward kernel, eps-prediction loss and deterministic DLIM reverse chain as in the
reference implementation; MLPModel backbone (their vector architecture, working
path only). No data normalization.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torch.distributions.exponential import Exponential


def build_dlpm_schedule(num_steps, alpha, s=0.008, device="cpu", dtype=torch.float32):
    t = torch.arange(0, num_steps, dtype=dtype, device=device)
    bar_alpha = torch.cos((t / num_steps + s) / (1 + s) * math.pi / 2) ** 2
    bar_alpha = bar_alpha / bar_alpha[0]
    betas = 1 - bar_alpha / torch.cat([bar_alpha[0:1], bar_alpha[:-1]])
    alphas_lin = 1 - betas
    gammas = alphas_lin ** (1.0 / alpha)
    bargammas = torch.cumprod(gammas, dim=0)
    sigmas = (1 - gammas**alpha) ** (1.0 / alpha)
    barsigmas = (1 - bargammas**alpha) ** (1.0 / alpha)
    return {"gammas": gammas, "bargammas": bargammas,
            "sigmas": sigmas, "barsigmas": barsigmas}


def dlpm_loss_term(pred, target):
    dims = list(range(1, pred.dim()))
    return torch.sqrt(
        nn.functional.mse_loss(pred, target, reduction="none").mean(dim=dims)
    )


class StableVariable:
    """Univariate alpha-stable S^alpha_{beta,sigma} (loc 0) via Chambers-Mallows-Stuck."""

    def __init__(self, alpha, beta=0.0, sigma=1.0, dtype=torch.float32):
        assert 0 < alpha <= 2.0, "alpha must be in (0, 2]"
        assert -1.0 <= beta <= 1.0, "beta must be in [-1, 1]"
        assert sigma > 0
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.dtype = dtype

    def sample(self, size, device):
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        x = self._cms_standard(self.alpha, self.beta, n, device)
        return (x * self.sigma).reshape(size).to(self.dtype)

    @staticmethod
    def _cms_standard(alpha, beta, size, device):
        TH = torch.rand(size, dtype=torch.float64, device=device) * math.pi - math.pi / 2.0
        W = (Exponential(torch.tensor([1.0], dtype=torch.float64, device=device))
             .sample([size]).reshape(-1))
        aTH, bTH = alpha * TH, beta * TH
        cosTH, tanTH = torch.cos(TH), torch.tan(TH)

        if alpha == 1:
            return 2 / math.pi * (
                (math.pi / 2 + bTH) * tanTH
                - beta * torch.log((math.pi / 2 * W * cosTH) / (math.pi / 2 + bTH))
            )
        elif beta == 0:
            return (W / (cosTH / torch.tan(aTH) + torch.sin(TH))
                    * ((torch.cos(aTH) + torch.sin(aTH) * tanTH) / W) ** (1 / alpha))
        else:
            val0 = beta * math.tan(math.pi * alpha / 2)
            th0 = math.atan(val0) / alpha
            val3 = W / (cosTH / torch.tan(alpha * (th0 + TH)) + torch.sin(TH))
            return val3 * (
                (torch.cos(aTH) + torch.sin(aTH) * tanTH
                 - val0 * (torch.sin(aTH) - torch.cos(aTH) * tanTH)) / W
            ) ** (1 / alpha)


class LevyNoise:
    """Positive alpha/2-stable mixing variable A; eps = sqrt(A)*z is S_alpha."""

    def __init__(self, alpha, isotropic=True, apply_cA=None, dtype=torch.float32):
        assert 1.0 < alpha <= 2.0
        self.alpha = alpha
        self.a_index = alpha / 2.0
        self.isotropic = isotropic
        self.apply_cA = isotropic if apply_cA is None else apply_cA
        self.cA = 2.0 * math.cos(math.pi * alpha / 4.0) ** (2.0 / alpha)
        self.stable = StableVariable(self.a_index, beta=1.0, sigma=1.0, dtype=dtype)
        self.n_negative = 0

    def sample(self, shape, device):
        if isinstance(shape, int):
            shape = (shape,)
        if self.isotropic:
            n = shape[0]
            a = self.stable.sample((n,), device).view(n, *([1] * (len(shape) - 1)))
        else:
            a = self.stable.sample(shape, device)
        if self.apply_cA:
            a = a * self.cA
        self.n_negative += int((a < 0).sum())
        return a.clamp_min(0.0)


class DiffusionBlockConditioned(nn.Module):
    """Shariatian et al. DiffusionBlockConditioned, time-conditioning path only."""

    def __init__(self, nunits, time_emb_size, dropout_rate, skip_connection, group_norm):
        super().__init__()
        self.skip_connection = skip_connection
        self.act = nn.SiLU(inplace=False)
        drop = nn.Dropout(p=dropout_rate)
        norm1 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        norm2 = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.mlp_1 = nn.Sequential(drop, nn.Linear(nunits, nunits), norm1)
        self.t_proj = nn.Sequential(drop, nn.Linear(time_emb_size, nunits), self.act)
        self.mlp_2 = nn.Sequential(drop, nn.Linear(nunits, nunits), norm2)

    def forward(self, x, t_emb):
        x_skip = x
        x = self.act(self.mlp_1(x))
        x = x + self.t_proj(t_emb)
        x = self.mlp_2(x)
        if self.skip_connection:
            x = x + x_skip
        return self.act(x)


class DLPMMlp(nn.Module):
    """Shariatian et al. MLPModel, working path only: no_a=True, a_pos_emb=False,
    learn_variance=False, time_emb_type='learnable'. Data [B, d], t [B] in (0,1)."""

    def __init__(self, nfeatures, nunits, nblocks, time_emb_size,
                 skip_connection=True, group_norm=True, dropout_rate=0.0):
        super().__init__()
        self.act = nn.SiLU(inplace=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_size), self.act,
            nn.Linear(time_emb_size, time_emb_size), self.act,
        )
        norm_in = nn.LayerNorm([nunits]) if group_norm else nn.Identity()
        self.inblock = nn.Sequential(nn.Linear(nfeatures, nunits), norm_in, self.act)
        self.midblocks = nn.ModuleList([
            DiffusionBlockConditioned(nunits, time_emb_size, dropout_rate,
                                      skip_connection, group_norm)
            for _ in range(nblocks)
        ])
        self.outblock = DiffusionBlockConditioned(nunits, time_emb_size, dropout_rate,
                                                  skip_connection, group_norm)
        self.outlinear = nn.Linear(nunits, nfeatures)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.reshape(-1, 1).float())
        h = self.inblock(x)
        for blk in self.midblocks:
            h = blk(h, t_emb)
        return self.outlinear(self.outblock(h, t_emb))


class AbstractDiffusionLevy(L.LightningModule):
    def __init__(self, model_config, loss_config, optim_config, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.alpha = float(loss_config["alpha"])
        self.T = int(loss_config["T"])
        self.isotropic = bool(loss_config.get("isotropic", True))
        self.rescale_timesteps = bool(loss_config.get("rescale_timesteps", True))
        self.val_probe_steps = list(
            loss_config.get("val_probe_steps", [200, 400, 2000, 3600])
        )
        self.optim_config = optim_config
        self.batch_size = batch_size

        self.model = DLPMMlp(**model_config)

        sched = build_dlpm_schedule(self.T, self.alpha)
        self.register_buffer("gammas", sched["gammas"], persistent=True)
        self.register_buffer("bargammas", sched["bargammas"], persistent=True)
        self.register_buffer("barsigmas", sched["barsigmas"], persistent=True)

        self.noise = LevyNoise(self.alpha, isotropic=self.isotropic, apply_cA=True)

    def forward(self, x, t):
        return self.model(x, t)

    @staticmethod
    def _x0(batch):
        if isinstance(batch, (list, tuple)):
            return batch[0]
        if isinstance(batch, dict):
            return batch["data_sample"]
        return batch

    def _noise_batch(self, x0):
        A = self.noise.sample(tuple(x0.shape), x0.device)
        return A.sqrt() * torch.randn_like(x0)

    def training_step(self, batch, batch_idx):
        x0 = self._x0(batch).float()
        nd = x0.dim() - 1

        t = torch.randint(1, self.T, size=(x0.shape[0],), device=x0.device)
        eps_t = self._noise_batch(x0)

        bg = self.bargammas[t].view(-1, *([1] * nd))
        bs = self.barsigmas[t].view(-1, *([1] * nd))
        x_t = bg * x0 + bs * eps_t

        t_in = t.float() / self.T if self.rescale_timesteps else t.float()
        loss = dlpm_loss_term(self.model(x_t, t_in), eps_t).mean()

        self.log("train/loss_nonfinite", (~torch.isfinite(loss)).float(), on_step=True)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = self._x0(batch).float()
        nd = x0.dim() - 1
        eps_t = self._noise_batch(x0)

        def probe(net, tag):
            for t_val in self.val_probe_steps:
                t = torch.full((x0.shape[0],), int(t_val),
                               device=x0.device, dtype=torch.long)
                bg = self.bargammas[t].view(-1, *([1] * nd))
                bs = self.barsigmas[t].view(-1, *([1] * nd))
                x_t = bg * x0 + bs * eps_t
                t_in = t.float() / self.T if self.rescale_timesteps else t.float()
                self.log(f"val/loss_{tag}_t_{t_val}",
                         dlpm_loss_term(net(x_t, t_in), eps_t).mean(),
                         on_step=False, on_epoch=True, sync_dist=True)

        probe(self.model, "online")
        ema_cbs = [c for c in self.trainer.callbacks if isinstance(c, EMACallbackLevy)]
        if ema_cbs and ema_cbs[0].shadow_model is not None:
            ema = ema_cbs[0].shadow_model
            if next(ema.parameters()).device != x0.device:
                ema.to(x0.device)
            probe(ema, "ema")

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(L.pytorch.utilities.grad_norm(self.model, norm_type=2),
                      on_step=True, on_epoch=False)
        n_bad = sum(int((~torch.isfinite(p.grad)).sum())
                    for p in self.model.parameters() if p.grad is not None)
        self.log("train/n_nonfinite_grads", float(n_bad), on_step=True, on_epoch=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.optim_config["lr"]),
            weight_decay=float(self.optim_config.get("weight_decay", 1e-4)),
            betas=(0.9, 0.999), eps=1e-8,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.optim_config["lr_step_size"]),
                gamma=float(self.optim_config["lr_gamma"]),
            ),
            "interval": self.optim_config.get("sched_interval", "epoch"),
            "frequency": 1,
        }
        return [optimizer], [scheduler]


class EMACallbackLevy(L.Callback):
    def __init__(self, rate=0.9999):
        super().__init__()
        self.rate = rate
        self.shadow_model = None

    def on_train_start(self, trainer, pl_module):
        if self.shadow_model is None:
            self.shadow_model = copy.deepcopy(pl_module.model)
            self.shadow_model.to(pl_module.device)
            self.shadow_model.eval()
            for p in self.shadow_model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for op, sp in zip(pl_module.model.parameters(), self.shadow_model.parameters()):
            sp.lerp_(op, 1.0 - self.rate)
        for ob, sb in zip(pl_module.model.buffers(), self.shadow_model.buffers()):
            sb.copy_(ob)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_shadow_weights"] = self.shadow_model.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_shadow_weights" in checkpoint:
            if self.shadow_model is None:
                self.shadow_model = copy.deepcopy(pl_module.model)
                self.shadow_model.eval()
                for p in self.shadow_model.parameters():
                    p.requires_grad_(False)
            self.shadow_model.load_state_dict(checkpoint["ema_shadow_weights"])


@torch.no_grad()
def dlim_sampling(x, noise_predictor, gammas, barsigmas):
    """Deterministic DLIM chain from t=K-1 down to 1. NFE = K-1."""
    K = gammas.shape[0]
    for t in range(K - 1, 0, -1):
        if not torch.isfinite(x).all():
            raise FloatingPointError(
                f"non-finite x at step t={t}: "
                f"{(~torch.isfinite(x)).sum().item()} bad entries"
            )
        t_in = torch.full((x.shape[0],), t / K, device=x.device, dtype=torch.float32)
        eps = noise_predictor(x, t_in)
        x = (x - barsigmas[t] * eps) / gammas[t] + barsigmas[t - 1] * eps
    return x


@torch.no_grad()
def sample_levy_prior(n, dim, levy_noise, prior_scale, device):
    A = levy_noise.sample((n, dim), device)
    z = torch.randn(n, dim, device=device)
    return (prior_scale * A.sqrt() * z).float()


@torch.no_grad()
def generate_levy(denoiser, n_samples, dim, alpha, n_steps, device,
                  isotropic=True, batch_size=100_000, log_fn=None):
    sched = build_dlpm_schedule(n_steps, alpha, device=device)
    gammas = sched["gammas"].to(torch.float32)
    barsigmas = sched["barsigmas"].to(torch.float32)
    prior_scale = float(barsigmas[-1])
    levy_noise = LevyNoise(alpha, isotropic=isotropic, apply_cA=True)

    out = np.empty((n_samples, dim), dtype=np.float32)
    done = 0
    while done < n_samples:
        b = min(batch_size, n_samples - done)
        x = sample_levy_prior(b, dim, levy_noise, prior_scale, device)
        x = dlim_sampling(x, denoiser, gammas, barsigmas)
        out[done : done + b] = x.float().cpu().numpy()
        done += b
        if log_fn is not None:
            log_fn(done, n_samples)
    if levy_noise.n_negative:
        print(f"  WARNING: {levy_noise.n_negative} negative A clamped to 0")
    return out
