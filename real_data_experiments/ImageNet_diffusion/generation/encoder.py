import os
import warnings
import numpy as np
import torch
import json
import tempfile  # Necessario per make_cache_dir_path
import torch.nn as nn

# Importazione di dnnlib non necessaria se si usa l'implementazione sotto
# Rimuovi l'importazione di dnnlib da load_stability_vae se usi l'implementazione qui sotto

_constant_cache = dict()

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", "`resume_download` is deprecated")


# --- Implementazione minimale di dnnlib.make_cache_dir_path ---
# Questa funzione è necessaria per load_stability_vae
def make_cache_dir_path(*paths: str) -> str:
    """Implementazione minimale per trovare un percorso di cache."""
    # Priorità: Variabile d'ambiente, cartella utente, temp dir
    if "HF_HOME" in os.environ:
        return os.path.join(os.environ["HF_HOME"], *paths)
    if "HOME" in os.environ:
        return os.path.join(os.environ["HOME"], ".cache", "dnnlib", *paths)
    return os.path.join(tempfile.gettempdir(), ".cache", "dnnlib", *paths)


# -----------------------------------------------------------------


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    # ... (Funzione 'constant' omessa per brevità, è corretta) ...
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    # ... (Funzione 'const_like' omessa per brevità, è corretta) ...
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(
        value, shape=shape, dtype=dtype, device=device, memory_format=memory_format
    )


class Encoder(nn.Module):
    # ... (Classe 'Encoder' e 'StandardRGBEncoder' omesse, sono corrette) ...
    def __init__(self):
        super().__init__()
        pass

    def init(self, device):  # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x):  # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x):  # raw pixels => raw latents
        raise NotImplementedError  # to be overridden by subclass

    def encode_latents(self, x):  # raw latents => final latents
        raise NotImplementedError  # to be overridden by subclass

    def decode(self, x):  # final latents => raw pixels
        raise NotImplementedError  # to be overridden by subclass


class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x):  # raw pixels => raw latents
        return x

    def encode_latents(self, x):  # raw latents => final latents
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x):  # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


class StabilityVAEEncoder(Encoder):
    # ... (Classe 'StabilityVAEEncoder' omessa, è corretta) ...
    def __init__(
        self,
        vae_name="stabilityai/sd-vae-ft-mse",  # Name of the VAE to use.
        raw_mean=[5.81, 3.25, 0.12, -2.15],  # Assumed mean of the raw latents.
        raw_std=[
            4.17,
            4.62,
            3.71,
            3.28,
        ],  # Assumed standard deviation of the raw latents.
        final_mean=0,  # Desired mean of the final latents.
        final_std=0.5,  # Desired standard deviation of the final latents.
        batch_size=8,  # Batch size to use when running the VAE.
    ):
        super().__init__()
        self.vae_name = vae_name
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._vae = None

    def init(self, device):  # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_stability_vae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None)  # do not pickle the vae

    def _run_vae_encoder(self, x):
        d = self._vae.encode(x)["latent_dist"]
        return torch.cat([d.mean, d.std], dim=1)

    def _run_vae_decoder(self, x):
        return self._vae.decode(x)["sample"]

    def encode_pixels(self, x):  # raw pixels => raw latents
        self.init(x.device)
        x = x.to(torch.float32) / 255
        x = torch.cat(
            [self._run_vae_encoder(batch) for batch in x.split(self.batch_size)]
        )
        return x

    def encode_latents(self, x):  # raw latents => final latents
        mean, std = x.to(torch.float32).chunk(2, dim=1)
        x = mean + torch.randn_like(mean) * std
        x = x * const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + const_like(x, self.bias).reshape(1, -1, 1, 1)
        return x

    def decode(self, x):  # final latents => raw pixels
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - const_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = torch.cat(
            [self._run_vae_decoder(batch) for batch in x.split(self.batch_size)]
        )
        x = x.clamp(0, 1).mul(255).to(torch.uint8)
        return x


# ----------------------------------------------------------------------------


def load_stability_vae(
    vae_name="stabilityai/sd-vae-ft-mse", device=torch.device("cpu")
):
    # import dnnlib # Rimosso! Usiamo la funzione make_cache_dir_path definita sopra

    # Usiamo la funzione make_cache_dir_path definita all'inizio dello script
    cache_dir = make_cache_dir_path("diffusers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HOME"] = cache_dir

    import diffusers  # Richiede: pip install diffusers

    try:
        # First try with local_files_only to avoid consulting tfhub metadata if the model is already in cache.
        vae = diffusers.models.AutoencoderKL.from_pretrained(
            vae_name, cache_dir=cache_dir, local_files_only=True
        )
    except:
        # Could not load the model from cache; try without local_files_only.
        vae = diffusers.models.AutoencoderKL.from_pretrained(
            vae_name, cache_dir=cache_dir
        )
    return vae.eval().requires_grad_(False).to(device)


# ----------------------------------------------------------------------------
