import os
import random
from functools import partial

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sampling import edm_sampling_images_vector


# -------------------- Generator Settings -------------------- #
def setting_generator_classic_edm(denoiser_fn, sigmas, device):
    """Classic EDM generator from Gaussian noise."""
    generator = partial(
        edm_sampling_images_vector,
        denoiser_fn=denoiser_fn,
        sigmas=sigmas,
        device=device,
    )

    def generate(z):
        init = sigmas[0] * z
        return init, generator(init)

    return generate


def setting_generator_empirical_dist(
    train_set_dir, denoiser_fn, sigmas_shorter, device
):
    """EDM generator using empirical training data."""
    files = sorted(f for f in os.listdir(train_set_dir) if f.endswith((".png", ".jpg")))
    training_data = torch.stack(
        [
            torch.tensor(
                np.array(Image.open(os.path.join(train_set_dir, f)).convert("RGB")),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            / 127.5
            - 1.0
            for f in files
        ]
    )
    print(f"Loaded {training_data.size(0)} images, shape: {training_data.shape}")

    generator = partial(
        edm_sampling_images_vector,
        denoiser_fn=denoiser_fn,
        sigmas=sigmas_shorter,
        device=device,
    )

    def generate(z):
        sigma_z = sigmas_shorter[0] * z
        indices = torch.randint(0, training_data.size(0), (z.shape[0],))
        batch_train = training_data[indices].to(device)
        init = batch_train + sigma_z
        return init, generator(init)

    return generate


def setting_generator_trained_dist(denoiser_fn, sigmas, sigmas_shorter, device):
    """EDM generator starting from classic EDM output, then refining with shorter sigma schedule."""
    generator_full = partial(
        edm_sampling_images_vector,
        denoiser_fn=denoiser_fn,
        sigmas=sigmas,
        device=device,
    )
    generator_short = partial(
        edm_sampling_images_vector,
        denoiser_fn=denoiser_fn,
        sigmas=sigmas_shorter,
        device=device,
    )

    def generate(z):
        z_start = sigmas[0] * torch.randn_like(z)
        x_hat_0 = generator_full(z_start)
        init = x_hat_0 + sigmas_shorter[0] * z
        return init, generator_short(init)

    return generate


def setting_generator_flow_classic(
    denoiser_fn, sigmas_shorter, small_noise_flow, device
):
    """Classic flow generator with small injected noise."""
    generator = partial(
        edm_sampling_images_vector,
        denoiser_fn=denoiser_fn,
        sigmas=sigmas_shorter,
        device=device,
    )

    def generate(z):
        init = z + (sigmas_shorter[0] - small_noise_flow) * torch.randn_like(z)
        return init, generator(init)

    return generate


def setting_generator_flow_noised(denoiser_fn, sigmas_shorter, device):
    """Flow generator using pre-noised input."""
    generator = partial(
        edm_sampling_images_vector,
        denoiser_fn=denoiser_fn,
        sigmas=sigmas_shorter,
        device=device,
    )

    def generate(z):
        return z, generator(z)

    return generate


# -------------------- Data Generation -------------------- #
def generating_data(
    denoiser=None,
    gen_size=50_000,
    sigmas=None,
    sigmas_shorter=None,
    initialization=None,
    batch_size_gen=128,
    generator_type="edm_classic",
    train_set_dir=None,
    small_noise_flow=0.05,
    device="cuda",
):
    """
    Generate data using different EDM-based generators and return both initialization and generated samples.

    Returns:
        init_data: Tensor of initial inputs used for generation [gen_size, C, H, W]
        generated_data: Tensor of generated samples [gen_size, C, H, W]
    """
    # --- Define generator function ---
    gen_fn_map = {
        "edm_classic": lambda: setting_generator_classic_edm(denoiser, sigmas, device),
        "empirical": lambda: setting_generator_empirical_dist(
            train_set_dir, denoiser, sigmas_shorter, device
        ),
        "trained": lambda: setting_generator_trained_dist(
            denoiser, sigmas, sigmas_shorter, device
        ),
        "flow_classic": lambda: setting_generator_flow_classic(
            denoiser, sigmas_shorter, small_noise_flow, device
        ),
        "flow_noised": lambda: setting_generator_flow_noised(
            denoiser, sigmas_shorter, device
        ),
    }

    if generator_type not in gen_fn_map:
        raise ValueError(f"Unknown generator type: {generator_type}")
    if generator_type == "empirical" and train_set_dir is None:
        raise ValueError("train_set_dir must be provided for empirical generator")

    gen_fn = gen_fn_map[generator_type]()

    # --- Generate data in batches ---
    init_list, gen_list = [], []
    n_batches = (gen_size + batch_size_gen - 1) // batch_size_gen

    for _ in tqdm(range(n_batches), desc=f"Generating {generator_type} data"):
        current_batch_size = min(
            batch_size_gen, gen_size - len(init_list) * batch_size_gen
        )

        # --- Initialize input ---
        if initialization is not None:
            z = initialization[:current_batch_size].to(device)
            initialization = initialization[current_batch_size:]
        else:
            z = torch.randn(current_batch_size, 3, 64, 64, device=device)

        # --- Generate batch ---
        init_batch, gen_batch = gen_fn(z)
        init_list.append(init_batch.cpu())
        gen_list.append(gen_batch.cpu())

    # --- Concatenate all batches ---
    init_data = torch.cat(init_list, dim=0)[:gen_size]
    generated_data = torch.cat(gen_list, dim=0)[:gen_size]

    return init_data, generated_data


# -------------------- Utility Functions -------------------- #
def tensor_to_images(tensor, to_pil=True):
    """
    Convert a batch of tensors in [-1, 1] to images [0, 255].

    Args:
        tensor: torch.Tensor of shape [B, C, H, W], values in [-1,1]
        to_pil: if True, returns list of PIL Images, else numpy arrays

    Returns:
        List of PIL.Image or np.ndarray
    """
    tensor = tensor.clamp(-1, 1)
    tensor = ((tensor + 1.0) * 127.5).to(torch.uint8)
    tensor = tensor.permute(0, 2, 3, 1)
    images = tensor.cpu().numpy()
    return [Image.fromarray(img) for img in images] if to_pil else images


def images_to_tensor(images):
    """
    Convert a batch of images [0, 255] to torch tensors in [-1, 1].

    Args:
        images: list of PIL.Image or np.ndarray of shape [H, W, C]

    Returns:
        torch.Tensor of shape [B, C, H, W], values in [-1,1]
    """
    if isinstance(images[0], Image.Image):
        images = [np.array(img) for img in images]
    tensor = torch.stack(
        [torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in images]
    )
    return tensor / 127.5 - 1.0


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
