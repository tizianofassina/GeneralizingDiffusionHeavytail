import torch
import numpy as np
from functools import partial
from tqdm import tqdm
from PIL import Image
from typing import Dict, Any

from . import dnnlib
import random
import json
import pickle
import os
from . import dnnlib
from . import torch_utils
from diffusers import AutoencoderKL


def decode_dict_clean(gen_dict, batchsize, device="cuda"):
    print(f"🚀 Initializing Decoding on device: {device}")

    # 1. Setup Path
    # Based on your 'ls', the folder name is 'stability_encoder_decoder'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vae_path = os.path.join(current_dir, "stability_encoder_decoder")

    if os.path.exists(vae_path):
        print(f"✅ Loading local VAE from: {vae_path}")
        # Load using Safetensors in offline mode for cluster compatibility
        vae = AutoencoderKL.from_pretrained(
            vae_path, local_files_only=True, use_safetensors=True
        ).to(device)
    else:
        # Fallback for PC mode
        print(f"🌐 Local VAE not found at {vae_path}. Downloading from Hugging Face...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    vae.eval()

    # NVIDIA Constants for denormalizing EDM2 latents
    RAW_MEAN = torch.tensor([5.81, 3.25, 0.12, -2.15]).view(1, 4, 1, 1).to(device)
    RAW_STD = torch.tensor([4.17, 4.62, 3.71, 3.28]).view(1, 4, 1, 1).to(device)
    scale = 0.5 / RAW_STD
    bias = -RAW_MEAN * scale

    decoded_dict = {}

    for class_id, latents in gen_dict.items():
        print(f"📦 Decoding Class {class_id}...")
        batches = latents.split(batchsize)
        decoded_images = []

        for batch in batches:
            batch = batch.to(device).float()
            with torch.no_grad():
                # A. Denormalization (Bring latents back to VAE scale)
                z = (batch - bias) / scale

                # B. Decoding (Conversion from Latent Space to RGB Space)
                output = vae.decode(z).sample
                # C. Post-processing (Map from [-1, 1] to [0, 255])
                # Shift range from [-1, 1] to [0, 1] then to [0, 255]
                img = output.clamp(0, 1).mul(255).to(torch.uint8)
                decoded_images.append(img.cpu())

        decoded_dict[class_id] = torch.cat(decoded_images, dim=0)

    return decoded_dict


def load_model_from_pickle(path: str, device: str) -> Any:
    if not isinstance(path, str):
        model = path
    else:
        with dnnlib.util.open_url(path) as f:
            data = pickle.load(f)
        model = data.get("ema") or data.get("model") or data
    model.eval()
    model = model.to(device)
    return model


def edm2_sampler_schedule(
    net,
    init,
    sigmas,
    labels=None,
    gnet=None,
    guidance=1.90,
    dtype=torch.float32,
    device="cpu",
):

    net.to(device)
    if gnet is not None:
        gnet.to(device)

    def denoise(x, t):
        Dx = net(x.to(dtype), t.to(dtype), labels).to(dtype)

        if guidance == 1:
            return Dx

        ref_Dx = gnet(x.to(dtype), t.to(dtype), labels).to(dtype)

        return ref_Dx.lerp(Dx, guidance)

    sigmas = sigmas.to(device).to(dtype)
    t_steps = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

    x_next = init.to(device).to(dtype)
    num_steps = sigmas.shape[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

        t_cur = t_cur.to(device)
        t_next = t_next.to(device)

        x_cur = x_next
        t_hat = t_cur
        x_hat = x_cur

        t_hat_batch = t_hat.expand(x_hat.shape[0]).to(dtype)
        t_next_batch = t_next.expand(x_hat.shape[0]).to(dtype)

        d_cur = (x_hat - denoise(x_hat, t_hat_batch)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next_batch)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def generate_data(
    net_path: str,
    gnet_path: str,
    device: str,
    class_map_path: str,
    pre_allocated_init: dict = None,
    init_mode: str = "gaussian",
    batch_size: int = 16,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    guidance: float = 1.9,
    num_to_generate=None,
    num_steps: int = 32,
    dtype: torch.dtype = torch.float32,
):

    with open(class_map_path, "r") as file:
        class_map_config = json.load(file)

    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    sigmas = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigmas = sigmas.to(device).to(dtype)

    print(f"Loading NET: {net_path} and GNET: {gnet_path} on {device}...")
    net = load_model_from_pickle(net_path, device)
    gnet = load_model_from_pickle(gnet_path, device)

    generator_partial = partial(
        edm2_sampler_schedule,
        net=net,
        gnet=gnet,
        sigmas=sigmas,
        guidance=guidance,
        dtype=dtype,
        device=device
    )

    all_generated_images_dict = {}
    all_init_data_dict = {}

    class_map = {
        str(k): v for k, v in class_map_config["class_map_birds"].items()
    }  # This changes from birds to dogs

    subset_indices_to_process = sorted(class_map.keys())

    for subset_class_idx in subset_indices_to_process:

        original_class_idx = str(class_map[subset_class_idx])

        current_init_idx = 0

        all_generated_class_data = []
        all_init_class_data = []

        if init_mode == "class_preallocated":

            if original_class_idx not in pre_allocated_init:
                print(
                    f"Skipping class {original_class_idx}: Pre-allocated data not found."
                )
                continue

            class_tensor = pre_allocated_init[original_class_idx].to(device).to(dtype)
            available_samples = class_tensor.shape[0]
            if num_to_generate is not None:
                n_gen = min(num_to_generate, available_samples)
            else:
                n_gen = available_samples

        elif init_mode == "gaussian":
            n_gen = num_to_generate if num_to_generate is not None else 1000
            class_tensor = None

        else:
            raise ValueError(
                f"Unknown init_mode: {init_mode}. Use 'gaussian' or 'class_preallocated'."
            )

        num_batches = (n_gen + batch_size - 1) // batch_size

        print(
            f"Generating {n_gen} images for Original ID {original_class_idx} (Subset ID {subset_class_idx})..."
        )

        for batch_i in tqdm(
            range(num_batches), desc=f"Class {original_class_idx} batches"
        ):

            current_batch_size = min(batch_size, n_gen - batch_i * batch_size)

            original_indices = torch.full(
                (current_batch_size,),
                int(original_class_idx),
                dtype=torch.long,
                device=device,
            )

            labels = (
                torch.nn.functional.one_hot(original_indices, num_classes=1000)
                .to(device)
                .to(dtype)
            )

            if init_mode == "gaussian":
                init_final = sigmas[0] * torch.randn(
                    (current_batch_size, 4, 64, 64), device=device, dtype=dtype
                )

            elif init_mode == "class_preallocated":
                start_idx = current_init_idx
                end_idx = current_init_idx + current_batch_size

                init_final = class_tensor[start_idx:end_idx]

                current_init_idx = end_idx

            generated_batch = generator_partial(init=init_final, labels=labels)

            all_generated_class_data.append(generated_batch.cpu())
            all_init_class_data.append(init_final.cpu())

        if all_generated_class_data:
            all_generated_images_dict[str(original_class_idx)] = torch.cat(
                all_generated_class_data, dim=0
            )
            all_init_data_dict[str(original_class_idx)] = torch.cat(
                all_init_class_data, dim=0
            )

    return all_init_data_dict, all_generated_images_dict


# -------------------- Utility Functions -------------------- #
def sample_init_data_for_display(init_dict: dict, samples_per_class: int = 3) -> dict:
    sampled_dict = {}
    for class_id, full_tensor in init_dict.items():
        num_available = full_tensor.shape[0]
        k = min(samples_per_class, num_available)

        if k > 0:
            random_indices = torch.randperm(num_available)[:k]
            sampled_dict[class_id] = full_tensor[random_indices].to(device)

    return sampled_dict


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


def rescale_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.clamp(-1, 1)
    tensor = ((tensor + 1.0) * 127.5).to(torch.uint8)
    return tensor


def rescale_dict_to_image(
    images_dict: Dict[Any, torch.Tensor],
) -> Dict[Any, torch.Tensor]:

    rescaled_dict = {}

    for class_id, full_tensor in images_dict.items():
        print(
            f"Rescaling tensor for Class ID {class_id} (Shape: {full_tensor.shape})..."
        )
        rescaled_tensor = rescale_tensor(full_tensor)

        rescaled_dict[class_id] = rescaled_tensor
        print(f"   -> Rescaled shape for class {class_id}: {rescaled_tensor.shape}")

    return rescaled_dict


def save_images_to_png(images_dict: Dict[Any, torch.Tensor], save_dir: str, seed: int):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n🚀 Starting PNG in {save_dir}...")

    total_images_saved = 0

    sorted_class_ids = sorted(images_dict.keys())

    for class_id in sorted_class_ids:
        class_tensor = images_dict[class_id]

        numpy_images = class_tensor.permute(0, 2, 3, 1).numpy()

        for idx, img_array in enumerate(
            tqdm(numpy_images, desc=f"Saving Class {class_id}")
        ):

            filename = os.path.join(
                save_dir, f"c{int(class_id):04d}_{int(idx):06d}_{seed}.png"
            )
            img = Image.fromarray(img_array)
            img.save(filename)
            total_images_saved += 1

    print(f"✅ Saved images : {total_images_saved}")
