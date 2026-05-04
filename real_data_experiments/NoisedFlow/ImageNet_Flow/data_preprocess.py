import os
import json
import numpy as np
import torch
from pathlib import Path

# ============================================================
## 📂 Path and I/O Configuration
# ============================================================
LATENT_DATA_ROOT = Path("ImageNet/img512_encoded_right")
JSON_FILENAME = "flow_class_map.json"
OUTPUT_RAW = "datasets/train_set_tensor_birds.pt"  
OUTPUT_NORM = "datasets/train_set_tensor_birds_normalized.pt"

# ============================================================
## 🧪 Karras Normalization Constants
# ============================================================
# These constants are derived from the Stability AI VAE (Karras et al.)
# to center and scale the latent distributions.
RAW_MEAN = torch.tensor([5.81, 3.25, 0.12, -2.15]).view(1, 4, 1, 1)
RAW_STD = torch.tensor([4.17, 4.62, 3.71, 3.28]).view(1, 4, 1, 1)

# Target distribution for the Flow input: centered with reduced variance (0.5)
# This helps the Normalizing Flow converge faster on the latent manifold.
FINAL_STD = 0.5
FINAL_MEAN = 0.0


def normalize_tensors(raw_dict):
    """
    Applies Karras normalization to the mean (mu) channels of the VAE latent.
    This shifts the ImageNet latent distribution to a zero-centered, fixed-scale space.
    """
    norm_dict = {}
    # Linearly transform: z_norm = (z_raw - RAW_MEAN) * (FINAL_STD / RAW_STD) + FINAL_MEAN
    scale = FINAL_STD / RAW_STD
    bias = FINAL_MEAN - RAW_MEAN * scale

    print(
        f"\nApplying Normalization (Scale: {scale.flatten().tolist()}, Bias: {bias.flatten().tolist()})"
    )

    for class_id, tensor in raw_dict.items():
        # CHANNEL LOGIC:
        # Standard VAE encoders output 8 channels: [0:4] are 'mu' (mean), [4:8] are 'logvar'.
        # For TarFlow training, we only need the 'mu' channels (the actual latent representation).
        z_raw = tensor[:, :4, :, :].float()
        z_norm = z_raw * scale + bias
        norm_dict[str(class_id)] = z_norm
    return norm_dict


def create_datasets(key: str = "class_map_birds"):
    """
    Groups specific ImageNet classes into a single dataset (e.g., all bird species).
    Saves both the raw 8-channel latents and the normalized 4-channel latents.
    """
    # 1. Load JSON Map (Mapping new sequential IDs to original ImageNet Class IDs)
    try:
        with open(JSON_FILENAME, "r") as f:
            data_map = json.load(f)
        class_map = data_map.get(key, data_map)
        print(f"Loaded class map with {len(class_map)} classes.")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    raw_data = {}

    # 2. Aggregate Data from Disk
    # Iterates through the class hierarchy to stack individual .npy files into PyTorch tensors.
    for new_id, original_id in class_map.items():
        class_dir = LATENT_DATA_ROOT / f"{int(original_id):03d}"
        if not class_dir.exists():
            print(f"Skipping {original_id}: Path not found.")
            continue

        all_latents = []
        for file_path in class_dir.glob("*.npy"):
            try:
                # Expected shape from VAE encoder: (8, 64, 64)
                arr = np.load(file_path)  
                all_latents.append(torch.from_numpy(arr).float())
            except Exception as e:
                continue

        if all_latents:
            # Resulting tensor: [N_samples, 8, 64, 64]
            raw_data[str(original_id)] = torch.stack(all_latents)
            print(f"Class {original_id}: {len(all_latents)} samples loaded.")

    if not raw_data:
        print("No data found!")
        return

    # 3. Save RAW Dataset (Full 8 channels, includes log-variance for possible VAE reconstruction)
    print(f"\nSaving RAW dataset to {OUTPUT_RAW}...")
    torch.save(raw_data, OUTPUT_RAW)

    # 4. Normalize and Save NORM Dataset (4 channels)
    # This is the primary training set for the TarFlow module.
    normalized_data = normalize_tensors(raw_data)
    print(f"Saving NORMALIZED dataset to {OUTPUT_NORM}...")
    torch.save(normalized_data, OUTPUT_NORM)

    print("\n*** Processing Complete! Both files generated. ***")


if __name__ == "__main__":
    # Ensure local 'datasets' folder exists
    Path("datasets").mkdir(parents=True, exist_ok=True)
    # Example execution: creating a subset for specific ImageNet birds
    create_datasets("class_map_birds")