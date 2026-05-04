import os
import json
import shutil
from tqdm import tqdm

# ============================================================
## 📂 Path Configuration
# ============================================================
# Mapping file containing [filename, class_index] pairs
JSON_PATH = "ImageNet_512_processed/datasets/img512_encoded_dir/dataset.json"
# Source: Unstructured directory containing all latent vectors (.npy)
SOURCE_LATENTS_DIR = (
    "/ImageNet_512_processed/datasets/img512_encoded_dir"
)
# Destination: Structured directory organized by ImageNet class folders
OUTPUT_DIR = (
    "ImageNet_512_processed/datasets/img512_encoded_right"
)
LATENT_EXT = ".npy"


def reorganize_latents():
    """
    Standardizes the ImageNet latent dataset into a class-aware folder structure.
    This is required for efficient loading during class-conditional TarFlow training.
    """
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: Cannot find mapping file at {JSON_PATH}")
        return

    print(f"Loading metadata from {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        # 'labels' usually contains the list of [relative_path, class_label]
        image_list = data.get("labels", [])

    if not image_list:
        print("ERROR: JSON file is empty or invalid.")
        return

    # Create subdirectories for each unique ImageNet class (000-999)
    unique_classes = set(item[1] for item in image_list if item is not None)
    for cls_idx in unique_classes:
        os.makedirs(os.path.join(OUTPUT_DIR, f"{cls_idx:03d}"), exist_ok=True)

    # Track count per class to generate sequential filenames
    class_counters = {cls_idx: 0 for cls_idx in unique_classes}

    print(f"Reorganizing latents (Resume mode active)...")

    success_count = 0
    error_count = 0
    skipped_count = 0

    for item in tqdm(image_list, desc="Processing Latents"):
        if item is None:
            continue

        rel_path, class_idx = item

        # Convert image extension (e.g., .jpg) to latent extension (.npy)
        latent_rel_path = os.path.splitext(rel_path)[0] + LATENT_EXT
        src_path = os.path.join(SOURCE_LATENTS_DIR, latent_rel_path)

        class_counters[class_idx] += 1
        current_num = class_counters[class_idx]

        # Construct a standardized filename for deterministic loading
        new_filename = f"latent_{class_idx:03d}_{current_num:04d}{LATENT_EXT}"
        dst_path = os.path.join(OUTPUT_DIR, f"{cls_idx:03d}", new_filename)

        # --- RESUME LOGIC ---
        # Skips copying if the file already exists in the destination to save time on HPC clusters
        if os.path.exists(dst_path):
            success_count += 1
            skipped_count += 1
            continue

        try:
            if os.path.exists(src_path):
                # Copy the latent vector and preserve metadata (copy2)
                shutil.copy2(src_path, dst_path)
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"\nError copying latent {latent_rel_path}: {e}", flush=True)
            error_count += 1

    # --- Summary Statistics ---
    print(f"\n--- REORGANIZATION COMPLETE ---")
    print(f"Total processed/present: {success_count}")
    print(f"Files already existing (skipped): {skipped_count}")
    print(f"Latents missing from source: {error_count}")
    print(f"Check your data at: {OUTPUT_DIR}")


if __name__ == "__main__":
    reorganize_latents()