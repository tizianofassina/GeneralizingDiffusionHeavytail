import os
import json
import shutil
from tqdm import tqdm

# --- PATH CONFIGURATION ---
# Base directory containing the raw unstructured ImageNet images
BASE_DIR = "ImageNet_512_processed/datasets/img512"
# JSON mapping providing the ground truth class labels for each image
JSON_PATH = os.path.join(BASE_DIR, "dataset.json")
# Output directory where images will be stored in a class-indexed hierarchy
OUTPUT_DIR = "ImageNet_512_processed/datasets/img512_right"


def reorganize_and_rename_dataset():
    """
    Standardizes the raw ImageNet-512 image dataset.
    Renames images to a fixed format (img_CLASS_INDEX.png) and sorts them 
    into subdirectories for class-conditional data loading.
    """
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: Cannot find {JSON_PATH}")
        return

    print(f"Loading metadata from {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        # Expects a list of [relative_path, class_idx]
        image_list = data.get("labels", [])

    if not image_list:
        print("ERROR: JSON file is empty.")
        return

    # 1. Prepare class folders
    # Identifies all unique labels to create directory structure (000/ to 999/)
    unique_classes = set(item[1] for item in image_list if item is not None)
    for cls_idx in unique_classes:
        os.makedirs(os.path.join(OUTPUT_DIR, f"{cls_idx:03d}"), exist_ok=True)

    # 2. Counter for each class to ensure sequential naming within subfolders
    class_counters = {cls_idx: 0 for cls_idx in unique_classes}

    print(f"Starting process (Resume mode active)...")

    success_count = 0
    error_count = 0
    skipped_count = 0

    # 3. Process images
    for item in tqdm(image_list, desc="Processing"):
        if item is None:
            continue

        rel_path, class_idx = item
        src_path = os.path.join(BASE_DIR, rel_path)

        # Increment local class counter for filename generation
        class_counters[class_idx] += 1
        current_img_num = class_counters[class_idx]

        # Standardized naming convention for consistent dataset indexing
        new_filename = f"img_{class_idx:03d}_{current_img_num:04d}.png"
        dst_path = os.path.join(OUTPUT_DIR, f"{cls_idx:03d}", new_filename)

        # --- RESUME LOGIC ---
        # Checks if file already exists in destination; prevents redundant I/O 
        # during re-runs or interrupted processes.
        if os.path.exists(dst_path):
            success_count += 1
            skipped_count += 1
            continue

        try:
            if os.path.exists(src_path):
                # copy2 preserves metadata (timestamps, etc.) while moving the file
                shutil.copy2(src_path, dst_path)
                success_count += 1
            else:
                # Logs missing source files without crashing the entire pipeline
                error_count += 1
        except Exception as e:
            print(f"\nError processing {rel_path}: {e}", flush=True)
            error_count += 1

    # --- Summary Report ---
    print(f"\n--- PROCESSING COMPLETE ---")
    print(f"Total processed/present: {success_count}")
    print(f"Files already existing (skipped): {skipped_count}")
    print(f"Files missing from source: {error_count}")
    print(f"Dataset location: {OUTPUT_DIR}")


if __name__ == "__main__":
    reorganize_and_rename_dataset()
