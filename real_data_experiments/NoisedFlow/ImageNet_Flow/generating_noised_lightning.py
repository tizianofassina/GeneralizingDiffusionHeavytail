import torch
import json
import os
from TarFlowImageNet.architecture import Model, TarFlowModule
from TarFlowImageNet.utils import set_random_seed
from tqdm import tqdm
import sys
from datetime import datetime
from typing import Dict, List

# ============================================================
## ⚙️ Initial Configuration and Hardware
# ============================================================
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Accepts dynamical flag and seed from CLI for parallel job arrays
dynamical = bool(int(sys.argv[1]))

# ===== File Setup =====
# Selection between the 'Fixed Sigma' or 'Dynamical Sigma' model checkpoints
if dynamical:
    CKPT_NAME = "TarFlow_ImageNet_birds_best_dynamical.ckpt"
else:
    CKPT_NAME = "TarFlow_ImageNet_birds_best.ckpt"

CHECKPOINT_PATH = os.path.join("flow_models", CKPT_NAME)
LABEL_JSON_FILE = "flow_class_map.json"

print(f"Using model : {CKPT_NAME}")

# ===== Parameters (Sampling Settings) =====
seed = int(sys.argv[2])
set_random_seed(seed)
batch_size = 200 
rescale_factor = 1 / 14 
sigma_max = 7
SAMPLES_PER_CLASS = 1000  # Target number of initializations for each category

# Classifier-Free Guidance (CFG) Weight
# Values > 1.0 push the generation further away from the unconditional distribution
# toward the class-specific manifold.
if len(sys.argv) > 1:
    GUIDANCE_WEIGHT = float(sys.argv[3])
    print(f"Using GUIDANCE_WEIGHT from command line: {GUIDANCE_WEIGHT}")
else:
    GUIDANCE_WEIGHT = 1.0
    print(f"Using default GUIDANCE_WEIGHT: {GUIDANCE_WEIGHT}")

# ===== Architecture Definition =====
img_size = 64
in_channels = 4
patch_size = 2
channels = 128
num_blocks = 1
layers_per_block = 8
nvp = True

num_patches = (img_size // patch_size) ** 2
Z_SHAPE_TEMPLATE = (batch_size, num_patches, in_channels * patch_size**2)

# ============================================================
## 1. Load Class Map & ID Resolution
# ============================================================
print(f"🔍 Loading class map configuration from {LABEL_JSON_FILE}...")
with open(LABEL_JSON_FILE, "r") as f:
    config = json.load(f)

# Reconstruct the mapping used during training: Sequential ID -> Original ImageNet ID
SUBSET_TO_ORIGINAL_MAP = {int(k): v for k, v in config["class_map_birds"].items()}
NUM_CLASSES = len(SUBSET_TO_ORIGINAL_MAP)
SUBSET_INDICES_TO_GENERATE = sorted(SUBSET_TO_ORIGINAL_MAP.keys())

NUM_SAMPLES_TOTAL = NUM_CLASSES * SAMPLES_PER_CLASS
num_batches_per_class = SAMPLES_PER_CLASS // batch_size

print(f"**NUM_CLASSES = {NUM_CLASSES}**. Batches per class: {num_batches_per_class}.")

# ============================================================
## 2. Model Loading (Conditional Mode)
# ============================================================
model = Model(
    in_channels=in_channels,
    img_size=img_size,
    patch_size=patch_size,
    channels=channels,
    num_blocks=num_blocks,
    layers_per_block=layers_per_block,
    nvp=nvp,
    num_classes=NUM_CLASSES,
)

print(f"🔍 Loading Lightning checkpoint: {CHECKPOINT_PATH}")

# Note: drop_label=True is required to enable the CFG (unconditional) path in the model
module = TarFlowModule.load_from_checkpoint(
    CHECKPOINT_PATH, model=model, map_location=device, drop_label=True
)
model = module.model.to(device)
model.eval()
reverse_fn = getattr(model, "reverse", None)

print("✅ Conditional model loaded successfully!")

# ============================================================
## 3. Generation Loop
# ============================================================
# Stores output indexed by ORIGINAL ImageNet ID for downstream evaluation compatibility
generated_data_dict: Dict[int, torch.Tensor] = {}

factor_val = int(1 / rescale_factor)
run_name_type = "dynamical" if dynamical else "static"
run_name = f"flow_IN_birds_CFG_{GUIDANCE_WEIGHT}_sigma_{sigma_max}_factor_{factor_val}_big_{channels}_{run_name_type}"

save_path = f"{run_name}_seed_{seed}.pt"
print(f"🚀 Starting generation of {NUM_SAMPLES_TOTAL} total images.")

# 

for subset_class_idx in SUBSET_INDICES_TO_GENERATE:
    original_class_id = SUBSET_TO_ORIGINAL_MAP[subset_class_idx]
    current_class_batches: List[torch.Tensor] = []

    # Create the conditioning label tensor for the current batch
    y_cond = torch.full(
        (batch_size,), subset_class_idx, dtype=torch.long, device=device
    )

    print(f"\nGenerating subset index {subset_class_idx} (Original ID: {original_class_id})...")

    for i in tqdm(range(num_batches_per_class), desc=f"ID {original_class_id}"):
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            # 1. Start from Standard Normal latent noise
            z = torch.randn(Z_SHAPE_TEMPLATE, device=device)

            # 2. Reverse Flow with CFG
            # The 'guidance' parameter linearly interpolates between the conditional 
            # and unconditional score/log-likelihood estimates.
            x_gen = reverse_fn(x=z, y=y_cond, guidance=GUIDANCE_WEIGHT) / rescale_factor

            current_class_batches.append(x_gen.cpu())

    if current_class_batches:
        # Group all class samples: [SAMPLES_PER_CLASS, Channels, LatentH, LatentW]
        class_tensor_concatenated = torch.cat(current_class_batches, dim=0)
        generated_data_dict[str(original_class_id)] = class_tensor_concatenated

# ============================================================
## 4. Save Aggregated Output
# ============================================================
total_images_generated = sum(t.shape[0] for t in generated_data_dict.values())

# Save as a PyTorch dictionary (.pt) for easy loading in diffusion training scripts
torch.save(generated_data_dict, save_path)
print(f"\n✅ {total_images_generated} images generated and saved to '{save_path}'")