import torch
from TarFlowFFHQ.architecture import Model, TarFlowModule
from TarFlowFFHQ.utils import set_random_seed
from tqdm import tqdm
from datetime import datetime
import os
import gc

# ============================================================
## ⚙️ Initial Configuration and Hardware
# ============================================================
# Use "high" precision for TensorFloat-32 (TF32) on supported GPUs (Ampere+)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
## 🛠️ Generation and Model Parameters
# ============================================================
# I/O and Sampling Parameters
BATCH_SIZE = 500  # High batch size for high-throughput generation
NUM_SAMPLES_TOTAL = 70000  # Matches the size of the original FFHQ dataset
FACTOR = 14  # Normalization factor used during training to scale pixel intensities
RESCALE_FACTOR = 1 / FACTOR  # Inverse scale to bring flow outputs back to data range
SIGMA_MAX = 7  # Terminal noise level used in the targeted flow formulation

# Model Architecture Parameters (Vision Transformer style patching)
IMG_SIZE = 64
IN_CHANNELS = 3
PATCH_SIZE = 2
CHANNELS = 128
NUM_BLOCKS = 1
LAYERS_PER_BLOCK = 8
NVP = True  # Using RealNVP-style Affine Coupling layers (Non-Volume Preserving)
NUM_CLASSES = 0  # 0 indicates unconditional generation (p(x) vs p(x|y))

# Derived Parameters: Flow operates in the flattened patch space
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
# Latent space shape: [Batch, Patches, Pixel-dim-per-patch]
Z_SHAPE = (BATCH_SIZE, NUM_PATCHES, IN_CHANNELS * PATCH_SIZE**2)

# ============================================================
## 💾 Model Loading
# ============================================================

# 1️⃣ Initialize the skeleton architecture
model = Model(
    in_channels=IN_CHANNELS,
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    channels=CHANNELS,
    num_blocks=NUM_BLOCKS,
    layers_per_block=LAYERS_PER_BLOCK,
    nvp=NVP,
    num_classes=NUM_CLASSES,
)

# 2️⃣ Define path to the pre-trained Lightning checkpoint
CKPT_NAME = f"TarFlow_FFHQ_noised_with_factor_14_big_128_dynamical.ckpt"
CHECKPOINT_PATH = os.path.join("flow_models", CKPT_NAME)
print(f"🔍 Loading Lightning checkpoint: {CHECKPOINT_PATH}")

# 3️⃣ Load state_dict via Lightning wrapper and extract the core torch model
module = TarFlowModule.load_from_checkpoint(
    CHECKPOINT_PATH, model=model, map_location=device
)
model = module.model.to(device)
model.eval()  # Disable Dropout/Batchnorm for inference

print("✅ Lightning wrapper and model loaded successfully!")

# ============================================================
## 🔁 Unconditional Generation Loop
# ============================================================

# The flow's 'reverse' method maps Latent Noise -> Image Data
reverse_fn = getattr(model, "reverse", None)
if reverse_fn is None:
    raise AttributeError("The TarFlow Model does not have a 'reverse' method defined.")

# Run generation for multiple seeds to assess variability or build different datasets
for j in range(3):
    RANDOM_SEED = j
    set_random_seed(RANDOM_SEED)

    # Output naming convention for seed-specific generation
    SAVE_PATH = CKPT_NAME.replace(".ckpt", f"_seed_{j}.pt")
    print(f"\n📂 Generation for Seed {j} -> {SAVE_PATH}")

    all_tensors = []
    num_batches = NUM_SAMPLES_TOTAL // BATCH_SIZE
    print(f"🚀 Starting unconditional generation (Total Batches: {num_batches})...")

    for i in tqdm(range(num_batches), desc="Unconditional Generation"):
        # Use mixed precision (bf16) for faster inference without losing dynamic range
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            # 1. Sample from base distribution (Standard Normal in latent space)
            z = torch.randn(Z_SHAPE, device=device)

            # 2. Pass through Flow and rescale intensities
            # Output is divided by RESCALE_FACTOR to reverse the training normalization
            x_gen = reverse_fn(x=z) / RESCALE_FACTOR

            # 3. Quality control: check for numerical instability
            if torch.isnan(x_gen).any():
                print(f"\n⚠️ NaN detected in batch {i}")
            else:
                pass

            # Move to CPU immediately to manage GPU VRAM
            all_tensors.append(x_gen.cpu())

    # ============================================================
    ## 📦 Save Output
    # ============================================================

    # Concatenate all batches into a single large tensor [70000, 1024, 12]
    final_tensors = torch.cat(all_tensors, dim=0)
    torch.save(final_tensors, SAVE_PATH)
    print(f"\n✅ {final_tensors.shape[0]} tensors generated and saved to '{SAVE_PATH}'")
    
    # Explicit memory cleanup for next seed loop
    del final_tensors, all_tensors
    gc.collect()