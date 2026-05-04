import torch
from TarFlowFFHQ.architecture import TarFlowModule, Model, TarFlowFFHQDataModule
from TarFlowFFHQ.utils import set_random_seed
from datetime import datetime
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os

# ============================================================
## ⚙️ Initial Configuration and Hardware
# ============================================================
# Enable TensorFloat-32 (TF32) on supported NVIDIA GPUs for faster matmuls
torch.set_float32_matmul_precision("high")

# Ensure reproducibility across different training runs
RANDOM_SEED = 200
set_random_seed(RANDOM_SEED)

# ============================================================
## 🛠️ Training Parameters and Hyperparameters
# ============================================================
# Core Training Parameters
BATCH_SIZE = 500  # Global batch size; optimized for GPU utilization
EPOCHS = 700      # Extended training schedule for complex image manifolds
LEARNING_RATE = 3e-4
ACCUMULATION_STEPS = 1  # Standard training; increase for effective larger batch sizes
ENABLE_AMP = True       # Automatic Mixed Precision for memory efficiency

# Data/Scaling Parameters
# FACTOR is a critical hyperparameter that scales the target distribution 
# to a range more suitable for the flow's affine coupling stability.
FACTOR = 14  
RESCALE_FACTOR = 1 / FACTOR  
SIGMA_MAX = 7  # Level of terminal noise injected into the target manifold
DATA_PATH = "datasets/train_set_tensor.pt"

print(f"⚙️ Using FACTOR: {FACTOR}")

# ============================================================
## 🏗️ Model Architecture Parameters
# ============================================================
IMG_SIZE = 64
IN_CHANNELS = 3
PATCH_SIZE = 2
# Network capacity: 128 channels with 8 layers per block
CHANNELS = 128  
NUM_BLOCKS = 1  
LAYERS_PER_BLOCK = 8
NVP = True  # Enforces Non-Volume Preserving transformations via Jacobian determinant
NUM_CLASSES = 0  # 0 indicates the model learns the unconditional distribution p(x)

# ============================================================
## 💾 Data and Model Setup
# ============================================================

# 1️⃣ Initialize the Data Module (Handles loading and on-the-fly noising)
data_module = TarFlowFFHQDataModule(
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    sigma_max=SIGMA_MAX,
)
os.makedirs("flow_models", exist_ok=True)

# 2️⃣ Initialize the Core Model (The actual Flow architecture)
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

# 3️⃣ Initialize the Lightning Module (Training logic wrapper)
# Encapsulates optimizer setup and the forward-KLD loss calculation
tarflow_module = TarFlowModule(
    model=model,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    accum_steps=ACCUMULATION_STEPS,
    rescale_factor=RESCALE_FACTOR,
    enable_amp=ENABLE_AMP,
    sigma_max=SIGMA_MAX,
)

print("✅ Data, Model, and Lightning Module initialized.")

# ============================================================
## 📝 Logging and Checkpointing
# ============================================================
# Dynamic run naming for TensorBoard experiment tracking
RUN_NAME = f"FFHQ_noised_factor_{FACTOR}_SOTA_update_prior_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = os.path.join("runs", RUN_NAME)

logger = TensorBoardLogger(save_dir="runs", name=RUN_NAME)

# Save the best model state; monitor can be added (e.g., validation loss)
SAVE_PATH = f"TarFlow_FFHQ_noised_with_factor_{FACTOR}_SOTA_update_prior"
checkpoint_callback = ModelCheckpoint(
    dirpath="flow_models",
    filename=SAVE_PATH,
    save_top_k=1,
    save_last=False,
    monitor=None,
)
print(f"💾 Checkpoint will be saved as: {SAVE_PATH}.ckpt")

# ============================================================
## 🚀 Trainer Initialization and Training
# ============================================================

# Initialize the high-level Lightning Trainer
trainer = L.Trainer(
    accelerator="gpu",
    devices=torch.cuda.device_count(),
    # Using bf16-mixed to maintain numerical stability during training 
    # while gaining the speed of half-precision.
    precision="bf16-mixed",  
    max_epochs=EPOCHS,
    accumulate_grad_batches=ACCUMULATION_STEPS,
    default_root_dir=LOG_DIR,
    log_every_n_steps=10,
    logger=logger,
    callbacks=[checkpoint_callback],
)

# ===== Execution =====
print(f"🔥 Starting Training for {EPOCHS} epochs...")
# Connects the model logic with the data stream
trainer.fit(tarflow_module, datamodule=data_module)
print("\n✅ Training complete!")