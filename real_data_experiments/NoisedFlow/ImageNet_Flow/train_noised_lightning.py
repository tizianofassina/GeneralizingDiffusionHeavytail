import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from TarFlowImageNet.architecture import TarFlowModule, Model, TarFlowImageNetDataModule
from TarFlowImageNet.utils import set_random_seed
from datetime import datetime
import os, json
from lightning.pytorch.strategies import DDPStrategy
import sys

# High-precision matmuls for Ampere/Hopper GPUs
torch.set_float32_matmul_precision("high")

# ===== File Configuration =====
LABEL_JSON_FILE = "flow_class_map.json"

# ===== Hyperparameters =====
batch_size = 1000
epochs = 700
lr = 3e-4
factor = 14
rescale_factor = 1 / factor
accum_steps = 1
enable_amp = True
sigma_max = 7
num_workers = 6
print(f"Using factor: {factor}")

# Dynamical training: adjusts the noise level sigma dynamically during training steps
dynamical = bool(int(sys.argv[1]))

# ===== Reproducibility =====
set_random_seed(200)
os.makedirs("flow_models", exist_ok=True)

# ===== Dynamic Class Mapping =====
# This logic remaps sparse ImageNet class IDs (e.g., 10, 153, 281) to a 
# dense range (0, 1, 2...) for the classifier/conditioning embedding layer.
print(f"🔍 Loading class map configuration from {LABEL_JSON_FILE}...")
with open(LABEL_JSON_FILE, "r") as f:
    config = json.load(f)

SUBSET_TO_ORIGINAL_MAP = {
    int(k): v for k, v in config["class_map_birds"].items()
} 
ORIGINAL_TO_SUBSET_MAP = {v: k for k, v in SUBSET_TO_ORIGINAL_MAP.items()}
NUM_CLASSES_SUBSET = len(SUBSET_TO_ORIGINAL_MAP)
print(f"**Dynamically set NUM_CLASSES = {NUM_CLASSES_SUBSET}** (from JSON).")

# ===== Data Module =====
# Uses the normalized 4-channel latents created in the previous preprocessing step.
data_module = TarFlowImageNetDataModule(
    data_path="datasets/train_set_tensor_birds_normalized.pt", 
    batch_size=batch_size,
    sigma_max=sigma_max,
    num_workers=num_workers,
    original_to_subset_map=ORIGINAL_TO_SUBSET_MAP,
    dynamical=dynamical,
)

# ===== Model Setup =====
model = Model(
    in_channels=4,          # Working with normalized 'mu' latents
    img_size=64,            # Latent spatial resolution (derived from 512px input)
    patch_size=2,           # Vision-Transformer style patching for the flow
    channels=128,           
    num_blocks=1,
    layers_per_block=8,
    nvp=True,               # Non-Volume Preserving (Affine Coupling)
    num_classes=NUM_CLASSES_SUBSET, # Conditioned on the bird subset
)

tarflow_module = TarFlowModule(
    model=model,
    batch_size=batch_size,
    lr=lr,
    sigma_max=sigma_max,
    accum_steps=accum_steps,
    rescale_factor=rescale_factor,
    enable_amp=enable_amp,
    drop_label=True,        # Enables classifier-free guidance training (CFG)
    dynamical=dynamical,
)

# ===== Logging & Versioning =====
if dynamical:
    run_name = f"ImageNet_birds_factor_{factor}_big_128_dynamical_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
else:
    run_name = f"ImageNet_birds_factor_{factor}_big_128_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

log_dir = os.path.join(os.getcwd(), "runs", run_name)
logger = TensorBoardLogger(save_dir="runs", name=run_name)

if dynamical:
    save_path = f"TarFlow_ImageNet_birds_best_dynamical"
else:
    save_path = f"TarFlow_ImageNet_birds_best"

checkpoint_callback = ModelCheckpoint(
    dirpath="flow_models",
    filename=save_path,
    save_top_k=1,
    save_last=False,
    monitor=None,
)

# ===== Trainer Configuration (HPC / SLURM Optimized) =====
# Pulls environment variables from SLURM to automatically configure 
# multi-node DDP (Distributed Data Parallel) training.
num_gpus_on_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 4))
num_nodes = int(os.environ.get("SLURM_NNODES", 1))

trainer = L.Trainer(
    accelerator="gpu",
    devices=num_gpus_on_node,
    num_nodes=num_nodes,
    strategy="ddp",         # Distributed Data Parallel for synchronization
    precision="bf16-mixed", # Mixed precision for memory and speed
    max_epochs=epochs,
    accumulate_grad_batches=accum_steps,
    logger=logger,
    callbacks=[checkpoint_callback],
)

# ===== Training Execution =====
# Only the master rank (0) prints the starting status to avoid console clutter.
if trainer.global_rank == 0:
    print(f"Starting training on {num_nodes} node(s) with {num_gpus_on_node} GPUs each...")

trainer.fit(tarflow_module, datamodule=data_module)

if trainer.global_rank == 0:
    print("Finish training")