import os
import pickle
import sys
from generation import dnnlib
from generation import torch_utils
import gc

# -------------------- Global Setup -------------------- #
# Injecting local modules into sys.modules for compatibility with EDM2 pickled models
sys.modules["dnnlib"] = dnnlib
sys.modules["torch_utils"] = torch_utils

import numpy as np
import torch
import json
import random
from generation.sampling import (
    generate_data,
    tensor_to_images,
    set_seed,
    sample_init_data_for_display,
    rescale_dict_to_image,
    decode_dict_clean,
    save_images_to_png,
)

gc.collect()

# -------------------- Device & Arguments -------------------- #
device = "cuda"
# SEED_GENERATION is passed as the first CLI argument for reproducibility
SEED_GENERATION = int(sys.argv[1])

# -------------------- Path & Hyperparameter Configuration -------------------- #
model = sys.argv[2]  # typically "s" for the small EDM2 variant
NET_PATH = f"generation/model_weights/edm2-img512-{model}-guid-dino.pkl"
GNET_PATH = f"generation/model_weights/edm2-img512-s-guid-dino_gnet.pkl"
MAP_PATH = "../NoisedFlow/ImageNet_Flow/flow_class_map.json"

BATCH_SIZE = 200
GUIDANCE = 1.9 # Classifier-free guidance scale for EDM2

# ---- Configuration: Standard EDM (Classic Baseline) ----
sigma_max, sigma_min = 80.0, 0.002
n_steps = 32
rho = 7.0

# ---- Configuration: Proposed Method (Shortened Trajectory) ----
# Higher precision initialization allows for fewer ODE solver steps
sigma_max_shorter = 7.0
n_steps_shorter = 20
rho_shorter = 5.0

num_to_generate = 1000  # Target count per ImageNet class

# Directory setup for visual inspection samples
sample_dir = "generation/some_samples_IN_birds"
os.makedirs(sample_dir, exist_ok=True)

# Directory setup for full tensor storage (init and final generation)
generated_dir = "generation/init_&_gen"
os.makedirs(generated_dir, exist_ok=True)

# -------------------- Main Experimental Loop -------------------- #
# Testing different Flow CFG scales and dynamical vs static noise manifolds
for name in [
    f"flow_IN_birds_CFG_1.0_sigma_7_factor_14_big_128_dynamical_seed_{SEED_GENERATION}",
    f"flow_IN_birds_CFG_1.5_sigma_7_factor_14_big_128_dynamical_seed_{SEED_GENERATION}",
    f"flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_dynamical_seed_{SEED_GENERATION}",
    f"flow_IN_birds_CFG_2.0_sigma_7_factor_14_big_128_dynamical_seed_{SEED_GENERATION}",
    f"flow_IN_birds_CFG_1.0_sigma_7_factor_14_big_128_seed_{SEED_GENERATION}",
    f"flow_IN_birds_CFG_1.5_sigma_7_factor_14_big_128_seed_{SEED_GENERATION}",
    f"flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_seed_{SEED_GENERATION}",
    f"flow_IN_birds_CFG_2.0_sigma_7_factor_14_big_128_seed_{SEED_GENERATION}",
    f"empirical_birds_seed_{SEED_GENERATION}",
    f"edm_classic_birds_seed_{SEED_GENERATION}",
    f"gaussian_birds_sigma_7_seed_{SEED_GENERATION}",
]:

    set_seed(SEED_GENERATION)

    init = None
    init_mode = None
    some_samples_init = None
    some_samples_num = 3 # Small subset for preview saving

    # Default to standard schedule
    current_sigma_max = sigma_max
    current_n_steps = n_steps
    current_rho = rho
    current_num_to_gen = num_to_generate

    # Logic for Loading Flow-based Noised Priors
    if "flow" in name:
        init = torch.load("../NoisedFlow/ImageNet_Flow/" + name + ".pt")
        init_mode = "class_preallocated"
        print(init.keys())

        # --- Data Integrity: Checking for NaNs in Flow Prior ---
        total_nan_init = 0
        total_inf_init = 0
        for class_id, tensor in init.items():
            total_nan_init += torch.isnan(tensor).sum().item()
            total_inf_init += torch.isinf(tensor).sum().item()
        print(f"Init data: NaN={total_nan_init}, Inf={total_inf_init}")

        # Set shortened sampling schedule for the proposed method
        current_num_to_gen = num_to_generate
        current_sigma_max = sigma_max_shorter
        current_n_steps = n_steps_shorter
        current_rho = rho_shorter

    # Logic for Empirical Initialization (Noised training data)
    if "empirical" in name:
        current_sigma_max = sigma_max_shorter
        current_n_steps = n_steps_shorter
        current_rho = rho_shorter
        init_mode = "class_preallocated"
        real_data = torch.load(
            "../NoisedFlow/ImageNet_Flow/datasets/train_set_tensor_birds_normalized.pt"
        )
        init = {}
        for label in real_data.keys():
            current_num_to_gen = num_to_generate
            idx = random.choices(range(real_data[label].shape[0]), k=current_num_to_gen)
            # Add noise at sigma=7.0 to real data
            init[label] = real_data[label][idx] + current_sigma_max * torch.randn_like(
                real_data[label][idx]
            )

    # Logic for Gaussian Baseline (Starting from pure noise at sigma=7)
    if "gaussian" in name:
        current_num_to_gen = num_to_generate
        current_sigma_max = sigma_max_shorter
        current_n_steps = n_steps_shorter
        current_rho = rho_shorter
        init_mode = "gaussian"

    # Logic for EDM Classic Baseline (Full 32-step trajectory from sigma=80)
    if "edm_classic" in name:
        current_num_to_gen = num_to_generate
        current_sigma_max = sigma_max
        current_n_steps = n_steps
        current_rho = rho
        init_mode = "gaussian"

    # Create model-specific subdirectory for visual samples
    os.makedirs(
        "generation/some_samples_IN_birds/" + name, exist_ok=True
    )

    # --- PHASE 1: DEBUG/PREVIEW RUN ---
    # Generates a small batch of images to verify quality and visual correctness
    print(f"\n--- DEBUG RUN: {name} ---")
    _, some_samples = generate_data(
        net_path=NET_PATH,
        gnet_path=GNET_PATH,
        device=device,
        class_map_path=MAP_PATH,
        pre_allocated_init=init,
        init_mode=init_mode,
        batch_size=BATCH_SIZE,
        sigma_min=sigma_min,
        sigma_max=current_sigma_max,
        rho=current_rho,
        guidance=GUIDANCE,
        num_to_generate=some_samples_num,
        num_steps=current_n_steps,
    )

    some_samples = decode_dict_clean(some_samples, batchsize=BATCH_SIZE, device=device)
    save_images_to_png(
        some_samples,
        save_dir="generation/some_samples_IN_birds/" + name,
        seed=SEED_GENERATION,
    )

    # --- PHASE 2: FULL DATASET GENERATION ---
    # Generates the full 1000 images per class for final metric calculation (FID/KID)
    print(f"\n--- FULL RUN: {name} ---")
    init_full, gen_full = generate_data(
        net_path=NET_PATH,
        gnet_path=GNET_PATH,
        device=device,
        class_map_path=MAP_PATH,
        pre_allocated_init=init,
        init_mode=init_mode,
        batch_size=BATCH_SIZE,
        sigma_min=sigma_min,
        sigma_max=current_sigma_max,
        rho=current_rho,
        guidance=GUIDANCE,
        num_to_generate=current_num_to_gen,
        num_steps=current_n_steps,
    )

    OUTPUT_DIR = "generation/init_&_gen"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Persist the initial noise/flow states and the final decoded images
    torch.save(init_full, os.path.join(OUTPUT_DIR, f"init_{name}.pt"))
    torch.save(gen_full, os.path.join(OUTPUT_DIR, f"generation_{name}.pt"))

    print(f"✅ Save data for {name}. Total class generated : {len(gen_full)}")
    
    # Explicit memory management to prevent OOM during multi-model loops
    del init, init_full, gen_full, some_samples
    if "real_data" in locals():
        del real_data
    gc.collect()
    torch.cuda.empty_cache()