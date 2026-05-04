import os
import pickle
import dnnlib
import numpy as np
import torch
import gc

from generation.gen_functions import generating_data, tensor_to_images, set_seed

# -------------------- Device -------------------- #
device = "cuda"

# -------------------- Seed -------------------- #
# Initial global seed setup
SEED = 1000

# -------------------- Denoiser -------------------- #
model = "ve"  # Variance Exploding (VE) or Variance Preserving (VP)
network_pkl = f"generation/model_weights/edm-ffhq-64x64-uncond-{model}.pkl"
with dnnlib.util.open_url(network_pkl) as f:
    # Load the pre-trained Karras et al. (EDM) denoiser
    denoiser_fn = pickle.load(f)["ema"].to(device).float()
denoiser_fn.eval()

# -------------------- Training Data Directory -------------------- #
train_set_dir = "datasets/train_set"

# -------------------- Sigma Schedules -------------------- #
# Standard EDM Full sigma schedule (used for classic EDM baseline)
sigma_max, sigma_min = 80.0, 0.002
n_steps = 40
rho = 7.0

t = torch.linspace(0, 1, n_steps, device=device)
inv_rho = 1.0 / rho
sigmas = (sigma_max**inv_rho + t * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho
sigmas = sigmas.to(device)

# Shorter sigma schedule (Used for TarFlow and specialized noise-level tests)
sigma_max_shorter = 7 
n_steps_shorter = 20  
rho_shorter = 5.0

t_shorter = torch.linspace(0, 1, n_steps_shorter, device=device)
inv_rho_shorter = 1.0 / rho_shorter
sigmas_shorter = (
    sigma_max_shorter**inv_rho_shorter
    + t_shorter * (sigma_min**inv_rho_shorter - sigma_max_shorter**inv_rho_shorter)
) ** rho_shorter
sigmas_shorter = sigmas_shorter.to(device)


print("Using shorter sigma schedule:", sigmas_shorter)
print("Model:", network_pkl)

# -------------------- Generation Parameters -------------------- #
gen_size = 70_000
batch_size_gen = 250
SEED = 0 # Final seed override for loop start

# -------------------- Output Folders -------------------- #
sample_dir = "generation/some_samples"
os.makedirs(sample_dir, exist_ok=True)

generated_dir = "generation/init_&_gen"
os.makedirs(generated_dir, exist_ok=True)

# -------------------- Generation Loop -------------------- #
# Compare TarFlow (SOTA) against standard generative baselines
for name in [
    "TarFlow_FFHQ_noised_with_factor_14_SOTA_update_prior_seed",
    #"TarFlow_FFHQ_noised_with_factor_14_big_128_dynamical_seed",
    "gaussian_sigma_7",
    "edm_classic",
    "empirical",
    "trained",
]:
    for j in range(3): # Conduct 3 independent trials per model
        used_seed = SEED + j
        set_seed(used_seed)
        generator = name
        init = None
        
        # Logic to load TarFlow/Flow-based initializations
        if "FFHQ" in name or "flow" in name:
            try:
                init = torch.load(f"../NoisedFlow/FFHQ_Flow/{name}_{j}.pt")
                print("Correctly loaded")
            except (FileNotFoundError, OSError):
                try:
                    init = torch.load(
                        f"../NoisedFlow/FFHQ_Flow/{name}_{j}.pt"
                    )
                    print("Correctly loaded")
                except Exception as e:
                    print(f"❌ File {name} not found")
                    exit()
            print("Correctly loaded initialization.")
            generator = "flow_noised"
            
        # Baseline: Standard EDM sampling
        if name == "edm_classic":
            generator = "edm_classic"
            
        # Baseline: Empirical noise (Real data + Gaussian)
        if name == "empirical":
            real_data = torch.load("../NoisedFlow/FFHQ_Flow/datasets/train_set_tensor.pt")
            init = real_data + sigma_max_shorter * torch.randn_like(real_data)
            generator = "flow_noised"
            
        # Baseline: Data generated from EDM + additional noise
        if name == "trained":
            edm = torch.load(
                f"generation/init_&_gen/generation_edm_classic_{model}_{j}.pt"
            )
            init = edm + sigma_max_shorter * torch.randn_like(edm)
            generator = "flow_noised"
            
        # Baseline: Pure Gaussian noise at sigma=7
        if name == "gaussian_sigma_7":
            init = 7 * torch.randn(gen_size, 3, 64, 64)
            generator = "flow_noised"

        # Data Integrity Check
        print("Check nan and inf in init data")
        if init is not None:
            num_nan_init = torch.isnan(init).sum().item()
            num_inf_init = torch.isinf(init).sum().item()
            print(f"Init data: NaN={num_nan_init}, Inf={num_inf_init}")
            init[torch.isnan(init)] = 0.0
            init[torch.isinf(init)] = 0.0
            print("After cleaning NaN and Inf in init data")
            num_nan_init = torch.isnan(init).sum().item()
            num_inf_init = torch.isinf(init).sum().item()
            print(f"Init data: NaN={num_nan_init}, Inf={num_inf_init}")

        # --- Sub-loop: Generate and save small visual samples (n=30) ---
        sample_init = (
            init[:30].to(device)
            if init is not None
            else torch.randn(30, 3, 64, 64, device=device)
        )
        print(sigmas)
        _, sample_generated = generating_data(
            denoiser=denoiser_fn,
            gen_size=30,
            sigmas=sigmas,
            sigmas_shorter=sigmas_shorter,
            initialization=sample_init,
            batch_size_gen=30,
            generator_type=generator,
            train_set_dir=train_set_dir,
            small_noise_flow=0.05,
            device=device,
        )

        sample_images = tensor_to_images(sample_generated, to_pil=True)
        for i, img in enumerate(sample_images):
            img.save(os.path.join(sample_dir, f"image_{name}_{model}_{j}_{i:03d}.png"))

        print(f"Saved 30 sample images for generator '{name}' in '{sample_dir}'")

        # --- Main Generation: Generate full dataset (n=70,000) ---
        init_data, generated_data = generating_data(
            denoiser=denoiser_fn,
            gen_size=gen_size,
            sigmas=sigmas,
            sigmas_shorter=sigmas_shorter,
            initialization=init,
            batch_size_gen=batch_size_gen,
            generator_type=generator,
            train_set_dir=train_set_dir,
            small_noise_flow=0.05,
            device=device,
        )
        
        # Save results using specific naming conventions for EDM vs Others
        if "edm_classic" == name:
            extension = f"{name}_{model}_{j}"
            torch.save(init_data, os.path.join(generated_dir, f"init_{extension}.pt"))
            torch.save(
                generated_data,
                os.path.join(generated_dir, f"generation_{extension}.pt"),
            )
        else:
            extension = f"{name}_{n_steps_shorter}_{model}_{j}"
            torch.save(init_data, os.path.join(generated_dir, f"init_{extension}.pt"))
            torch.save(
                generated_data,
                os.path.join(generated_dir, f"generation_{extension}.pt"),
            )
            print(f"Saved init_{extension}.pt and generation_{extension}.pt")

        # Memory management to avoid CUDA OOM during the 70k generation
        del init, init_data, generated_data, sample_init, sample_generated
        torch.cuda.empty_cache()
        gc.collect()