import os
import math
import csv
import numpy as np
import torch
from tqdm import tqdm
import gc
from generation.gen_functions import set_seed
from metric_FFHQ import (
    compute_swd,
    compute_swd_new,
    compute_maxwd,
    compute_optimized_maxwd,
    compute_mu_sigma_fid,
    compute_mu_sigma_fid_dino,
    fid_score,
    compute_swd_fid,
    compute_maxwd_fid,
    compute_kid,
    extract_inception_features,
)

# -------------------- Device Configuration -------------------- #
device = "cuda"

# -------------------- Test Parameters -------------------- #
n_tests, n_tests_real = 4, 4
n_batch_proj = 100
size = 70_000
half_size = size // 2
fid_size = 50_000

# -------------------- Results Directory & CSV Setup -------------------- #
modality = "ve"
os.makedirs("results", exist_ok=True)
output_path_txt = f"results/results_{modality}_final.txt"
output_path_csv = f"results/results_{modality}_final.csv"

# Define CSV Header
headers = [
    "model_name",
    "sigma_max",
    "KID",
    "FID_computed_real",
    "FID_karras_ffhq",
    "DINOv2_FID",
    "SWD_gen_data_mean",
    "SWD_gen_data_std",
    "MaxWD_gen_data_mean",
    "MaxWD_gen_data_std",
    "SWD_init_mean",
    "SWD_init_std",
    "MaxWD_init_mean",
    "MaxWD_init_std",
    "SWD_FID_gen",
    "MaxWD_FID_gen",
    "SWD_real_baseline",
    "MaxWD_real_baseline",
]

file_exists = os.path.isfile(output_path_csv)

if not os.path.isfile(output_path_csv) or os.path.getsize(output_path_csv) == 0:
    with open(output_path_csv, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)

with open(output_path_txt, "a") as f_txt:

    def log(s=""):
        print(s)
        f_txt.write(s + "\n")
        f_txt.flush()

    for name in [
        "TarFlow_FFHQ_noised_with_factor_14_big_128_dynamical_seed_20_ve_0",
        "TarFlow_FFHQ_noised_with_factor_14_big_128_dynamical_seed_20_ve_1",
        "TarFlow_FFHQ_noised_with_factor_14_big_128_dynamical_seed_20_ve_2",
        "TarFlow_FFHQ_noised_with_factor_14_SOTA_update_prior_seed_20_ve_0",
        "TarFlow_FFHQ_noised_with_factor_14_SOTA_update_prior_seed_20_ve_1",
        "TarFlow_FFHQ_noised_with_factor_14_SOTA_update_prior_seed_20_ve_2",
        "edm_classic_ve_0",
        "edm_classic_ve_1",
        "edm_classic_ve_2",
        "empirical_20_ve_0",
        "empirical_20_ve_1",
        "empirical_20_ve_2",
        "gaussian_sigma_7_20_ve_0",
        "gaussian_sigma_7_20_ve_1",
        "gaussian_sigma_7_20_ve_2",
        "trained_20_ve_0",
        "trained_20_ve_1",
        "trained_20_ve_2",
    ]:
        log("\n" + "=" * 80)
        log(f"Model: {name}")
        log("=" * 80)

        # --- Sigma Max Logic (Unchanged) ---
        if any(x in name for x in ["sigma_5", "factor_12_small", "factor_10_big"]):
            sigma_max = 5.4710636138916015625
        elif name == "flow_noised_sigma_2":
            sigma_max = 2.2943
        elif "edm_classic" in name:
            sigma_max = 80.0
        elif any(x in name for x in ["sigma_7, FFHQ"]):
            sigma_max = 7.0
        elif "trained" in name:
            sigma_max = 7.0
        elif "empirical" in name:
            sigma_max = 7.0
        else:
            sigma_max = 7.0

        print("Sigma max : ", sigma_max)

        set_seed(300_000)

        # --- Load Generated and Initialization Data (Unchanged) ---
        gen_data = torch.load(f"generation/init_&_gen/generation_{name}.pt")[
            :size
        ].clip(-1.0, 1.0)
        p_theta = torch.load(f"generation/init_&_gen/init_{name}.pt")

        # --- Load Real Data and create variants ---
        real_data = torch.load("datasets/train_set_tensor.pt")[:size]
        p_T = real_data + sigma_max * torch.randn_like(real_data)
        gaussian = math.sqrt(sigma_max**2 + 0.333) * torch.randn_like(real_data)

        indices = torch.randperm(size)[:fid_size]

        # --- Compute KID ---
        feats_real = extract_inception_features(
            real_data[indices], batch_size=1000, device=device
        )
        feats_gen = extract_inception_features(
            gen_data[indices], batch_size=1000, device=device
        )
        kid_val = compute_kid(feats_real, feats_gen, n_max=fid_size, device="cpu")
        log(f"KID value: {kid_val:.6f}")

        # --- Compute Classical FID ---
        mu_real_computed, sigma_real_computed = compute_mu_sigma_fid(
            real_data[indices], batch_size=1000, device=device
        )

        # Load pre-computed Karras statistics
        data_karras = np.load("generation/train_info/ffhq-64x64.npz")
        mu_karras, sigma_karras = data_karras["mu"].astype(np.float64), data_karras[
            "sigma"
        ].astype(np.float64)

        mu_gen, sigma_gen = compute_mu_sigma_fid(
            gen_data[indices], batch_size=1000, device=device
        )

        fid_comp, _, _ = fid_score(
            mu_real_computed, sigma_real_computed, mu_gen, sigma_gen
        )
        fid_karras, _, _ = fid_score(mu_karras, sigma_karras, mu_gen, sigma_gen)

        log(f"FID vs computed real: {fid_comp:.4f}")
        log(f"FID vs Karras FFHQ: {fid_karras:.4f}")

        # --- DINOv2 FID ---
        mu_r_dino, sigma_r_dino = compute_mu_sigma_fid_dino(
            real_data[indices], batch_size=1000, device=device
        )
        mu_g_dino, sigma_g_dino = compute_mu_sigma_fid_dino(
            gen_data[indices], batch_size=1000, device=device
        )
        fid_dino, _, _ = fid_score(mu_r_dino, sigma_r_dino, mu_g_dino, sigma_g_dino)
        log(f"DINOv2 FID: {fid_dino:.4f}")

        # --- SWD / MaxWD (using n_proj = 20,000 as per your script) ---
        n_proj = 20_000

        # SWD Init

        init_gen_mean_swd, init_gen_std_swd = compute_swd_new(
            p_theta,
            p_T,
            n_proj=n_proj,
            n_batch_proj=n_batch_proj,
            n_tests=n_tests,
            n_data=17500,
            random=True,
        )
        init_real_mean_swd, init_real_std_swd = compute_swd_new(
            p_T,
            p_T,
            n_proj=n_proj,
            n_batch_proj=n_batch_proj,
            n_tests=n_tests,
            n_data=17500,
            random=True,
        )
        # SWD data
        gen_gen_mean_swd, gen_gen_std_swd = compute_swd_new(
            gen_data,
            real_data,
            n_proj=n_proj,
            n_batch_proj=n_batch_proj,
            n_tests=n_tests,
            n_data=17500,
            random=True,
        )

        gen_real_mean_swd, gen_real_std_swd = compute_swd_new(
            real_data,
            real_data,
            n_proj=n_proj,
            n_batch_proj=n_batch_proj,
            n_tests=n_tests,
            n_data=17500,
            random=True,
        )

        # Max init
        init_gen_mean_maxwd, init_gen_std_maxwd = compute_optimized_maxwd(
            p_theta, p_T, n_tests=n_tests_real, random=True, n_data=17500
        )

        init_real_mean_maxwd, init_real_std_maxwd = compute_optimized_maxwd(
            p_T, p_T, n_tests=n_tests_real, random=True, n_data=17500
        )

        # Max data
        gen_gen_mean_maxwd, gen_gen_std_maxwd = compute_optimized_maxwd(
            gen_data, real_data, n_tests=n_tests_real, random=True, n_data=17500
        )
        gen_real_mean_maxwd, gen_real_std_maxwd = compute_optimized_maxwd(
            real_data,
            real_data,
            n_tests=n_tests_real,
            random=True,
            n_data=17500,
        )
        # FID-like metrics (Wasserstein distance on Inception features)
        gen_fid_swd = compute_swd_fid(
            gen_data[:fid_size],
            real_data[:fid_size],
            n_proj=n_proj,
            n_tests=1,
            n_batch_proj=n_batch_proj,
        )
        gen_fid_maxwd = compute_maxwd_fid(
            gen_data[:fid_size],
            real_data[:fid_size],
            n_proj=n_proj,
            n_tests=1,
            n_batch_proj=n_batch_proj,
        )

        # --- CSV WRITING ---
        row = [
            name,
            f"{sigma_max:.4f}",
            f"{kid_val:.8f}",
            f"{fid_comp:.4f}",
            f"{fid_karras:.4f}",
            f"{fid_dino:.4f}",
            f"{gen_gen_mean_swd:.4f}",  # SWD data mean
            f"{gen_gen_std_swd:.4f}",  # SWD data std
            f"{gen_gen_mean_maxwd:.4f}",  # Max data mean
            f"{gen_gen_std_maxwd:.4f}",  # Max data std
            f"{init_gen_mean_swd:.4f}",  # SWD init mean
            f"{init_gen_std_swd:.4f}",  # SWD init std
            f"{init_gen_mean_maxwd:.4f}",  # Max init mean
            f"{init_gen_std_maxwd:.4f}",  # Max init std
            f"{gen_fid_swd[0] if isinstance(gen_fid_swd, (list, tuple)) else gen_fid_swd:.4f}",
            f"{gen_fid_maxwd[0] if isinstance(gen_fid_maxwd, (list, tuple)) else gen_fid_maxwd:.4f}",
            f"{gen_real_mean_swd:.4f}",  # Baseline SWD real
            f"{gen_real_mean_maxwd:.4f}",  # Baseline MaxWD real
        ]

        with open(output_path_csv, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(row)

        log(f"-> Model {name} data written to CSV.")
        del gen_data, real_data, p_theta, p_T, feats_real, feats_gen, mu_gen, sigma_gen
        gc.collect()
        torch.cuda.empty_cache()

log(f"\nAll processes completed. CSV saved at: {output_path_csv}")
