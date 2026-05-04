import os
import math
import numpy as np
import torch
from tqdm import tqdm
import csv  # Added for CSV export
import gc
from generation import dnnlib
from generation import torch_utils
import sys

# Maintain compatibility with dnnlib dependencies
sys.modules["dnnlib"] = dnnlib
sys.modules["torch_utils"] = torch_utils
from metric_fid_wass_class import *
from generation.sampling import set_seed
from generation.sampling import decode_dict_clean

# -------------------- Global Configuration -------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results", exist_ok=True)
output_path = (
    f"results/results_ImageNet_dog_final.txt"  # qualitative log
)
csv_path = f"results/results_ImageNet_dogs_final.csv"  # quantitative table

# Initialize CSV file with comprehensive headers for statistical reporting
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Model",
                "SWD_Global",
                "SWD_Global_Std",
                "Real_SWD_Global",
                "Real_SWD_Global_Std",
                "Init_SWD_Global",
                "Init_SWD_Global_Std",
                "Real_Init_SWD_Global",
                "Real_Init_SWD_Global_Std",
                "MaxWD_Global",
                "MaxWD_Global_Std",
                "Real_MaxWD_Global",
                "Real_MaxWD_Global_Std",
                "Init_MaxWD_Global",
                "Init_MaxWD_Global_Std",
                "Real_Init_MaxWD_Global",
                "Real_Init_MaxWD_Global_Std",
                "FID_Global",
                "KID_Global",
                "DINO_FID_Global",
            ]
        )

with open(output_path, "a") as f:

    def log(s=""):
        print(s)
        f.write(s + "\n")
        f.flush()

    # Loop through all saved model outputs
    for name in [
        "edm_classic_seed_0", 
        "edm_classic_seed_1", 
        "edm_classic_seed_2", 
        "empirical_seed_0", 
        "empirical_seed_1", 
        "empirical_seed_2", 
        "flow_IN_CFG_0.5_sigma_7_factor_14_big_128_seed_0", 
        "flow_IN_CFG_0.5_sigma_7_factor_14_big_128_seed_1", 
        "flow_IN_CFG_0.5_sigma_7_factor_14_big_128_seed_2", 
        "flow_IN_CFG_1.0_sigma_7_factor_14_big_128_seed_0", 
        "flow_IN_CFG_1.0_sigma_7_factor_14_big_128_seed_1", 
        "flow_IN_CFG_1.0_sigma_7_factor_14_big_128_seed_2", 
        "flow_IN_CFG_1.5_sigma_7_factor_14_big_128_seed_0", 
        "flow_IN_CFG_1.5_sigma_7_factor_14_big_128_seed_1", 
        "flow_IN_CFG_1.5_sigma_7_factor_14_big_128_seed_2", 
        "flow_IN_CFG_2.0_sigma_7_factor_14_big_128_seed_0", 
        "flow_IN_CFG_2.0_sigma_7_factor_14_big_128_seed_1", 
        "flow_IN_CFG_2.0_sigma_7_factor_14_big_128_seed_2", 
        "flow_IN_CFG_0.5_sigma_7_factor_14_big_128_dynamical_seed_0", 
        "flow_IN_CFG_0.5_sigma_7_factor_14_big_128_dynamical_seed_1", 
        "flow_IN_CFG_0.5_sigma_7_factor_14_big_128_dynamical_seed_2", 
        "flow_IN_CFG_1.0_sigma_7_factor_14_big_128_dynamical_seed_0", 
        "flow_IN_CFG_1.0_sigma_7_factor_14_big_128_dynamical_seed_1", 
        "flow_IN_CFG_1.0_sigma_7_factor_14_big_128_dynamical_seed_2", 
        "flow_IN_CFG_1.5_sigma_7_factor_14_big_128_dynamical_seed_0", 
        "flow_IN_CFG_1.5_sigma_7_factor_14_big_128_dynamical_seed_1", 
        "flow_IN_CFG_1.5_sigma_7_factor_14_big_128_dynamical_seed_2", 
        "flow_IN_CFG_2.0_sigma_7_factor_14_big_128_dynamical_seed_0", 
        "flow_IN_CFG_2.0_sigma_7_factor_14_big_128_dynamical_seed_1", 
        "flow_IN_CFG_2.0_sigma_7_factor_14_big_128_dynamical_seed_2", 
        "gaussian_sigma_7_seed_0", 
        "gaussian_sigma_7_seed_1", 
        "gaussian_sigma_7_seed_2", 
    ]:

        log("\n" + "=" * 80)
        log(f"Model: {name}")
        log("=" * 80)

        csv_row = {"Model": name}

        # Define noise scale based on model type
        if "edm_classic" in name:
            sigma_max = 80.0
        else:
            sigma_max = 7.0

        print("Sigma_max:", sigma_max)
        set_seed(300_000)

        # Load normalized ImageNet Dog tensors
        real_data = torch.load(
            "../NoisedFlow/ImageNet_Flow/datasets/train_set_tensor_IN_dogs_normalized.pt"
        )

        gen_data = torch.load(f"generation/init_&_gen/generation_{name}.pt")
        p_theta = torch.load(f"generation/init_&_gen/init_{name}.pt")

        if real_data.keys() != gen_data.keys() or real_data.keys() != p_theta.keys():
            raise ValueError("Generated data and real data have different classes.")

        # Reconstruct the theoretical prior p_T
        p_T = {}
        for class_id, tensor in real_data.items():
            p_T[class_id] = tensor + sigma_max * torch.randn_like(tensor)

        log(f"→ Loaded data. sigma_max = {sigma_max}")

        # -------------------- Sliced Wasserstein Evaluation -------------------- #
        log("Studying encoded data with Wasserstein Distance...")
        for proj in [20_000]:
            # SWD calculations for generated vs real and initial vs prior distributions
            gen_swd_class = compute_class_swd(
                gen_data, real_data, n_tests=4, n_proj=proj, device=device
            )
            gen_swd_global = compute_global_swd(
                gen_data, real_data, n_tests=4, n_proj=proj, n_data=17500, device=device
            )
            real_swd_global = compute_global_swd(
                real_data,
                real_data,
                n_tests=4,
                n_proj=proj,
                n_data=17500,
                device=device,
                random=True,
            )
            real_swd_class = compute_class_swd(
                real_data, real_data, n_tests=4, n_proj=proj, device=device, random=True
            )
            gen_init_swd_class = compute_class_swd(
                p_theta, p_T, n_tests=4, n_proj=proj, device=device
            )
            gen_init_swd_global = compute_global_swd(
                p_theta, p_T, n_tests=4, n_proj=proj, n_data=17500, device=device
            )
            real_init_swd_class = compute_class_swd(
                p_T, p_T, n_tests=4, n_proj=proj, device=device, random=True
            )
            real_init_swd_global = compute_global_swd(
                p_T,
                p_T,
                n_tests=4,
                n_proj=proj,
                device=device,
                n_data=17500,
                random=True,
            )

            # Assign SWD metrics to the CSV buffer
            csv_row["SWD_Global"] = gen_swd_global[0]
            csv_row["SWD_Global_Std"] = gen_swd_global[1]
            csv_row["Real_SWD_Global"] = real_swd_global[0]
            csv_row["Real_SWD_Global_Std"] = real_swd_global[1]
            csv_row["Init_SWD_Global"] = gen_init_swd_global[0]
            csv_row["Init_SWD_Global_Std"] = gen_init_swd_global[1]
            csv_row["Real_Init_SWD_Global"] = real_init_swd_global[0]
            csv_row["Real_Init_SWD_Global_Std"] = real_init_swd_global[1]

            log(f"SWD WD Results (Projections: {proj}):")
            log(f"Generated SWD: {gen_swd_global[0]:.4f} ± {gen_swd_global[1]:.4f}")
            log(f"Real SWD: {real_swd_global[0]:.4f} ± {real_swd_global[1]:.4f}")

            gc.collect()
            torch.cuda.empty_cache()

            # -------------------- Max Wasserstein Evaluation -------------------- #
            gen_maxwd_class = compute_class_maxwd(
                gen_data, real_data, n_tests=4, n_proj=proj, device=device
            )
            gen_maxwd_global = compute_global_maxwd(
                gen_data, real_data, n_tests=4, n_proj=proj, n_data=17500, device=device
            )
            real_maxwd_global = compute_global_maxwd(
                real_data, real_data, n_tests=4, n_proj=proj, n_data=17500, device=device, random=True
            )
            real_maxwd_class = compute_class_maxwd(
                real_data, real_data, n_tests=4, n_proj=proj, device=device, random=True
            )
            gen_init_maxwd_class = compute_class_maxwd(
                p_theta, p_T, n_tests=4, n_proj=proj, device=device
            )
            gen_init_maxwd_global = compute_global_maxwd(
                p_theta, p_T, n_tests=4, n_proj=proj, n_data=17500, device=device
            )
            real_init_maxwd_class = compute_class_maxwd(
                p_T, p_T, n_tests=4, n_proj=proj, device=device, random=True
            )
            real_init_maxwd_global = compute_global_maxwd(
                p_T, p_T, n_tests=4, n_proj=proj, device=device, n_data=17500, random=True
            )

            # Assign MaxWD metrics to CSV buffer
            csv_row["MaxWD_Global"] = gen_maxwd_global[0]
            csv_row["MaxWD_Global_Std"] = gen_maxwd_global[1]
            csv_row["Real_MaxWD_Global"] = real_maxwd_global[0]
            csv_row["Real_MaxWD_Global_Std"] = real_maxwd_global[1]
            csv_row["Init_MaxWD_Global"] = gen_init_maxwd_global[0]
            csv_row["Init_MaxWD_Global_Std"] = gen_init_maxwd_global[1]
            csv_row["Real_Init_MaxWD_Global"] = real_init_maxwd_global[0]
            csv_row["Real_Init_MaxWD_Global_Std"] = real_init_maxwd_global[1]

            log(f"Max WD Results (Projections: {proj}):")
            log(f"Generated MaxWD: {gen_maxwd_global[0]:.4f} ± {gen_maxwd_global[1]:.4f}")

            gc.collect()
            torch.cuda.empty_cache()

        # -------------------- Inception / Perceptual Evaluation -------------------- #
        log("Studying encoded data with FID/DINO...")
        imgs_gen = None

        if os.path.exists(f"features/feats_{name}_Inception.pt"):
            feats_gen_class = torch.load(f"features/feats_{name}_Inception.pt")
        else:
            imgs_gen = decode_dict_clean(gen_data, batchsize=16, device=device)
            feats_gen_class = extract_class_inception_feats(
                data_dict=imgs_gen, batch_size=128, device=device
            )
            torch.save(feats_gen_class, f"features/feats_{name}_Inception.pt")

        feats_real = torch.load("features/feats_real_Inception.pt")

        # FID calculation
        fid_global, mean_fid_global, sigma_fid_global = compute_global_fid(
            feats_gen_class, feats_real
        )
        csv_row["FID_Global"] = fid_global
        log(f"FID (global): {fid_global:.4f}")

        # KID calculation
        kid_class = compute_class_kid(feats_gen_class, feats_real)
        kid_global = compute_global_kid(feats_gen_class, feats_real)
        csv_row["KID_Global"] = kid_global
        log(f"KID (global): {kid_global}")

        gc.collect()
        torch.cuda.empty_cache()

        # -------------------- DINOv2 Perceptual Evaluation -------------------- #
        if os.path.exists(f"features/feats_{name}_Dino.pt"):
            feats_dino_gen_class = torch.load(f"features/feats_{name}_Dino.pt")
            if "gen_data" in locals(): del gen_data
        else:
            if imgs_gen is None:
                imgs_gen = decode_dict_clean(gen_data, batchsize=16, device=device)
            if "gen_data" in locals(): del gen_data
            with torch.cuda.amp.autocast():
                feats_dino_gen_class = extract_class_dinov2_feats(
                    data_dict=imgs_gen, batch_size=32, device=device
                )
            torch.save(feats_dino_gen_class, f"features/feats_{name}_Dino.pt")

        feats_dino_real = torch.load("features/feats_real_Dino.pt")

        # Cast DINO features to float32 for metric stability
        for k in feats_dino_gen_class: feats_dino_gen_class[k] = feats_dino_gen_class[k].to(torch.float32)
        for k in feats_dino_real: feats_dino_real[k] = feats_dino_real[k].to(torch.float32)

        fid_global_dino, _, _ = compute_global_fid(feats_dino_gen_class, feats_dino_real)
        csv_row["DINO_FID_Global"] = fid_global_dino
        log(f"DINO FID (global): {fid_global_dino:.4f}")

        # Commit final metrics to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([csv_row.get(h) for h in [
                "Model", "SWD_Global", "SWD_Global_Std", "Real_SWD_Global", "Real_SWD_Global_Std",
                "Init_SWD_Global", "Init_SWD_Global_Std", "Real_Init_SWD_Global", "Real_Init_SWD_Global_Std",
                "MaxWD_Global", "MaxWD_Global_Std", "Real_MaxWD_Global", "Real_MaxWD_Global_Std",
                "Init_MaxWD_Global", "Init_MaxWD_Global_Std", "Real_Init_MaxWD_Global", "Real_Init_MaxWD_Global_Std",
                "FID_Global", "KID_Global", "DINO_FID_Global"
            ]])

        gc.collect()
        torch.cuda.empty_cache()