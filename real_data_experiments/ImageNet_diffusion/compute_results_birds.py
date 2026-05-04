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

# -------------------- Global Setup -------------------- #
# Ensuring compatibility for EDM2/DNNLIB dependencies
sys.modules["dnnlib"] = dnnlib
sys.modules["torch_utils"] = torch_utils
from metric_fid_wass_class import *
from generation.sampling import set_seed
from generation.sampling import decode_dict_clean

# -------------------- Global Configuration -------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results", exist_ok=True)
# Target paths for qualitative (txt) and quantitative (csv) metric logging
output_path = f"results/results_ImageNet_birds_final.txt"  # This has to be changed from birds to dogs
csv_path = f"results/results_ImageNet_birds_final.csv"  # This has to be changed from birds to dogs

# Initialize CSV file with headers if it doesn't exist to prevent overwrite errors
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

    # Iterate through all experimental runs (Classic EDM vs Empirical vs TarFlow variants)
    for name in [
        "edm_classic_birds_seed_0",  # This has to be changed from birds to dogs
        "edm_classic_birds_seed_1",  # This has to be changed from birds to dogs
        "edm_classic_birds_seed_2",  # This has to be changed from birds to dogs
        "empirical_birds_seed_0",  # This has to be changed from birds to dogs
        "empirical_birds_seed_1",  # This has to be changed from birds to dogs
        "empirical_birds_seed_2",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_seed_0",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_seed_1",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_seed_2",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.0_sigma_7_factor_14_big_128_seed_0",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.0_sigma_7_factor_14_big_128_seed_1",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.0_sigma_7_factor_14_big_128_seed_2",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.5_sigma_7_factor_14_big_128_seed_0",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.5_sigma_7_factor_14_big_128_seed_1",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.5_sigma_7_factor_14_big_128_seed_2",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_2.0_sigma_7_factor_14_big_128_seed_0",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_2.0_sigma_7_factor_14_big_128_seed_1",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_2.0_sigma_7_factor_14_big_128_seed_2",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_dynamical_seed_0",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_dynamical_seed_1",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_dynamical_seed_2",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.0_sigma_7_factor_14_big_128_dynamical_seed_0",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.0_sigma_7_factor_14_big_128_dynamical_seed_1",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.0_sigma_7_factor_14_big_128_dynamical_seed_2",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.5_sigma_7_factor_14_big_128_dynamical_seed_0",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.5_sigma_7_factor_14_big_128_dynamical_seed_1",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_1.5_sigma_7_factor_14_big_128_dynamical_seed_2",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_2.0_sigma_7_factor_14_big_128_dynamical_seed_0",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_2.0_sigma_7_factor_14_big_128_dynamical_seed_1",  # This has to be changed from birds to dogs
        "flow_IN_birds_CFG_2.0_sigma_7_factor_14_big_128_dynamical_seed_2",  # This has to be changed from birds to dogs
        "gaussian_birds_sigma_7_seed_0",  # This has to be changed from birds to dogs
        "gaussian_birds_sigma_7_seed_1",  # This has to be changed from birds to dogs
        "gaussian_birds_sigma_7_seed_2",  # This has to be changed from birds to dogs
    ]:

        log("\n" + "=" * 80)
        log(f"Model: {name}")
        log("=" * 80)

        # Buffer for current model results to write into CSV row
        csv_row = {"Model": name}

        # Match sigma_max to the initialization used in the generation script
        if "edm_classic" in name:
            sigma_max = 80.0
        else:
            sigma_max = 7.0

        print("Sigma_max:", sigma_max)
        set_seed(300_000)

        # Load ground truth and generated data for comparison
        real_data = torch.load(
            "../NoisedFlow/ImageNet_Flow/datasets/train_set_tensor_birds_normalized.pt"
        )

        gen_data = torch.load(f"generation/init_&_gen/generation_{name}.pt")
        p_theta = torch.load(f"generation/init_&_gen/init_{name}.pt")

        if real_data.keys() != gen_data.keys() or real_data.keys() != p_theta.keys():
            raise ValueError("Generated data and real data have different classes.")

        # Construct the theoretical noise distribution (p_T) for initialization comparison
        p_T = {}
        for class_id, tensor in real_data.items():
            p_T[class_id] = tensor + sigma_max * torch.randn_like(tensor)

        log(f"→ Loaded data. sigma_max = {sigma_max}")

        # -------------------- Wasserstein Metric Suite -------------------- #
        # Calculating Sliced Wasserstein Distance (SWD) and Maximum WD
        log("Studying encoded data with Wasserstein Distance...")
        for proj in [20_000]: # Number of random projections for SWD
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

            # Store SWD global metrics and their standard deviations
            csv_row["SWD_Global"] = gen_swd_global[0]
            csv_row["SWD_Global_Std"] = gen_swd_global[1]
            csv_row["Real_SWD_Global"] = real_swd_global[0]
            csv_row["Real_SWD_Global_Std"] = real_swd_global[1]
            csv_row["Init_SWD_Global"] = gen_init_swd_global[0]
            csv_row["Init_SWD_Global_Std"] = gen_init_swd_global[1]
            csv_row["Real_Init_SWD_Global"] = real_init_swd_global[0]
            csv_row["Real_Init_SWD_Global_Std"] = real_init_swd_global[1]

            log(f"SWD WD Results:")
            log(f"--- Projections: {proj} ---")
            log(
                f"Generated SWD (global): {gen_swd_global[0]:.4f} ± {gen_swd_global[1]:.4f}"
            )
            log(
                f"Real SWD (global): {real_swd_global[0]:.4f} ± {real_swd_global[1]:.4f}"
            )
            log(
                f"Generation Init SWD (global): {gen_init_swd_global[0]:.4f} ± {gen_init_swd_global[1]:.4f}"
            )
            log(
                f"Real Init SWD (global): {real_init_swd_global[0]:.4f} ± {real_init_swd_global[1]:.4f}"
            )

            # Cleanup to avoid GPU memory fragmentation
            del (
                gen_swd_class,
                gen_swd_global,
                real_swd_global,
                real_swd_class,
                gen_init_swd_class,
                gen_init_swd_global,
                real_init_swd_class,
                real_init_swd_global,
            )
            gc.collect()
            torch.cuda.empty_cache()

            # Global Maximum Wasserstein Distance evaluation
            gen_maxwd_global = compute_global_maxwd(
                gen_data, real_data, n_tests=4, n_proj=proj, n_data=17500, device=device
            )
            real_maxwd_global = compute_global_maxwd(
                real_data,
                real_data,
                n_tests=4,
                n_proj=proj,
                n_data=17500,
                device=device,
                random=True,
            )

            gen_init_maxwd_global = compute_global_maxwd(
                p_theta, p_T, n_tests=4, n_proj=proj, n_data=17500, device=device
            )
            real_init_maxwd_global = compute_global_maxwd(
                p_T,
                p_T,
                n_tests=4,
                n_proj=proj,
                device=device,
                n_data=17500,
                random=True,
            )

            # Store MaxWD global metrics with Std
            csv_row["MaxWD_Global"] = gen_maxwd_global[0]
            csv_row["MaxWD_Global_Std"] = gen_maxwd_global[1]
            csv_row["Real_MaxWD_Global"] = real_maxwd_global[0]
            csv_row["Real_MaxWD_Global_Std"] = real_maxwd_global[1]
            csv_row["Init_MaxWD_Global"] = gen_init_maxwd_global[0]
            csv_row["Init_MaxWD_Global_Std"] = gen_init_maxwd_global[1]
            csv_row["Real_Init_MaxWD_Global"] = real_init_maxwd_global[0]
            csv_row["Real_Init_MaxWD_Global_Std"] = real_init_maxwd_global[1]

            log(f"Max WD Results:")
            log(f"--- Projections: {proj} ---")
            log(
                f"Generated MaxWD (global): {gen_maxwd_global[0]:.4f} ± {gen_maxwd_global[1]:.4f}"
            )
            log(
                f"Real MaxWD (global): {real_maxwd_global[0]:.4f} ± {real_maxwd_global[1]:.4f}"
            )
            log(
                f"Generation Init MaxWD (global): {gen_init_maxwd_global[0]:.4f} ± {gen_init_maxwd_global[1]:.4f}"
            )
            log(
                f"Real Init MaxWD (global): {real_init_maxwd_global[0]:.4f} ± {real_init_maxwd_global[1]:.4f}"
            )

            del (
                gen_maxwd_global,
                real_maxwd_global,
                gen_init_maxwd_global,
                real_init_maxwd_global,
            )
            gc.collect()
            torch.cuda.empty_cache()

        # -------------------- Perceptual Metrics (FID/DINO) -------------------- #
        log("Studying encoded data with FID/DINO...")

        imgs_gen = None

        # Check for cached Inception features to speed up repetitive runs
        if os.path.exists(f"features/feats_{name}_Inception.pt"):
            feats_gen_class = torch.load(f"features/feats_{name}_Inception.pt")
        else:
            imgs_gen = decode_dict_clean(gen_data, batchsize=16, device=device)
            feats_gen_class = extract_class_inception_feats(
                data_dict=imgs_gen, batch_size=128, device=device
            )
            torch.save(feats_gen_class, f"features/feats_{name}_Inception.pt")

        feats_real = torch.load(
            "features/feats_real_birds_Inception.pt"
        )  # This has to be changed from birds to dogs

        # Compute Global Fréchet Inception Distance
        fid_global, mean_fid_global, sigma_fid_global = compute_global_fid(
            feats_gen_class, feats_real
        )

        csv_row["FID_Global"] = fid_global

        log(
            f"FID (global): {fid_global:.4f} (mean: {mean_fid_global}, sigma: {sigma_fid_global})"
        )

        # Kernel Inception Distance (KID) evaluation
        kid_class = compute_class_kid(feats_gen_class, feats_real)
        kid_global = compute_global_kid(feats_gen_class, feats_real)

        csv_row["KID_Global"] = kid_global

        log(f"KID (global): {kid_global}")

        del (
            real_data,
            p_T,
            p_theta,
            feats_gen_class,
            feats_real,
            fid_global,
            mean_fid_global,
            sigma_fid_global,
            kid_class,
            kid_global,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # DinoV2 feature extraction for modern perceptual similarity check
        if os.path.exists(f"features/feats_{name}_Dino.pt"):
            feats_dino_gen_class = torch.load(f"features/feats_{name}_Dino.pt")
            if "gen_data" in locals():
                del gen_data
                gc.collect()
                torch.cuda.empty_cache()
        else:
            if imgs_gen is None:
                imgs_gen = decode_dict_clean(gen_data, batchsize=16, device=device)
            if "gen_data" in locals():
                del gen_data
                gc.collect()
                torch.cuda.empty_cache()
            with torch.cuda.amp.autocast():
                feats_dino_gen_class = extract_class_dinov2_feats(
                    data_dict=imgs_gen, batch_size=32, device=device
                )
            torch.save(feats_dino_gen_class, f"features/feats_{name}_Dino.pt")

        feats_dino_real = torch.load(
            "features/feats_real_birds_Dino.pt"
        )  # This has to be changed from birds to dogs

        # Standardizing Dino features to float32 for metric calculations
        for k in feats_dino_gen_class:
            feats_dino_gen_class[k] = feats_dino_gen_class[k].to(torch.float32)

        for k in feats_dino_real:
            feats_dino_real[k] = feats_dino_real[k].to(torch.float32)

        fid_global_dino, mean_fid_global_dino, sigma_fid_global_dino = (
            compute_global_fid(feats_dino_gen_class, feats_dino_real)
        )

        csv_row["DINO_FID_Global"] = fid_global_dino

        log(
            f"DINO FID (global): {fid_global_dino:.4f} (mean: {mean_fid_global_dino}, sigma: {sigma_fid_global_dino})"
        )

        # Final step: Persist the full row of metrics to the CSV results file
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    csv_row.get("Model"),
                    csv_row.get("SWD_Global"),
                    csv_row.get("SWD_Global_Std"),
                    csv_row.get("Real_SWD_Global"),
                    csv_row.get("Real_SWD_Global_Std"),
                    csv_row.get("Init_SWD_Global"),
                    csv_row.get("Init_SWD_Global_Std"),
                    csv_row.get("Real_Init_SWD_Global"),
                    csv_row.get("Real_Init_SWD_Global_Std"),
                    csv_row.get("MaxWD_Global"),
                    csv_row.get("MaxWD_Global_Std"),
                    csv_row.get("Real_MaxWD_Global"),
                    csv_row.get("Real_MaxWD_Global_Std"),
                    csv_row.get("Init_MaxWD_Global"),
                    csv_row.get("Init_MaxWD_Global_Std"),
                    csv_row.get("Real_Init_MaxWD_Global"),
                    csv_row.get("Real_Init_MaxWD_Global_Std"),
                    csv_row.get("FID_Global"),
                    csv_row.get("KID_Global"),
                    csv_row.get("DINO_FID_Global"),
                ]
            )

        del (
            feats_dino_gen_class,
            feats_dino_real,
            fid_global_dino,
            mean_fid_global_dino,
            sigma_fid_global_dino,
            imgs_gen,
        )
        imgs_gen = None
        gc.collect()
        torch.cuda.empty_cache()