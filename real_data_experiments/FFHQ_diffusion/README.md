# Generating and evaluating different initialization approaches for FFHQ
This repository contains the framework for **generating and evaluating data** using a pretrained **VE denoiser for diffusion models** and different approaches described in the paper.  
It includes steps for downloading model weights, generating data, and computing evaluation metrics.

---

## 1. Model Preparation

1. **Download the VE denoiser**  
   Download the pretrained VE denoiser weights and place them in the folder:

   ```
   model_weights/
   ```

   > Ensure the weights are compatible with the scripts in this folder.

---

## 2. Data Generation

Once the VE denoiser is ready and the initialization data are prepared in the `NoisedFlow` folder, generate samples:

```bash
python3 generate_data.py
```

- This script generates **3 independent samples** of data per sampling approach.

---

## 3. Computing Results and Metrics

After generating data, compute evaluation metrics:

```bash
python3 compute_results.py
```

- This script calculates:
  - **FIDO**  
  - **KID**  
  - **DINO**  
  - **FID**  
  - **SWD** (Sliced Wasserstein Distance)  
  - **MaxSWD**  

- Inception model and Dino model have to be downloaded to get results.
- The results are saved in a structured format for further reporting or analysis.

---

## 4. Requirements

The required Python environment is the same as used in the **Karras pipeline** of EDM:
---

## Notes

- Ensure the `NoisedFlow` folder contains correctly prepared initialization data before running `generate_data.py`.
- The workflow is designed for **single-machine execution** — no Slurm scripts are required.
