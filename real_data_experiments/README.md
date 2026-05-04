# Paper code repository

This repository contains code for reproducing experiments and generating samples using **score-based generative models** across different datasets.  
It is organized into four main folders:

---

## Repository Structure

- **`toy/`**  
  Contains code for reproducing **toy experiments**.  
  Implements all functions used for toy distributions and provides scripts for training, sampling, and computing metrics.

- **`NoisedFlow/`**  
  Code for training **flows on noised data** for both FFHQ and ImageNet.  
  > Note: Data must be downloaded and preprocessed before training to be usable.

- **`FFHQ_diffusion/`**  
  Code for generating and evaluating images on **FFHQ** using all approaches proposed in the article.  
  > Evaluation metrics (e.g., Inception, DINO, FID) require downloading the corresponding pretrained models.

- **`ImageNet_diffusion/`**  
  Code for generating and evaluating images on **ImageNet subsets** using all approaches proposed in the article.  
  Evaluation metrics (e.g., Inception, DINO, FID) require downloading the corresponding pretrained models.  

> **`toy/`** repository is indipendent from others.
> `NoisedFlow/ImageNet_Flow` and `ImageNet_diffusion` are configured for the **birds dataset**.  
> To work on the **dogs dataset**, several minor modifications are required.

## Notes

- Each folder contains its own **README.md** with further details on scripts, data preparation, and evaluation procedures.  
- Each subfolder contains its own requirements.txt specific to that experiment. Please note that the code was originally developed and tuned on a high-performance computing (HPC) cluster. Consequently, minor environment adjustments might be necessary when running on local machines or different hardware architectures.
-    All scripts set random seeds to ensure reproducible results across runs.
