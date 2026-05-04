# ImageNet Flow: Noised TarFlow Training
---

This repository contains the experimental framework for training of Noised **TarFlow** on **ImageNet**.  
The pipeline includes dataset preparation, preprocessing, training, and generation of initialization data.

The code is set for birds subset.


## 1. Dataset Preparation

1. **Download ImageNet**  
   Follow the instructions on the official [EDM2 repository](https://github.com/NVlabs/edm2) to obtain the ImageNet dataset.

2. **Convert to encoded images**  
   Convert the dataset to the  encoded as described in Karras et al. This ensures compatibility with the downstream preprocessing and training scripts.


This operations should produce a folder `ImageNet_512_processed/datasets/` that is ready to be reordered (images and latents) using the following scripts.
The space needed for this operations is around 2T.
  
   
4. **Organize data by category**  
   Run the scripts 
    
    ```bash
   python reordering_latent.py
   
   python reordering_data.py
   ```
   
   > This scripts arranges the dataset in category folders for easier subset selection and preprocessing.

---

## 2. Subset Selection and Preprocessing

Once the dataset is organized:

1. **Prepare the precise subset (dogs, birds or other subset)**  
   The subset can be defined in `flow_class_map.json` using the notation $i : label[i]$. This allows focusing on specific classes for your experiments.

2. **Run preprocessing**  
   ```bash
   python preprocess_data.py   
   ```

   > This script prepares the data for TarFlow training, applying necessary transformations and saving processed tensors. the correct map of the json file has to be specified. 

---

## 3. TarFlow Training

After preprocessing, train the model:
For fix training (constant noise) pass the argument 0, for dynamical pass 1. 

```bash 
python training_noised_lightning.py 0
```

- This will run the **Noised TarFlow training** using the preprocessed ImageNet subset.
- Training logs and checkpoints will be saved according to the configuration in the script.

---

## 4. Generating Initialization Data

Once the model is trained, generate data for initialization. Parameters to be specified are dynamical (we use dynamically trained model), seed value, CFG parameter.
Use 0,1,2 as seed values for reproducing results of the article. 
For fixed, seed = 0 , CFG = 0.5 run 
```bash
python generate_noised_lightning.py 0 0 0.5
```
- Seeds used in the article are = 0,1,2.
- The generated data can be used for further experiments, model evaluation, or as a starting point for additional training.


## Notes

- All scripts are designed for **Slurm-based HPC clusters**. 

This project uses on Python libraries that can require a specific version. These versions are expected to be compatible with the code. 

```text
torch==2.5.1
lightning==2.5.6
pytorch-lightning==2.5.6
```

