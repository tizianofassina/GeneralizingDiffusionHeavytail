# Noised TarFlow Training on FFHQ

This repository contains the framework for training Noise ** TarFlow models** on the **FFHQ dataset**.  
The pipeline supports both model training and generation of initialization data.

---

## 1. Dataset Preparation

1. **Download FFHQ**  
   Follow the official instructions to download the FFHQ dataset:  
   [FFHQ Dataset](https://github.com/NVlabs/edm)
   
## 2. Model Training

Train the Noised TarFlow model on FFHQ:

```bash
python3 train_noised_lightning.py
```
The model is set here for fixed training. 
To modify this behavior, edit the training loop in the architecture and the datamodule by simply commenting or uncommenting the relevant lines.
---

## 3. Generating Initialization Data

Once the model is trained, generate initialization samples:

```bash
python3 generating_noised_lightning.py
```


---


This project uses on Python libraries that can require a specific version. These versions are expected to be compatible with the code. 

```text
torch==2.5.1
lightning==2.5.6
pytorch-lightning==2.5.6
```

