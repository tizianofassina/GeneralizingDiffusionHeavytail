# EDM2 Denoiser Data Generation and Evaluation

This repository contains the framework for **generating and evaluating data** using a pretrained ** denoiser for diffusion models** of size S from EDM2.  
It includes steps for downloading model weights, generating data, and computing evaluation metrics.
The code is set for birds subset.


## 1. Model Preparation

1. **Download the S denoiser**  
   Download the pretrained S denoiser weights and place them in the folder:

   ```
   model_weights/
   ```
   > Download the conditional and the gnet model.
   > Ensure the weights are compatible with the scripts in this folder.
   

---

## 2. Data Generation

Once the S denoiser is ready and the initialization data are prepared in the `NoisedFlow` folder, generate samples is possible
 by specifying the argument seed (f.e. 0 ) and running : 
Code is set for birds generation. 

```bash
python generate_data.py 0
```

- Attention seed number must match the seed number used in the generation of initialization data.  
- Generated data will be saved in the configured output folder for further analysis `generation/init_\&_gen`.
- For generating dogs or other subclasses edit the generate_data and sampling script. 

---

## 3. Computing Results and Metrics

After generating data, compute evaluation metrics:

```bash
python3 compute_results_birds.py 
```

Before the computations of metrics, it is needed to compute FID and DINO FD features for the training data.
To do so  run : 

```bash
python3 feats_real_data.py 
```


- This script calculates:
  - **FIDO**  
  - **KID**  
  - **DINO**  
  - **FID**  
  - **SWD** (Sliced Wasserstein Distance)  
  - **MaxSWD**  

- Inception and Dino model have to be downloaded to compute the results. 
- The results are saved in a structured format for further reporting or analysis.


---

## 4. Requirements

The required Python environment is the same as used in the **Karras pipeline** of EDM2:

> For full compatibility, please refer to the versions used in Karras’s codebase.

---

## Notes

- Ensure the `NoisedFlow` folder contains correctly prepared initialization data before running `generate_data.py`.
