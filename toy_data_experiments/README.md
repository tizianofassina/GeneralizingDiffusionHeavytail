# Toy Data — Experiments

Toy 2D experiments comparing **diffusion-based denoising** strategies (NN
denoiser vs MCMC denoisers — HMC, Barker, NUTS) starting from three different
initial distributions (`gaussian`, `p_t`, `p_theta`).

Two parallel setups:
- **Heavy-tail**: 4-component mixture of multivariate Student-t (df=3).
- **GMM**: 5×5 = 25-mode Gaussian mixture.

Both follow the same logic; the GMM side is a bit lighter (only the analytic
denoiser, no NN/MCMC variants) and uses an additional **Monte Carlo KL**
evaluation against the analytic noised density.

This is a research playground, not production code. Things may break in subtle
ways if you change paths around — most file lookups are fail-loud (`assert`).


## Pipelines

### Heavy-tail
Run **in this order**:

```
1. create_train_test_data_ht.py  →  ground-truth train/test sets
2. gen_flow_ht.py                →  trains heavy-tail flow, generates p_theta
3. gen_init_ht.py                →  generates p_inf (gaussian) and p_t inits
4. diffusion_generation_nn.py    →  trains NN diffusion + DDPM sampling
5. diffusion_generation_mcmc.py  →  DDPM sampling with MCMC denoisers (Slurm)
6. results_ht.py                 →  evaluation, writes a single CSV

Bonus : evaluating_mcmc.py      → For evaluating the quality of MCMC estimation
```

### GMM
Run **in this order**:

```
1. create_train_test_data_gmm.py  →  ground-truth train/test sets
2. gen_flow_gmm.py                →  trains GMM flow (gaussian-base), saves p_theta
3. diffusion_generation_gmm.py    →  DDPM sampling with the analytic GMM score
4. results_gmm.py                 →  global MSW + Monte Carlo KL, two CSVs
```

For the GMM side, `gaussian` and `p_t` initialisations are generated **on the
fly** inside the denoising script (no separate `gen_init_gmm.py`). `p_theta`
is loaded from the flow checkpoint trained at the matching sigma.

Skipping a step makes the next one crash. That's intentional.


## Setup

```bash
pip install -e .
```

Plus you'll need: `jax`, `flowjax`, `equinox`, `optax`, `numpyro`, `blackjax`,
`pytorch-lightning`, `torch`, `tqdm`, `diptest`, `pyyaml`, `pandas`.

You also need `VE_SGM` somewhere on `PYTHONPATH` (the diffusion architecture
lives there) — only needed for the heavy-tail NN diffusion step.


---

## Heavy-tail steps

### 1) Generate ground-truth data

```bash
python create_train_test_data_ht.py 1000
python create_train_test_data_ht.py 10000
python create_train_test_data_ht.py 100000
```

Saves training sets of size 10³, 10⁴, 10⁵ and two independent test sets of size
10⁷ in `data/dim_2/`. Seeds (34/35/39) are hardcoded — don't change them, the
downstream scripts assume them.

### 2) Train the heavy-tail flow + generate `p_theta`

```bash
python gen_flow_ht.py dynamic    # or "fix"
```

Trains the flow on a grid of `(sigma, n_train)` combinations, samples 10⁷ points
per checkpoint, saves them to `data/data_gen_noised/dim_2/`. Models go in
`model_flow/`.

`dynamic` resamples noise every minibatch; `fix` uses noise drawn once at the
start. The downstream scripts default to `dynamic`.

### 3) Generate `p_inf` and `p_t` initialisations

```bash
python gen_init_ht.py
```

Iterates over the three sigmas (1.24, 1.05, 0.80), saves
`samples_p_inf_sigma_*` and `samples_pt_sigma_*` in `data/data_gen_noised/dim_2/`.
Same seed for every sigma, so the noise pattern is identical and only the scale
changes.

### 4) NN diffusion: train + DDPM sampling

```bash
python diffusion_generation_nn.py
```

Trains 3 diffusion models (one per `train_size ∈ {1000, 10000, 100000}`),
then for each `(train_size, sigma, init)` runs DDPM with the NN denoiser
truncated at `sigma`. Outputs go to `data/data_gen_diffusion_nn/dim_2/`.

Hyperparameters live in `config_diffusion.yaml`.

This is the slowest part. 27 generations × 1M samples each, plus 3 model
trainings.

### 5) MCMC diffusion: DDPM sampling

This one is parametrised, intended for a Slurm array:

```bash
sbatch diffusion_generation_mcmc.sh
```

The array runs all `(sigma, init, mcmc)` combinations for
`mcmc ∈ {hmc, barker, nuts}`. Each task calls:

```bash
python diffusion_generation_mcmc.py <init> <sigma> <mcmc>
```

Outputs go to `data/data_gen_diffusion_mcmc/dim_2/`.

If you don't have Slurm, you can just run the python script in a loop manually.

### 6) Evaluation

```bash
python results_ht.py
```

For every method + sigma + init combination:
- Computes **bulk Max-Sliced Wasserstein** (samples masked to `[0.1, 0.9]`
  quantiles per dim, 10 draws against the reference for variance).
- Finds a **worst direction** on `p_t` (per `(method, sigma)`, also per
  `train_size` for the NN), then computes **tail quantiles** along that
  direction.

Also computes:
- A reference vs reference baseline (noise floor).
- A flow trained with **σ=0** on the fly (just to compare against the noised
  flows).

Everything is dumped in one big CSV at `results/results_diffusion_ht.csv`.


### Bonus 

```bash
python evaluating_mcmc.py
```

For every MCMC perform an evaluation of the quality of the MCMC posterior estimation comparing to an oracle. 
Everything is saved in `results_mcmc_quality.txt`

---

## GMM steps

### 1) Generate ground-truth data

```bash
python create_train_test_data_gmm.py 1000
python create_train_test_data_gmm.py 10000
python create_train_test_data_gmm.py 100000
```

Same seeds and sizes as the heavy-tail version, but saved in `data_gmm/dim_2/`.
The `tail_3` in the filenames is just compatibility cruft — the GMM has no
heavy tails, but the filename convention is shared.

### 2) Train the GMM flow + generate `p_theta`

```bash
python gen_flow_gmm.py dynamic    # or "fix"
```

Same structure as the heavy-tail flow (same sigma grid, same `n_train` grid,
same hyperparameters), but the flow uses a **gaussian** base distribution
instead of a Student-t. Samples 10⁷ points per checkpoint, saves them to
`data_gen_noised_gmm/dim_2/`. Models go in `model_flow_gmm/`.

### 3) Diffusion: DDPM sampling with the analytic score

```bash
python diffusion_generation_gmm.py
```

For each `(sigma, init)` in `{1.24, 1.05, 0.80} × {gaussian, p_t, p_theta}` runs
DDPM with the **analytic GMM-convolved score** (no NN, no MCMC — we already know
the true density). Outputs go to `data_gen_diffusion_gmm/dim_2/`.

`gaussian` and `p_t` are sampled on the fly (no separate init script).
`p_theta` is loaded from the flow checkpoint trained at the matching sigma —
it expects `n_train=10000`, `dynamic` modality. If you trained with `fix` or
a different `n_train`, edit the `FLOW_*` constants at the top of the script.

### 4) Evaluation

```bash
python results_gmm.py dynamic     # or "fix"
```

Two parts in one script:

- **Part A — global MSW**: same logic as the heavy-tail evaluation but on the
  **whole** distribution (no bulk masking, no tail directions). 10 chunks of
  1M against the reference, mean ± std. Also a reference-vs-reference baseline.
  Output: `results_global_gmm.csv`.

- **Part B — Monte Carlo KL**: for every `(n_train, sigma)` cell on the full
  flow grid (24 × 13), loads the flow checkpoint and estimates
  `KL(p_t || p_theta) ≈ E_{x ~ test_set + sigma·N(0,I)} [log p_t(x) - log p_theta(x)]`
  where `log p_t` is the **analytic noised GMM density** and `log p_theta` is
  the flow's `log_prob`. Missing checkpoints become `NaN` cells (no crash).
  Output: `results_kl_gmm_{dynamic|fix}.csv`, rows = `n_train`, columns = `sigma`.


---

## File / directory layout

```
toy_data/
├── config_diffusion.yaml             ← NN diffusion hyperparams (HT only)
│
├── create_train_test_data_ht.py      ← HT data
├── gen_flow_ht.py                    ← HT flow (Student-t base)
├── gen_init_ht.py
├── diffusion_generation_nn.py
├── diffusion_generation_mcmc.py
├── diffusion_generation_mcmc.sh
├── results_ht.py
│
├── create_train_test_data_gmm.py     ← GMM data
├── gen_flow_gmm.py                   ← GMM flow (Gaussian base)
├── diffusion_generation_gmm.py       ← analytic-score denoising
├── results_gmm.py                    ← global MSW + KL MC
│
├── pyproject.toml
├── src/                              ← installed as `hf_toy` package
│   ├── __init__.py
│   ├── bulk_metrics.py               ← max-sliced Wasserstein
│   ├── create_data.py                ← HT reference distribution
│   ├── create_data_gmm.py            ← GMM reference distribution
│   ├── diffusion_samplers.py         ← DDPM samplers
│   ├── mc_denoisers.py               ← HMC / Barker / NUTS denoisers
│   └── find_unimodality.py           ← unimodality onset analysis
│
├── data/                             ← HT, created at runtime
│   ├── dim_2/                        ← train/test sets
│   ├── data_gen_noised/dim_2/        ← p_inf, p_t, flow samples (p_theta)
│   ├── data_gen_diffusion_nn/dim_2/
│   └── data_gen_diffusion_mcmc/dim_2/
│
├── data_gmm/                         ← GMM, created at runtime
│   └── dim_2/                        ← train/test sets
├── data_gen_noised_gmm/dim_2/        ← GMM flow samples (p_theta)
├── data_gen_diffusion_gmm/dim_2/     ← GMM denoised samples
│
├── model_flow/                       ← HT flow checkpoints
├── model_flow_gmm/                   ← GMM flow checkpoints
└── model_diffusion/                  ← HT NN diffusion checkpoints
```


## Choices & seeds

- **HT distribution**: 4-component mixture of multivariate Student-t, df=3,
  global_scale=4, scale=0.1.
- **GMM distribution**: 5×5 grid of 25 isotropic Gaussian modes, scale=0.1,
  global_scale=4.
- **Sigma values used (denoising)**: `[1.24404052, 1.05527906, 0.8012415]` —
  points on the EDM schedule (sigma_max=3, sigma_min=0.0002, rho=2) bracketing
  the unimodality onset (~σ≈1).
- **Sigma grid (flow + KL)**: 13 sigmas from 2.56 down to 0.01, same list for
  HT and GMM.
- **n_train grid (flow + KL)**: 24 sizes from 100 to 100000, same list for
  HT and GMM.
- **Seeds**: distribution=0, train=34, test=35, second test=39, init=53,
  flow_init=42, flow_train=42, flow_sample=999, DDPM=43, evaluation=42,
  KL noise base=1. Same seeds across HT and GMM where it matters.

If you change a seed somewhere, make sure to change it everywhere or things
silently won't match.


## Known gotchas

- `0.801241500` and `0.8012415` are the same float in Python — file paths use
  the latter (Python's default str repr). If you hardcode the long form in a
  string lookup, nothing matches.
- `p_theta` paths bake in the flow training modality (`dynamic` / `fix`),
  number of training samples, and number of epochs. If you retrain the flow
  with different settings, update the `FLOW_*` constants in the diffusion
  scripts (both HT and GMM).
- The diffusion NN scripts skip training if the checkpoint file already exists.
  Delete the checkpoint to force retraining.
- Generation scripts skip outputs if the file already exists (NN, GMM diffusion)
  or overwrite (MCMC, results scripts). Be aware.
- The GMM `results_gmm.py` KL evaluation expects flow checkpoints from
  **all** 24 × 13 = 312 cells. Missing ones become `NaN` rather than crashing,
  but if you only ran a subset of `gen_flow_gmm.py`, expect lots of `NaN`s.
- HT and GMM live in **separate directory trees** (`data/` vs `data_gmm/`,
  `model_flow/` vs `model_flow_gmm/`, etc.). Don't try to share them.


## Reproducing the results

Just run the steps in order, for the side(s) you care about. With the default
seeds, you should get the same numbers we report. If you don't — check the
seeds first, then check that you didn't accidentally rerun the data-generation
script with a different size.