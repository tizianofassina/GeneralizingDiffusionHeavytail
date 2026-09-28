"""
Fixed-sigma comparison at sigma_T = 2.3 — generation + full evaluation.
dim 10, Student-t mixture with 20 modes, tail index 3.

Every method starts from the SAME noise level and runs the SAME Karras grid
(2.3 -> 0.002, rho=3, 40 steps). Only the initialisation and the sampler differ:

    DDPM  + {gaussian, p_t, p_theta}    ancestral,  1 NFE/step
    Heun  + {gaussian, p_t, p_theta}    PF-ODE,     2 NFE/step
    t-EDM + Student-t prior             PF-ODE,     2 NFE/step

"EDM classic" is NOT generated separately: at sigma_T = 2.3 from a Gaussian
prior it coincides exactly with "Heun + gaussian init" (same prior, same
sampler, same denoiser).

DLPM (Levy) enters as a baseline in its NATIVE configuration (full chain,
K = 25, alpha-stable prior), loaded from the file produced by
nn_levy_diffusion_dim_10.py: its forward process is VP with alpha-stable
noise, so it has no sigma comparable to the VE ones to truncate at.

Denoisers are NOT trained here: run the training scripts first.

"""

import os
import csv
import math
import time

import numpy as np
import torch

from VE_SGM.architecture import AbstractDiffusion
from VE_SGM.architecture_t_student import AbstractDiffusionStudentT, sample_t_prior
from VE_SGM.sampling import heun_sampling


# ===========================================================================
# CONFIG
# ===========================================================================
DIM        = 10
N_MIXTURE  = 20
TAIL_INDEX = 3

TRAIN_SEED = 34
SEED_TEST  = 35          # evaluation reference
SEED_TEST2 = 39          # independent clean sample, used to build p_t
DIST_SEED  = 0
TRAIN_SIZE = 10_000
TEST_SIZE  = 10_000_000

# --- fixed-sigma schedule --------------------------------------------------
SIGMA     = 2.3
SIGMA_MIN = 0.002
RHO       = 3
N_STEPS   = 25           # -> 41 nodes

N_GEN     = 1_000_000
BATCH_GEN = 100_000
GEN_SEED  = 43
INIT_SEED = 7            # initial samples, shared by DDPM and Heun

# --- t-EDM -----------------------------------------------------------------
NU, IID, LOG_TIME = 3, False, True

# --- DLPM (native run, already generated elsewhere) -----------------------
DLPM_ALPHA, DLPM_K, DLPM_NUNITS, DLPM_GN = 1.7, 25, 464, 1

# --- flow used as p_theta: MUST have been trained at sigma = 2.3 ----------
FLOW_LAYERS, FLOW_NN_WIDTH, FLOW_NN_DEPTH = 5, 50, 3
FLOW_INF, FLOW_N_TRAIN, FLOW_N_EPOCHS = 1, 10000, 40_000
FLOW_TRAIN_MODALITY = "dynamic"

# --- flow sigma = 0 baseline, trained inside this script -------------------
FLOW_KEY, FLOW_TRAIN_SEED, FLOW_SAMPLE_SEED = 42, 42, 999
FLOW0_N_EPOCHS, FLOW0_LR, FLOW0_BATCH = 20000, 0.001, 1000

# --- metrics ---------------------------------------------------------------
CHUNK_SIZE, N_CHUNKS, MSW_SEED = 1_000_000, 10, 79
TAIL_QUANTILES = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]

DATA_INPUT_DIR = f"data/dim_{DIM}"
NOISED_DIR     = "data/data_gen_noised"
GEN_DIR        = f"data/data_gen_diffusion/dim_{DIM}"
MODEL_DIR      = "model_diffusion"
OUT_DIR        = "results_dim_10_final"

os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

VESGM_RUN = f"vesgm_dim_{DIM}_size_{TRAIN_SIZE}_tail_{TAIL_INDEX}"
TEDM_RUN  = f"tedm_dim_{DIM}_size_{TRAIN_SIZE}_tail_{TAIL_INDEX}_nu_{NU}"

TEST_FILE  = (f"{DATA_INPUT_DIR}/test_set_dim_{DIM}_size_{TEST_SIZE}"
              f"_tail_{TAIL_INDEX}_seed_{SEED_TEST}_dist_seed_{DIST_SEED}.npy")
TEST2_FILE = (f"{DATA_INPUT_DIR}/test_set_dim_{DIM}_size_{TEST_SIZE}"
              f"_tail_{TAIL_INDEX}_seed_{SEED_TEST2}_dist_seed_{DIST_SEED}.npy")
TRAIN_FILE = (f"{DATA_INPUT_DIR}/train_set_dim_{DIM}_size_{TRAIN_SIZE}"
              f"_tail_{TAIL_INDEX}_seed_{TRAIN_SEED}_dist_seed_{DIST_SEED}.npy")

CONFIG_VESGM = {
    "diffusion_config": {"sigma_min": 0.002, "sigma_max": 80.0,
                         "log_mean": -1.0, "log_std": 1.2, "num_workers": 6},
    "optim_config": {"lr": 1e-4},
    "denoiser_config": {"sigma_data": 1, "sigma_min": 0.002, "sigma_max": 80.0,
                        "sigma_disc": 1000, "input_dim": DIM, "embed_dim": 256,
                        "channel_mult": [2, 4, 4, 2]},
    "trainer_config": {"batch_size": 1000},
}
CONFIG_TEDM = {
    "diffusion_config": {"sigma_min": 0.002, "sigma_max": 80.0,
                         "log_mean": -1.0, "log_std": 1.2, "iid": IID,
                         "num_workers": 6},
    "optim_config": {"lr": 1e-4},
    "denoiser_config": {"sigma_data": 1, "sigma_min": 0.002, "sigma_max": 80.0,
                        "sigma_disc": 1000, "input_dim": DIM, "embed_dim": 256,
                        "channel_mult": [2, 4, 4, 2], "nu": NU,
                        "log_time": LOG_TIME},
    "trainer_config": {"batch_size": 1000},
}

LOG, TAIL = [], []


def log(m=""):
    print(m, flush=True)
    LOG.append(m)


def tail_out(m=""):
    TAIL.append(m)


def fmt(sec):
    s = int(sec); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


# ===========================================================================
# PATHS
# ===========================================================================
INIT_TAG = {"gaussian": "gaussian", "p_t": "pt", "p_theta": "p_theta"}


def karras_nodes(sigma_hi, sigma_lo, rho, n_nodes):
    """Ascending Karras grid; sigma_hi IS a node by construction."""
    inv = 1.0 / rho
    t = np.linspace(0.0, 1.0, n_nodes)
    return np.sort((sigma_hi ** inv + t * (sigma_lo ** inv - sigma_hi ** inv)) ** rho)


def flow_ptheta_path():
    return (f"{NOISED_DIR}/samples_flow_flow_dim{DIM}_sig{SIGMA}"
            f"_tail{TAIL_INDEX}_layers{FLOW_LAYERS}_w{FLOW_NN_WIDTH}"
            f"_d{FLOW_NN_DEPTH}_inf{FLOW_INF}_n{FLOW_N_TRAIN}"
            f"_ep{FLOW_N_EPOCHS}_{FLOW_TRAIN_MODALITY}.npy")


def out_path(init, sampler):
    """samples_{tag}_{sampler}_sigma_2.3_size_10000_1000000_tail_3.npy"""
    tag = INIT_TAG.get(init, init)
    return os.path.join(
        GEN_DIR, f"samples_{tag}_{sampler}_sigma_{SIGMA}_size_{TRAIN_SIZE}"
                 f"_{N_GEN}_tail_{TAIL_INDEX}.npy")


TEDM_OUT = os.path.join(
    GEN_DIR, f"samples_tedm_nu_{NU}_heun_sigma_{SIGMA}_size_{TRAIN_SIZE}"
             f"_{N_GEN}_tail_{TAIL_INDEX}.npy")

DLPM_PATH = os.path.join(
    GEN_DIR, f"samples_dlpm_alpha_{DLPM_ALPHA}_K_{DLPM_K}_nu_{DLPM_NUNITS}"
             f"_gn_{DLPM_GN}_size_{TRAIN_SIZE}_{N_GEN}_tail_{TAIL_INDEX}.npy")


# ===========================================================================
# PART A — GENERATION
# ===========================================================================
def load_vesgm(device):
    ckpt = os.path.join(MODEL_DIR, f"{VESGM_RUN}_model.ckpt")
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"
    meta = torch.load(os.path.join(MODEL_DIR, f"{VESGM_RUN}_meta.pt"),
                      map_location="cpu")
    cfg = dict(CONFIG_VESGM["denoiser_config"])
    cfg["sigma_data"] = meta["sigma_data"]
    if "denoiser_config" in meta:
        for k in ("sigma_min", "sigma_max", "sigma_disc", "input_dim",
                  "embed_dim", "channel_mult"):
            assert meta["denoiser_config"][k] == cfg[k], (
                f"{k}: training={meta['denoiser_config'][k]} != gen={cfg[k]}")
    model = AbstractDiffusion.load_from_checkpoint(
        ckpt, diffusion_config=CONFIG_VESGM["diffusion_config"],
        optim_config=CONFIG_VESGM["optim_config"], denoiser_config=cfg,
        validation_sigmas=[0.1, 1.0, 5.0, 20.0, 80.0],
        batch_size=CONFIG_VESGM["trainer_config"]["batch_size"],
    ).to(device=device, dtype=torch.float32)
    model.eval()
    log(f"  VE-SGM denoiser loaded (sigma_data={meta['sigma_data']:.4f})")
    return model.denoiser.eval().to(device)


def load_tedm(device):
    ckpt = os.path.join(MODEL_DIR, f"{TEDM_RUN}_model.ckpt")
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"
    meta = torch.load(os.path.join(MODEL_DIR, f"{TEDM_RUN}_meta.pt"),
                      map_location="cpu")
    assert meta.get("nu", NU) == NU and meta.get("iid", IID) == IID
    assert meta.get("log_time", LOG_TIME) == LOG_TIME
    cfg = dict(CONFIG_TEDM["denoiser_config"])
    cfg["sigma_data"] = meta["sigma_data"]
    model = AbstractDiffusionStudentT.load_from_checkpoint(
        ckpt, diffusion_config=CONFIG_TEDM["diffusion_config"],
        optim_config=CONFIG_TEDM["optim_config"], denoiser_config=cfg,
        batch_size=CONFIG_TEDM["trainer_config"]["batch_size"],
    ).to(device=device, dtype=torch.float32)
    model.eval()
    log(f"  t-EDM denoiser loaded (sigma_data={meta['sigma_data']:.4f}, nu={NU})")
    return model.denoiser.eval().to(device)


def make_inits():
    """Built once and shared by DDPM and Heun, so the sampler is the only
    variable between the two."""
    inits = {}
    g = torch.Generator().manual_seed(INIT_SEED)

    inits["gaussian"] = SIGMA * torch.randn(N_GEN, DIM, generator=g,
                                            dtype=torch.float32)

    # p_t = clean reference sample (seed 39, independent of the evaluation
    # set seed 35) + sigma * N(0, I)  ->  exactly p_T at sigma = 2.3
    if os.path.isfile(TEST2_FILE):
        clean = torch.tensor(np.asarray(np.load(TEST2_FILE, mmap_mode="r")[:N_GEN]),
                             dtype=torch.float32)
        inits["p_t"] = clean + SIGMA * torch.randn(N_GEN, DIM, generator=g,
                                                   dtype=torch.float32)
    else:
        log(f"  MISSING (p_t skipped): {TEST2_FILE}")

    fp = flow_ptheta_path()
    if os.path.isfile(fp):
        arr = np.asarray(np.load(fp, mmap_mode="r")[:N_GEN])
        assert arr.shape == (N_GEN, DIM), arr.shape
        inits["p_theta"] = torch.tensor(arr, dtype=torch.float32)
    else:
        log(f"  MISSING (p_theta skipped): {fp}")
        log(f"  -> retrain the flow at sigma={SIGMA} first: "
            f"gen_flow_dim_10_ht.py with sigma=[{SIGMA}]")

    for k, v in inits.items():
        log(f"  init {k:<9} std {v.std().item():.4f}  "
            f"|x|max {v.abs().max().item():.4g}")
    return inits


@torch.no_grad()
def ddpm_generate(denoiser, x_init, sig_asc, device, seed):
    """x_{t-1} = (1-r)*x0_hat + r*x_t + N(0, std^2),  r = (s_{t-1}/s_t)^2"""
    n = x_init.shape[0]
    out = torch.empty((n, DIM), dtype=torch.float32)
    n_steps = len(sig_asc) - 1
    gen = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, n, BATCH_GEN):
        end = min(start + BATCH_GEN, n)
        x = x_init[start:end].to(device)
        t0 = time.time()
        for i in range(n_steps):
            s_t, s_tm1 = float(sig_asc[-(i + 1)]), float(sig_asc[-(i + 2)])
            sig_vec = torch.full((x.shape[0],), s_t, device=device,
                                 dtype=torch.float32)
            pred_x0 = denoiser(x, sig_vec)
            r = (s_tm1 ** 2) / (s_t ** 2)
            mean = (1.0 - r) * pred_x0 + r * x
            std = (s_tm1 / s_t) * math.sqrt(max(s_t ** 2 - s_tm1 ** 2, 0.0))
            x = mean + std * torch.randn(x.shape, device=device, generator=gen,
                                         dtype=torch.float32)
        out[start:end] = x.cpu()
        print(f"    [{end}/{n}]  {fmt(time.time() - t0)}", flush=True)
    return out.numpy()


@torch.no_grad()
def heun_generate(denoiser, x_init, sig_desc, device):
    n = x_init.shape[0]
    out = np.empty((n, DIM), dtype=np.float32)
    for start in range(0, n, BATCH_GEN):
        end = min(start + BATCH_GEN, n)
        t0 = time.time()
        x = x_init[start:end].to(device)
        out[start:end] = heun_sampling(x, denoiser, sig_desc).float().cpu().numpy()
        print(f"    [{end}/{n}]  {fmt(time.time() - t0)}", flush=True)
    return out


def run_one(dst, label, sampler, x_init, denoiser, sig_asc, sig_desc, device):
    if os.path.exists(dst):
        log(f"  already exists, skipping: {os.path.basename(dst)}")
        return
    log(f"\n--- {label} ---")
    t0 = time.time()
    s = (ddpm_generate(denoiser, x_init, sig_asc, device, GEN_SEED)
         if sampler == "nn"
         else heun_generate(denoiser, x_init, sig_desc, device))
    ok = np.isfinite(s).all(axis=1)
    np.save(dst, s)
    log(f"  saved {s.shape} in {fmt(time.time() - t0)}  "
        f"non-finite rows: {(~ok).sum()}")
    log(f"    std {s[ok].std():.4f}  |x|max {np.abs(s[ok]).max():.4g}")
    log(f"  -> {dst}")


def generate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_asc = karras_nodes(SIGMA, SIGMA_MIN, RHO, N_STEPS + 1)
    sig_desc = torch.tensor(sig_asc[::-1].copy(), dtype=torch.float32,
                            device=device)

    log("=" * 80)
    log(f"PART A - GENERATION   dim={DIM} ({N_MIXTURE} modes)  "
        f"tail_index={TAIL_INDEX}  train_size={TRAIN_SIZE}")
    log(f"  sigma_T={SIGMA}  sigma_min={SIGMA_MIN}  rho={RHO}  "
        f"{N_STEPS} steps ({len(sig_asc)} nodes)")
    log(f"  schedule: {sig_asc[-1]:.6f} -> {sig_asc[0]:.6f}")
    log(f"  NOTE: 'EDM classic' == 'Heun + gaussian' at this sigma")
    log("=" * 80)

    log("\nBuilding initialisations...")
    inits = make_inits()

    log("\nLoading denoisers...")
    den = load_vesgm(device)
    for name, x0 in inits.items():
        run_one(out_path(name, "nn"), f"DDPM + init={name}", "nn",
                x0, den, sig_asc, sig_desc, device)
        run_one(out_path(name, "heun"), f"HEUN + init={name}", "heun",
                x0, den, sig_asc, sig_desc, device)
    del den, inits

    den_t = load_tedm(device)
    torch.manual_seed(INIT_SEED)
    torch.cuda.manual_seed_all(INIT_SEED)
    x_t = sample_t_prior(N_GEN, DIM, NU, SIGMA, device, iid=IID).cpu()
    log(f"\n  init tprior   std {x_t.std().item():.4f}  "
        f"|x|max {x_t.abs().max().item():.4g}")
    run_one(TEDM_OUT, f"t-EDM (nu={NU}) + Student-t prior", "heun",
            x_t, den_t, sig_asc, sig_desc, device)
    del den_t, x_t

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===========================================================================
# PART B — EVALUATION
# ===========================================================================
NAMES = [
    ("p_t_nn",       "nn",   "p_t",          out_path("p_t", "nn")),
    ("p_theta_nn",   "nn",   "p_theta",      out_path("p_theta", "nn")),
    ("p_inf_nn",     "nn",   "gaussian",     out_path("gaussian", "nn")),
    ("p_t_heun",     "heun", "p_t",          out_path("p_t", "heun")),
    ("p_theta_heun", "heun", "p_theta",      out_path("p_theta", "heun")),
    ("p_inf_heun",   "heun", "gaussian",     out_path("gaussian", "heun")),
    ("tedm",         "tedm", "student_t",    TEDM_OUT),
    ("dlpm",         "dlpm", "alpha_stable", DLPM_PATH),
]

_MSW_CACHE = {}
_HITS = _MISSES = 0


def evaluate():
    import jax
    import jax.numpy as jnp
    import jax.random as jrn
    from jax import nn as jnn
    import equinox as eqx
    import optax
    from flowjax.flows import coupling_flow
    from flowjax import distributions as flow_dist
    from flowjax.bijections import TriangularAffine
    from hf_toy.bulk_metrics import max_sliced_wasserstein

    global _HITS, _MISSES

    def msw(X, Y, seed=MSW_SEED):
        global _HITS, _MISSES
        key = (id(X), id(Y), seed)
        hit = _MSW_CACHE.get(key)
        if hit is not None:
            _HITS += 1
            return hit
        _MISSES += 1
        n = min(len(X), len(Y))
        v, d = max_sliced_wasserstein(
            jnp.array(X[:n]), jnp.array(Y[:n]), key=jrn.key(seed),
            tol=1e-7, lr=1e-3, max_iter=10_000, disable_pbar=True)
        res = (float(v), np.asarray(d))
        _MSW_CACHE[key] = res
        return res

    def scan_finite(path, chunk=1_000_000):
        a = np.load(path, mmap_mode="r")
        nan = pos = neg = bad = 0
        mx = 0.0
        for i in range(0, len(a), chunk):
            c = np.asarray(a[i:i + chunk], dtype=np.float32)
            nan += int(np.isnan(c).sum())
            pos += int(np.isposinf(c).sum())
            neg += int(np.isneginf(c).sum())
            m = np.isfinite(c).all(axis=1)
            bad += int((~m).sum())
            if m.any():
                mx = max(mx, float(np.abs(c[m]).max()))
        return len(a), nan, pos, neg, bad, mx

    def load_array(path, label):
        if not os.path.isfile(path):
            log(f"  MISSING: {path}")
            return None
        a = np.asarray(np.load(path), dtype=np.float32)
        fin = np.isfinite(a).all(axis=1)
        if fin.sum() < len(a):
            log(f"  {label}: dropped {len(a) - fin.sum()} non-finite rows")
            a = a[fin]
        log(f"  loaded {label}: {a.shape}")
        log(f"    std per coord: {np.round(a.std(axis=0), 4)}")
        log(f"    max|x| {np.abs(a).max():.4g}")
        return a

    class HeavyTailFlow(eqx.Module):
        flow: eqx.Module
        dim: int
        sigma: float

        def __init__(self, dim, sigma, tail_index, flow_layers, nn_width,
                     nn_depth, key):
            self.dim, self.sigma = dim, sigma
            base = flow_dist.Transformed(
                base_dist=flow_dist.StudentT(
                    df=tail_index, loc=jnp.zeros((dim,)),
                    scale=sigma if sigma != 0.0 else 1.0),
                bijection=TriangularAffine(loc=jnp.zeros((dim,)),
                                           arr=jnp.eye(dim)))
            self.flow = coupling_flow(
                key=jrn.key(key), base_dist=base, flow_layers=flow_layers,
                nn_activation=jnn.relu6, nn_width=nn_width, nn_depth=nn_depth)

        def sample(self, rng, n):
            return self.flow.sample(rng, (n,))

        def train_dynamic(self, rng, data, n_epochs, lr, bs_gen):
            rng_train = jrn.split(rng, 2)[1]
            n = len(data)
            bs = min(bs_gen, n)
            nb = n // bs
            sched = optax.cosine_decay_schedule(init_value=lr,
                                                decay_steps=n_epochs * nb)
            opt = optax.adam(sched)
            st = opt.init(eqx.filter(self.flow, eqx.is_inexact_array))
            flow, sigma = self.flow, self.sigma

            @eqx.filter_jit
            def run_epoch(flow, st, data, rng_ep):
                rp, rn = jrn.split(rng_ep, 2)
                idx = jrn.permutation(rp, jnp.arange(n))[:nb * bs]
                batches = data[idx.reshape(nb, bs)]
                keys = jrn.split(rn, nb)
                dyn, static = eqx.partition(flow, eqx.is_inexact_array)

                def step(carry, inp):
                    dyn, st = carry
                    batch, k = inp
                    f_ = eqx.combine(dyn, static)

                    def loss_fn(f):
                        return -jnp.mean(f.log_prob(
                            batch + sigma * jrn.normal(k, batch.shape)))

                    loss, gr = eqx.filter_value_and_grad(loss_fn)(f_)
                    upd, st = opt.update(gr, st)
                    f_ = eqx.apply_updates(f_, upd)
                    dyn, _ = eqx.partition(f_, eqx.is_inexact_array)
                    return (dyn, st), loss

                (dyn, st), losses = jax.lax.scan(step, (dyn, st),
                                                 (batches, keys))
                return eqx.combine(dyn, static), st, losses

            for ep in range(n_epochs):
                flow, st, l = run_epoch(flow, st, data,
                                        jrn.fold_in(rng_train, ep))
                if (ep + 1) % 500 == 0 or ep == 0:
                    log(f"    Epoch {ep+1:5d}/{n_epochs} | "
                        f"Loss: {float(jnp.mean(l)):.6f}")
            return eqx.tree_at(lambda m: m.flow, self, flow)

    # ── PART 0 — finiteness check ──────────────────────────────────────────
    log("\n" + "=" * 96)
    log("PART B - EVALUATION")
    log("=" * 96)
    hdr = (f"{'dataset':<16}{'rows':>11}{'NaN':>9}{'+Inf':>8}{'-Inf':>8}"
           f"{'bad':>9}{'%bad':>8}{'max|x|':>14}")
    log(hdr); log("-" * len(hdr))
    n_missing = 0
    for lab, _, _, path in NAMES + [("test_set", "", "", TEST_FILE),
                                    ("train_set", "", "", TRAIN_FILE)]:
        if not os.path.isfile(path):
            log(f"{lab:<16}{'MISSING':>11}    {path}")
            n_missing += 1
            continue
        n, nan, pos, neg, bad, mx = scan_finite(path)
        log(f"{lab:<16}{n:>11}{nan:>9}{pos:>8}{neg:>8}{bad:>9}"
            f"{100.0*bad/n:>8.4f}{mx:>14.4g}")
    log("-" * len(hdr))
    log(f"  missing files: {n_missing}")

    # ── reference chunks ───────────────────────────────────────────────────
    log(f"\nLoading test set: {TEST_FILE}")
    pool = np.load(TEST_FILE, mmap_mode="r")
    log(f"  pool shape: {pool.shape}")
    ref = [np.asarray(pool[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE],
                      dtype=np.float32) for i in range(N_CHUNKS)]
    ref_slog = [slog(c) for c in ref]
    log(f"  built {N_CHUNKS} contiguous chunks of {CHUNK_SIZE}")
    log(f"  ref std per coord: {np.round(ref[0].std(axis=0), 4)}")
    log(f"  ref max|x|: {np.abs(ref[0]).max():.4g}")

    # ── load generated samples ─────────────────────────────────────────────
    log("\nLoading generated samples...")
    samples, samples_slog, meta = {}, {}, {}
    for lab, sampler, init, path in NAMES:
        a = load_array(path, lab)
        samples[lab] = a
        samples_slog[lab] = slog(a) if a is not None else None
        meta[lab] = (sampler, init)

    # ── flow sigma = 0 baseline ────────────────────────────────────────────
    log("\n--- flow (sigma=0), trained here as a baseline ---")
    if os.path.isfile(TRAIN_FILE):
        log(f"  train data: {TRAIN_FILE}")
        log(f"  using the first {TRAIN_SIZE} rows, {FLOW0_N_EPOCHS} epochs")
        td = jnp.asarray(np.load(TRAIN_FILE, mmap_mode="r")[:TRAIN_SIZE])
        t0 = time.time()
        fm = HeavyTailFlow(dim=DIM, sigma=0.0, tail_index=TAIL_INDEX,
                           flow_layers=FLOW_LAYERS, nn_width=FLOW_NN_WIDTH,
                           nn_depth=FLOW_NN_DEPTH, key=FLOW_KEY)
        fm = fm.train_dynamic(rng=jrn.key(FLOW_TRAIN_SEED), data=td,
                              n_epochs=FLOW0_N_EPOCHS, lr=FLOW0_LR,
                              bs_gen=FLOW0_BATCH)
        log(f"  trained in {fmt(time.time() - t0)}")
        fs = np.asarray(fm.sample(jrn.key(FLOW_SAMPLE_SEED), N_GEN),
                        dtype=np.float32)
        fin = np.isfinite(fs).all(axis=1)
        if fin.sum() < len(fs):
            log(f"  dropped {len(fs) - fin.sum()} non-finite rows")
        fs = fs[fin]
        log(f"  flow samples: {fs.shape}")
        log(f"    std per coord: {np.round(fs.std(axis=0), 4)}")
        log(f"    max|x| {np.abs(fs).max():.4g}")
        samples["flow_sigma0"] = fs
        samples_slog["flow_sigma0"] = slog(fs)
        meta["flow_sigma0"] = ("flow_sigma0", "-")
    else:
        log(f"  MISSING train file, flow baseline skipped: {TRAIN_FILE}")

    order = [l for l in samples if samples[l] is not None]

    # ── PART 1 — worst direction + tail quantiles ──────────────────────────
    log("\n" + "=" * 70)
    log("PART 1 - worst direction from p_T, tail quantiles for ALL datasets")
    log("=" * 70)

    tail_out("=" * 78)
    tail_out(f"Fixed-sigma comparison, dim {DIM} ({N_MIXTURE} modes), "
             f"tail_index={TAIL_INDEX}")
    tail_out(f"sigma_T={SIGMA}  sigma_min={SIGMA_MIN}  rho={RHO}  "
             f"{N_STEPS} steps  train_size={TRAIN_SIZE}")
    tail_out(f"Worst direction: max MSW between p_T (1M) and {N_CHUNKS} "
             f"reference chunks of {CHUNK_SIZE}, natural space")
    tail_out("Every dataset is projected on EACH worst direction (nn, heun)")
    tail_out("DLPM is in its native configuration (full chain, K=25)")
    tail_out("=" * 78)

    saved_dirs, quant_rows = {}, []

    for sampler in ("nn", "heun"):
        pt = samples.get(f"p_t_{sampler}")
        if pt is None:
            log(f"\n[{sampler}] p_t missing -> no worst direction")
            tail_out(f"\n[{sampler}] p_T MISSING -> no worst direction, "
                     f"no quantiles")
            continue

        log(f"\n[{sampler}] searching worst direction on p_t ...")
        best_v, best_d = -np.inf, None
        for i, R in enumerate(ref):
            v, d = msw(pt, R)
            log(f"    {sampler} dir chunk {i}: MSW={v:.6f}")
            if v > best_v:
                best_v, best_d = v, d
        log(f"    {sampler} worst direction: MSW={best_v:.6f}")
        log(f"      dir = {np.round(best_d, 6).tolist()}")
        saved_dirs[f"dir_{sampler}"] = best_d
        saved_dirs[f"msw_{sampler}"] = np.array(best_v)

        rq = np.array([np.quantile(c @ best_d, TAIL_QUANTILES) for c in ref])
        rm, rs = rq.mean(axis=0), rq.std(axis=0)

        curves = {}
        for lab in order:
            g = samples[lab]
            curves[lab] = (np.quantile(g @ best_d, TAIL_QUANTILES), len(g))
            for i, q in enumerate(TAIL_QUANTILES):
                val, m_ = float(curves[lab][0][i]), float(rm[i])
                quant_rows.append({
                    "direction": sampler, "dataset": lab, "quantile": q,
                    "value": val, "ref_mean": m_, "ref_std": float(rs[i]),
                    "rel_err": (val - m_) / m_ if m_ != 0 else float("nan"),
                    "n_gen": curves[lab][1]})
            log(f"  {lab}: quantiles done ({curves[lab][1]} samples)")
        for i, q in enumerate(TAIL_QUANTILES):
            quant_rows.append({
                "direction": sampler, "dataset": "reference", "quantile": q,
                "value": float(rm[i]), "ref_mean": float(rm[i]),
                "ref_std": float(rs[i]), "rel_err": 0.0, "n_gen": CHUNK_SIZE})

        tail_out("")
        tail_out("-" * 96)
        tail_out(f"[{sampler}]  worst direction found on p_T   "
                 f"max-SW = {best_v:.6f}")
        tail_out(f"  direction = {np.round(best_d, 6).tolist()}")
        tail_out("-" * 96)
        head = f"{'dataset':<18}" + "".join(f"{'q='+format(q,'.4f'):>13}"
                                            for q in TAIL_QUANTILES)
        tail_out(head); tail_out("-" * len(head))
        tail_out(f"{'reference mean':<18}"
                 + "".join(f"{rm[i]:>13.4f}" for i in range(len(TAIL_QUANTILES))))
        tail_out(f"{'reference std':<18}"
                 + "".join(f"{rs[i]:>13.4f}" for i in range(len(TAIL_QUANTILES))))
        tail_out("-" * len(head))
        for lab in order:
            tail_out(f"{lab:<18}"
                     + "".join(f"{curves[lab][0][i]:>13.4f}"
                               for i in range(len(TAIL_QUANTILES))))
        tail_out("-" * len(head))
        tail_out("relative error (value - ref_mean) / ref_mean")
        for lab in order:
            tail_out(f"{lab:<18}"
                     + "".join(f"{(curves[lab][0][i]-rm[i])/rm[i]:>13.4f}"
                               for i in range(len(TAIL_QUANTILES))))
        tail_out("-" * len(head))
        for lab in order:
            tail_out(f"  {lab}: n = {curves[lab][1]}")

    # ── PART 2 — MSW ───────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("PART 2 - MSW, natural and signed-log space")
    log("=" * 70)

    rows = []
    for lab in list(samples.keys()):
        sampler, init = meta[lab]
        g = samples[lab]
        if g is None:
            log(f"\n--- {lab}: MISSING, skipped ---")
            rows.append({"method": sampler, "init": init,
                         "mean_msw_nat": "MISSING", "std_msw_nat": "MISSING",
                         "mean_msw_slog": "MISSING", "std_msw_slog": "MISSING",
                         "n_gen": ""})
            continue
        log(f"\n--- {lab} ---")
        vn = [msw(g, R)[0] for R in ref]
        vs = [msw(samples_slog[lab], R)[0] for R in ref_slog]
        rows.append({"method": sampler, "init": init,
                     "mean_msw_nat": float(np.mean(vn)),
                     "std_msw_nat": float(np.std(vn)),
                     "mean_msw_slog": float(np.mean(vs)),
                     "std_msw_slog": float(np.std(vs)), "n_gen": len(g)})
        log(f"  => natural  mean={np.mean(vn):.6f}  std={np.std(vn):.6f}")
        log(f"  => slog     mean={np.mean(vs):.6f}  std={np.std(vs):.6f}")

    log("\n--- reference vs reference ---")
    rn, rs_ = [], []
    for i in range(N_CHUNKS):
        j = (i + 1) % N_CHUNKS
        a, b = msw(ref[i], ref[j])[0], msw(ref_slog[i], ref_slog[j])[0]
        rn.append(a); rs_.append(b)
        log(f"    chunk {i} vs {j}:  natural={a:.6f}   slog={b:.6f}")
    rows.append({"method": "reference", "init": "-",
                 "mean_msw_nat": float(np.mean(rn)),
                 "std_msw_nat": float(np.std(rn)),
                 "mean_msw_slog": float(np.mean(rs_)),
                 "std_msw_slog": float(np.std(rs_)), "n_gen": CHUNK_SIZE})
    floor_n, floor_s = float(np.mean(rn)), float(np.mean(rs_))
    log(f"  => natural  mean={floor_n:.6f}  std={np.std(rn):.6f}")
    log(f"  => slog     mean={floor_s:.6f}  std={np.std(rs_):.6f}")
    log(f"\n  MSW cache: {_HITS} hits, {_MISSES} misses "
        f"({_HITS} redundant computations avoided)")

    # ── summary ────────────────────────────────────────────────────────────
    log("\n" + "=" * 92)
    log("SUMMARY - max-sliced Wasserstein")
    log("=" * 92)
    h = (f"{'method':<14}{'init':<14}{'MSW nat':>14}{'std':>12}{'ratio':>9}"
         f"{'MSW slog':>13}{'std':>12}{'ratio':>9}")
    log(h); log("-" * len(h))
    for r in rows:
        if r["mean_msw_nat"] == "MISSING":
            log(f"{r['method']:<14}{str(r['init']):<14}{'MISSING':>14}")
            continue
        log(f"{r['method']:<14}{str(r['init']):<14}"
            f"{r['mean_msw_nat']:>14.6g}{r['std_msw_nat']:>12.4g}"
            f"{r['mean_msw_nat']/floor_n:>9.2f}"
            f"{r['mean_msw_slog']:>13.6f}{r['std_msw_slog']:>12.6f}"
            f"{r['mean_msw_slog']/floor_s:>9.2f}")

    # ── write out ──────────────────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, "msw_slog.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "init", "mean_msw_nat",
                                          "std_msw_nat", "mean_msw_slog",
                                          "std_msw_slog", "n_gen"])
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(OUT_DIR, "tail_quantiles.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direction", "dataset", "quantile",
                                          "value", "ref_mean", "ref_std",
                                          "rel_err", "n_gen"])
        w.writeheader(); w.writerows(quant_rows)

    if saved_dirs:
        np.savez(os.path.join(OUT_DIR, "worst_directions.npz"), **saved_dirs)
    with open(os.path.join(OUT_DIR, "tail_quantiles.txt"), "w") as f:
        f.write("\n".join(TAIL) + "\n")


# ===========================================================================
if __name__ == "__main__":
    t_start = time.time()
    generate_all()
    evaluate()
    log(f"\nTotal wall time {fmt(time.time() - t_start)}")
    with open(os.path.join(OUT_DIR, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"\n-> {OUT_DIR}/  (msw_slog.csv, tail_quantiles.csv, "
          f"tail_quantiles.txt, worst_directions.npz, run_log.txt)")
