"""
Training-set size study at sigma_T = 2.3: p_T initialisation + ancestral DDPM,
one VE-SGM denoiser per training-set size.

Generation reproduces EXACTLY the p_T + DDPM branch of gen_and_eval_sigma23.py
(same p_T sample, same Karras grid, same GEN_SEED): only the checkpoint
changes. For N = 10000 the output file coincides with the one written by
gen_and_eval_sigma23.py and is reused.

Metrics (same definitions as the HMC / nn / nn_heun evaluation script):
  PART 1  ONE worst direction, searched on the p_T dataset of the denoiser
          trained on DIR_TRAIN_SIZE = 100000 samples (max MSW over the 10
          reference chunks, natural space). Every dataset is projected on
          that single direction and tail quantiles are computed along it.
  PART 2  MSW in natural and signed-log space against 10 reference chunks of
          1M (natural-space values of the size-100000 dataset are served from
          the PART-1 cache), plus the reference-vs-reference floor.

"""

import os
import csv
import math
import time
import argparse

import numpy as np
import torch

from VE_SGM.architecture import AbstractDiffusion


# ===========================================================================
# CONFIG
# ===========================================================================
DIM        = 10
N_MIXTURE  = 20
TAIL_INDEX = 3

TRAIN_SEED = 34
SEED_TEST  = 35          # evaluation reference
SEED_TEST2 = 39          # independent clean sample, used to build p_T
DIST_SEED  = 0
TEST_SIZE  = 10_000_000

TRAIN_SIZES    = [1000, 5000, 10000, 50000, 100000]
DIR_TRAIN_SIZE = 100000  # the ONLY dataset used to search the worst direction

# --- sampling: identical to gen_and_eval_sigma23.py ------------------------
SIGMA     = 2.3
SIGMA_MIN = 0.002
RHO       = 3
N_STEPS   = 25           # -> 26 nodes

N_GEN     = 1_000_000
BATCH_GEN = 100_000
GEN_SEED  = 43
INIT_SEED = 7

# --- metrics: identical to the HMC evaluation script -----------------------
CHUNK_SIZE, N_CHUNKS, MSW_SEED = 1_000_000, 10, 79
TAIL_QUANTILES = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]

DATA_INPUT_DIR = f"data/dim_{DIM}"
GEN_DIR        = f"data/data_gen_diffusion/dim_{DIM}"
MODEL_DIR      = "model_diffusion"
OUT_DIR        = "results_dim_10_size_sigma23"

TEST_FILE  = (f"{DATA_INPUT_DIR}/test_set_dim_{DIM}_size_{TEST_SIZE}"
              f"_tail_{TAIL_INDEX}_seed_{SEED_TEST}_dist_seed_{DIST_SEED}.npy")
TEST2_FILE = (f"{DATA_INPUT_DIR}/test_set_dim_{DIM}_size_{TEST_SIZE}"
              f"_tail_{TAIL_INDEX}_seed_{SEED_TEST2}_dist_seed_{DIST_SEED}.npy")

OUTPUT_CSV  = os.path.join(OUT_DIR, "msw_slog.csv")
OUTPUT_QCSV = os.path.join(OUT_DIR, "tail_quantiles.csv")
OUTPUT_TXT  = os.path.join(OUT_DIR, "tail_quantiles.txt")
OUTPUT_DIR_ = os.path.join(OUT_DIR, "worst_direction.npz")
OUTPUT_LOG  = os.path.join(OUT_DIR, "run_log.txt")

CONFIG_VESGM = {
    "diffusion_config": {"sigma_min": 0.002, "sigma_max": 80.0,
                         "log_mean": -1.0, "log_std": 1.2, "num_workers": 6},
    "optim_config": {"lr": 1e-4},
    "denoiser_config": {"sigma_data": 1, "sigma_min": 0.002, "sigma_max": 80.0,
                        "sigma_disc": 1000, "input_dim": DIM, "embed_dim": 256,
                        "channel_mult": [2, 4, 4, 2]},
    "trainer_config": {"batch_size": 1000},
}

LOG, TAIL = [], []


# ===========================================================================
# UTILS
# ===========================================================================
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


def vesgm_run(n):
    return f"vesgm_dim_{DIM}_size_{n}_tail_{TAIL_INDEX}"


def ckpt_path(n):
    return os.path.join(MODEL_DIR, f"{vesgm_run(n)}_model.ckpt")


def meta_path(n):
    return os.path.join(MODEL_DIR, f"{vesgm_run(n)}_meta.pt")


def gen_path(n):
    """Same convention as out_path('p_t', 'nn') in gen_and_eval_sigma23.py."""
    return os.path.join(
        GEN_DIR, f"samples_pt_nn_sigma_{SIGMA}_size_{n}"
                 f"_{N_GEN}_tail_{TAIL_INDEX}.npy")


def label(n):
    return f"size_{n}"


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


def load_array(path, lab):
    if not os.path.isfile(path):
        log(f"  MISSING: {path}")
        return None
    a = np.asarray(np.load(path), dtype=np.float32)
    fin = np.isfinite(a).all(axis=1)
    if fin.sum() < len(a):
        log(f"  {lab}: dropped {len(a) - fin.sum()} non-finite rows")
        a = a[fin]
    log(f"  loaded {lab}: {a.shape}")
    log(f"    std per coord: {np.round(a.std(axis=0), 4)}")
    log(f"    max|x| {np.abs(a).max():.4g}")
    return a


# ===========================================================================
# PART A - GENERATION (p_T + ancestral DDPM, one denoiser per size)
# ===========================================================================
def karras_nodes(sigma_hi, sigma_lo, rho, n_nodes):
    """Ascending Karras grid; sigma_hi IS a node by construction."""
    inv = 1.0 / rho
    t = np.linspace(0.0, 1.0, n_nodes)
    return np.sort((sigma_hi ** inv + t * (sigma_lo ** inv - sigma_hi ** inv)) ** rho)


def load_vesgm(device, n):
    ckpt = ckpt_path(n)
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"
    meta = torch.load(meta_path(n), map_location="cpu")
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
    log(f"  VE-SGM denoiser size {n} loaded (sigma_data={meta['sigma_data']:.4f})")
    return model.denoiser.eval().to(device)


def make_pt_init():
    """p_T = clean sample (seed 39) + SIGMA * N(0, I), bit-identical to the
    p_t init of gen_and_eval_sigma23.make_inits. There the gaussian init is
    drawn FIRST from the same generator: draw it here too and discard it,
    otherwise the p_T noise would be different."""
    g = torch.Generator().manual_seed(INIT_SEED)
    _ = torch.randn(N_GEN, DIM, generator=g, dtype=torch.float32)
    clean = torch.tensor(np.asarray(np.load(TEST2_FILE, mmap_mode="r")[:N_GEN]),
                         dtype=torch.float32)
    x = clean + SIGMA * torch.randn(N_GEN, DIM, generator=g, dtype=torch.float32)
    log(f"  init p_T  std {x.std().item():.4f}  |x|max {x.abs().max().item():.4g}")
    return x


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


def generate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_asc = karras_nodes(SIGMA, SIGMA_MIN, RHO, N_STEPS + 1)

    log("=" * 80)
    log(f"PART A - GENERATION   dim={DIM} ({N_MIXTURE} modes)  "
        f"tail_index={TAIL_INDEX}")
    log(f"  init = p_T, sampler = ancestral DDPM, sizes = {TRAIN_SIZES}")
    log(f"  sigma_T={SIGMA}  sigma_min={SIGMA_MIN}  rho={RHO}  "
        f"{N_STEPS} steps ({len(sig_asc)} nodes)")
    log(f"  schedule: {sig_asc[-1]:.6f} -> {sig_asc[0]:.6f}")
    log(f"  INIT_SEED={INIT_SEED}  GEN_SEED={GEN_SEED}")
    log("=" * 80)

    todo = []
    for n in TRAIN_SIZES:
        if os.path.exists(gen_path(n)):
            log(f"  already exists, skipping: {os.path.basename(gen_path(n))}")
        elif not os.path.exists(ckpt_path(n)):
            log(f"  MISSING checkpoint, size {n} not generated: {ckpt_path(n)}")
        else:
            todo.append(n)
    if not todo:
        return

    if not os.path.isfile(TEST2_FILE):
        raise SystemExit(f"cannot build p_T, missing {TEST2_FILE}")

    log("\nBuilding p_T initialisation (shared by all sizes)...")
    x_init = make_pt_init()

    for n in todo:
        log(f"\n--- DDPM + p_T, denoiser trained on {n} samples ---")
        den = load_vesgm(device, n)
        t0 = time.time()
        s = ddpm_generate(den, x_init, sig_asc, device, GEN_SEED)
        ok = np.isfinite(s).all(axis=1)
        np.save(gen_path(n), s)
        log(f"  saved {s.shape} in {fmt(time.time() - t0)}  "
            f"non-finite rows: {(~ok).sum()}")
        log(f"    std {s[ok].std():.4f}  |x|max {np.abs(s[ok]).max():.4g}")
        log(f"  -> {gen_path(n)}")
        del den
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del x_init


# ===========================================================================
# PART B - EVALUATION
# ===========================================================================
def evaluate():
    import jax.numpy as jnp
    import jax.random as jrn
    from hf_toy.bulk_metrics import max_sliced_wasserstein

    # (id(X), id(Y), seed) -> (value, direction). Valid because every array
    # passed to msw() stays alive for the whole evaluation.
    cache, stats = {}, {"hits": 0, "misses": 0}

    def msw(X, Y, seed=MSW_SEED):
        key = (id(X), id(Y), seed)
        if key in cache:
            stats["hits"] += 1
            return cache[key]
        stats["misses"] += 1
        n = min(len(X), len(Y))
        v, d = max_sliced_wasserstein(
            jnp.array(X[:n]), jnp.array(Y[:n]), key=jrn.key(seed),
            tol=1e-7, lr=1e-3, max_iter=10_000, disable_pbar=True)
        res = (float(v), np.asarray(d))
        cache[key] = res
        return res

    # ── PART 0 - finiteness check ──────────────────────────────────────────
    log("\n" + "=" * 96)
    log("PART B - EVALUATION")
    log("PART 0 - finiteness check")
    log("=" * 96)
    hdr = (f"{'dataset':<18}{'rows':>11}{'NaN':>10}{'+Inf':>8}{'-Inf':>8}"
           f"{'bad rows':>11}{'%bad':>8}{'max|x|':>14}")
    log(hdr); log("-" * len(hdr))
    n_missing = n_dirty = 0
    for lab, path in ([(label(n), gen_path(n)) for n in TRAIN_SIZES]
                      + [("test_set", TEST_FILE)]):
        if not os.path.isfile(path):
            log(f"{lab:<18}{'MISSING':>11}    {path}")
            n_missing += 1
            continue
        n, nan, pos, neg, bad, mx = scan_finite(path)
        log(f"{lab:<18}{n:>11}{nan:>10}{pos:>8}{neg:>8}"
            f"{bad:>11}{100.0*bad/n:>8.4f}{mx:>14.4g}")
        if bad:
            n_dirty += 1
    log("-" * len(hdr))
    log(f"  missing files: {n_missing}   files with non-finite rows: {n_dirty}")

    # fail BEFORE any expensive computation
    if not os.path.isfile(TEST_FILE):
        raise SystemExit("cannot proceed without the test set")
    if not os.path.isfile(gen_path(DIR_TRAIN_SIZE)):
        raise SystemExit(f"the size-{DIR_TRAIN_SIZE} dataset is required to "
                         f"define the projection direction, but it is missing")

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

    # ── generated samples ──────────────────────────────────────────────────
    log("\nLoading generated samples...")
    samples, samples_slog = {}, {}
    for n in TRAIN_SIZES:
        a = load_array(gen_path(n), label(n))
        samples[n] = a
        samples_slog[n] = slog(a) if a is not None else None
    order = [n for n in TRAIN_SIZES if samples[n] is not None]

    # ── PART 1 - single worst direction + tail quantiles ───────────────────
    log("\n" + "=" * 70)
    log(f"PART 1 - worst direction, searched on the size-{DIR_TRAIN_SIZE} "
        f"p_T dataset")
    log("=" * 70)

    pt = samples[DIR_TRAIN_SIZE]
    best_v, best_d = -np.inf, None
    for i, R in enumerate(ref):
        v, d = msw(pt, R)
        log(f"    size_{DIR_TRAIN_SIZE} dir chunk {i}: MSW={v:.6f}")
        if v > best_v:
            best_v, best_d = v, d
    log(f"  worst direction: max-SW={best_v:.6f}")
    log(f"    dir = {np.round(best_d, 6).tolist()}")
    np.savez(OUTPUT_DIR_, direction=best_d, msw=np.array(best_v),
             train_size=np.array(DIR_TRAIN_SIZE))

    rq = np.array([np.quantile(c @ best_d, TAIL_QUANTILES) for c in ref])
    rm, rs = rq.mean(axis=0), rq.std(axis=0)

    quant_rows, curves = [], {}
    for n in order:
        g = samples[n]
        curves[n] = (np.quantile(g @ best_d, TAIL_QUANTILES), len(g))
        for i, q in enumerate(TAIL_QUANTILES):
            val, m_ = float(curves[n][0][i]), float(rm[i])
            quant_rows.append({
                "train_size": n, "quantile": q, "value": val,
                "ref_mean": m_, "ref_std": float(rs[i]),
                "rel_err": (val - m_) / m_ if m_ != 0 else float("nan"),
                "n_gen": curves[n][1]})
        log(f"  size {n}: quantiles done ({curves[n][1]} samples)")
    for i, q in enumerate(TAIL_QUANTILES):
        quant_rows.append({
            "train_size": "reference", "quantile": q, "value": float(rm[i]),
            "ref_mean": float(rm[i]), "ref_std": float(rs[i]),
            "rel_err": 0.0, "n_gen": CHUNK_SIZE})

    tail_out("=" * 78)
    tail_out(f"Training-set size study, dim {DIM} ({N_MIXTURE} modes), "
             f"tail_index={TAIL_INDEX}")
    tail_out(f"init = p_T, ancestral DDPM, sigma_T={SIGMA}  "
             f"sigma_min={SIGMA_MIN}  rho={RHO}  {N_STEPS} steps")
    tail_out(f"Single worst direction, searched on the size-{DIR_TRAIN_SIZE} "
             f"p_T dataset against {N_CHUNKS} reference chunks of "
             f"{CHUNK_SIZE}, natural space; max-SW = {best_v:.6f}")
    tail_out("All datasets are projected on this same direction.")
    tail_out(f"Reference: mean and std over the {N_CHUNKS} chunks")
    tail_out("=" * 78)
    tail_out("")
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
    for n in order:
        tail_out(f"{'size ' + str(n):<18}"
                 + "".join(f"{curves[n][0][i]:>13.4f}"
                           for i in range(len(TAIL_QUANTILES))))
    tail_out("-" * len(head))
    tail_out("relative error (value - ref_mean) / ref_mean")
    for n in order:
        tail_out(f"{'size ' + str(n):<18}"
                 + "".join(f"{(curves[n][0][i]-rm[i])/rm[i]:>13.4f}"
                           for i in range(len(TAIL_QUANTILES))))
    tail_out("-" * len(head))
    for n in order:
        tail_out(f"  size {n}: n = {curves[n][1]}")

    # write PART 1 outputs right away, so they survive a crash in PART 2
    with open(OUTPUT_QCSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["train_size", "quantile", "value",
                                          "ref_mean", "ref_std", "rel_err",
                                          "n_gen"])
        w.writeheader(); w.writerows(quant_rows)
    with open(OUTPUT_TXT, "w") as f:
        f.write("\n".join(TAIL) + "\n")

    # ── PART 2 - MSW, natural and signed-log space ─────────────────────────
    log("\n" + "=" * 70)
    log("PART 2 - MSW, natural and signed-log space")
    log("=" * 70)

    rows = []
    for n in TRAIN_SIZES:
        g = samples[n]
        if g is None:
            log(f"\n--- size {n}: MISSING, skipped ---")
            rows.append({"train_size": n, "mean_msw_nat": "MISSING",
                         "std_msw_nat": "MISSING", "mean_msw_slog": "MISSING",
                         "std_msw_slog": "MISSING", "n_gen": ""})
            continue
        log(f"\n--- size {n} ---")
        vn, vs = [], []
        for i, R in enumerate(ref):
            v, _ = msw(g, R)          # size 100000: served from PART-1 cache
            vn.append(v)
            log(f"    size_{n} [natural] chunk {i}: MSW={v:.6f}")
        for i, R in enumerate(ref_slog):
            v, _ = msw(samples_slog[n], R)
            vs.append(v)
            log(f"    size_{n} [slog] chunk {i}: MSW={v:.6f}")
        rows.append({"train_size": n,
                     "mean_msw_nat": float(np.mean(vn)),
                     "std_msw_nat": float(np.std(vn)),
                     "mean_msw_slog": float(np.mean(vs)),
                     "std_msw_slog": float(np.std(vs)),
                     "n_gen": len(g)})
        log(f"  => natural  mean={np.mean(vn):.6f}  std={np.std(vn):.6f}")
        log(f"  => slog     mean={np.mean(vs):.6f}  std={np.std(vs):.6f}")

    log("\n--- reference vs reference ---")
    rn, rs_ = [], []
    for i in range(N_CHUNKS):
        j = (i + 1) % N_CHUNKS
        a, _ = msw(ref[i], ref[j])
        b, _ = msw(ref_slog[i], ref_slog[j])
        rn.append(a); rs_.append(b)
        log(f"    chunk {i} vs {j}:  natural={a:.6f}   slog={b:.6f}")
    rows.append({"train_size": "reference",
                 "mean_msw_nat": float(np.mean(rn)),
                 "std_msw_nat": float(np.std(rn)),
                 "mean_msw_slog": float(np.mean(rs_)),
                 "std_msw_slog": float(np.std(rs_)),
                 "n_gen": CHUNK_SIZE})
    floor_n, floor_s = float(np.mean(rn)), float(np.mean(rs_))
    log(f"  => natural  mean={floor_n:.6f}  std={np.std(rn):.6f}")
    log(f"  => slog     mean={floor_s:.6f}  std={np.std(rs_):.6f}")
    log(f"\n  MSW cache: {stats['hits']} hits, {stats['misses']} misses "
        f"({stats['hits']} redundant computations avoided)")

    # ── summary ────────────────────────────────────────────────────────────
    log("\n" + "=" * 88)
    log(f"SUMMARY - max-sliced Wasserstein by training-set size "
        f"(p_T + DDPM, sigma_T={SIGMA})")
    log("=" * 88)
    h = (f"{'train size':<14}{'MSW nat':>14}{'std':>12}{'ratio':>9}"
         f"{'MSW slog':>13}{'std':>12}{'ratio':>9}")
    log(h); log("-" * len(h))
    for r in rows:
        if r["mean_msw_nat"] == "MISSING":
            log(f"{str(r['train_size']):<14}{'MISSING':>14}")
            continue
        log(f"{str(r['train_size']):<14}"
            f"{r['mean_msw_nat']:>14.6g}{r['std_msw_nat']:>12.4g}"
            f"{r['mean_msw_nat']/floor_n:>9.2f}"
            f"{r['mean_msw_slog']:>13.6f}{r['std_msw_slog']:>12.6f}"
            f"{r['mean_msw_slog']/floor_s:>9.2f}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["train_size", "mean_msw_nat",
                                          "std_msw_nat", "mean_msw_slog",
                                          "std_msw_slog", "n_gen"])
        w.writeheader(); w.writerows(rows)


# ===========================================================================
# MAIN
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Training-set size study, p_T + DDPM at sigma_T = 2.3")
    p.add_argument("--skip-gen", action="store_true",
                   help="skip generation, evaluate existing files only")
    p.add_argument("--skip-eval", action="store_true",
                   help="generate only, no evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(GEN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    try:
        if not args.skip_gen:
            generate_all()
        if not args.skip_eval:
            evaluate()
    finally:
        log(f"\nTotal wall time {fmt(time.time() - t_start)}")
        with open(OUTPUT_LOG, "w") as f:
            f.write("\n".join(LOG) + "\n")

    print(f"\nCSV  -> {OUTPUT_CSV}")
    print(f"QCSV -> {OUTPUT_QCSV}")
    print(f"TXT  -> {OUTPUT_TXT}")
    print(f"DIR  -> {OUTPUT_DIR_}")
    print(f"LOG  -> {OUTPUT_LOG}")