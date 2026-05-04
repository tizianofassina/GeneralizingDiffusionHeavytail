
import time
import jax
import jax.numpy as jnp
import jax.random as jrn
from jax import jit, vmap

from hf_toy.create_data import build_heavy_tail_ref_dist
from hf_toy.mc_denoisers import nuts_denoiser, barker_denoiser, hmc_denoiser

DIM = 2
MAX_RANK = 10
N_REPEATS = 200

pi_ref = build_heavy_tail_ref_dist(
    rng=jrn.key(0),
    student_df=3.0,
    dim=DIM,
    scale=0.1,
    global_scale=4.,
    n_mixture=4,
)

direction = jnp.ones((DIM,))
direction = direction / jnp.linalg.vector_norm(direction)

SIGMAS = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
X_T_NORMS = [0., 0.1, 1., 3., 5., 7., 9., 11., 13., 15., 17.]

ORACLE = (10_000, 1000)  # always NUTS

CONFIGS = [
    (10_000, 1000),
    (10_000, 500),
    (1000, 200),
    (1000, 100),
    (500, 100),
    (250, 100),
    (200, 100),
]

DENOISERS = {
    "nuts": nuts_denoiser,
    "barker": barker_denoiser,
    "hmc": hmc_denoiser,
}

OUT_FILE = "results_mcmc_quality.txt"

N_NORMS = len(X_T_NORMS)
TOTAL_PER_SIGMA = N_NORMS * N_REPEATS

all_x_ts = jnp.concatenate([
    jnp.tile(direction * norm, (N_REPEATS, 1)) for norm in X_T_NORMS
])


def run_batch(denoiser_fn, n_mcmc, n_warmup, sigma, x_ts, rngs):
    def single(x_t, rng):
        return denoiser_fn(
            x_t=x_t,
            sigma=sigma,
            rng=rng,
            pi_ref=pi_ref,
            adaptation="low_rank",
            num_warmup=n_warmup,
            num_samples=n_mcmc,
            max_rank=MAX_RANK,
            debug=False,
        )
    return jit(vmap(single))(x_ts, rngs)


def log(msg, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


with open(OUT_FILE, "w") as f:
    log("=" * 80, f)
    log("DENOISER QUALITY COMPARISON: NUTS vs BARKER vs HMC", f)
    log("=" * 80, f)
    log(f"N_REPEATS={N_REPEATS}, DIM={DIM}", f)
    log(f"Oracle: NUTS with n_mcmc={ORACLE[0]}, warmup={ORACLE[1]}", f)
    log(f"Sigmas: {SIGMAS}", f)
    log(f"X_T_NORMS: {X_T_NORMS}", f)
    log(f"Configs tested: {CONFIGS}", f)
    log(f"Denoisers: {list(DENOISERS.keys())}", f)
    log("", f)

    for sigma in SIGMAS:
        log(f"{'#' * 80}", f)
        log(f"# SIGMA = {sigma}", f)
        log(f"{'#' * 80}", f)
        log("", f)

        seed_base = int(sigma * 10000)
        rngs_oracle = jrn.split(jrn.key(seed_base), TOTAL_PER_SIGMA)
        rngs_oracle2 = jrn.split(jrn.key(seed_base + 99), TOTAL_PER_SIGMA)

        # Oracle: always NUTS
        t0 = time.perf_counter()
        oracle_all = run_batch(nuts_denoiser, ORACLE[0], ORACLE[1], sigma, all_x_ts, rngs_oracle)
        oracle_all.block_until_ready()
        t_oracle = time.perf_counter() - t0
        log(f"Oracle (NUTS {ORACLE[0]}/{ORACLE[1]}) computed in {t_oracle:.1f}s", f)

        oracle2_all = run_batch(nuts_denoiser, ORACLE[0], ORACLE[1], sigma, all_x_ts, rngs_oracle2)
        oracle2_all.block_until_ready()

        oracle_by_norm = oracle_all.reshape(N_NORMS, N_REPEATS, DIM)
        oracle2_by_norm = oracle2_all.reshape(N_NORMS, N_REPEATS, DIM)

        # Run all denoisers × all configs
        all_results = {}
        for denoiser_name, denoiser_fn in DENOISERS.items():
            for n_mcmc, n_warmup in CONFIGS:
                key_tuple = (denoiser_name, n_mcmc, n_warmup)
                rngs_test = jrn.split(jrn.key(seed_base + hash(denoiser_name) % 10000 + n_mcmc + n_warmup), TOTAL_PER_SIGMA)
                t0 = time.perf_counter()
                results = run_batch(denoiser_fn, n_mcmc, n_warmup, sigma, all_x_ts, rngs_test)
                results.block_until_ready()
                elapsed = time.perf_counter() - t0
                all_results[key_tuple] = (results.reshape(N_NORMS, N_REPEATS, DIM), elapsed)
                log(f"  {denoiser_name:>8} ({n_mcmc}/{n_warmup}) done in {elapsed:.1f}s", f)

        log("", f)

        # Print results per norm
        for i, x_t_norm in enumerate(X_T_NORMS):
            oracle_mean = oracle_by_norm[i].mean(axis=0)
            oracle2_mean = oracle2_by_norm[i].mean(axis=0)
            noise_L2 = float(jnp.linalg.norm(oracle_mean - oracle2_mean))

            log("=" * 90, f)
            log(f"sigma={sigma}  ||x_t|| = {x_t_norm:.1f}", f)
            log("=" * 90, f)
            log(f"Oracle mean: [{float(oracle_mean[0]):.4f}, {float(oracle_mean[1]):.4f}]", f)
            log(f"Oracle noise floor: L2 = {noise_L2:.6f}", f)
            log("", f)

            log(f"{'denoiser':>10} {'n_mcmc':>8} {'warmup':>8} {'mean_L2':>10} {'std_L2':>10} "
                f"{'max_L2':>10} {'vs_oracle':>12}", f)
            log("-" * 78, f)

            for denoiser_name in DENOISERS:
                for n_mcmc, n_warmup in CONFIGS:
                    key_tuple = (denoiser_name, n_mcmc, n_warmup)
                    res_i = all_results[key_tuple][0][i]
                    diffs = jnp.linalg.norm(res_i - oracle_mean, axis=-1)
                    config_mean = res_i.mean(axis=0)
                    vs_oracle = float(jnp.linalg.norm(config_mean - oracle_mean))

                    log(f"{denoiser_name:>10} {n_mcmc:>8} {n_warmup:>8} {float(jnp.mean(diffs)):>10.4f} "
                        f"{float(jnp.std(diffs)):>10.4f} {float(jnp.max(diffs)):>10.4f} "
                        f"{vs_oracle:>12.6f}", f)
                log("", f)  

            log("", f)

    log("Done.", f)

print(f"\nAll results saved to {OUT_FILE}")