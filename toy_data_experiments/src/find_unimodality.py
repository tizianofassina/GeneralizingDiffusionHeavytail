
from __future__ import annotations
import math
import time
import numpy as np
import diptest
from scipy.special import erf
from jax import random as jrn, numpy as jnp


_NULL_CACHE: dict[tuple[int, int], tuple[float, float]] = {}


def _uniform_dip_null(n: int, n_repeats: int = 100, seed: int = 0) -> tuple[float, float]:
    key = (n, n_repeats)
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    rng = np.random.default_rng(seed)
    dips = np.empty(n_repeats)
    for i in range(n_repeats):
        dips[i] = diptest.diptest(rng.random(n))[0]
    m, s = float(dips.mean()), float(dips.std())
    _NULL_CACHE[key] = (m, s)
    return m, s


def _z_dip_from_samples_1d(x: np.ndarray, null_mean: float, null_std: float) -> float:
    d, _ = diptest.diptest(x)
    return (d - null_mean) / null_std


def worst_case_onset(
    pi_ref,
    dim: int,
    n_slices: int,
    n_slices_verify: int | None = None,
    n_samples: int = 50_000,
    z_threshold: float = 0.0,
    sigma_min: float = 0.1,
    sigma_max: float = 5.0,
    bisect_iters: int = 12,
    n_null_repeats: int = 500,
    rng_key=None,
    verbose: bool = True,
):
    
    if rng_key is None:
        rng_key = jrn.key(0)
    if n_slices_verify is None:
        n_slices_verify = n_slices

    null_mean, null_std = _uniform_dip_null(n_samples, n_repeats=n_null_repeats)

    k_ref, k_noise, k_dirs, k_ref_v, k_noise_v, k_dirs_v = jrn.split(rng_key, 6)

    if verbose:
        t0 = time.time()

    x_ref = np.asarray(pi_ref.sample(k_ref, (n_samples,)))                       
    eps   = np.asarray(jrn.normal(k_noise, (n_samples, dim)))                   
    dirs  = np.array(jrn.normal(k_dirs, (n_slices, dim)))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)                   

    proj_x   = x_ref @ dirs.T     
    proj_eps = eps   @ dirs.T    

    if verbose:
        print(f"[calib] sampled + projected in {time.time()-t0:.2f}s")

    lo = np.full(n_slices, sigma_min, dtype=np.float64)
    hi = np.full(n_slices, sigma_max, dtype=np.float64)

    def z_at(sigma_vec):
        mix = proj_x + sigma_vec[None, :] * proj_eps    
        out = np.empty(n_slices)
        for j in range(n_slices):
            d, _ = diptest.diptest(mix[:, j])
            out[j] = (d - null_mean) / null_std
        return out

    z_lo = z_at(lo)   
    z_hi = z_at(hi) 

    bad_lo = z_lo < z_threshold       
    bad_hi = z_hi >= z_threshold        
    if verbose and (bad_lo.any() or bad_hi.any()):
        print(f"[calib] bracket warnings: {bad_lo.sum()} too-small-min, "
              f"{bad_hi.sum()} too-small-max")

    log_lo, log_hi = np.log(lo), np.log(hi)
    for it in range(bisect_iters):
        log_mid = 0.5 * (log_lo + log_hi)
        z_mid = z_at(np.exp(log_mid))
        move_up = z_mid >= z_threshold
        log_lo = np.where(move_up, log_mid, log_lo)
        log_hi = np.where(move_up, log_hi, log_mid)

    onset_per_slice = np.exp(log_hi)  
    onset_per_slice = np.where(bad_hi, sigma_max, onset_per_slice)
    worst_sigma = float(onset_per_slice.max())

    if verbose:
        print(f"[calib] worst_sigma = {worst_sigma:.4f} "
              f"(Median Sigma for having onset = {np.median(onset_per_slice):.4f}) "
              f"in {time.time()-t0:.2f}s")
        t1 = time.time()

    x_v   = np.asarray(pi_ref.sample(k_ref_v, (n_samples,)))
    eps_v = np.asarray(jrn.normal(k_noise_v, (n_samples, dim)))
    dirs_v = np.array(jrn.normal(k_dirs_v, (n_slices_verify, dim)))
    dirs_v /= np.linalg.norm(dirs_v, axis=-1, keepdims=True)

    proj_mix_v = (x_v + worst_sigma * eps_v) @ dirs_v.T  
    verify_z_all = np.empty(n_slices_verify)
    for j in range(n_slices_verify):
        d, _ = diptest.diptest(proj_mix_v[:, j])
        verify_z_all[j] = (d - null_mean) / null_std

    verify_z_min = float(verify_z_all.min())   
    verify_z_max = float(verify_z_all.max())  
    verify_ok = bool(verify_z_max < z_threshold)

    if verbose:
        print(f"[verify] fresh z_dip in [{verify_z_min:.3f}, {verify_z_max:.3f}], "
              f"ok={verify_ok} in {time.time()-t1:.2f}s")
    if verbose:
        
        return {
            "worst_sigma": worst_sigma,
            "onset_per_slice": onset_per_slice,
            "dirs" : dirs,
            "verify_z_all": verify_z_all,
            "verify_z_min": verify_z_min,
            "verify_z_max": verify_z_max,
            "verify_ok": verify_ok,
        }
    else:
        return worst_sigma

