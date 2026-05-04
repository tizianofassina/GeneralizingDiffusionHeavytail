import torch
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import pickle
import generation.dnnlib
import torch.nn.functional as F
from PIL import Image
import gc
import os


def update_stats(mu, sigma, n_total, batch_features):
    """
    Incrementally update mean and covariance using Chan–Golub–LeVeque formula
    with float64 precision for numerical stability.

    Parameters
    ----------
    mu : np.ndarray or None
        Current mean vector (shape [D,]).
    sigma : np.ndarray or None
        Current covariance matrix (shape [D, D]).
    n_total : int
        Total number of samples seen so far.
    batch_features : torch.Tensor
        Features from the current batch (shape [n_b, D]).

    Returns
    -------
    new_mu : np.ndarray
        Updated mean.
    new_sigma : np.ndarray
        Updated covariance (unnormalized, i.e., multiplied by n_total - 1).
    new_n : int
        Updated total number of samples.
    """
    # Convert to float64 NumPy array for precision
    batch_features = batch_features.detach().cpu().numpy().astype(np.float64)

    n_b = batch_features.shape[0]
    mu_b = batch_features.mean(axis=0)
    sigma_b = np.cov(batch_features, rowvar=False, bias=False) * (
        n_b - 1
    )  # unnormalized

    if mu is None:
        # First batch initialization
        return mu_b, sigma_b, n_b

    # Merge statistics using Chan–Golub–LeVeque formula
    delta = mu_b - mu
    new_n = n_total + n_b
    new_mu = (n_total * mu + n_b * mu_b) / new_n
    new_sigma = sigma + sigma_b + np.outer(delta, delta) * (n_total * n_b / new_n)

    return new_mu, new_sigma, new_n


def fid_score(mu_gen, sigma_gen, mu_real, sigma_real, eps=1e-6):
    """
    Compute the Fréchet Inception Distance (FID) between two Gaussian distributions.
    Uses double precision throughout for numerical stability.
    """

    # ensure double precision
    mu_gen = np.atleast_1d(mu_gen).astype(np.float64)
    mu_real = np.atleast_1d(mu_real).astype(np.float64)
    sigma_gen = np.atleast_2d(sigma_gen).astype(np.float64)
    sigma_real = np.atleast_2d(sigma_real).astype(np.float64)

    diff = mu_gen - mu_real
    m = diff @ diff

    # Add a small diagonal regularizer for numerical stability
    epsI = np.eye(sigma_gen.shape[0], dtype=np.float64) * eps
    cov_prod = (sigma_gen + epsI) @ (sigma_real + epsI)

    s = sqrtm(cov_prod)

    # Handle small imaginary parts due to numerical error
    if np.iscomplexobj(s):
        s = s.real

    # Compute the trace term in double precision
    sigma = np.trace(sigma_gen + sigma_real - 2 * s)

    # Convert only the final scalar to float (or leave as float64)
    fid = float(m + sigma)
    return fid, float(m), float(sigma)


@torch.no_grad()
def compute_swd(
    data_1: torch.Tensor,
    data_real: torch.Tensor,
    n_tests: int = 5,
    n_proj: int = 32,
    p: int = 2,
    n_batch_proj: int | None = None,
    device: str = "cuda",
    n_data=None,
    random: bool = False,
) -> tuple[float, float]:
    """
    Compute the Sliced Wasserstein Distance (SWD) between two datasets.

    Args:
        data_1: Tensor of shape [N, D] or higher, first dataset.
        data_real: Tensor of shape [N, D] or higher, reference dataset.
        n_tests: Number of batches to split the data into.
        n_proj: Total number of random projections.
        p: Power for Wasserstein computation.
        n_batch_proj: Batch size for projections; defaults to n_proj.
        device: Device to run computations on.
        n_data: If specified, uses this number of samples per batch with random sampling.

    Returns:
        mean_wass: Mean SWD over batches.
        std_wass: Standard deviation of SWD over batches.
    """
    # Move data to device and ensure float32
    data_1 = data_1.to(device=device, dtype=torch.float32, non_blocking=True)
    data_real = data_real.to(device=device, dtype=torch.float32, non_blocking=True)

    # Flatten if data has more than 2 dimensions
    if data_1.dim() > 2:
        data_1 = data_1.flatten(1)
        data_real = data_real.flatten(1)

    N = min(len(data_1), len(data_real))
    D = data_1.shape[1]
    if n_data is not None:
        random = True
        batch_size = n_data
    else:
        batch_size = N // n_tests
    if batch_size == 0:
        raise ValueError(f"Datasets are too small for n_tests={n_tests}.")

    wass_values = torch.empty(n_tests, device=device, dtype=torch.float32)
    n_batch_proj = n_batch_proj or n_proj

    # --- Compute SWD for each batch ---

    for i in range(n_tests):
        if random:
            indices_1 = torch.randperm(len(data_1), device=device)[:batch_size]
            indices_real = torch.randperm(len(data_real), device=device)[:batch_size]
            x = data_1[indices_1]
            y = data_real[indices_real]
        else:
            x = data_1[i * batch_size : (i + 1) * batch_size]
            y = data_real[i * batch_size : (i + 1) * batch_size]

        total_sum = 0.0

        # Process projections in batches
        for j in range(0, n_proj, n_batch_proj):
            k = min(n_batch_proj, n_proj - j)

            proj = torch.randn(D, k, device=device, dtype=torch.float32)
            proj /= torch.linalg.norm(proj, dim=0, keepdim=True)

            x_proj = torch.sort(x @ proj, dim=0).values
            y_proj = torch.sort(y @ proj, dim=0).values

            diff = torch.abs(x_proj - y_proj).pow(p)
            total_sum += (torch.mean(diff, dim=0) ** (1 / p)).sum()

        wass_values[i] = total_sum / n_proj

    return float(wass_values.mean()), float(wass_values.std(unbiased=True))


@torch.no_grad()
def compute_maxwd(
    data_1: torch.Tensor,
    data_real: torch.Tensor,
    n_tests: int = 5,
    n_proj: int = 32,
    p: int = 2,
    n_batch_proj: int | None = None,
    device: str = "cuda",
    n_data=None,
    random: bool = False,
) -> tuple[float, float]:
    """
    Compute the Maximum Wasserstein Distance (MaxWD) between two datasets.

    Args:
        data_1: Tensor of shape [N, D] or higher, first dataset.
        data_real: Tensor of shape [N, D] or higher, reference dataset.
        n_tests: Number of batches to split the data into.
        n_proj: Total number of random projections.
        p: Power for Wasserstein computation.
        n_batch_proj: Batch size for projections; defaults to n_proj.
        device: Device to run computations on.
        n_data: If specified, uses this number of samples per batch with random sampling.

    Returns:
        mean_maxwd: Mean MaxWD over batches.
        std_maxwd: Standard deviation of MaxWD over batches.
    """
    # Move data to device and ensure float32
    data_1 = data_1.to(device=device, dtype=torch.float32, non_blocking=True)
    data_real = data_real.to(device=device, dtype=torch.float32, non_blocking=True)

    # Flatten if data has more than 2 dimensions
    if data_1.dim() > 2:
        data_1 = data_1.flatten(1)
        data_real = data_real.flatten(1)

    N = min(len(data_1), len(data_real))
    D = data_1.shape[1]

    if n_data is not None:
        random = True
        batch_size = n_data
    else:

        batch_size = N // n_tests

    if batch_size == 0:
        raise ValueError(f"Datasets are too small for n_tests={n_tests}.")

    maxwd_values = torch.empty(n_tests, device=device, dtype=torch.float32)
    n_batch_proj = n_batch_proj or n_proj

    # --- Compute MaxWD for each batch ---
    for i in range(n_tests):
        if random:
            indices_1 = torch.randperm(len(data_1), device=device)[:batch_size]
            indices_real = torch.randperm(len(data_real), device=device)[:batch_size]
            x = data_1[indices_1]
            y = data_real[indices_real]
        else:
            x = data_1[i * batch_size : (i + 1) * batch_size]
            y = data_real[i * batch_size : (i + 1) * batch_size]

        proj_vals = []

        # Process projections in batches
        for j in range(0, n_proj, n_batch_proj):
            k = min(n_batch_proj, n_proj - j)

            proj = torch.randn(D, k, device=device, dtype=torch.float32)
            proj /= torch.linalg.norm(proj, dim=0, keepdim=True)

            x_proj = torch.sort(x @ proj, dim=0).values
            y_proj = torch.sort(y @ proj, dim=0).values

            diff = torch.abs(x_proj - y_proj).pow(p)
            proj_vals.append(torch.mean(diff, dim=0) ** (1 / p))

        # Take the maximum over all projections
        maxwd_values[i] = torch.cat(proj_vals).max()

    return float(maxwd_values.mean()), float(maxwd_values.std(unbiased=True))


@torch.compile()
def _from_param_to_angle(param: torch.Tensor) -> torch.Tensor:
    """
    Stereographic projection (https://en.wikipedia.org/wiki/Stereographic_projection)
    Args:
        param (torch.Tensor): d-1 tensor

    Returns:
        torch.Tensor: d tensor with unit norm
    """
    s2 = (param**2).sum()
    x_0 = (s2 - 1) / (s2 + 1)
    other_coords = 2 * param / (s2 + 1)
    angle = torch.cat((x_0 * torch.ones((1,)).to(param.device), other_coords))
    return angle


@torch.compile()
def _1D_wasserstein(
    samples_1: torch.Tensor, samples_2: torch.Tensor, p: int = 2
) -> torch.Tensor:
    """
    Calculate 1d wasserstein distance
    Args:
        samples_1 (torch.Tensor): Samples 1
        samples_2 (torch.Tensor): Samples 2
        p (int): p

    Returns:
        torch.Tensor: Wasserstein
    """
    diffs = torch.sort(samples_1, dim=0)[0] - torch.sort(samples_2, dim=0)[0]

    wasserstein_distance = torch.pow(torch.abs(diffs), p).mean() ** (1 / p)
    return wasserstein_distance


def max_sliced_wasserstein_optimization(
    samples_1: torch.Tensor,
    samples_2: torch.Tensor,
    tol: float = 1e-5,
    p: int = 2,
    max_iter: int = 10_000,
    disable_pbar=False,
) -> tuple[float, torch.Tensor]:
    """
    Calculate maximum Wasserstein distance by solving optimization problem.
    Args:
        samples_1 (torch.Tensor): First set of samples (n_samples, dim)
        samples_2 (torch.Tensor): Second set of samples (n_samples, dim)
        tol (float, optional): Tolerance of the optimization procedure. Defaults to 1e-4.
        p (int, optional): Wassertein parameter. Defaults to 2.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10_000.
        disable_pbar (bool, optional): Disable tqdm pbar. Defaults to False.

    Returns:
        Tuple[float, Torch.Tensor]: Returns the value of the wasserstein and the optimal direction
    """
    device = samples_1.device
    assert device == samples_2.device
    n_1, d_1 = samples_1.shape
    n_2, d_2 = samples_2.shape
    assert d_1 == d_2
    d_x = d_1

    # initialization
    latent_param = torch.randn((d_x - 1,), device=device)
    latent_param = latent_param.requires_grad_(True)
    optim = torch.optim.Adam([latent_param], lr=1e-3, maximize=True)

    err = 0 * torch.ones((1,)).to(device)
    old_err = -10 * torch.ones((1,)).to(device)
    angle = torch.empty((d_x,), device=device)
    pbar = tqdm(range(max_iter), desc="max_sw", disable=disable_pbar)
    for i in pbar:
        optim.zero_grad()
        angle = _from_param_to_angle(latent_param)
        err = _1D_wasserstein(samples_1 @ angle, samples_2 @ angle, p=p)

        if (err - old_err).abs() < tol:
            break
        else:
            old_err = err
            err.backward()
            optim.step()
        if (i % 10) == 9:
            pbar.set_postfix({"sw": err.item()})

    return err.detach().item(), angle.detach()


@torch.no_grad()
def compute_optimized_maxwd(
    data_1: torch.Tensor,
    data_real: torch.Tensor,
    n_tests: int = 5,
    p: int = 2,
    tol: float = 1e-5,
    max_iter: int = 10_000,
    device: str = "cuda",
    n_data: int | None = None,
    random: bool = False,
    disable_pbar: bool = True,
) -> tuple[float, float]:
    """
    Compute the Maximum Sliced Wasserstein Distance (MaxWD) between two datasets
    using optimization (Adam) for each test batch.
    """
    # Move data to device and ensure float32
    data_1 = data_1.to(device=device, dtype=torch.float32, non_blocking=True)
    data_real = data_real.to(device=device, dtype=torch.float32, non_blocking=True)

    # Flatten if data has more than 2 dimensions (standard for feature vectors)
    if data_1.dim() > 2:
        data_1 = data_1.flatten(1)
        data_real = data_real.flatten(1)

    N = min(len(data_1), len(data_real))

    if n_data is not None:
        random = True
        batch_size = n_data
    else:
        batch_size = N // n_tests

    if batch_size == 0:
        raise ValueError(f"Datasets are too small for n_tests={n_tests}.")

    maxwd_values = torch.empty(n_tests, device=device, dtype=torch.float32)

    # --- Compute Optimized MaxWD for each test/batch ---
    for i in range(n_tests):
        # 1. Slice the data for this test
        if random:
            indices_1 = torch.randperm(len(data_1), device=device)[:batch_size]
            indices_real = torch.randperm(len(data_real), device=device)[:batch_size]
            x = data_1[indices_1]
            y = data_real[indices_real]
        else:
            x = data_1[i * batch_size : (i + 1) * batch_size]
            y = data_real[i * batch_size : (i + 1) * batch_size]

        # 2. Use the professor's optimization logic
        # We temporarily enable grads for the internal optimization loop
        with torch.enable_grad():
            val, _ = max_sliced_wasserstein_optimization(
                x, y, tol=tol, p=p, max_iter=max_iter, disable_pbar=disable_pbar
            )

        maxwd_values[i] = val

    return float(maxwd_values.mean()), float(maxwd_values.std(unbiased=True))


@torch.no_grad()
def compute_class_swd(
    data_gen: dict[int, torch.Tensor],
    data_real: dict[int, torch.Tensor],
    n_tests: int = 5,
    n_proj: int = 32,
    p: int = 2,
    device: str = "cuda",
    random: bool = False,
    n_batch_proj: int = 100,
) -> dict[int, tuple[float, float]]:
    results = {}
    for class_id in data_gen:
        if class_id in data_real:
            mean_wass, std_wass = compute_swd(
                data_gen[class_id],
                data_real[class_id],
                n_tests=n_tests,
                n_proj=n_proj,
                n_batch_proj=n_batch_proj,
                p=p,
                device=device,
                random=random,
            )
            results[class_id] = (mean_wass, std_wass)
    return results


@torch.no_grad()
def compute_global_swd(
    data_gen: dict[int, torch.Tensor],
    data_real: dict[int, torch.Tensor],
    n_tests: int = 5,
    n_proj: int = 32,
    n_batch_proj: int = 100,
    n_data=None,
    p: int = 2,
    device: str = "cuda",
    random: bool = False,
) -> tuple[float, float]:
    all_gen = torch.cat(list(data_gen.values()), dim=0)
    all_real = torch.cat(list(data_real.values()), dim=0)
    all_gen = all_gen[torch.randperm(len(all_gen), device=all_gen.device)]
    all_real = all_real[torch.randperm(len(all_real), device=all_real.device)]

    mean_wass, std_wass = compute_swd(
        all_gen,
        all_real,
        n_tests=n_tests,
        n_proj=n_proj,
        p=p,
        n_batch_proj=n_batch_proj,
        n_data=n_data,
        device=device,
        random=random,
    )
    return mean_wass, std_wass


@torch.no_grad()
def compute_class_maxwd(
    data_gen: dict[int, torch.Tensor],
    data_real: dict[int, torch.Tensor],
    n_tests: int = 5,
    n_proj: int = 32,
    n_batch_proj: int = 100,
    p: int = 2,
    device: str = "cuda",
    random: bool = False,
) -> dict[int, tuple[float, float]]:
    results = {}
    for class_id in data_gen:
        if class_id in data_real:
            mean_maxwd, std_maxwd = compute_optimized_maxwd(
                data_gen[class_id],
                data_real[class_id],
                n_tests=n_tests,
                # n_proj=n_proj,
                p=p,
                # n_batch_proj=n_batch_proj,
                device=device,
                random=random,
            )
            results[class_id] = (mean_maxwd, std_maxwd)
    return results


@torch.no_grad()
def compute_global_maxwd(
    data_gen: dict[int, torch.Tensor],
    data_real: dict[int, torch.Tensor],
    n_tests: int = 5,
    n_proj: int = 32,
    n_batch_proj: int = 100,
    n_data=None,
    p: int = 2,
    device: str = "cuda",
    random: bool = False,
) -> tuple[float, float]:
    all_gen = torch.cat(list(data_gen.values()), dim=0)
    all_real = torch.cat(list(data_real.values()), dim=0)
    all_gen = all_gen[torch.randperm(len(all_gen), device=all_gen.device)]
    all_real = all_real[torch.randperm(len(all_real), device=all_real.device)]

    mean_maxwd, std_maxwd = compute_optimized_maxwd(
        all_gen,
        all_real,
        n_tests=n_tests,
        # n_proj=n_proj,
        p=p,
        # n_batch_proj=n_batch_proj,
        n_data=n_data,
        device=device,
        random=random,
    )
    return mean_maxwd, std_maxwd


@torch.no_grad()
def extract_class_inception_feats(
    data_dict: dict[int, torch.Tensor], batch_size: int = 64, device: str = "cuda"
) -> dict[int, torch.Tensor]:
    local_path = "generation/inception_weights/inception-2015-12-05.pkl"
    with open(local_path, "rb") as f:
        inception_net = pickle.load(f).to(device)
    inception_net.eval()

    feature_dict = {}

    for class_id, data in data_dict.items():
        all_feats = []
        N = len(data)

        for start in tqdm(range(0, N, batch_size), desc=f"Features Class {class_id}"):
            cur_batch = min(batch_size, N - start)
            batch_images = data[start : start + cur_batch].to(device).to(torch.float32)
            # batch_images = batch_images.clamp(-1.0, 1.0) # The decoder outputs [0,255]
            # batch_images = (batch_images * 127.5 + 128)
            batch_images = batch_images.clamp(0, 255)

            batch_images = torch.nn.functional.interpolate(
                batch_images, size=(299, 299), mode="bilinear", align_corners=False
            )

            with torch.no_grad():
                feats = inception_net(batch_images, return_features=True).to(
                    torch.float32
                )
                all_feats.append(feats.cpu())

        feature_dict[class_id] = torch.cat(all_feats, dim=0)

    return feature_dict


def compute_class_mu_sigma_from_feats(
    class_feats_dict: dict[int, torch.Tensor], batch_size: int = 64
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    class_stats = {}

    for class_id, feats in class_feats_dict.items():
        mu, sigma, n_total = None, None, 0
        N = len(feats)

        for start in tqdm(
            range(0, N, batch_size), desc=f"Computing Stats Class {class_id}"
        ):
            batch_feats = feats[start : start + batch_size]
            mu, sigma, n_total = update_stats(mu, sigma, n_total, batch_feats)

        if n_total > 1:
            sigma /= n_total - 1

        class_stats[class_id] = (mu, sigma)

    return class_stats


def to_numpy(arr):
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return arr


def compute_class_fid(
    stats_gen: dict[int, tuple],
    stats_real: dict[int, tuple],
    eps: float = 1e-6,
) -> dict[int, tuple[float, float, float]]:
    class_fids = {}

    for class_id in stats_gen:
        if class_id in stats_real:
            mu_gen, sigma_gen = stats_gen[class_id]
            mu_real, sigma_real = stats_real[class_id]

            mu_gen_np = to_numpy(mu_gen)
            sigma_gen_np = to_numpy(sigma_gen)
            mu_real_np = to_numpy(mu_real)
            sigma_real_np = to_numpy(sigma_real)

            fid, m_term, sigma_term = fid_score(
                mu_gen_np, sigma_gen_np, mu_real_np, sigma_real_np, eps=eps
            )

            class_fids[class_id] = (fid, m_term, sigma_term)

    return class_fids


@torch.no_grad()
def compute_mu_sigma_from_feats_global(
    feats: torch.Tensor, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:

    mu, sigma, n_total = None, None, 0
    N = len(feats)

    for start in range(0, N, batch_size):
        batch_feats = feats[start : start + batch_size]
        mu, sigma, n_total = update_stats(mu, sigma, n_total, batch_feats)

    if n_total > 1:
        sigma /= n_total - 1

    return mu, sigma


@torch.no_grad()
def compute_global_fid(
    feats_gen: dict[int, torch.Tensor],
    feats_real: dict[int, torch.Tensor],
    batch_size: int = 64,
    eps: float = 1e-6,
) -> tuple[float, float, float]:

    all_gen = torch.cat(list(feats_gen.values()), dim=0)
    if all_gen.shape[0] > 50_000:
        idx = torch.randperm(all_gen.shape[0])[:50_000]
        all_gen = all_gen[idx]

    mu_gen_global, sigma_gen_global = compute_mu_sigma_from_feats_global(
        all_gen, batch_size=batch_size
    )

    all_real = torch.cat(list(feats_real.values()), dim=0)
    mu_real_global, sigma_real_global = compute_mu_sigma_from_feats_global(
        all_real, batch_size=batch_size
    )

    fid, m_term, sigma_term = fid_score(
        mu_gen_global, sigma_gen_global, mu_real_global, sigma_real_global, eps=eps
    )

    return fid, m_term, sigma_term


@torch.no_grad()
def extract_class_dinov2_feats(
    data_dict: dict[int, torch.Tensor],
    batch_size: int = 64,
    device: str = "cuda",
    resize_mode: str = "torch",
) -> dict[int, torch.Tensor]:

    os.environ["XFORMERS_DISABLED"] = "1"
    torch.hub.set_dir("./generation/torch_hub_cache")
    dino_net = torch.hub.load(
        "facebookresearch/dinov2:main",
        "dinov2_vitl14",
        trust_repo=True,
        verbose=False,
        skip_validation=True,
    ).to(device)
    dino_net.eval().requires_grad_(False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feature_dict = {}
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=True, enable_mem_efficient=False
    ):
        for class_id, data in data_dict.items():
            all_feats = []
            N = len(data)

            for start in tqdm(
                range(0, N, batch_size), desc=f"DINOv2 Feats Class {class_id}"
            ):
                cur_batch = min(batch_size, N - start)
                batch = data[start : start + cur_batch].to(device).to(torch.float32)
                if resize_mode == "torch":
                    batch = F.interpolate(
                        batch, size=(224, 224), mode="bicubic", antialias=True
                    )
                elif resize_mode == "pil":
                    imgs = []
                    for img in batch.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy():
                        img = Image.fromarray(img, "RGB").resize(
                            (224, 224), Image.Resampling.BICUBIC
                        )
                        imgs.append(np.asarray(img))
                    batch = (
                        torch.from_numpy(np.stack(imgs))
                        .permute(0, 3, 1, 2)
                        .float()
                        .to(device)
                    )

                batch = batch / 255.0
                batch = (batch - mean) / std

                with torch.no_grad():
                    feats = dino_net(batch)
                all_feats.append(feats.to(torch.float32).cpu())
                del batch, feats
                gc.collect()
                torch.cuda.empty_cache()

            feature_dict[class_id] = torch.cat(all_feats, dim=0)
    del dino_net
    torch.cuda.empty_cache()
    return feature_dict


@torch.no_grad()
def extract_inception_features(data, batch_size=64, device="cuda"):
    """
    Extract Inception features for a dataset of images in [-1,1].

    Returns a tensor of shape [N, 2048].
    """
    local_path = "generation/inception_weights/inception-2015-12-05.pkl"
    with open(local_path, "rb") as f:
        inception_net = pickle.load(f).to(device)
    inception_net.eval()

    N = len(data)
    feats_tensor = torch.empty((N, 2048), device=device, dtype=torch.float32)

    for start in tqdm(range(0, N, batch_size), desc="Extracting Inception features"):
        cur_batch = min(batch_size, N - start)
        batch_images = data[start : start + cur_batch].to(device)
        batch_images = batch_images.clamp(-1.0, 1.0)

        # Convert [-1,1] -> [0,255]
        batch_images = (batch_images * 127.5 + 128).clamp(0, 255)

        # Resize to 299x299
        batch_images = torch.nn.functional.interpolate(
            batch_images, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Extract features
        feats_tensor[start : start + cur_batch] = inception_net(
            batch_images, return_features=True
        ).to(torch.float32)

    return feats_tensor


def polynomial_mmd_batched(
    features_real, features_fake, degree=3, coef=1.0, gamma=None, batch_size=1024
):
    if gamma is None:
        gamma = features_real.size(1)

    def get_sums(X, Y, is_same):
        N = X.shape[0]
        M = Y.shape[0]
        total_sum = 0.0
        diag_sum = 0.0
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            dot = X[i:end_i] @ Y.T
            kernel_stripe = (dot / gamma + coef) ** degree
            total_sum += kernel_stripe.sum().item()
            if is_same:
                diag_sum += torch.diag(kernel_stripe[:, i:end_i]).sum().item()
        return total_sum, diag_sum

    N = features_real.shape[0]
    M = features_fake.shape[0]

    sum_rr, diag_rr = get_sums(features_real, features_real, is_same=True)
    sum_ff, diag_ff = get_sums(features_fake, features_fake, is_same=True)
    sum_rf, _ = get_sums(features_real, features_fake, is_same=False)

    term_rr = (sum_rr - diag_rr) / (N * (N - 1))
    term_ff = (sum_ff - diag_ff) / (M * (M - 1))
    term_rf = sum_rf / (N * M)

    return term_rr + term_ff - 2 * term_rf


@torch.no_grad()
def compute_kid(feats_1, feats_2, n_max=50_000, device="cpu"):
    N = min(n_max, feats_1.shape[0], feats_2.shape[0])
    idx_1 = torch.randperm(feats_1.size(0))[:N]
    idx_2 = torch.randperm(feats_2.size(0))[:N]
    f1 = feats_1[idx_1].to(device)
    f2 = feats_2[idx_2].to(device)
    kid_val = polynomial_mmd_batched(f1, f2, batch_size=1024)
    return kid_val


@torch.no_grad()
def compute_class_kid(
    feats_gen: dict[int, torch.Tensor],
    feats_real: dict[int, torch.Tensor],
    min_test_size: int = 256,
) -> dict[int, float]:
    class_kids = {}
    for class_id in tqdm(feats_gen, desc="Computing Class KID (Adaptive)"):
        if class_id in feats_real:
            gen = feats_gen[class_id]
            real = feats_real[class_id]
            kid_val = compute_kid(gen, real, n_max=1000, device="cpu")
            class_kids[class_id] = kid_val
    return class_kids


@torch.no_grad()
def compute_global_kid(
    feats_gen: dict[int, torch.Tensor],
    feats_real: dict[int, torch.Tensor],
    min_test_size: int = 256,
) -> float:
    all_gen = torch.cat(list(feats_gen.values()), dim=0)
    all_real = torch.cat(list(feats_real.values()), dim=0)
    print("Computing global kid")
    kid_val = compute_kid(all_gen, all_real, n_max=50_000, device="cpu")
    return kid_val
