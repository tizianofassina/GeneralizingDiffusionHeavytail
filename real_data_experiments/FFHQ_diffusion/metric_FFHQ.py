import torch
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
import pickle
import dnnlib
import torch.nn.functional as F
from PIL import Image

import torch


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
def compute_swd_new(
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
    batch_size = N // n_tests
    if batch_size == 0:
        raise ValueError(f"Datasets are too small for n_tests={n_tests}.")

    maxwd_values = torch.empty(n_tests, device=device, dtype=torch.float32)
    n_batch_proj = n_batch_proj or n_proj

    # --- Compute MaxWD for each batch ---
    for i in range(n_tests):
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

    return float(maxwd_values.mean()), float(maxwd_values.std(unbiased=False))


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


# --- Incremental stats update (numerically stable)
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


# --- Main function to compute Inception statistics
def compute_mu_sigma_fid(data, batch_size=64, device="cuda"):
    """Compute Inception statistics (mu, sigma) from a dataset of images in [-1,1]."""

    # Load NVIDIA's Inception network (as used by StyleGAN3/EDM)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    with dnnlib.util.open_url(url) as f:
        inception_net = pickle.load(f).to(device)
    inception_net.eval()

    mu, sigma, n_total = None, None, 0
    N = len(data)
    for start in tqdm(range(0, N, batch_size), desc="Features via Inception"):
        cur_batch = min(batch_size, N - start)
        batch_images = data[start : start + cur_batch].to(device)
        batch_images = batch_images.clamp(-1.0, 1.0)

        # Convert from [-1,1] → [0,255]
        batch_images = (batch_images * 127.5 + 128).clamp(0, 255)

        # Resize to 299×299
        batch_images = torch.nn.functional.interpolate(
            batch_images, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Extract Inception features
        with torch.no_grad():
            feats = inception_net(batch_images, return_features=True).to(torch.float32)

        mu, sigma, n_total = update_stats(mu, sigma, n_total, feats)

    # Normalize covariance
    sigma /= n_total - 1

    return mu, sigma


def compute_mu_sigma_fid_dino(data, batch_size=64, device="cuda", resize_mode="torch"):
    """
    Compute DINOv2 feature statistics (mu, sigma) for a dataset of images in [-1, 1].

    Parameters
    ----------
    data : torch.Tensor
        Dataset tensor of shape [N, 3, H, W] with values in [-1, 1].
    batch_size : int
        Batch size for feature extraction.
    device : str
        Device ('cuda' or 'cpu').
    resize_mode : str
        'torch' (fast) or 'pil' (slow exact match with NVIDIA's implementation).
    """

    # --- 1. Load DINOv2 model ---
    torch.hub.set_dir("./torch_hub_cache")
    dino = torch.hub.load(
        "facebookresearch/dinov2:main",
        "dinov2_vitl14",
        trust_repo=True,
        verbose=False,
        skip_validation=True,
    ).to(device)
    dino.eval().requires_grad_(False)

    # --- 2. Mean/Std ImageNet for normalization ---
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    mu, sigma, n_total = None, None, 0
    N = len(data)

    for start in tqdm(range(0, N, batch_size), desc="Features via DINOv2"):
        cur_batch = min(batch_size, N - start)
        batch = data[start : start + cur_batch].to(device)

        # Convert [-1,1] → [0,255]
        batch = (batch * 127.5 + 128).clamp(0, 255)

        # --- 4. Resize ---
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
                torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(device)
            )
        else:
            raise ValueError(f"resize_mode '{resize_mode}' not supported")

        # Normalizzazione DINOv2
        batch = batch / 255.0
        batch = (batch - mean) / std

        # Extraction feature
        with torch.no_grad():
            feats = dino(batch)
        feats = feats.to(torch.float64)

        # Update stats
        mu, sigma, n_total = update_stats(mu, sigma, n_total, feats)

    sigma /= n_total - 1

    return mu, sigma


def compute_swd_fid(
    data_1, data_2, n_proj, n_tests=2, batch_size=64, n_batch_proj=500, device="cuda"
):
    """Compute Inception statistics (mu, sigma) from a dataset of images in [-1,1]."""

    # Load NVIDIA's Inception network (as used by StyleGAN3/EDM)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    with dnnlib.util.open_url(url) as f:
        inception_net = pickle.load(f).to(device)
    inception_net.eval()

    N = min(data_1.shape[0], data_2.shape[0])

    feats_1 = torch.empty((N, 2048)).to(device)
    feats_2 = torch.empty((N, 2048)).to(device)

    for start in tqdm(
        range(0, N, batch_size), desc="Features via Inception -  First Data"
    ):
        cur_batch = min(batch_size, N - start)
        batch_images = data_1[start : start + cur_batch].to(device)
        batch_images = batch_images.clamp(-1.0, 1.0)

        # Convert from [-1,1] → [0,255]
        batch_images = (batch_images * 127.5 + 128).clamp(0, 255)

        # Resize to 299×299
        batch_images = torch.nn.functional.interpolate(
            batch_images, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Extract Inception features
        with torch.no_grad():
            feats_1[start : start + cur_batch] = inception_net(
                batch_images, return_features=True
            ).to(torch.float32)

    for start in tqdm(
        range(0, N, batch_size), desc="Features via Inception - Second Data"
    ):
        cur_batch = min(batch_size, N - start)
        batch_images = data_2[start : start + cur_batch].to(device)
        batch_images = batch_images.clamp(-1.0, 1.0)

        # Convert from [-1,1] → [0,255]
        batch_images = (batch_images * 127.5 + 128).clamp(0, 255)

        # Resize to 299×299
        batch_images = torch.nn.functional.interpolate(
            batch_images, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Extract Inception features
        with torch.no_grad():
            feats_2[start : start + cur_batch] = inception_net(
                batch_images, return_features=True
            ).to(torch.float32)

    mean_wasserstein, std_wasserstein = compute_swd(
        feats_1, feats_2, n_proj=n_proj, n_batch_proj=n_batch_proj, n_tests=n_tests
    )

    return mean_wasserstein, std_wasserstein


def compute_maxwd_fid(
    data_1, data_2, n_proj, n_tests=2, batch_size=64, n_batch_proj=500, device="cuda"
):
    """Compute Inception statistics (mu, sigma) from a dataset of images in [-1,1]."""

    # Load NVIDIA's Inception network (as used by StyleGAN3/EDM)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    with dnnlib.util.open_url(url) as f:
        inception_net = pickle.load(f).to(device)
    inception_net.eval()

    N = min(data_1.shape[0], data_2.shape[0])

    feats_1 = torch.empty((N, 2048)).to(device)
    feats_2 = torch.empty((N, 2048)).to(device)

    for start in tqdm(
        range(0, N, batch_size), desc="Features via Inception - First Data"
    ):
        cur_batch = min(batch_size, N - start)
        batch_images = data_1[start : start + cur_batch].to(device)
        batch_images = batch_images.clamp(-1.0, 1.0)

        # Convert from [-1,1] → [0,255]
        batch_images = (batch_images * 127.5 + 128).clamp(0, 255)

        # Resize to 299×299
        batch_images = torch.nn.functional.interpolate(
            batch_images, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Extract Inception features
        with torch.no_grad():
            feats_1[start : start + cur_batch] = inception_net(
                batch_images, return_features=True
            ).to(torch.float32)

    for start in tqdm(
        range(0, N, batch_size), desc="Features via Inception - Second Data"
    ):
        cur_batch = min(batch_size, N - start)
        batch_images = data_2[start : start + cur_batch].to(device)
        batch_images = batch_images.clamp(-1.0, 1.0)

        # Convert from [-1,1] → [0,255]
        batch_images = (batch_images * 127.5 + 128).clamp(0, 255)

        # Resize to 299×299
        batch_images = torch.nn.functional.interpolate(
            batch_images, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Extract Inception features
        with torch.no_grad():
            feats_2[start : start + cur_batch] = inception_net(
                batch_images, return_features=True
            ).to(torch.float32)

    mean_wasserstein, std_wasserstein = compute_maxwd(
        feats_1, feats_2, n_proj=n_proj, n_batch_proj=n_batch_proj, n_tests=n_tests
    )

    return mean_wasserstein, std_wasserstein


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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Dati sintetici
    torch.manual_seed(0)
    data1 = torch.load("train_set_tensor.pt").to("cpu")  # dataset 1
    data2 = data1 + torch.randn_like(data1) * 0.05  # dataset 2 leggermente shiftato
    print("1")
    # --- SWD / MaxWD
    data1_flat = data1.flatten(1)
    data2_flat = data2.flatten(1)
    mean_swd, std_swd = compute_swd(
        data1_flat, data2_flat, n_tests=4, n_proj=1000, n_batch_proj=4, device=device
    )
    print("SWD:", mean_swd, "+/-", std_swd)
    print("2")
    mean_maxwd, std_maxwd = compute_maxwd(
        data1_flat, data2_flat, n_tests=4, n_proj=1000, n_batch_proj=4, device=device
    )
    print("MaxWD:", mean_maxwd, "+/-", std_maxwd)
    print("3")
    # --- SWD FID / MaxWD FID
    mean_swd_fid, std_swd_fid = compute_swd_fid(
        data1, data2, n_proj=1000, n_tests=4, n_batch_proj=4, device=device
    )
    print("SWD FID:", mean_swd_fid, "+/-", std_swd_fid)
    print("4")
    mean_maxwd_fid, std_maxwd_fid = compute_maxwd_fid(
        data1, data2, n_proj=1000, n_tests=4, n_batch_proj=4, device=device
    )
    print("MaxWD FID:", mean_maxwd_fid, "+/-", std_maxwd_fid)
    print("5")
    # --- FID classico
    mu1, sigma1 = compute_mu_sigma_fid(data1, batch_size=64, device=device)
    mu2, sigma2 = compute_mu_sigma_fid(data2, batch_size=64, device=device)
    fid = fid_score(mu1, sigma1, mu2, sigma2)
    print("FID :", fid)


def batch_matmul(X, Y, batch_size=1024):
    """
    Compute X @ Y^T in batches along the second dimension of Y.
    Returns the full result as a matrix.
    """
    N, d = X.shape
    M, _ = Y.shape
    result = torch.zeros((N, M), device=X.device, dtype=X.dtype)

    for i in range(0, M, batch_size):
        # if (i // batch_size) % 10 == 0:
        # print(f"Processing batch {i} to {min(i + batch_size, M)} / {M}")
        j = min(i + batch_size, M)
        result[:, i:j] = X @ Y[i:j].T
    return result


def polynomial_mmd_batched(
    features_real, features_fake, degree=3, coef=1.0, gamma=None, batch_size=1024
):
    if gamma is None:
        gamma = features_real.size(1)

    K_rr = (
        batch_matmul(features_real, features_real, batch_size) / gamma + coef
    ) ** degree
    K_ff = (
        batch_matmul(features_fake, features_fake, batch_size) / gamma + coef
    ) ** degree
    K_rf = (
        batch_matmul(features_real, features_fake, batch_size) / gamma + coef
    ) ** degree

    N = K_rr.size(0)
    M = K_ff.size(0)
    K_rr = K_rr - torch.diag(torch.diag(K_rr))
    K_ff = K_ff - torch.diag(torch.diag(K_ff))

    mmd = K_rr.sum() / (N * (N - 1)) + K_ff.sum() / (M * (M - 1)) - 2 * K_rf.mean()
    return mmd.item()


@torch.no_grad()
def extract_inception_features(data, batch_size=64, device="cuda"):
    """
    Extract Inception features for a dataset of images in [-1,1].

    Returns a tensor of shape [N, 2048].
    """
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    with dnnlib.util.open_url(url) as f:
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


@torch.no_grad()
def compute_kid(feats_1, feats_2, n_max=50_000, device="cpu"):
    idx_1 = torch.randperm(feats_1.size(0))[:n_max]
    idx_2 = torch.randperm(feats_2.size(0))[:n_max]
    feats_1 = feats_1[idx_1].to(device)
    feats_2 = feats_2[idx_2].to(device)
    kid_val = polynomial_mmd_batched(feats_1, feats_2, batch_size=50)
    return kid_val
