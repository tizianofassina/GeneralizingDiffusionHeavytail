import torch
from torch import distributions as dist
from tqdm import tqdm
import lightning as L
from functools import partial
from typing import Dict, Any
import os
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from lightning import LightningDataModule
from torch import Generator, float as torch_float, randn, randn_like
from torchvision import transforms
from torch.utils.data import TensorDataset

torch_float = torch.float
randn = torch.randn
randn_like = torch.randn_like


def to_dict(item):
    """
    Convert a single data item to a dictionary with a torch tensor.

    Args:
        item: Input data item (could be list, tuple, or np.ndarray).

    Returns:
        dict: Dictionary containing a single key 'data_sample' as a float32 tensor.
    """
    if isinstance(item, (tuple, list)):
        x = item[0]
    else:
        x = item
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    return {"data_sample": x.to(dtype=torch.float32)}


def add_ve_noise(item, sigma_max, sigma_min, log_mean, log_std, **kwargs):
    dtype = torch.float32
    device = item["data_sample"].device  # <-- usa lo stesso device del tensore

    # Parametri su device corretto
    sigma_max = torch.tensor(sigma_max, dtype=dtype, device=device)
    sigma_min = torch.tensor(sigma_min, dtype=dtype, device=device)
    log_mean = torch.tensor(log_mean, dtype=dtype, device=device)
    log_std = torch.tensor(log_std, dtype=dtype, device=device)

    # Noise level
    noise_level = (
        (torch.randn(size=(1,), dtype=dtype, device=device) * log_std + log_mean)
        .exp()
        .clip(sigma_min, sigma_max)
    )

    noisy = (
        item["data_sample"]
        + torch.randn_like(item["data_sample"], device=device) * noise_level
    )

    return {
        **item,
        "noise_level": noise_level.squeeze(0),
        "noisy_sample": noisy,
    }


class VEDataset(Dataset):
    """
    PyTorch Dataset applying VE noise dynamically on top of a base dataset.

    Each data item is converted to a dictionary with a 'data_sample' key
    and a corresponding 'noisy_sample' key with applied VE noise.
    """

    def __init__(self, base_dataset, diffusion_cfg, **kwargs):
        super().__init__(**kwargs)
        self.base_dataset = base_dataset
        # Compose transformations: conversion to dict + VE noise
        methods = [to_dict]
        methods += [partial(add_ve_noise, **diffusion_cfg)]
        self.transform = transforms.Compose(methods)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index) -> Dict:
        """
        Retrieve a transformed data sample.

        Args:
            index (int): Index of the data sample.

        Returns:
            dict: Dictionary containing 'data_sample', 'noisy_sample', and 'noise_level'.
        """
        base_data_instance = self.base_dataset[index]
        result = self.transform(base_data_instance)

        # Ensure all tensors are float32
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(dtype=torch.float32)

        # Sanity checks
        assert (
            result["data_sample"].device == result["noisy_sample"].device
        ), "Device mismatch!"
        assert (
            result["data_sample"].dtype == torch.float32
        ), f"Wrong dtype: {result['data_sample'].dtype}"
        return result


class VEDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for Variance Exploding datasets.

    Handles splitting into training and validation sets, DataLoader setup, and
    integrates VEDataset for dynamic noise application.
    """

    def __init__(
        self,
        base_dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        n_procs: int = os.cpu_count(),
        val_ptg: float = 0.05,
        diffusion_cfg: Dict[str, Any] = {},
        return_data: bool = False,
        add_maps: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.n_procs = n_procs
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.ve_dataset_params = {"diffusion_cfg": diffusion_cfg}
        self.add_maps = add_maps
        self.return_data = return_data
        self.val_ptg = val_ptg
        self.num_workers = num_workers
        self.use_pin_memory = torch.cuda.is_available()

    def setup(self, stage: str):
        """
        Split dataset into training and validation sets.

        Args:
            stage (str): Stage of training ('fit', 'test', etc.)
        """
        if stage == "fit":
            final_dataset = VEDataset(
                base_dataset=self.base_dataset, **self.ve_dataset_params
            )
            n_val = int(len(final_dataset) * self.val_ptg)
            n_train = len(final_dataset) - n_val
            self.train, self.val = random_split(
                final_dataset,
                lengths=[n_train, n_val],
                generator=Generator().manual_seed(42),
            )

    def train_dataloader(self):
        """
        Returns:
            DataLoader: DataLoader for the training set.
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.use_pin_memory,
            prefetch_factor=None if self.num_workers == 0 else 2,
        )

    def val_dataloader(self):
        """
        Returns:
            DataLoader: DataLoader for the validation set.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.use_pin_memory,
            prefetch_factor=None if self.num_workers == 0 else 2,
        )
