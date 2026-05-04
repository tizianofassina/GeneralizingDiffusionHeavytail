import os
import torch
import pickle
from generation import dnnlib, torch_utils
import numpy as np
import json
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import generation
import sys

# --- CONFIG ---
BASE_PATH = "ImageNet_512_processed/datasets/img512_right/"
INCEPTION_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"

sys.modules["torch_utils"] = generation.torch_utils
sys.modules["dnnlib"] = generation.dnnlib


class ImageFolderSubset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".png")]
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255.0),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        return self.transform(img)


@torch.no_grad()
def extract_features(class_map: dict, batch_size: int = 64, device: str = "cuda"):
    local_path = "generation/inception_weights/inception-2015-12-05.pkl"
    print(f"Loading Inception from {local_path}...")
    with open(local_path, "rb") as f:
        inception_net = pickle.load(f).to(device)
    inception_net.eval()

    print("Loading DINOv2 (vitl14)...")
    torch.hub.set_dir("./generation/torch_hub_cache")
    dino_net = torch.hub.load(
        "facebookresearch/dinov2:main",
        "dinov2_vitl14",
        trust_repo=True,
        verbose=False,
        skip_validation=True,
    ).to(device)
    dino_net.eval()

    dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    inception_dict, dinov2_dict = {}, {}

    for _, target_class in class_map.items():
        folder_str = str(target_class).zfill(3)
        class_key = str(target_class)
        full_folder_path = os.path.join(BASE_PATH, folder_str)

        if not os.path.exists(full_folder_path):
            print(f"Skipping: {full_folder_path} not found.")
            continue

        print(f"\nProcessing Folder {folder_str} (Target Class: {class_key})")
        dataset = ImageFolderSubset(full_folder_path)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        inc_feats_list, dino_feats_list = [], []
        for batch in tqdm(dataloader, desc="Extracting"):
            batch = batch.to(device)
            # Inception
            img_inc = F.interpolate(
                batch, size=(299, 299), mode="bilinear", align_corners=False
            ).clamp(0, 255)
            feat_inc = inception_net(img_inc, return_features=True).to(torch.float32)
            inc_feats_list.append(feat_inc.cpu())
            # DINOv2
            img_dino = F.interpolate(
                batch, size=(224, 224), mode="bicubic", antialias=True
            )
            img_dino = (img_dino / 255.0 - dino_mean) / dino_std
            feat_dino = dino_net(img_dino).to(torch.float32)
            dino_feats_list.append(feat_dino.cpu())

        if inc_feats_list:
            inception_dict[class_key] = torch.cat(inc_feats_list, dim=0)
            dinov2_dict[class_key] = torch.cat(dino_feats_list, dim=0)

    return inception_dict, dinov2_dict


if __name__ == "__main__":
    MAP_PATH = "../../NoisedFlow/NoisedFlow/ImageNet_Flow/flow_class_map.json"
    os.makedirs("features", exist_ok=True)

    with open(MAP_PATH, "r") as f:
        config = json.load(f)
    class_map = config["class_map_birds"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inc_features, dino_features = extract_features(
        class_map, batch_size=10, device=device
    )

    print("\nSaving results...")
    torch.save(inc_features, "features/feats_real_birds_Inception.pt")
    torch.save(dino_features, "features/feats_real_birds_Dino.pt")

    print(f"Process complete. Extracted {len(inc_features)} classes.")
