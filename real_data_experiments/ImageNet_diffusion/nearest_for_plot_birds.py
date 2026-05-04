import os
import torch
import gc
import sys
from tqdm import tqdm
from PIL import Image
import random
from generation.sampling import decode_dict_clean
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

map_path = "../NoisedFlow/ImageNet_Flow/flow_class_map.json"
with open(map_path, "r") as file:
    class_map_config = json.load(file)
all_dog_labels = [int(v) for k, v in class_map_config["class_map_birds"].items()]
n_to_plot = 10
random.seed(0)
LABELS_TO_CHECK = random.sample(all_dog_labels, n_to_plot)  # [160, 258, 251, 239, 157]
NUM_QUERIES_PER_CLASS = 10
BATCH_SIZE_DECODE = 16
SAVE_BASE_DIR = "photos_IN_birds_for_plot"  # This changes from 300 to dogs

DATASETS_TO_SEARCH = {  # This changes from 300 to dogs
    "real_train": "../NoisedFlow/ImageNet_Flow/datasets/train_set_tensor_birds_normalized.pt",
    "edm_classic": "generation/init_&_gen/generation_edm_classic_birds_seed_0.pt",
    "flow_cfg_0.5": "generation/init_&_gen/generation_flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_seed_0.pt",
    "flow_cfg_0.5_dynamical": "generation/init_&_gen/generation_flow_IN_birds_CFG_0.5_sigma_7_factor_14_big_128_dynamical_seed_0.pt",
    "gaussian_sigma_7": "generation/init_&_gen/generation_gaussian_birds_sigma_7_seed_0.pt",
    "empirical": "generation/init_&_gen/generation_empirical_birds_seed_0.pt",
}

os.makedirs(SAVE_BASE_DIR, exist_ok=True)


def find_nearest_neighbors(query_tensor, search_dataset):
    q = query_tensor.flatten(1)
    s = search_dataset.flatten(1)
    dists = torch.cdist(q, s, p=2)
    idx = torch.argmin(dists, dim=1)
    return search_dataset[idx]


train_data = torch.load(DATASETS_TO_SEARCH["real_train"])

for label in LABELS_TO_CHECK:
    label_str = str(label)
    if label_str not in train_data:
        print(f"⚠️ Label {label_str} missing. Skipping.")
        continue

    print(f"\n🔍 Class {label_str}")
    torch.manual_seed(42)
    num_samples = train_data[label_str].shape[0]
    indices = torch.randperm(num_samples)[:NUM_QUERIES_PER_CLASS]
    queries = train_data[label_str][indices].to(device)
    to_decode = {f"class_{label}_queries": queries.cpu()}

    for ds_name, ds_path in DATASETS_TO_SEARCH.items():
        if ds_name == "real_train":
            continue

        print(f"   -> {ds_name}")
        search_data_dict = torch.load(ds_path)

        if label_str in search_data_dict:
            search_pool = search_data_dict[label_str].to(device)
            neighbors = find_nearest_neighbors(queries, search_pool)
            to_decode[f"class_{label}_nn_{ds_name}"] = neighbors.cpu()
            del search_pool

        del search_data_dict
        gc.collect()
        torch.cuda.empty_cache()

    print(f"🎨 Decoding...")
    decoded_results = decode_dict_clean(
        to_decode, batchsize=BATCH_SIZE_DECODE, device=device
    )

    for key, images_tensor in decoded_results.items():
        class_save_dir = os.path.join(SAVE_BASE_DIR, key)
        os.makedirs(class_save_dir, exist_ok=True)

        for i in range(images_tensor.shape[0]):
            img_array = images_tensor[i].permute(1, 2, 0).numpy()
            img = Image.fromarray(img_array)
            img.save(os.path.join(class_save_dir, f"sample_{i:02d}.png"))

    del queries, to_decode, decoded_results
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n✅ Done: {SAVE_BASE_DIR}")
