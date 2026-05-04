import torch
import os
from torchvision.utils import save_image


def find_nearest_and_save():
    base_path = "generation/init_&_gen"
    train_path = "datasets/train_set_tensor.pt"
    output_root = "photos_FFHQ_for_plot"

    gen_files = [
        "generation_edm_classic_ve_0.pt",
        "generation_empirical_20_ve_0.pt",
        "generation_TarFlow_FFHQ_noised_with_factor_14_SOTA_update_prior_seed_20_ve_0.pt",
        #"generation_TarFlow_FFHQ_noised_with_factor_14_big_128_dynamical_seed_20_ve_0.pt",
        "generation_gaussian_sigma_7_20_ve_0.pt",
    ]
    train_set = torch.load(train_path)
    torch.manual_seed(42)
    num_samples = train_set.shape[0]
    indices = torch.randperm(num_samples)[:10]
    train_set = train_set[indices]

    os.makedirs(os.path.join(output_root, "original_train"), exist_ok=True)
    for i, img in enumerate(train_set):
        save_image(
            img,
            os.path.join(output_root, "original_train", f"train_{i}.jpg"),
            normalize=True,
            value_range=(-1, 1),
        )

    for f_name in gen_files:
        folder_name = f_name.replace(".pt", "")
        target_dir = os.path.join(output_root, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        gen_data = torch.load(os.path.join(base_path, f_name)).float()

        train_flat = train_set.view(10, -1)
        gen_flat = gen_data.view(gen_data.size(0), -1)

        for i in range(10):
            dist = torch.cdist(train_flat[i : i + 1], gen_flat)[0]
            nearest_idx = torch.argmin(dist)
            save_image(
                gen_data[nearest_idx],
                os.path.join(target_dir, f"nearest_to_train_{i}.jpg"),
                normalize=True,
                value_range=(-1, 1),
            )


if __name__ == "__main__":
    find_nearest_and_save()
