import torch
import matplotlib.pyplot as plt

from src.data.datasets import get_stl10_pretrain_dataset


# Same normalization constants used in augmentations.py
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(image):
    """
    Undo normalization so the image can be displayed correctly.
    """
    image = image * std + mean
    return image.clamp(0, 1)


dataset = get_stl10_pretrain_dataset()

n_images = 5

fig, axes = plt.subplots(n_images, 2, figsize=(6, 3 * n_images))

for i in range(n_images):

    (view1, view2), _ = dataset[i]

    view1 = denormalize(view1)
    view2 = denormalize(view2)

    # PyTorch: [C, H, W]
    # Matplotlib: [H, W, C]
    view1 = view1.permute(1, 2, 0)
    view2 = view2.permute(1, 2, 0)

    axes[i, 0].imshow(view1)
    axes[i, 1].imshow(view2)

    axes[i, 0].set_title(f"Image {i} — View 1")
    axes[i, 1].set_title(f"Image {i} — View 2")

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")


plt.tight_layout()

plt.savefig(
    "figures/stl10_positive_pairs.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()