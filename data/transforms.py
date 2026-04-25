import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from PIL import Image


# EuroSAT RGB statistics (computed from full dataset)
EUROSAT_MEAN = [0.3444, 0.3803, 0.4078]
EUROSAT_STD  = [0.2038, 0.1367, 0.1152]


class RandomRotation90:
    """
    Randomly rotate image by 0, 90, 180, or 270 degrees.
    More natural for satellite imagery than arbitrary-angle rotation —
    a top-down view has no canonical orientation.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice([0, 90, 180, 270])
        img_tensor = TF.to_tensor(img)
        rotated_tensor = TF.rotate(img_tensor, angle, expand=False)
        return TF.to_pil_image(rotated_tensor)


class RandomGaussianNoise:
    """
    Add small Gaussian noise to simulate sensor noise in satellite imagery.
    Applied after ToTensor so operates on float tensors.
    """
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomChannelDrop:
    """
    Randomly zero out one channel with low probability.
    Simulates cloud cover or sensor dropout in one spectral band.
    """
    def __init__(self, prob: float = 0.05):
        self.prob = prob

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            ch = random.randint(0, tensor.shape[0] - 1)
            tensor = tensor.clone()
            tensor[ch] = 0.0
        return tensor


def get_train_transforms() -> T.Compose:
    """Full augmentation pipeline for training."""
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomRotation90(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        RandomGaussianNoise(std=0.01),
        RandomChannelDrop(prob=0.05),
        T.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
    ])


def get_val_transforms() -> T.Compose:
    """Minimal pipeline for validation — no augmentation."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
    ])


def get_mae_transforms() -> T.Compose:
    """
    Transforms for MAE pretraining Phase 1.
    Slightly heavier augmentation — model must reconstruct
    despite distortion, forcing robust feature learning.
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomRotation90(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        T.ToTensor(),
        RandomGaussianNoise(std=0.02),
        T.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD),
    ])


if __name__ == "__main__":
    import numpy as np

    print("Testing RS transforms...")

    # Create a dummy PIL image
    dummy_pil = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    for name, transform in [
        ("train",   get_train_transforms()),
        ("val",     get_val_transforms()),
        ("mae",     get_mae_transforms()),
    ]:
        out = transform(dummy_pil)
        assert out.shape == (3, 224, 224), f"{name} shape mismatch!" #type: ignore
        print(f"  {name:6s} transform: {out.shape}  "#type: ignore
              f"min={out.min():.2f}  max={out.max():.2f}")#type: ignore

    print("\nRandomRotation90 test...")
    rot = RandomRotation90()
    for _ in range(8):
        out = rot(dummy_pil)
    print("  Rotation check passed.")

    print("\nAll transform checks passed.")