import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# EuroSAT class names — used to build VQA templates later
EUROSAT_CLASSES = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial Buildings",
    "Pasture",
    "Permanent Crop",
    "Residential Buildings",
    "River",
    "SeaLake",
]

def get_eurosat_transforms(train: bool = True) -> transforms.Compose:
    """
    RS-specific augmentations.
    No horizontal flip bias — satellite images have no canonical orientation.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),       # valid for satellite imagery
            transforms.RandomRotation(90),         # 90deg rotations are natural
            transforms.ColorJitter(0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3444, 0.3803, 0.4078],     # EuroSAT RGB mean
                std=[0.2038, 0.1367, 0.1152],      # EuroSAT RGB std
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3444, 0.3803, 0.4078],
                std=[0.2038, 0.1367, 0.1152],
            ),
        ])


class EuroSATDataset(Dataset):
    """
    EuroSAT dataset wrapper.

    Expects the EuroSAT folder structure:
        data/EuroSAT/
            AnnualCrop/  Forest/  HerbaceousVegetation/ ...

    Download: https://madm.dfki.de/files/sentinel/EuroSAT.zip
    """

    def __init__(
        self,
        root: str = "data/EuroSAT",
        train: bool = True,
        train_split: float = 0.8,
        gsd: float = 10.0,       # Sentinel-2 GSD
    ):
        self.gsd = gsd
        transform = get_eurosat_transforms(train)

        # torchvision ImageFolder handles the class subfolder structure
        full_dataset = datasets.ImageFolder(root=root, transform=transform)

        # Manual train/val split (EuroSAT has no official split)
        total = len(full_dataset)
        train_size = int(total * train_split)
        val_size = total - train_size

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        self.dataset = train_ds if train else val_ds
        self.classes = full_dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx] #type: ignore
        return {
            "image": image,                              # [3, 224, 224]
            "label": torch.tensor(label, dtype=torch.long),
            "gsd": torch.tensor(self.gsd, dtype=torch.float32),
            "class_name": EUROSAT_CLASSES[label],
        }


def mae_mask_patches(
    feature_map: torch.Tensor,
    mask_ratio: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask patches in a CNN feature map for MAE pretraining.

    Args:
        feature_map: [B, C, H, W]  — output of CNN stem
        mask_ratio:  fraction of patches to mask (default 0.75)

    Returns:
        masked:      [B, C, H, W]  — feature map with masked patches zeroed
        mask:        [B, H, W]     — binary mask (1 = masked, 0 = visible)
    """
    B, C, H, W = feature_map.shape
    num_patches = H * W
    num_masked = int(num_patches * mask_ratio)

    mask = torch.zeros(B, num_patches, device=feature_map.device)

    for i in range(B):
        # Random indices to mask
        masked_idx = torch.randperm(num_patches)[:num_masked]
        mask[i, masked_idx] = 1.0

    mask_2d = mask.view(B, H, W)                        # [B, H, W]
    mask_expanded = mask_2d.unsqueeze(1).expand_as(feature_map)  # [B, C, H, W]

    masked_feature_map = feature_map * (1 - mask_expanded)

    return masked_feature_map, mask_2d


def get_eurosat_dataloader(
    root: str = "data/EuroSAT",
    train: bool = True,
    batch_size: int = 32,
    num_workers: int = 2,
) -> DataLoader:
    dataset = EuroSATDataset(root=root, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    print("Testing MAE masking (no dataset download needed)...")

    # Test mae_mask_patches with a dummy feature map
    dummy_feature_map = torch.randn(2, 256, 28, 28)
    masked, mask = mae_mask_patches(dummy_feature_map, mask_ratio=0.75)

    print(f"Feature map:    {dummy_feature_map.shape}")
    print(f"Masked map:     {masked.shape}")
    print(f"Mask:           {mask.shape}")

    masked_ratio = mask.float().mean().item()
    print(f"Actual masked:  {masked_ratio:.2%}  (should be ~75%)")
    assert masked.shape == dummy_feature_map.shape, "Shape mismatch!"
    print("MAE masking check passed.")

    print("\nTo test the full dataloader, download EuroSAT and run:")
    print("  loader = get_eurosat_dataloader('data/EuroSAT', train=True)")