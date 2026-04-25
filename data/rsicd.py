import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_rsicd_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3444, 0.3803, 0.4078],
                std=[0.2038, 0.1367, 0.1152],
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


class RSICDDataset(Dataset):
    """
    RSICD image captioning dataset.

    Expects this folder structure:
        data/RSICD/
            images/      
            dataset_rsicd.json
    """

    def __init__(
        self,
        root: str = "data/RSICD",
        train: bool = True,
        train_split: float = 0.9,
        gsd: float = 10.0,
    ):
        self.image_dir = os.path.join(root, "images")
        self.gsd = gsd
        self.transform = get_rsicd_transforms(train)

        #loading the  annotations
        json_path = os.path.join(root, "dataset_rsicd.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        #building the  flat list of (filename, caption) pairs
        #each image has 5 captions .... we use all of them as separate samples
        all_pairs = []
        for item in data["images"]:
            filename = item["filename"]
            for sentence in item["sentences"]:
                caption = sentence["raw"].strip()
                if caption:
                    all_pairs.append((filename, caption))

        #train and  val split
        total = len(all_pairs)
        split_idx = int(total * train_split)

        # Deterministic shuffle before split
        torch.manual_seed(42)
        indices = torch.randperm(total).tolist()
        all_pairs = [all_pairs[i] for i in indices]

        self.pairs = all_pairs[:split_idx] if train else all_pairs[split_idx:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]
        clean_name=os.path.basename(filename)
        img_path = os.path.join(self.image_dir, clean_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,     # [3, 224, 224]
            "caption": caption,  # raw string
            "gsd": torch.tensor(self.gsd, dtype=torch.float32),
        }


def get_rsicd_dataloader(
    root: str = "data/RSICD",
    train: bool = True,
    batch_size: int = 16,
    num_workers: int = 2,
) -> DataLoader:
    dataset = RSICDDataset(root=root, train=train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    print("Testing RSICD dataset structure (no download needed)...")

    # Test transform pipeline with a dummy image
    dummy_image = torch.randn(3, 224, 224)
    print(f"Dummy image shape: {dummy_image.shape}")

    transform = get_rsicd_transforms(train=True)
    from PIL import Image
    import numpy as np
    dummy_pil = Image.fromarray(
        (torch.randint(0, 255, (224, 224, 3)).numpy()).astype("uint8")
    )
    out = transform(dummy_pil)
    print(f"Transform output:  {out.shape}")  # type: ignore
    assert out.shape == (3, 224, 224), "Shape mismatch!" #type: ignore
    print("RSICD transform check passed.")

    print("\nTo test the full dataloader, download RSICD and run:")
    print("  loader = get_rsicd_dataloader('data/RSICD', train=True)")