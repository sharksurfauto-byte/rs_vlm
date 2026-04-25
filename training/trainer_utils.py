import torch
import os
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load a yaml config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: str = None,
):
    """Save model + optimizer state to disk."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch{epoch:03d}.pt"

    path = os.path.join(checkpoint_dir, filename)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint — returns (epoch, loss)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["loss"]


def get_device() -> torch.device:
    """Returns CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def get_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
):
    """
    Linear warmup then cosine decay scheduler.
    Warmup prevents large early updates from destabilising pretrained weights.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    print("Testing trainer utils...")

    # Test config loading
    cfg = load_config("configs/colab_config.yaml")
    print(f"Loaded config — phase1 batch_size: {cfg['phase1']['batch_size']}")

    # Test device detection
    device = get_device()

    # Test scheduler
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=30)
    print(f"Scheduler created: {scheduler}")

    print("\nTrainer utils check passed.")