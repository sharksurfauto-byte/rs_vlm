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
    filename: str | None = None,
):
    """Save model + optimizer state to disk."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch{epoch:03d}.pt"

    path = os.path.join(checkpoint_dir, filename)

    # Handle DataParallel
    model_to_save = model.module if hasattr(model, "module") else model

    torch.save({
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

    print(f"Checkpoint saved: {path}")

    # --- Disk Cleanup to prevent 'No space left on device' (Kaggle 20GB limit) ---
    import glob
    if "step" in filename:
        step_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "*_step*.pt")))
        while len(step_ckpts) > 1:  # Keep only the latest 1 step checkpoint
            os.remove(step_ckpts.pop(0))
    elif "epoch" in filename:
        epoch_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "*_epoch*.pt")))
        while len(epoch_ckpts) > 1:  # Keep only the latest 1 epoch checkpoint
            os.remove(epoch_ckpts.pop(0))
            
    return path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[int, float]:
    """Load checkpoint — returns (epoch, loss)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(ckpt["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["loss"]


def get_device() -> torch.device:
    """Returns CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        print(f"  Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
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
            return (epoch + 1) / max(1, warmup_epochs)
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