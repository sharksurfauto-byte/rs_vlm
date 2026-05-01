# This is Phase1 of the pretraining
# only the vision encoder is trained , the LLM and the proj are frozen
# we feed the image through CNN stem and get feat map
# randomly mask 75% of the patches
# train the vit to reconstruct the masked patches
# no labels are needed . this is fully supervised on EuroSAT's 27k imgs

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder.hybrid_encoder import HybridEncoder
from data.eurosat import get_eurosat_dataloader, mae_mask_patches
# get_mae_transforms not needed here — eurosat dataloader handles its own transforms
from training.trainer_utils import(
    load_config, save_checkpoint, load_checkpoint,
    get_device, get_warmup_scheduler
)

class MAEDecoder(nn.Module):
    # this is a lightweight decoder for MAE pretraining. this reconstructs the masked CNN feat map patches from ViT tokens
    # x shape: [B,785,384] -vit tokens
    # output shape : [B,256,28,28] reconstructed feat map

    def __init__(self, embed_dim=384, out_channels=256):
        super().__init__()
        self.decoder=nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, out_channels) 
        )

    def forward(self, tokens:torch.Tensor) -> torch.Tensor:
        # drop the CLS token, keep the patches only
        patch_tokens=tokens[:,1:,:] #[B,784,384]
        reconstructed = self.decoder(patch_tokens) #[B,784,256]

        B,N,C = reconstructed.shape
        H=W=int(N**0.5) #28
        reconstructed = reconstructed.transpose(1,2).view(B,C,H,W)
        return reconstructed #[B,256,28,28]
    
def mae_loss(
        original,
        reconstructed,
        mask,
)->torch.Tensor:
    # this is the mae loss computed only on masked patches
    # we only penalise reconstruction of the patches the mdoel couldnt see= computing loss on visiable patches would be very ezzz lol

    #original and reconstructed shape: [B,256,28,28]
    # mask shape = [B,28,28] where 1=masked and 0=visible

    mask_expanded = mask.unsqueeze(1).expand_as(original) #[B,256,28,28]
    diff = (original - reconstructed) ** 2
    # mean over masked pos only
    loss=(diff*mask_expanded).sum() / mask_expanded.sum()
    return loss

def train_phase1(config_path:str = "configs/colab_config.yaml",
                 resume_from:str|None = None):
    cfg = load_config(config_path)
    p1 = cfg["phase1"]
    device=get_device()

    #model
    encoder = HybridEncoder(
        cnn_pretrained=cfg["model"]["cnn_pretrained"],
        embed_dim=cfg["model"]["encoder_dim"],  # encoder_dim only lives under [model] in the yaml
        num_blocks=cfg["encoder"]["num_blocks"],
        num_heads=cfg["encoder"]["num_heads"],
        mlp_ratio=cfg["encoder"]["mlp_ratio"],
        dropout=cfg["encoder"]["dropout"],
    ).to(device)

    decoder = MAEDecoder(
        embed_dim=cfg["model"]["encoder_dim"],
        out_channels=256
    ).to(device)

    #only enc and dec are trainable in p1
    params= list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=p1["lr"], weight_decay=p1["weight_decay"])

    scheduler = get_warmup_scheduler(optimizer, p1["warmup_epochs"], p1["epochs"])
    
    # resume from checkpoint if provided
    start_epoch=0
    if resume_from:
        start_epoch,_ = load_checkpoint(encoder, optimizer, resume_from, device)

    #dataloader
    loader = get_eurosat_dataloader(
        root = cfg["data"]["eurosat_root"],
        train=True,
        batch_size=p1["batch_size"],
        num_workers= cfg["data"]["num_workers"], # how many sub processes to use for retrieval of data
    )

    #training loop
    print(f"\nPhase 1 - MAE pretraining")
    print(f"Epochs: {p1['epochs']}, Batch size: {p1['batch_size']}, LR: {p1['lr']}")
    print(f"Dataset size: {len(loader)} images\n")

    avg_loss = 0.0
    for epoch in range(start_epoch, p1["epochs"]):
        encoder.train()
        decoder.train()
        epoch_loss=0.0
        num_batches=0
        for batch in loader:
            images = batch["image"].to(device) #[B,3,224,224]
            gsd = batch["gsd"].to(device) #[B]
            #step1: get cnn feat map
            feature_map = encoder.cnn_stem(images) #[B,256,28,28]
            #step2: mask 75% patches
            masked_map, mask=mae_mask_patches(
                feature_map,
                mask_ratio=p1["mask_ratio"]
            )
            #step3: pass masked map through vit body — gsd_bias goes in as an arg now, no param mutation
            gsd_bias = encoder.gsd_adapter(gsd)                         #[B,1,384]
            tokens = encoder.vit_body(masked_map, gsd_bias=gsd_bias)    #[B,785,384]
            #step4:: decode and compute the loss oon the masked pathces only
            reconstructed = decoder(tokens) #[B,256,28,28]
            loss=mae_loss(feature_map, reconstructed, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        scheduler.step()
        avg_loss = epoch_loss/num_batches

        print(f"Epoch [{epoch+1}/{p1['epochs']}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        #save checkpoint
        if (epoch+1)%p1["save_every"]==0:
            save_checkpoint(
                encoder, optimizer, epoch+1, avg_loss,
                p1["checkpoint_dir"],
                filename=f"encoder_epoch{epoch+1:03d}.pt"
            )

    #save fginal encoder
    save_checkpoint(
        encoder, optimizer, p1["epochs"], avg_loss,
        p1["checkpoint_dir"],
        filename=f"encoder_final.pt"
    )

    print("\nPhase 1 complete.")
    return encoder

if __name__ == "__main__":
    # Smoke test with dummy data (no dataset needed)
    print("Running Phase 1 smoke test (CPU, dummy data)...")

    device = get_device()

    encoder = HybridEncoder(cnn_pretrained=False).to(device)
    decoder = MAEDecoder().to(device)

    dummy_images = torch.randn(2, 3, 224, 224).to(device)
    dummy_gsd = torch.tensor([10.0, 0.3]).to(device)

    # One forward pass
    feature_map = encoder.cnn_stem(dummy_images)
    masked_map, mask = mae_mask_patches(feature_map, mask_ratio=0.75)

    gsd_bias = encoder.gsd_adapter(dummy_gsd)                      #[B,1,384]
    tokens = encoder.vit_body(masked_map, gsd_bias=gsd_bias)        #[B,785,384]

    reconstructed = decoder(tokens)
    loss = mae_loss(feature_map.detach(), reconstructed, mask)

    print(f"Feature map:    {feature_map.shape}")
    print(f"Masked map:     {masked_map.shape}")
    print(f"Tokens:         {tokens.shape}")
    print(f"Reconstructed:  {reconstructed.shape}")
    print(f"MAE loss:       {loss.item():.4f}")
    assert reconstructed.shape == feature_map.shape, "Shape mismatch!"
    print("\nPhase 1 smoke test passed.")