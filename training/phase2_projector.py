import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rs_vlm import RSVLM
from data.rsicd import get_rsicd_dataloader
from training.trainer_utils import (
    load_config, save_checkpoint, load_checkpoint,
    get_device, get_warmup_scheduler
)

def train_phase2(
        config_path = "configs/colab_config.yaml",  # fixed: was "config/" (missing s)
        phase1_checkpoint:str|None=None,
        resume_from:str|None=None,
):
    cfg=load_config(config_path)
    p2=cfg["phase2"]
    device = get_device()  # fixed: was hardcoded torch.device('cpu') — would skip GPU on Colab

    #model
    model=RSVLM(
        llm_name=cfg["model"]["llm_name"],
        enc_dim=cfg["model"]["encoder_dim"],
        llm_dim=cfg["model"]["llm_dim"],
        cnn_pretrained=cfg["model"]["cnn_pretrained"],
        use_lora=False,   # LLM fully frozen in Phase 2, LoRA not needed
        lora_r=cfg["model"]["lora_r"],
    ).to(device)

    #load p1 encoder weights if avail
    if phase1_checkpoint:
        ckpt=torch.load(phase1_checkpoint, map_location=device)
        model.encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded Phase 1 checkpoint: {phase1_checkpoint}")

    # gradient checkpointing on LLM — critical for T4
    # gradients flow THROUGH the frozen LLM back to the projector, so without this
    # all 22 layer activations stay in VRAM → OOM. checkpointing recomputes them during backward
    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False
    print("Gradient checkpointing enabled on LLM")

    # freeze CNN stem + ViT body but keep GSDAdapter trainable
    model.freeze_encoder_expect_gsd()
    for param in model.llm.parameters():
        param.requires_grad=False

    model.unfreeze_projector()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} (projector + gsd adapter)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(p2["lr"]),                    # cast — yaml.dump can serialize 1e-4 as a string
        weight_decay=float(p2["weight_decay"])
    )

    scheduler=get_warmup_scheduler(
        optimizer, 
        p2["warmup_epochs"],
        p2["epochs"]
    )

    #dataloader
    loader = get_rsicd_dataloader(
        root=cfg["data"]["rsicd_root"],
        train=True,
        batch_size=p2["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    #ressume
    start_epoch=0
    if resume_from:
        start_epoch,_ = load_checkpoint(model, optimizer, resume_from, device)
        
    global_step = start_epoch * len(loader)  # derive from epoch after loader is built

    #scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    #trainingloop
    print(f"\nPhase 2 — Projector alignment")
    print(f"Epochs: {p2['epochs']}  |  Batch: {p2['batch_size']}  |  LR: {p2['lr']}")
    print(f"Dataset size: {len(loader.dataset)} samples\n")
    
    save_steps = 1000 # Save checkpoint every 1000 batches
    print_steps = 100 # Print progress every 100 batches
    
    avg_loss=0.0
    for epoch in range(start_epoch, p2["epochs"]):
        model.train()
        model.encoder.eval()
        epoch_loss=0.0
        num_batches=0
        for batch in loader:
            images=batch["image"].to(device) #[B,3,224,224]
            gsd=batch["gsd"].to(device) #[B]
            captions=batch["caption"]

            encoded=model.tokenizer(
                captions,
                return_tensors="pt",  # fixed: was return_tensor (missing s) — would crash on .to(device)
                padding=True,
                truncation=True,
                max_length=64,
            )

            input_ids=encoded["input_ids"].to(device)
            attention_mask=encoded["attention_mask"].to(device)

            #labels-> input_ids (causla lm loss)
            labels=input_ids.clone()
            labels[attention_mask == 0] = -100  # ignore padding in loss
            
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                outputs=model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gsd = gsd,
                    labels=labels
                )
                loss=outputs.loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p :p.requires_grad, model.parameters()), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            if global_step % print_steps == 0:
                print(f"  Epoch {epoch+1} | Step {global_step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                
            if global_step % save_steps == 0:
                save_checkpoint(
                    model, optimizer, epoch + 1, epoch_loss/num_batches,
                    p2["checkpoint_dir"],
                    filename=f"model_step{global_step}.pt"
                )
                print(f"  [Checkpoint saved at step {global_step}]")

        scheduler.step()
        avg_loss=epoch_loss/num_batches

        print(f"Epoch [{epoch+1:03d}/{p2['epochs']}]  "
              f"Loss: {avg_loss:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Still save at end of epoch just in case
        save_checkpoint(
            model, optimizer, epoch + 1, avg_loss,
            p2["checkpoint_dir"],
            filename=f"model_epoch{epoch+1:03d}.pt"
        )

    save_checkpoint(
        model, optimizer, p2["epochs"], avg_loss,
        p2["checkpoint_dir"],
        filename="model_phase2_final.pt"
    )
    print("\nPhase 2 complete.")
    return model

if __name__ == "__main__":
    print("Running Phase 2 smoke test (CPU, dummy data)...")

    device = get_device()

    model = RSVLM(cnn_pretrained=False, use_lora=False).to(device)
    model.freeze_encoder_expect_gsd()  # keep GSDAdapter trainable
    for param in model.llm.parameters():
        param.requires_grad = False
    model.unfreeze_projector()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}  (projector + GSD adapter)")

    # Dummy batch
    images = torch.randn(2, 3, 224, 224).to(device)
    gsd = torch.tensor([10.0, 0.3]).to(device)
    captions = [
        "A satellite image showing a dense residential area.",
        "An aerial view of a forested region with a river.",
    ]

    encoded = model.tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()

    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        gsd=gsd,
        labels=labels,
    )

    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits: {outputs.logits.shape}")
    print("\nPhase 2 smoke test passed.")