import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rs_vlm import RSVLM
from data.vqa_templates import EuroSATVQADataset
from data.rsicd import RSICDInstructionDataset
from torch.utils.data import DataLoader, ConcatDataset
from training.trainer_utils import(
    load_config, save_checkpoint, load_checkpoint, get_device, get_warmup_scheduler
)

def train_phase3(
    config_path: str="configs/colab_config.yaml",
    phase2_checkpoint: str|None=None,
    resume_from: str|None=None
):
    cfg=load_config(config_path)
    p3=cfg['phase3']
    device=get_device()

    #Model
    """
    Phase 3 uses LoRA...re-enable it here
    We load the full model with Lora and then load phase 2 weghits into the encoder and the projector
    Lora params start from random init as they havent been trained yet
    """

    model=RSVLM(
        llm_name=cfg["model"]["llm_name"],
        enc_dim=cfg["model"]["encoder_dim"],
        llm_dim=cfg["model"]["llm_dim"],
        cnn_pretrained=cfg["model"]["cnn_pretrained"],
        use_lora=True,            # LoRA back on for Phase 3
        lora_r=cfg["model"]["lora_r"],
    ).to(device)

    #laod phase 2 checkpoint
    """
    Phase 2 saved the full model state(encoder, projector and the frozen llm)
    we load it with the strict=False coz phase 2 model had no lora layers but phase 3 does. 
    the extra lora keys are simply ignored on load and stay random init
    """

    if phase2_checkpoint:
        ckpt=torch.load(phase2_checkpoint, map_location=device)
        missing,unexpected=model.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        print(f"Loaded Phase 2 checkpoint: {phase2_checkpoint}")
        print(f"  Missing keys (LoRA — expected): {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

    #gradient checkpointing
    """
    same as phase 2- needed to keep the LLM activations form filling the VRAM. Even thouigh Lora params are traininable, the base LLM layers are frozen and their activations still need to be recomputed during backward
    """
    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False   # required for gradient checkpointing
    print("gradient checkpointing enabled on llm")

    # Freeze unfreeze strategy
    """
    Encoder: freeze CNN stem and ViT body, keep GSD adapter trainable
    LLM: freeze base weights, only LoRA params train (handled by PEFT)
    Projector: trainable — continues to refine from Phase 2
    """
    model.freeze_encoder_expect_gsd()
    model.unfreeze_projector()

    # Multi-GPU Support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    #PEFT automatically keeps only lora params trainable in the llm
    # we verify this by confirming the trainiable params count
    # Note: access .module if wrapped in DataParallel
    model_internal = model.module if hasattr(model, "module") else model
    trainable = sum(p.numel() for p in model_internal.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_internal.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%) — projector + GSD adapter + LoRA")
    
    #optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(p3["lr"]),
        weight_decay=float(p3["weight_decay"]),
    )

    scheduler = get_warmup_scheduler(
        optimizer, p3["warmup_epochs"], p3["epochs"]
    )

    #data loader
    eurosat_dataset=EuroSATVQADataset(
        eurosat_root=cfg["data"]['eurosat_root'],
        train=True,
        tokenizer=model_internal.tokenizer,
        max_length=p3['max_length'],
    )
    
    rsicd_dataset=RSICDInstructionDataset(
        root=cfg["data"]['rsicd_root'],
        train=True,
        tokenizer=model_internal.tokenizer,
        max_length=p3['max_length'],
    )
    
    # Mix datasets (ConcatDataset interleaves them if shuffled)
    dataset = ConcatDataset([eurosat_dataset, rsicd_dataset])
    
    loader=DataLoader(
        dataset,
        batch_size=p3['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
    )

    #resume 
    start_epoch = 0
    if resume_from:
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from, device)

    global_step = start_epoch * len(loader)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") #mixed precision

    #training loop
    print(f"\nPhase 3 — LoRA instruction tuning")
    print(f"Epochs: {p3['epochs']}  |  Batch: {p3['batch_size']}  |  LR: {p3['lr']}")
    print(f"Dataset size: {len(dataset)} samples\n")

    save_steps=500
    print_steps=50

    avg_loss=0.0
    for epoch in range(start_epoch, p3['epochs']):
        model.train()
        # Ensure encoder stays in eval (access via .module if DP)
        if hasattr(model, "module"):
            model.module.encoder.eval()
        else:
            model.encoder.eval()

        epoch_loss=0.0
        num_batches=0

        for batch in loader:
            images=batch['image'].to(device)
            gsd=batch['gsd'].to(device)
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            #labels
            labels=batch['labels'].to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type=='cuda',
                dtype=torch.float16
            ):
                outputs=model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gsd=gsd,
                    labels=labels,
                )
                loss = outputs.loss
                # If using DataParallel, loss might be a vector of losses per GPU
                if loss.dim() > 0:
                    loss = loss.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p:p.requires_grad, model.parameters()),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            epoch_loss+=loss.item()
            num_batches+=1
            global_step+=1

            if global_step % print_steps == 0:
                print(f"  Epoch {epoch+1} | Step {global_step} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            if global_step % save_steps == 0:
                save_checkpoint(
                    model, optimizer, epoch + 1, epoch_loss / num_batches,
                    p3["checkpoint_dir"],
                    filename=f"model_step{global_step}.pt"
                )
                print(f"  [Checkpoint saved at step {global_step}]")

        scheduler.step()
        avg_loss=epoch_loss/num_batches
        print(f"Epoch [{epoch+1:03d}/{p3['epochs']}]  "
              f"Loss: {avg_loss:.4f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        save_checkpoint(
            model, optimizer, epoch + 1, avg_loss,
            p3["checkpoint_dir"],
            filename=f"model_epoch{epoch+1:03d}.pt"
        )
    save_checkpoint(
        model, optimizer, p3["epochs"], avg_loss,
        p3["checkpoint_dir"],
        filename="model_phase3_final.pt"
    )
    print("\nPhase 3 complete")
    return model

if __name__ == "__main__":
    print("Running Phase 3 smoke test (CPU, dummy data)...")

    device = torch.device("cpu")

    model = RSVLM(cnn_pretrained=False, use_lora=True)
    model.freeze_encoder_expect_gsd()
    model.unfreeze_projector()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}  (projector + GSD adapter + LoRA)")

    # Dummy VQA batch — simulate what EuroSATVQADataset produces
    B = 2
    images = torch.randn(B, 3, 224, 224)
    gsd = torch.tensor([10.0, 10.0])

    questions = [
        "<|user|>\nWhat type of land use is shown?\n</s>\n<|assistant|>\nThis is a forest.</s>",
        "<|user|>\nWhat does this satellite image show?\n</s>\n<|assistant|>\nThis is a highway.</s>",
    ]

    encoded = model.tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        gsd=gsd,
        labels=labels,
    )

    print(f"Loss:   {outputs.loss.item():.4f}")
    print(f"Logits: {outputs.logits.shape}")
    print("\nPhase 3 smoke test passed.")