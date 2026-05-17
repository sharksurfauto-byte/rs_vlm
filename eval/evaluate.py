import torch
import os
import sys
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rs_vlm import RSVLM
from data.eurosat import EUROSAT_CLASSES, get_eurosat_dataloader
from data.rsicd import get_rsicd_dataloader
from data.vqa_templates import format_prompt
from eval.metrics import CaptioningMetrics, VQAMetrics

def load_vlm(checkpoint_path: str, device: torch.device):
    model = RSVLM(cnn_pretrained=False, use_lora=True).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model

@torch.no_grad()
def evaluate_vqa(model, loader, device):
    metrics = VQAMetrics(EUROSAT_CLASSES)
    print(f"Evaluating VQA on {len(loader.dataset)} samples...")
    
    for batch in tqdm(loader):
        images = batch['image'].to(device)
        gsd = batch['gsd'].to(device)
        ground_truths = batch['class_name']
        
        # Prepare Batch Prompt
        questions = ["What type of land use is shown in this satellite image?"] * images.shape[0]
        prompts = [format_prompt(q, answer=None) for q in questions]
        
        encoded = model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            # 1. Vision Tokens
            visual_tokens = model.encoder(images, gsd)
            visual_tokens = model.projector(visual_tokens)
            
            # 2. Text Embeds
            embed_layer = model.llm.get_input_embeddings()
            text_embeds = embed_layer(encoded["input_ids"])
            
            # 3. Cat
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            
            # 4. Mask
            visual_mask = torch.ones(images.shape[0], visual_tokens.shape[1], device=device)
            full_mask = torch.cat([visual_mask, encoded["attention_mask"]], dim=1)
            
            # 5. Generate
            outputs = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=16,
                temperature=0.0,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )
            
        predictions = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        metrics.update(predictions, ground_truths)
        
    return metrics.compute()

@torch.no_grad()
def evaluate_captioning(model, loader, device):
    metrics = CaptioningMetrics()
    gts = {}
    res = {}
    
    print(f"Evaluating Captioning on {len(loader.dataset)} samples...")
    
    img_id = 0
    for batch in tqdm(loader):
        images = batch['image'].to(device)
        gsd = batch['gsd'].to(device)
        batch_captions = batch['caption']
        
        # RSICD loader returns one caption per sample, but images are repeated.
        # pycocoevalcap needs all refs for an image id.
        # For simplicity in this eval script, we treat each sample as a unique image ID 
        # because the loader might shuffle.
        
        prompts = ["<|user|>\nDescribe this satellite image.</s>\n<|assistant|>\n"] * images.shape[0]
        
        encoded = model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            visual_tokens = model.encoder(images, gsd)
            visual_tokens = model.projector(visual_tokens)
            embed_layer = model.llm.get_input_embeddings()
            text_embeds = embed_layer(encoded["input_ids"])
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            visual_mask = torch.ones(images.shape[0], visual_tokens.shape[1], device=device)
            full_mask = torch.cat([visual_mask, encoded["attention_mask"]], dim=1)
            
            outputs = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=64,
                temperature=0.2,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )
            
        predictions = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for p, gt in zip(predictions, batch_captions):
            gts[img_id] = [gt]
            res[img_id] = [p]
            img_id += 1
            
    return metrics.compute(gts, res)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["vqa", "caption", "both"], default="both")
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vlm(args.checkpoint, device)
    
    results = {}
    
    if args.task in ["vqa", "both"]:
        loader = get_eurosat_dataloader(root=os.path.join(args.data_root, "EuroSAT"), train=False, batch_size=32)
        results["vqa"] = evaluate_vqa(model, loader, device)
        
    if args.task in ["caption", "both"]:
        loader = get_rsicd_dataloader(root=os.path.join(args.data_root, "RSICD"), train=False, batch_size=32)
        results["captioning"] = evaluate_captioning(model, loader, device)
        
    print("\n--- Final Evaluation Results ---")
    print(json.dumps(results, indent=4))
    
    with open("eval_results.json", "w") as f:
        json.dump(results, indent=4)
