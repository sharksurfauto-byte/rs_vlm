import torch
import os
import sys
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator

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
def evaluate_vqa(model, loader, accelerator):
    device = accelerator.device
    metrics = VQAMetrics(EUROSAT_CLASSES)
    if accelerator.is_main_process:
        print(f"Evaluating VQA on {len(loader.dataset)} samples...")
    
    for batch in tqdm(loader, disable=not accelerator.is_main_process):
        images = batch['image']
        gsd = batch['gsd']
        ground_truths = batch['class_name']
        prompts = [format_prompt("What type of land use is shown in this satellite image?", answer=None)] * images.shape[0]
        
        encoded = model.module.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            visual_tokens = model.module.encoder(images, gsd)
            visual_tokens = model.module.projector(visual_tokens)
            embed_layer = model.module.llm.get_input_embeddings()
            text_embeds = embed_layer(encoded["input_ids"])
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            visual_mask = torch.ones(images.shape[0], visual_tokens.shape[1], device=device)
            full_mask = torch.cat([visual_mask, encoded["attention_mask"]], dim=1)
            outputs = model.module.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=16,
                temperature=0.0,
                do_sample=False,
                pad_token_id=model.module.tokenizer.pad_token_id,
                eos_token_id=model.module.tokenizer.eos_token_id,
            )
        predictions = model.module.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds_gathered = accelerator.gather_for_metrics(predictions)
        gts_gathered = accelerator.gather_for_metrics(ground_truths)
        if accelerator.is_main_process:
            metrics.update(preds_gathered, gts_gathered)
    return metrics.compute() if accelerator.is_main_process else None

@torch.no_grad()
def evaluate_captioning(model, loader, accelerator):
    device = accelerator.device
    metrics = CaptioningMetrics()
    all_gts = []
    all_res = []
    if accelerator.is_main_process:
        print(f"Evaluating Captioning on {len(loader.dataset)} samples...")
    
    for batch in tqdm(loader, disable=not accelerator.is_main_process):
        images = batch['image']
        gsd = batch['gsd']
        batch_captions = batch['caption']
        prompts = ["<|user|>\nDescribe this satellite image.</s>\n<|assistant|>\n"] * images.shape[0]
        
        encoded = model.module.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            visual_tokens = model.module.encoder(images, gsd)
            visual_tokens = model.module.projector(visual_tokens)
            embed_layer = model.module.llm.get_input_embeddings()
            text_embeds = embed_layer(encoded["input_ids"])
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            visual_mask = torch.ones(images.shape[0], visual_tokens.shape[1], device=device)
            full_mask = torch.cat([visual_mask, encoded["attention_mask"]], dim=1)
            outputs = model.module.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=64,
                temperature=0.2,
                do_sample=False,
                pad_token_id=model.module.tokenizer.pad_token_id,
                eos_token_id=model.module.tokenizer.eos_token_id,
            )
        predictions = model.module.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds_gathered = accelerator.gather_for_metrics(predictions)
        gts_gathered = accelerator.gather_for_metrics(batch_captions)
        if accelerator.is_main_process:
            all_res.extend(preds_gathered)
            all_gts.extend(gts_gathered)
            
    if accelerator.is_main_process:
        gts_dict = {i: [gt] for i, gt in enumerate(all_gts)}
        res_dict = {i: [p] for i, p in enumerate(all_res)}
        # Save raw predictions before potentially crashing on metrics
        with open("raw_captions.json", "w") as f:
            json.dump({"gts": gts_dict, "res": res_dict}, f)
        return metrics.compute(gts_dict, res_dict)
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["vqa", "caption", "both"], default="both")
    parser.add_argument("--eurosat_path", type=str, default=None)
    parser.add_argument("--rsicd_path", type=str, default=None)
    args = parser.parse_args()
    
    accelerator = Accelerator()
    device = accelerator.device
    model = load_vlm(args.checkpoint, device)
    
    results = {}
    if args.task in ["vqa", "both"] and args.eurosat_path:
        loader = get_eurosat_dataloader(root=args.eurosat_path, train=False, batch_size=16)
        model_prepared, loader_prepared = accelerator.prepare(model, loader)
        results["vqa"] = evaluate_vqa(model_prepared, loader_prepared, accelerator)
        
    if args.task in ["caption", "both"] and args.rsicd_path:
        loader = get_rsicd_dataloader(root=args.rsicd_path, train=False, batch_size=16)
        # Handle re-prepare gracefully
        loader_prepared = accelerator.prepare(loader)
        results["captioning"] = evaluate_captioning(model_prepared if 'model_prepared' in locals() else model, loader_prepared, accelerator)
        
    if accelerator.is_main_process:
        print("\n--- Final Evaluation Results ---")
        final_results = {k: v for k, v in results.items() if v is not None}
        print(json.dumps(final_results, indent=4))
        with open("eval_results.json", "w") as f:
            json.dump(final_results, f, indent=4)
