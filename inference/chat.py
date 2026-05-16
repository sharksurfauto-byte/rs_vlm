import torch
import sys
import os
from PIL import Image
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.rs_vlm import RSVLM
from data.vqa_templates import format_prompt

def load_vlm(checkpoint_path: str, device: torch.device):
    print("Loading TinyLlama and initializing RS-VLM architecture...")
    # Initialize with LoRA=True because Phase 3 trained LoRA weights!
    model = RSVLM(cnn_pretrained=False, use_lora=True).to(device)
    
    print(f"Loading weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # We use strict=False because the checkpoint only contains the trained parameters
    # (Encoder, Projector, LoRA). The base TinyLlama weights are loaded from HuggingFace.
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    model.eval()
    print("Model ready!")
    return model

def chat_with_image(model: RSVLM, image_path: str, question: str, gsd: float = 10.0):
    device = next(model.parameters()).device
    
    # 1. Prepare Image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.3444, 0.3803, 0.4078], 
            std=[0.2038, 0.1367, 0.1152]
        ),
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    gsd_tensor = torch.tensor([gsd], dtype=torch.float32).to(device)
    
    # 2. Prepare Text Prompt
    # Note: We set answer=None because we want the model to generate the answer!
    prompt = format_prompt(question, answer=None)
    
    encoded = model.tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # 3. Generate!
    print(f"\nUser: {question}")
    print("Assistant: ", end="", flush=True)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            # 1. Get visual tokens
            visual_tokens = model.encoder(img_tensor, gsd_tensor)
            visual_tokens = model.projector(visual_tokens)
            
            # 2. Get text embeddings
            embed_layer = model.llm.get_input_embeddings()
            text_embeds = embed_layer(input_ids)
            
            # 3. Concatenate them
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            
            # 4. Extend the attention mask
            visual_mask = torch.ones(img_tensor.shape[0], visual_tokens.shape[1], device=device)
            full_mask = torch.cat([visual_mask, attention_mask], dim=1)
            
            # 5. Generate
            outputs = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=32,
                temperature=0.2, # Low temperature for factual VQA
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )
            
    # Decode only the generated tokens
    answer = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)
    return answer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # UPDATE THIS PATH AFTER DOWNLOADING
    CHECKPOINT = "training\checkpoints\model_phase3_final.pt"
    
    if not os.path.exists(CHECKPOINT):
        print(f"Error: Could not find checkpoint at {CHECKPOINT}")
        sys.exit(1)
        
    model = load_vlm(CHECKPOINT, device)
    
    TEST_PATH = r"inference\test_images"  
    
    if os.path.isdir(TEST_PATH):
        print(f"\n--- Batch Testing images in {TEST_PATH} ---")
        for file in os.listdir(TEST_PATH):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(TEST_PATH, file)
                print(f"\n[{file}]")
                chat_with_image(
                    model, 
                    image_path=img_path, 
                    question="What type of land use is shown in this satellite image?",
                    gsd=10.0
                )
    elif os.path.exists(TEST_PATH):
        chat_with_image(
            model, 
            image_path=TEST_PATH, 
            question="What type of land use is shown in this satellite image?",
            gsd=10.0
        )
    else:
        print(f"\nPlease create a folder '{TEST_PATH}' and put some test images in it!")
