import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoder.hybrid_encoder import HybridEncoder
from projector.mlp_projector import MLPProjector
from model.lora_config import get_lora_config

class RSVLM(nn.Module):
    """
    the full pipeline: 
    hybrid enc -> MLP proj -> Frozen LLM (lora)

    the visual tokens are prepended to the text tokens and passed into the llm
    the llm generates the output logits autoregressively

    the trainables are:
    1. phase 1: ther encoder only (MAE pretraining)
    2. phase 2: the proj only (caption alignment)
    3. proj + lora (instruction tuning
    """

    def __init__(
            self,
            llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            enc_dim=384,
            llm_dim=2048,
            cnn_pretrained=True,
            use_lora=True,
            lora_r=8,
    ):
        super().__init__()

        #stage1: vision encoder
        self.encoder=HybridEncoder(
            cnn_pretrained=cnn_pretrained,
            embed_dim=enc_dim
        )

        # stag2: projector 
        self.projector=MLPProjector(
            enc_dim=enc_dim,
            llm_dim=llm_dim
        )

        #stage3: llm backbone
        print(f"loading llm : {llm_name}")
        self.llm=AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16
        )

        #freeze all llm params
        for param in self.llm.parameters():
            param.requires_grad=False

        #apply lora on atn layer
        if use_lora:
            lora_cfg=get_lora_config(r=lora_r)
            self.llm=get_peft_model(self.llm, lora_cfg)

        #tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

        self.llm_dim=llm_dim

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_projector(self):
        for param in self.projector.parameters():
            param.requires_grad = False

    def unfreeze_projector(self):
        for param in self.projector.parameters():
            param.requires_grad = True

    def freeze_encoder_expect_gsd(self):
        #freeze CNN stem
        for param in self.encoder.cnn_stem.parameters():
            param.requires_grad=False
        #freeze vit body
        for param in self.encoder.vit_body.parameters():
            param.requires_grad=False

    def forward(
            self,
            images,
            input_ids,
            attention_mask,
            gsd=None,
            labels=None
    ):
        """
        Args:
            images shape [b,3,224,224]
            input_ids shape [b,seq len]
            attn_mask shape [B, seq_len]
            gsd [B]
            labels [B,seq len]

        Retuens:
            loss if labels are provided else returns the logits
        """

        #step1: encode the image into vsual tokens
        visual_tokens=self.encoder(images, gsd) #[b,785,384]
        visual_tokens=self.projector(visual_tokens) #[b,785,2048]

        #sterp 2: get the text embeds form the LLM embed layer
        embed_layer=self.llm.get_input_embeddings() #type:ignore
        text_embeds=embed_layer(input_ids)

        #step 3: prepend the visual tokens to text embeds
        input_embeds=torch.cat([visual_tokens, text_embeds], dim=1) #[b, 785+seqw_len, 2048]

        #step4: extend the attn mask to cover visual tokens
        visual_mask=torch.ones(images.shape[0], visual_tokens.shape[1], device=attention_mask.device)
        full_mask=torch.cat([visual_mask, attention_mask], dim=1)

        #step5: extend the labels if provided and mask visual tokens pos with -100
        if labels is not None:
            visual_labels=torch.full(
                (images.shape[0], visual_tokens.shape[1]), fill_value=-100, device=labels.device, dtype=labels.dtype
            )
            labels=torch.cat([visual_labels, labels], dim=1)

        #step6: forward through llm
        outputs=self.llm(
            inputs_embeds=input_embeds,
            attention_mask=full_mask,
            labels=labels
        )

        return outputs
    
    def get_param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Trainable %:      {100 * trainable / total:.2f}%")

if __name__=="__main__":
    print("Building RSVLM....")
    model=RSVLM(cnn_pretrained=False)
    model.get_param_summary()

    #dummy fw pass
    B=2
    images=torch.randn(B,3,224,224)
    gsd=torch.tensor([10.0,0.3])

    #tokenize a dummy question
    questions=["What land use is shown?", "How many buildings are visible?"]
    tokens=model.tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )

    print(f"\nImage input:    {images.shape}")
    print(f"Text input_ids: {tokens['input_ids'].shape}")

    outputs = model(
        images=images,  
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        gsd=gsd,    
    )

    print(f"LLM output logits: {outputs.logits.shape}")
    print("\nRS-VLM full pipeline check passed.")


