from peft import LoraConfig, TaskType

def get_lora_config(
        r:int=8, # rank
        lora_alpha:int=16, # this is the scaling factor for lora updates
        lora_dropout:float=0.05,
        ) -> LoraConfig:
    #this only targes the attn proj matrcis - q,k,v, output as these are the most impacful for the model performance
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        inference_mode=False
    )

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model
    import torch

    print("loading tiny llama...")
    model=AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
    )

    total_before=sum(p.numel() for p in model.parameters())

    config=get_lora_config()
    model=get_peft_model(model,config)

    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable %: {trainable/total*100:.2f}%")
    print("lora config test passed!!")