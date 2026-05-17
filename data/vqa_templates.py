import random
import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.eurosat import EUROSAT_CLASSES

# Mapping classes to visual attributes to enrich the training data
CLASS_ATTRIBUTES = {
    "Annual Crop": "geometric field patterns and seasonal vegetation signatures",
    "Forest": "dense canopy cover and varied green spectral responses typical of woodland",
    "Herbaceous Vegetation": "low-lying natural greenery and grassland textures",
    "Highway": "linear transportation infrastructure with high-contrast road surfaces",
    "Industrial Buildings": "large-scale industrial rooftops and rectangular geometric structures",
    "Pasture": "grassy areas typically used for livestock grazing with uniform textures",
    "Permanent Crop": "perennial agricultural systems like orchards or vineyards with consistent patterns",
    "Residential Buildings": "clusters of small-scale rooftops and suburban road networks",
    "River": "winding linear water bodies with distinct spectral signatures against the terrain",
    "SeaLake": "expansive, homogeneous spectral regions representing significant water bodies"
}

CLASSIFICATION_QUESTIONS = [
    "What type of land use is shown in this satellite image?",
    "What does this satellite image show?",
    "Describe the land cover in this image.",
    "What scene is depicted in this remote sensing image?",
    "What type of area is visible in this satellite photo?",
    "Based on the visual features, how would you classify this scene?",
    "Provide a brief analysis of the land use in this satellite imagery.",
]

# Standard, concise templates
BASIC_ANSWER_TEMPLATES = [
    "This satellite image shows {class_name}.",
    "The image depicts {class_name}.",
    "This is a satellite image of {class_name}.",
    "The land cover in this image is {class_name}.",
]

# Expert, descriptive templates (The "Research-Grade" responses)
EXPERT_ANSWER_TEMPLATES = [
    "Based on the spectral patterns and textures, this scene is identified as {class_name}. The imagery exhibits {description}.",
    "An analysis of the spatial features reveals {class_name}. This is characterized by {description}.",
    "This remote sensing capture shows a {class_name} area, identified by the {description} present in the scene.",
    "The primary land use here is {class_name}. We can observe {description}, which is a key indicator for this classification.",
    "This is a satellite view of {class_name}. The visual signature matches {description}, typical for this category.",
]

# Richer, longer caption-style templates
DESCRIPTIVE_TEMPLATES = [
    "A satellite image showing {class_name} with typical land cover patterns.",
    "An aerial view of {class_name} captured by a Sentinel-2 satellite.",
    "This remote sensing image depicts {class_name} terrain.",
    "The image shows a {class_name} area with characteristic spatial patterns.",
    "A top-down view of {class_name} at 10 metres per pixel resolution.",
]

# Templates focusing on visual attributes
ATTRIBUTE_TEMPLATES = [
    {
        "question": "Describe what you see in this satellite image.",
        "answer": "The image shows {class_name}. You can observe {attributes}."
    },
    {
        "question": "What type of terrain is shown and what are its features?",
        "answer": "This is {class_name} terrain, identifiable by {attributes}."
    },
    {
        "question": "Explain the visual characteristics of this scene.",
        "answer": "This scene depicts {class_name}, which features {attributes}."
    }
]

# Multi-turn conversational templates
MULTI_TURN_TEMPLATES = [
    [
        {"role": "user", "content": "What is shown in this satellite image?"},
        {"role": "assistant", "content": "This is {class_name}."},
        {"role": "user", "content": "What visual features help identify it?"},
        {"role": "assistant", "content": "The imagery exhibits {attributes}."}
    ],
    [
        {"role": "user", "content": "Can you classify this remote sensing image?"},
        {"role": "assistant", "content": "Yes, this is {class_name}."},
        {"role": "user", "content": "How confident are you?"},
        {"role": "assistant", "content": "The {attributes} strongly indicate {class_name}."}
    ]
]

# Contrastive templates to improve discrimination
CONTRAST_TEMPLATES = [
    {
        "question": "Is this a {wrong_class} or {class_name}?",
        "answer": "This is {class_name}, not {wrong_class}. You can tell by the {attributes}."
    },
    {
        "question": "Does this image depict {class_name} or {wrong_class}?",
        "answer": "The image shows {class_name}. While it might be confused with {wrong_class}, the presence of {attributes} confirms its classification."
    }
]

def get_classification_qa(class_name: str, wrong_class: str = None, randomise: bool = True) -> list[dict]:
    """
    Generate a classification QA sequence (one or more turns) with expert-level detail.
    """
    attributes = CLASS_ATTRIBUTES.get(class_name, "general land cover features")
    
    if not randomise:
        # Default to a single expert-style turn
        question = CLASSIFICATION_QUESTIONS[0]
        answer = EXPERT_ANSWER_TEMPLATES[0].format(class_name=class_name, description=attributes)
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    # Randomly choose template type
    choice = random.random()
    
    if choice < 0.15: # 15% Basic
        question = random.choice(CLASSIFICATION_QUESTIONS)
        answer = random.choice(BASIC_ANSWER_TEMPLATES).format(class_name=class_name)
        turns = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    
    elif choice < 0.40: # 25% Expert
        question = random.choice(CLASSIFICATION_QUESTIONS)
        answer = random.choice(EXPERT_ANSWER_TEMPLATES).format(class_name=class_name, description=attributes)
        turns = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        
    elif choice < 0.60: # 20% Descriptive Caption
        question = "Describe this satellite image."
        answer = random.choice(DESCRIPTIVE_TEMPLATES).format(class_name=class_name)
        turns = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        
    elif choice < 0.80: # 20% Attribute Based
        tmpl = random.choice(ATTRIBUTE_TEMPLATES)
        question = tmpl["question"]
        answer = tmpl["answer"].format(class_name=class_name, attributes=attributes)
        turns = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        
    elif choice < 0.90: # 10% Contrastive
        if wrong_class is None:
            # Fallback to expert
            question = random.choice(CLASSIFICATION_QUESTIONS)
            answer = random.choice(EXPERT_ANSWER_TEMPLATES).format(class_name=class_name, description=attributes)
            turns = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        else:
            tmpl = random.choice(CONTRAST_TEMPLATES)
            question = tmpl["question"].format(class_name=class_name, wrong_class=wrong_class)
            answer = tmpl["answer"].format(class_name=class_name, wrong_class=wrong_class, attributes=attributes)
            turns = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
            
    else: # 10% Multi-turn
        turns_template = random.choice(MULTI_TURN_TEMPLATES)
        turns = []
        for t in turns_template:
            turns.append({
                "role": t["role"],
                "content": t["content"].format(class_name=class_name, attributes=attributes)
            })
            
    return turns


def format_prompt(turns: list[dict], add_final_answer: bool = True) -> str:
    """
    Format a list of QA turns into a TinyLlama chat-style prompt.
    """
    prompt = ""
    for i, turn in enumerate(turns):
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            prompt += f"<|user|>\n{content}</s>\n<|assistant|>\n"
        elif role == "assistant":
            if not add_final_answer and i == len(turns) - 1:
                break
            prompt += f"{content}</s>\n"
    return prompt


class EuroSATVQADataset(Dataset):
    """
    Wraps EuroSATDataset and converts labels to rich QA pairs.
    Used for Phase 3 instruction tuning.
    """

    def __init__(
        self,
        eurosat_root: str = "data/EuroSAT",
        train: bool = True,
        tokenizer=None,
        max_length: int = 128, # Increased default max_length for richer answers
    ):
        from data.eurosat import EuroSATDataset
        self.base_dataset = EuroSATDataset(root=eurosat_root, train=train)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image = sample["image"]
        class_name = sample["class_name"]
        gsd = sample["gsd"]

        # Pick a random wrong class for contrastive templates
        other_classes = [c for c in EUROSAT_CLASSES if c != class_name]
        wrong_class = random.choice(other_classes)

        # Generate Rich QA turns
        turns = get_classification_qa(class_name, wrong_class=wrong_class, randomise=True)
        prompt = format_prompt(turns, add_final_answer=True)

        item = {
            "image": image,
            "gsd": gsd,
            "turns": turns,
            "prompt": prompt,
            "question": turns[0]["content"],
            "answer": turns[-1]["content"],
        }

        # Tokenise if tokenizer provided
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)
            
            labels = encoded["input_ids"].squeeze(0).clone()
            labels[item["attention_mask"] == 0] = -100
            
            # Mask out everything except the final assistant response
            prompt_no_final_answer = format_prompt(turns, add_final_answer=False)
            encoded_no_final_answer = self.tokenizer(
                prompt_no_final_answer,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_len = encoded_no_final_answer["input_ids"].shape[1]
            labels[:prompt_len] = -100
            
            item["labels"] = labels

        return item


if __name__ == "__main__":
    print("Testing Expert VQA template generation...")

    # Test QA generation for all classes
    for class_name in EUROSAT_CLASSES:
        turns = get_classification_qa(class_name, randomise=False)
        print(f"\n[{class_name}]")
        for t in turns:
            print(f"  {t['role'].upper()}: {t['content']}")

    print("\nTesting random expert variation...")
    for _ in range(3):
        turns = get_classification_qa("Industrial Buildings", randomise=True)
        print("--- NEW SAMPLE ---")
        for t in turns:
            print(f"  {t['role'].upper()}: {t['content']}")

    print("\nExpert VQA template check passed.")
