import random
import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.eurosat import EUROSAT_CLASSES


# Multiple question phrasings per task — prevents the model from
# memorising a single question format
CLASSIFICATION_QUESTIONS = [
    "What type of land use is shown in this satellite image?",
    "What does this satellite image show?",
    "Describe the land cover in this image.",
    "What scene is depicted in this remote sensing image?",
    "What type of area is visible in this satellite photo?",
]

CLASSIFICATION_ANSWER_TEMPLATES = [
    "This satellite image shows {class_name}.",
    "The image depicts {class_name}.",
    "This is a satellite image of {class_name}.",
    "The land cover in this image is {class_name}.",
]

COUNTING_QUESTIONS = [
    "What is the primary land use category in this image?",
    "How would you classify this satellite image?",
]

CHANGE_QUESTIONS = [
    "Describe what you observe in this remote sensing image.",
    "What features are visible in this satellite image?",
]


def get_classification_qa(class_name: str, randomise: bool = True) -> dict:
    """
    Generate a classification QA pair for a given EuroSAT class.

    Args:
        class_name:  e.g. "forest", "residential area"
        randomise:   if True, randomly pick question/answer template

    Returns:
        dict with 'question' and 'answer' strings
    """
    if randomise:
        question = random.choice(CLASSIFICATION_QUESTIONS)
        answer_template = random.choice(CLASSIFICATION_ANSWER_TEMPLATES)
    else:
        question = CLASSIFICATION_QUESTIONS[0]
        answer_template = CLASSIFICATION_ANSWER_TEMPLATES[0]

    answer = answer_template.format(class_name=class_name)

    return {
        "question": question,
        "answer": answer,
    }


def format_prompt(question: str, answer: str | None = None) -> str:
    """
    Format a QA pair into a TinyLlama chat-style prompt.

    During training: includes the answer (for loss computation).
    During inference: omits the answer (model generates it).
    """
    prompt = f"<|user|>\n{question}</s>\n<|assistant|>\n"
    if answer is not None:
        prompt += f"{answer}</s>"
    return prompt


class EuroSATVQADataset(Dataset):
    """
    Wraps EuroSATDataset and converts labels to QA pairs.
    Used for Phase 3 instruction tuning.
    """

    def __init__(
        self,
        eurosat_root: str = "data/EuroSAT",
        train: bool = True,
        tokenizer=None,
        max_length: int = 64,
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

        # Generate QA pair
        qa = get_classification_qa(class_name, randomise=True)
        prompt = format_prompt(qa["question"], qa["answer"])

        item = {
            "image": image,
            "gsd": gsd,
            "question": qa["question"],
            "answer": qa["answer"],
            "prompt": prompt,
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
            # Labels = input_ids (causal LM — predict next token)
            labels = encoded["input_ids"].squeeze(0).clone()
            
            # Mask out padding tokens
            labels[item["attention_mask"] == 0] = -100
            
            # Mask out the user question (instruction tuning objective)
            # We only want the model to learn to generate the answer.
            prompt_no_answer = format_prompt(qa["question"], answer=None)
            encoded_no_answer = self.tokenizer(
                prompt_no_answer,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_len = encoded_no_answer["input_ids"].shape[1]
            labels[:prompt_len] = -100
            
            item["labels"] = labels

        return item


if __name__ == "__main__":
    print("Testing VQA template generation...")

    # Test QA generation for all classes
    for class_name in EUROSAT_CLASSES:
        qa = get_classification_qa(class_name, randomise=False)
        print(f"  [{class_name}]")
        print(f"    Q: {qa['question']}")
        print(f"    A: {qa['answer']}")

    print("\nTesting prompt formatting...")
    qa = get_classification_qa("forest", randomise=False)
    prompt = format_prompt(qa["question"], qa["answer"])
    print(prompt)

    print("\nTesting random template variation...")
    for _ in range(3):
        qa = get_classification_qa("residential area", randomise=True)
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']}")

    print("\nVQA template check passed.")