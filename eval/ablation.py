import json
import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_comparison_table(results_dict: dict):
    """
    results_dict: {
        "RS-VLM": {"bleu4": 0.45, "cider": 1.2, "vqa_acc": 0.92},
        "Ablation (No GSD)": {"bleu4": 0.41, "cider": 1.0, "vqa_acc": 0.88},
        "Baseline (Pure ViT)": {"bleu4": 0.38, "cider": 0.8, "vqa_acc": 0.85},
    }
    """
    df = pd.DataFrame(results_dict).T
    print("\n--- Model Comparison Table ---")
    print(df)
    df.to_csv("comparison_table.csv")
    return df

def plot_per_class_accuracy(vqa_results: dict, output_path: str = "per_class_acc.png"):
    """
    vqa_results: {"Forest": 0.95, "Highway": 0.88, ...}
    """
    classes = list(vqa_results.keys())
    accs = list(vqa_results.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes, accs, color='skyblue')
    plt.xlabel('Land Use Class')
    plt.ylabel('Top-1 Accuracy')
    plt.title('RS-VLM Per-Class Accuracy on EuroSAT')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Example logic
    if os.path.exists("eval_results.json"):
        with open("eval_results.json", "r") as f:
            data = json.load(f)
            
        if "vqa" in data:
            plot_per_class_accuracy(data["vqa"]["per_class_accuracy"])
            
        # Add logic here to compare with other models once they are evaluated
