import torch
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from typing import Dict, List, Any

class CaptioningMetrics:
    """
    Computes standard Remote Sensing captioning metrics:
    BLEU-1, BLEU-4, METEOR, CIDEr
    """
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]

    def compute(self, gts: Dict[int, List[str]], res: Dict[int, List[str]]) -> Dict[str, float]:
        """
        Args:
            gts: Dictionary of {img_id: [ref1, ref2, ...]}
            res: Dictionary of {img_id: [candidate]}
        """
        eval_results = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    eval_results[m] = sc
            else:
                eval_results[method] = score
        return eval_results

class VQAMetrics:
    """
    Computes Top-1 Accuracy and Per-Class breakdown for VQA on EuroSAT
    """
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.per_class_correct = {name: 0 for name in self.class_names}
        self.per_class_total = {name: 0 for name in self.class_names}

    def update(self, predictions: List[str], ground_truths: List[str]):
        """
        Args:
            predictions: List of generated class names from model
            ground_truths: List of actual class names
        """
        for pred, gt in zip(predictions, ground_truths):
            # Clean strings for robust comparison
            pred_clean = pred.lower().strip().replace(".", "").replace(" ", "")
            gt_clean = gt.lower().strip().replace(".", "").replace(" ", "")
            
            is_correct = gt_clean in pred_clean # flexible match
            
            self.total += 1
            self.per_class_total[gt] += 1
            if is_correct:
                self.correct += 1
                self.per_class_correct[gt] += 1

    def compute(self) -> Dict[str, Any]:
        overall_acc = self.correct / self.total if self.total > 0 else 0
        
        per_class_acc = {}
        for name in self.class_names:
            total = self.per_class_total[name]
            correct = self.per_class_correct[name]
            per_class_acc[name] = correct / total if total > 0 else 0
            
        return {
            "overall_accuracy": overall_acc,
            "per_class_accuracy": per_class_acc
        }

if __name__ == "__main__":
    # Test VQA Metrics
    classes = ["Forest", "Highway", "River"]
    vqa = VQAMetrics(classes)
    vqa.update(["This is a Forest.", "River"], ["Forest", "River"])
    vqa.update(["Industrial"], ["Highway"])
    print("VQA Results:", vqa.compute())

    # Test Captioning Metrics
    cap = CaptioningMetrics()
    gts = {0: ["a satellite image of a forest"], 1: ["a river flowing through a city"]}
    res = {0: ["a satellite image of a forest"], 1: ["a small stream in a urban area"]}
    print("Captioning Results:", cap.compute(gts, res))
