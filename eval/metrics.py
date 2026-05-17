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
        # We initialize scorers individually for better error handling
        self.bleu = Bleu(4)
        self.cider = Cider()
        self.meteor = Meteor()

    def compute(self, gts: Dict[int, List[str]], res: Dict[int, List[str]]) -> Dict[str, float]:
        eval_results = {}
        
        # 1. BLEU
        print("Computing BLEU scores...")
        try:
            score, scores = self.bleu.compute_score(gts, res)
            for i, s in enumerate(score):
                eval_results[f"Bleu_{i+1}"] = s
        except Exception as e:
            print(f"Warning: BLEU failed: {e}")

        # 2. CIDEr
        print("Computing CIDEr scores...")
        try:
            score, scores = self.cider.compute_score(gts, res)
            eval_results["CIDEr"] = score
        except Exception as e:
            print(f"Warning: CIDEr failed: {e}")

        # 3. METEOR (Most fragile due to Java)
        print("Computing METEOR scores...")
        try:
            score, scores = self.meteor.compute_score(gts, res)
            eval_results["METEOR"] = score
        except Exception as e:
            print(f"Warning: METEOR calculation failed (likely Java subprocess error). Skipping... Error: {e}")
            eval_results["METEOR"] = 0.0

        return eval_results

class VQAMetrics:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.per_class_correct = {name: 0 for name in self.class_names}
        self.per_class_total = {name: 0 for name in self.class_names}

    def update(self, predictions: List[str], ground_truths: List[str]):
        for pred, gt in zip(predictions, ground_truths):
            pred_clean = pred.lower().strip().replace(".", "").replace(" ", "")
            gt_clean = gt.lower().strip().replace(".", "").replace(" ", "")
            is_correct = gt_clean in pred_clean
            self.total += 1
            self.per_class_total[gt] += 1
            if is_correct:
                self.correct += 1
                self.per_class_correct[gt] += 1

    def compute(self) -> Dict[str, Any]:
        overall_acc = self.correct / self.total if self.total > 0 else 0
        per_class_acc = {name: (self.per_class_correct[name] / self.per_class_total[name] if self.per_class_total[name] > 0 else 0) for name in self.class_names}
        return {"overall_accuracy": overall_acc, "per_class_accuracy": per_class_acc}
