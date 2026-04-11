import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils import class_weight
from transformers import Trainer, TrainerCallback
from sklearn.metrics import accuracy_score
from config import LABEL_MAP, HARMLESS_BOOST


def compute_class_weights(y_train, device):
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    weights[0] *= HARMLESS_BOOST
    print("📊 Class weights (after Harmless boost):")
    for i, w in enumerate(weights):
        print(f"   {LABEL_MAP[i]:<18}  {w:.4f}")
    return torch.tensor(weights, dtype=torch.float).to(device)


def make_weighted_trainer(weights_tensor):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            loss    = nn.CrossEntropyLoss(weight=weights_tensor)(logits, labels)
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


class EpochSummaryCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch    = int(state.epoch) if state.epoch else "?"
        val_acc  = metrics.get("eval_accuracy", float("nan"))
        val_loss = metrics.get("eval_loss",     float("nan"))
        print(f"\n── Epoch {epoch} {'─'*40}")
        print(f"   Val loss:     {val_loss:.4f}")
        print(f"   Val accuracy: {val_acc * 100:.2f}%")
        print("─" * 50)


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}