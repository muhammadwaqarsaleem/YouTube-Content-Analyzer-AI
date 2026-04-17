import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from config import LABEL_MAP, NUM_LABELS, DRIVE_SAVE_DIR, VAL_SIZE, RANDOM_SEED


def run_evaluation(trainer, val_dataset):
    print("Generating predictions on validation set...")
    preds_output = trainer.predict(val_dataset)
    y_pred = preds_output.predictions.argmax(-1)
    y_true = preds_output.label_ids
    return preds_output, y_pred, y_true


def print_classification_report(y_true, y_pred):
    target_names = [LABEL_MAP[i] for i in range(NUM_LABELS)]
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_confusion_matrix(y_true, y_pred):
    target_names = [LABEL_MAP[i] for i in range(NUM_LABELS)]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Validation Set")
    plt.tight_layout()
    plt.savefig(f"{DRIVE_SAVE_DIR}/confusion_matrix.png", dpi=150)
    plt.show()


def plot_training_curves(trainer):
    log_history = trainer.state.log_history
    train_logs  = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs   = [e for e in log_history if "eval_loss" in e]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot([e["epoch"] for e in train_logs], [e["loss"] for e in train_logs],
             label="Train loss", color="steelblue", alpha=0.7)
    ax1.plot([e["epoch"] for e in eval_logs],  [e["eval_loss"] for e in eval_logs],
             label="Val loss", color="tomato", linewidth=2, marker="o")
    ax1.set_title("Loss curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot([e["epoch"] for e in eval_logs],
             [e["eval_accuracy"] * 100 for e in eval_logs],
             color="seagreen", linewidth=2, marker="o")
    ax2.set_title("Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(alpha=0.3)

    plt.suptitle("BaitRadar — RoBERTa-Base Training", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{DRIVE_SAVE_DIR}/training_curves.png", dpi=150)
    plt.show()


def run_error_analysis(preds_output, y_pred, df):
    probs      = torch.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()
    confidence = probs[np.arange(len(y_pred)), y_pred]

    _, val_df = train_test_split(
        df.reset_index(drop=True), test_size=VAL_SIZE,
        random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_df = val_df.reset_index(drop=True)
    val_df["predicted"]  = y_pred
    val_df["confidence"] = confidence
    val_df["correct"]    = (val_df["label"] == val_df["predicted"])

    mistakes = val_df[~val_df["correct"]].copy()
    print(f"Total val samples : {len(val_df)}")
    print(f"Mistakes          : {len(mistakes)}  ({len(mistakes)/len(val_df)*100:.1f}%)")

    print("\n── Top 5 High-Confidence Errors ─────────────────────────────────────────")
    for _, row in mistakes.nlargest(5, "confidence").iterrows():
        print(f"\n  Title:      {row['title'][:80]}")
        print(f"  True:       {LABEL_MAP[row['label']]}  →  Predicted: {LABEL_MAP[row['predicted']]}  (conf: {row['confidence']:.2f})")
        print(f"  Transcript: {str(row['transcript'])[:150]}...")

    print("\n── Most Confused Class Pairs ────────────────────────────────────────────")
    pair_counts = (
        mistakes.groupby(["label", "predicted"])
        .size().reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(8)
    )
    for _, row in pair_counts.iterrows():
        print(f"  {LABEL_MAP[row['label']]:<15} → {LABEL_MAP[row['predicted']]:<15}  ({int(row['count']):>3} times)")