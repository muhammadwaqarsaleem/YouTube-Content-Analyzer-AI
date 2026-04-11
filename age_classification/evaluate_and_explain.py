"""
evaluate_and_explain.py  —  File 4 / 4
========================================
Evaluation & Explainable-AI pipeline for the Hierarchical RoBERTa
YouTube Age Classifier.

Outputs
-------
results/plots/
    confusion_matrix.png
    roc_curves.png
    pr_curves.png
    confidence_histogram.png
    per_class_metrics.png
    attention_sample_<n>.png

results/reports/
    classification_report.txt
    error_analysis.csv
    eval_summary.json
    attention_report.html      ← publication-ready XAI viewer

Author : University AI Research Team
Project: YouTube Age Classification  —  File 4 / 4
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                    # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import RobertaTokenizer

from hierarchical_roberta import HierarchicalRobertaForClassification, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES: List[str] = ["General", "Teen", "Mature"]

# Palette kept consistent across every plot and the HTML report
PALETTE: Dict[str, str] = {
    "General": "#52b788",   # sage green
    "Teen":    "#4895ef",   # sky blue
    "Mature":  "#e63946",   # crimson
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    # Paths
    checkpoint_path: str  = "checkpoints/best_model.pt"
    test_data_path:  str  = "data/processed_tensors/test_dataset.pt"
    csv_path:        str  = "training_ready_dataset.csv"
    plots_dir:       str  = "results/plots"
    reports_dir:     str  = "results/reports"

    # Model mirrors (must match training config exactly)
    model_name:  str = "roberta-base"
    chunk_size:  int = 512
    max_chunks:  int = 20
    stride:      int = 256
    num_classes: int = 3

    # Inference
    batch_size:  int = 8
    device:      str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # XAI
    n_explain_samples: int = 6    # samples rendered in the HTML report
    n_error_samples:   int = 5    # top-N confident wrong predictions

    # Plot quality
    dpi: int = 180


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    """All data produced by explain_prediction() for one transcript."""
    transcript_snippet: str
    predicted_label:    str
    true_label:         Optional[str]
    confidence:         float
    probabilities:      Dict[str, float]
    chunk_texts:        List[str]
    attention_weights:  List[float]     # one weight per real chunk, sums ≈ 1
    top3_chunks:        List[Dict]      # [{rank, chunk_idx, weight, text}, …]
    num_chunks_used:    int


# ─────────────────────────────────────────────────────────────────────────────
# Chunking helper  (mirrors dataset_preparation.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_transcript(
    text:       str,
    tokenizer:  RobertaTokenizer,
    chunk_size: int = 512,
    max_chunks: int = 20,
    stride:     int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Tokenise *text* with the identical sliding-window used in
    dataset_preparation.py so that inference is perfectly consistent
    with how the model was trained.

    Returns
    -------
    input_ids      : LongTensor (max_chunks, chunk_size)
    attention_mask : LongTensor (max_chunks, chunk_size)
    chunk_texts    : list[str]  – human-readable text per real chunk
    """
    enc     = tokenizer(text, add_special_tokens=True,
                        truncation=False, return_tensors=None)
    all_ids = enc["input_ids"]
    all_msk = enc["attention_mask"]

    input_ids      = torch.full((max_chunks, chunk_size),
                                tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((max_chunks, chunk_size), dtype=torch.long)
    chunk_texts: List[str] = []

    start = chunk_idx = 0
    while start < len(all_ids) and chunk_idx < max_chunks:
        end    = min(start + chunk_size, len(all_ids))
        c_ids  = all_ids[start:end]
        c_msk  = all_msk[start:end]
        length = len(c_ids)

        input_ids[chunk_idx,      :length] = torch.tensor(c_ids)
        attention_mask[chunk_idx, :length] = torch.tensor(c_msk)
        chunk_texts.append(tokenizer.decode(c_ids, skip_special_tokens=True))

        start     += stride
        chunk_idx += 1

    return input_ids, attention_mask, chunk_texts


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: EvalConfig) -> HierarchicalRobertaForClassification:
    """Restore best-checkpoint weights and set to eval mode."""
    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)

    model_cfg = ModelConfig(
        num_classes=cfg.num_classes,
        max_chunks=cfg.max_chunks,
        chunk_size=cfg.chunk_size,
        use_gradient_checkpointing=False,   # disabled for inference
        pooling_strategy="attention",
    )
    model = HierarchicalRobertaForClassification(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg.device).eval()

    epoch  = ckpt.get("epoch", "?")
    metric = ckpt.get("best_metric", float("nan"))
    log.info(f"Checkpoint loaded  epoch={epoch}  best_metric={metric:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Test-set inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: HierarchicalRobertaForClassification,
    cfg:   EvalConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-pass every sample in the test set.

    Returns
    -------
    y_true  : int ndarray (N,)
    y_pred  : int ndarray (N,)
    y_probs : float ndarray (N, num_classes)
    """
    data    = torch.load(cfg.test_data_path, map_location="cpu")
    dataset = TensorDataset(
        data["input_ids"], data["attention_mask"], data["labels"]
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    y_true_l: List[int]            = []
    y_pred_l: List[int]            = []
    y_probs_l: List[np.ndarray]    = []

    for ids, mask, labels in tqdm(loader, desc="Test inference", unit="batch"):
        out   = model(ids.to(cfg.device), mask.to(cfg.device))
        probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)

        y_true_l.extend(labels.numpy().tolist())
        y_pred_l.extend(preds.tolist())
        y_probs_l.extend(probs)

    return np.array(y_true_l), np.array(y_pred_l), np.array(y_probs_l)


# ─────────────────────────────────────────────────────────────────────────────
# XAI core
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def explain_prediction(
    transcript_text: str,
    model:           HierarchicalRobertaForClassification,
    tokenizer:       RobertaTokenizer,
    cfg:             EvalConfig,
    true_label:      Optional[str] = None,
) -> ExplanationResult:
    """
    Explain a single prediction end-to-end:

      1. Chunk the raw text using the identical sliding-window as File 1.
      2. Forward-pass with return_attention=True to retrieve the learned
         self-attention weights that aggregated chunk representations.
      3. Re-normalise weights after trimming padding chunks.
      4. Surface the top-3 highest-weighted chunks as the model's
         'reasoning evidence'.

    Design note
    -----------
    We deliberately re-use chunk_transcript() rather than pre-tokenised
    tensors so this function can be called on *any* raw string at inference
    time — making it useful for a live demo or API endpoint.
    """
    ids, mask, chunk_texts = chunk_transcript(
        transcript_text, tokenizer,
        chunk_size=cfg.chunk_size,
        max_chunks=cfg.max_chunks,
        stride=cfg.stride,
    )

    # Add batch dim → (1, max_chunks, chunk_size)
    ids  = ids.unsqueeze(0).to(cfg.device)
    mask = mask.unsqueeze(0).to(cfg.device)

    out   = model(ids, mask, return_attention=True)
    probs = F.softmax(out["logits"], dim=-1)[0].cpu().numpy()   # (num_classes,)
    pred  = int(probs.argmax())

    # Retrieve attention weights; fall back to uniform if unavailable
    if "attention_weights" in out and out["attention_weights"] is not None:
        raw_w = out["attention_weights"][0].cpu().numpy()       # (max_chunks,)
    else:
        raw_w = np.ones(cfg.max_chunks) / cfg.max_chunks

    # Trim to real (non-padding) chunks and re-normalise
    n_chunks = len(chunk_texts)
    weights  = raw_w[:n_chunks]
    weights  = weights / (weights.sum() + 1e-12)

    # Rank chunks
    top3_idx = np.argsort(weights)[::-1][:3]
    top3 = [
        {
            "rank":      int(r + 1),
            "chunk_idx": int(i),
            "weight":    float(weights[i]),
            "text":      chunk_texts[i],
        }
        for r, i in enumerate(top3_idx)
    ]

    return ExplanationResult(
        transcript_snippet = transcript_text[:400],
        predicted_label    = LABEL_NAMES[pred],
        true_label         = true_label,
        confidence         = float(probs[pred]),
        probabilities      = {n: float(p) for n, p in zip(LABEL_NAMES, probs)},
        chunk_texts        = chunk_texts,
        attention_weights  = weights.tolist(),
        top3_chunks        = top3,
        num_chunks_used    = n_chunks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Saved → {path}")


def _dark_ax(ax: plt.Axes) -> None:
    """Shared dark-theme axis styling."""
    ax.set_facecolor("#18181c")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#383848")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, cfg: EvalConfig
) -> None:
    """Side-by-side raw-count and row-normalised confusion matrices."""
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0f0f13")
    fig.suptitle(
        "Confusion Matrix  —  Hierarchical RoBERTa",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )

    for ax, data, fmt, title, cmap in zip(
        axes,
        [cm, norm],
        ["d", ".1%"],
        ["Raw Counts", "Row-Normalised (Recall per class)"],
        ["YlOrBr", "RdYlGn"],
    ):
        ax.set_facecolor("#18181c")
        sns.heatmap(
            data, ax=ax, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            linewidths=1.2, linecolor="#0f0f13",
            annot_kws={"fontsize": 13, "fontweight": "bold"},
            cbar_kws={"shrink": 0.78},
        )
        ax.set_title(title, fontsize=11, color="white", pad=10)
        ax.set_xlabel("Predicted", fontsize=11, color="#aaa")
        ax.set_ylabel("True",      fontsize=11, color="#aaa")
        ax.tick_params(colors="white", labelsize=10)

    plt.tight_layout()
    _savefig(fig, Path(cfg.plots_dir) / "confusion_matrix.png", cfg.dpi)


def plot_roc_curves(
    y_true: np.ndarray, y_probs: np.ndarray, cfg: EvalConfig
) -> None:
    """One-vs-Rest ROC curves for all three classes."""
    y_bin   = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#0f0f13")
    _dark_ax(ax)

    for i, label in enumerate(LABEL_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[label], lw=2.5,
                label=f"{label}  (AUC = {roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.07, color=PALETTE[label])

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.35, label="Random baseline")
    ax.set(
        xlabel="False Positive Rate", ylabel="True Positive Rate",
        xlim=[0, 1], ylim=[0, 1.02],
        title="ROC Curves  —  One-vs-Rest",
    )
    ax.legend(
        loc="lower right", frameon=True,
        facecolor="#252530", edgecolor="#444", labelcolor="white", fontsize=10,
    )
    plt.tight_layout()
    _savefig(fig, Path(cfg.plots_dir) / "roc_curves.png", cfg.dpi)


def plot_pr_curves(
    y_true: np.ndarray, y_probs: np.ndarray, cfg: EvalConfig
) -> None:
    """Precision-Recall curves — especially informative for the imbalanced Mature class."""
    y_bin   = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#0f0f13")
    _dark_ax(ax)

    for i, label in enumerate(LABEL_NAMES):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap      = average_precision_score(y_bin[:, i], y_probs[:, i])
        ax.step(r, p, where="post", color=PALETTE[label], lw=2.5,
                label=f"{label}  (AP = {ap:.3f})")
        ax.fill_between(r, p, step="post", alpha=0.07, color=PALETTE[label])

    ax.set(
        xlabel="Recall", ylabel="Precision",
        xlim=[0, 1], ylim=[0, 1.05],
        title="Precision-Recall Curves  —  One-vs-Rest",
    )
    ax.legend(
        loc="upper right", frameon=True,
        facecolor="#252530", edgecolor="#444", labelcolor="white", fontsize=10,
    )
    plt.tight_layout()
    _savefig(fig, Path(cfg.plots_dir) / "pr_curves.png", cfg.dpi)


def plot_confidence_histogram(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_probs: np.ndarray,
    cfg:     EvalConfig,
) -> None:
    """
    Three-panel figure:
      Left   — correct vs wrong confidence distributions
      Centre — per-class confidence histograms
      Right  — reliability / calibration diagram
    """
    max_conf = y_probs.max(axis=1)
    correct  = y_true == y_pred
    bins     = np.linspace(0, 1, 26)

    fig = plt.figure(figsize=(15, 5), facecolor="#0f0f13")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ── Left: correct vs wrong ────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    _dark_ax(ax0)
    ax0.hist(max_conf[correct],  bins=bins, color="#52b788", alpha=0.78,
             label="Correct", edgecolor="#0f0f13", linewidth=0.3)
    ax0.hist(max_conf[~correct], bins=bins, color="#e63946", alpha=0.78,
             label="Wrong",   edgecolor="#0f0f13", linewidth=0.3)
    ax0.set(xlabel="Max Softmax Probability", ylabel="Count",
            title="Correct vs Incorrect", xlim=(0, 1))
    ax0.legend(facecolor="#252530", edgecolor="#444",
               labelcolor="white", fontsize=9)

    # ── Centre: per-class densities ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    _dark_ax(ax1)
    for i, label in enumerate(LABEL_NAMES):
        mask = y_true == i
        if mask.sum() > 1:
            ax1.hist(max_conf[mask], bins=16, color=PALETTE[label], alpha=0.58,
                     label=label, edgecolor="#0f0f13", linewidth=0.3, density=True)
    ax1.set(xlabel="Confidence", ylabel="Density",
            title="Confidence by True Class", xlim=(0, 1))
    ax1.legend(facecolor="#252530", edgecolor="#444",
               labelcolor="white", fontsize=9)

    # ── Right: reliability diagram ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    _dark_ax(ax2)
    edges = np.linspace(0, 1, 11)
    m_conf, m_acc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (max_conf >= lo) & (max_conf < hi)
        if mask.sum() > 0:
            m_conf.append(max_conf[mask].mean())
            m_acc.append(correct[mask].mean())

    ax2.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.35, label="Perfect calibration")
    ax2.plot(m_conf, m_acc, color="#f5c842", lw=2.5, marker="o",
             markersize=6, label="Model")
    ax2.fill_between(m_conf, m_acc, m_conf, alpha=0.13,
                     color="#f5c842", label="Gap")
    ax2.set(xlabel="Mean Confidence (bin)", ylabel="Fraction Correct",
            title="Reliability Diagram", xlim=(0, 1), ylim=(0, 1.05))
    ax2.legend(facecolor="#252530", edgecolor="#444",
               labelcolor="white", fontsize=9)

    fig.suptitle("Prediction Confidence Analysis", color="white",
                 fontsize=13, fontweight="bold", y=1.02)
    _savefig(fig, Path(cfg.plots_dir) / "confidence_histogram.png", cfg.dpi)


def plot_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, cfg: EvalConfig
) -> None:
    """Grouped bar: Precision / Recall / F1 per class."""
    from sklearn.metrics import precision_score, recall_score

    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true,    y_pred, average=None, zero_division=0)
    f1   = f1_score(y_true,        y_pred, average=None, zero_division=0)

    x, w = np.arange(len(LABEL_NAMES)), 0.26

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0f0f13")
    _dark_ax(ax)

    for vals, offset, color, label in [
        (prec, -w, "#4895ef", "Precision"),
        (rec,   0, "#52b788", "Recall"),
        (f1,   +w, "#f5c842", "F1-Score"),
    ]:
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, alpha=0.85,
                      edgecolor="#0f0f13", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.013,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color="white", fontweight="bold")

    ax.set(xticks=x, xticklabels=LABEL_NAMES, ylim=(0, 1.15),
           ylabel="Score",
           title="Per-Class Precision / Recall / F1")
    ax.axhline(0.5, color="white", lw=0.6, ls=":", alpha=0.25)
    ax.legend(facecolor="#252530", edgecolor="#444",
              labelcolor="white", fontsize=10, loc="upper right")

    plt.tight_layout()
    _savefig(fig, Path(cfg.plots_dir) / "per_class_metrics.png", cfg.dpi)


def plot_attention_bar(
    result:   ExplanationResult,
    out_path: Path,
    cfg:      EvalConfig,
) -> None:
    """Horizontal bar chart of per-chunk attention weights for one sample."""
    n       = result.num_chunks_used
    weights = result.attention_weights[:n]
    labels  = [f"Chunk {i:02d}" for i in range(n)]
    accent  = PALETTE.get(result.predicted_label, "#aaa")
    max_w   = max(weights) if weights else 1.0

    bar_colors = [
        "#f5c842" if abs(w - max_w) < 1e-9 else
        accent    if w >= float(np.percentile(weights, 66)) else
        "#2e2e3a"
        for w in weights
    ]

    fig, ax = plt.subplots(figsize=(9, max(3.5, n * 0.42)), facecolor="#0f0f13")
    _dark_ax(ax)

    bars = ax.barh(
        labels[::-1], weights[::-1],
        color=bar_colors[::-1], height=0.65,
        edgecolor="#0f0f13", linewidth=0.3,
    )
    for b, w in zip(bars, weights[::-1]):
        ax.text(b.get_width() + max_w * 0.01,
                b.get_y() + b.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=8.5, color="#ccc")

    ax.axvline(1.0 / n, color="white", ls="--", lw=1,
               alpha=0.3, label=f"Uniform ({1.0/n:.3f})")
    ax.set(
        xlabel="Attention Weight",
        title=(
            f"Chunk Attention Weights\n"
            f"Predicted: {result.predicted_label} "
            f"({result.confidence:.1%})  |  True: {result.true_label or '—'}"
        ),
    )
    ax.title.set_color(accent)
    ax.legend(facecolor="#252530", edgecolor="#444",
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    _savefig(fig, out_path, cfg.dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_error_analysis(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_probs: np.ndarray,
    cfg:     EvalConfig,
) -> pd.DataFrame:
    """
    Find the N most confident wrong predictions and export them to CSV.

    Rationale
    ---------
    Highly confident errors are the most damaging for a content-safety model.
    Surfacing them enables targeted error analysis: are they sarcasm, domain-
    specific slang, edge-case genres?  Answering this guides future data
    collection and post-processing rules.
    """
    wrong_mask = y_true != y_pred
    wrong_idx  = np.where(wrong_mask)[0]
    wrong_conf = y_probs[wrong_idx].max(axis=1)
    order      = np.argsort(wrong_conf)[::-1][: cfg.n_error_samples]

    rows = []
    for rank, pos in enumerate(order):
        idx = wrong_idx[pos]
        rows.append({
            "rank":            rank + 1,
            "sample_idx":      int(idx),
            "true_label":      LABEL_NAMES[y_true[idx]],
            "predicted_label": LABEL_NAMES[y_pred[idx]],
            "confidence":      round(float(y_probs[idx].max()), 4),
            "prob_General":    round(float(y_probs[idx][0]), 4),
            "prob_Teen":       round(float(y_probs[idx][1]), 4),
            "prob_Mature":     round(float(y_probs[idx][2]), 4),
        })

    df   = pd.DataFrame(rows)
    path = Path(cfg.reports_dir) / "error_analysis.csv"
    df.to_csv(path, index=False)
    log.info(f"  Error analysis → {path}")

    print(f"\n{'═'*68}")
    print(f"  TOP {cfg.n_error_samples} MOST CONFIDENT WRONG PREDICTIONS")
    print(f"{'═'*68}")
    print(df.to_string(index=False))
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HTML attention report
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def _prob_bars_html(probs: Dict[str, float]) -> str:
    out = ""
    for label, p in probs.items():
        col = PALETTE[label]
        out += (
            f'<div class="prob-row">'
            f'<span class="prob-lbl">{label}</span>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{p*100:.1f}%;background:{col}"></div>'
            f'</div>'
            f'<span class="prob-val">{p:.1%}</span>'
            f'</div>\n'
        )
    return out


def _attn_bars_html(weights: List[float], pred_label: str) -> str:
    accent  = PALETTE.get(pred_label, "#aaa")
    max_w   = max(weights) if weights else 1.0
    uniform = 1.0 / len(weights) if weights else 0
    out = ""
    for i, w in enumerate(weights):
        rel     = w / max_w * 100
        uni_rel = uniform / max_w * 100
        is_top  = abs(w - max_w) < 1e-9
        is_high = w >= sorted(weights)[-min(3, len(weights))]
        col     = "#f5c842" if is_top else (accent if is_high else "#2e2e3a")
        out += (
            f'<div class="abar-row">'
            f'<span class="abar-lbl">C{i:02d}</span>'
            f'<div class="abar-track">'
            f'<div class="abar-fill" style="width:{rel:.1f}%;background:{col}"></div>'
            f'<div class="abar-uni"  style="left:{uni_rel:.1f}%"></div>'
            f'</div>'
            f'<span class="abar-val">{w:.4f}</span>'
            f'</div>\n'
        )
    return out


def _chunk_cards_html(top3: List[Dict], pred_label: str) -> str:
    medals = ["🥇", "🥈", "🥉"]
    accent = PALETTE.get(pred_label, "#aaa")
    out    = ""
    for item in top3:
        medal   = medals[item["rank"] - 1]
        raw     = textwrap.fill(item["text"], width=110)
        escaped = _esc(raw[:900] + ("…" if len(raw) > 900 else ""))
        out += (
            f'<div class="chunk-card" style="border-left-color:{accent}">'
            f'<div class="chunk-meta">'
            f'<span class="chunk-medal">{medal}</span>'
            f'<span>Chunk {item["chunk_idx"]:02d}</span>'
            f'<span class="chunk-w">weight = {item["weight"]:.4f}</span>'
            f'</div>'
            f'<pre class="chunk-text">{escaped}</pre>'
            f'</div>\n'
        )
    return out


def _sample_card_html(res: ExplanationResult, idx: int) -> str:
    accent    = PALETTE.get(res.predicted_label, "#aaa")
    badge_cls = res.predicted_label.lower()
    correct   = res.true_label is None or res.predicted_label == res.true_label

    true_badge = ""
    if res.true_label:
        if correct:
            true_badge = '<span class="verdict correct">✓ correct</span>'
        else:
            true_badge = (
                f'<span class="verdict wrong">✗ true: {res.true_label}</span>'
            )

    snip = _esc(textwrap.shorten(res.transcript_snippet, 140, placeholder="…"))

    return f"""
<section class="sample-card" style="border-left-color:{accent}">
  <div class="card-header">
    <div class="card-meta">
      <p class="card-tag">Sample {idx}</p>
      <p class="card-snippet">"{snip}"</p>
    </div>
    <div class="badge-group">
      <span class="badge {badge_cls}">{res.predicted_label}</span>
      {true_badge}
    </div>
  </div>

  <div class="two-col">
    <div>
      <h4 class="sub-title">Class Probabilities</h4>
      {_prob_bars_html(res.probabilities)}
      <p class="conf-note">
        Confidence: <b>{res.confidence:.1%}</b> &nbsp;·&nbsp;
        Chunks used: <b>{res.num_chunks_used}</b>
      </p>
    </div>
    <div>
      <h4 class="sub-title">Chunk Attention Weights</h4>
      <div class="abar-scroll">
        {_attn_bars_html(res.attention_weights, res.predicted_label)}
      </div>
    </div>
  </div>

  <h4 class="sub-title" style="margin-top:22px">
    Top-3 Most Influential Chunks
  </h4>
  <div class="chunk-list">
    {_chunk_cards_html(res.top3_chunks, res.predicted_label)}
  </div>
</section>
"""


def generate_html_report(
    results: List[ExplanationResult],
    cfg:     EvalConfig,
) -> Path:
    """
    Write a self-contained, dark-themed HTML file with:
      - Animated probability bars
      - Per-chunk attention heat bars with uniform-baseline indicator
      - Full text of top-3 most influential chunks, colour-coded by class
    """
    ts         = datetime.now().strftime("%Y-%m-%d  %H:%M")
    cards_html = "\n".join(_sample_card_html(r, i + 1) for i, r in enumerate(results))
    n_correct  = sum(
        1 for r in results
        if r.true_label is not None and r.predicted_label == r.true_label
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>XAI Attention Report — YouTube Age Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
<style>
/* ── Reset & tokens ──────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:      #0c0c10;
  --surface: #141418;
  --surface2:#1b1b22;
  --border:  #28282f;
  --text:    #e2e0d8;
  --muted:   #666372;
  --general: #52b788;
  --teen:    #4895ef;
  --mature:  #e63946;
  --gold:    #f5c842;
  --r:       10px;
  --mono:'Space Mono',monospace;
  --sans:'Lato',sans-serif;
  --head:'Syne',sans-serif;
}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);
      font-weight:300;line-height:1.7;padding:52px 20px 110px;}}
.wrap{{max-width:1020px;margin:0 auto;}}

/* ── Header ──────────────────────────────────────────── */
.site-header{{
  border-bottom:1px solid var(--border);
  padding-bottom:38px;margin-bottom:54px;position:relative;
}}
.site-header::after{{
  content:'';position:absolute;top:-40px;right:-60px;
  width:340px;height:340px;border-radius:50%;
  background:radial-gradient(circle,rgba(72,149,239,.1) 0%,transparent 68%);
  pointer-events:none;
}}
.overtitle{{
  font-family:var(--mono);font-size:.66rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--muted);margin-bottom:14px;
}}
h1{{
  font-family:var(--head);font-size:clamp(2.1rem,5vw,3.4rem);
  font-weight:800;line-height:1.04;margin-bottom:14px;
  background:linear-gradient(118deg,#fff 0%,#9b8fd4 55%,var(--teen) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}}
.intro{{font-size:.93rem;color:var(--muted);max-width:640px;}}
.meta-row{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px;}}
.mpill{{
  font-family:var(--mono);font-size:.68rem;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:999px;padding:4px 14px;color:var(--muted);
}}
.mpill b{{color:var(--text);}}
.stat-row{{
  display:flex;gap:16px;flex-wrap:wrap;margin-top:28px;
}}
.stat-box{{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:8px;padding:14px 22px;min-width:120px;
}}
.stat-val{{
  font-family:var(--head);font-size:1.7rem;font-weight:800;
  line-height:1.1;
}}
.stat-lbl{{font-size:.72rem;color:var(--muted);margin-top:2px;}}

/* ── Section heading ────────────────────────────────── */
.sec-title{{
  font-family:var(--head);font-size:1.2rem;font-weight:700;
  margin:54px 0 20px;padding-bottom:10px;
  border-bottom:1px solid var(--border);letter-spacing:.01em;
}}

/* ── Sample card ─────────────────────────────────────── */
.sample-card{{
  background:var(--surface);border:1px solid var(--border);
  border-left:4px solid;border-radius:var(--r);
  padding:28px 30px;margin-bottom:28px;
}}
.card-header{{
  display:flex;justify-content:space-between;
  align-items:flex-start;gap:16px;flex-wrap:wrap;
  margin-bottom:22px;
}}
.card-tag{{
  font-family:var(--mono);font-size:.63rem;
  letter-spacing:.16em;text-transform:uppercase;
  color:var(--muted);margin-bottom:6px;
}}
.card-snippet{{
  font-size:.86rem;color:#9e9baa;font-style:italic;max-width:520px;
}}
.badge-group{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}}
.badge{{
  font-family:var(--mono);font-size:.7rem;font-weight:700;
  padding:5px 16px;border-radius:999px;border:1.5px solid;
  letter-spacing:.04em;
}}
.badge.general{{color:var(--general);border-color:var(--general);background:rgba(82,183,136,.1);}}
.badge.teen{{   color:var(--teen);   border-color:var(--teen);   background:rgba(72,149,239,.1);}}
.badge.mature{{ color:var(--mature); border-color:var(--mature); background:rgba(230,57,70,.1); }}
.verdict{{
  font-family:var(--mono);font-size:.7rem;
  padding:3px 10px;border-radius:4px;
}}
.verdict.correct{{color:var(--general);background:rgba(82,183,136,.12);}}
.verdict.wrong{{  color:var(--mature); background:rgba(230,57,70,.12); }}

/* ── Two-col ────────────────────────────────────────── */
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:28px;}}
@media(max-width:600px){{.two-col{{grid-template-columns:1fr;}}}}
.sub-title{{
  font-family:var(--mono);font-size:.64rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);margin-bottom:10px;
}}

/* ── Probability bars ───────────────────────────────── */
.prob-row{{display:flex;align-items:center;gap:8px;margin-bottom:7px;}}
.prob-lbl{{font-family:var(--mono);font-size:.68rem;width:56px;
           color:var(--muted);flex-shrink:0;}}
.prob-track{{flex:1;background:#1f1f27;border-radius:4px;height:9px;overflow:hidden;}}
.prob-fill{{height:100%;border-radius:4px;
           transition:width .55s cubic-bezier(.4,0,.2,1);}}
.prob-val{{font-family:var(--mono);font-size:.68rem;
          color:var(--muted);width:40px;text-align:right;}}
.conf-note{{font-size:.76rem;color:var(--muted);margin-top:10px;}}
.conf-note b{{color:var(--text);}}

/* ── Attention bars ─────────────────────────────────── */
.abar-scroll{{max-height:290px;overflow-y:auto;padding-right:2px;}}
.abar-row{{display:flex;align-items:center;gap:7px;margin-bottom:4px;}}
.abar-lbl{{font-family:var(--mono);font-size:.6rem;
          color:var(--muted);width:26px;flex-shrink:0;}}
.abar-track{{flex:1;background:#1b1b22;border-radius:3px;
            height:11px;position:relative;overflow:visible;}}
.abar-fill{{height:100%;border-radius:3px;transition:width .4s ease;}}
.abar-uni{{
  position:absolute;top:-3px;bottom:-3px;
  width:1.5px;background:rgba(255,255,255,.18);
}}
.abar-val{{font-family:var(--mono);font-size:.6rem;
          color:var(--muted);width:46px;text-align:right;}}

/* ── Chunk cards ─────────────────────────────────────── */
.chunk-list{{display:flex;flex-direction:column;gap:12px;}}
.chunk-card{{
  background:var(--surface2);border:1px solid var(--border);
  border-left:3px solid;border-radius:8px;padding:14px 18px;
  transition:transform .15s;
}}
.chunk-card:hover{{transform:translateX(4px);}}
.chunk-meta{{
  display:flex;align-items:center;gap:14px;
  font-family:var(--mono);font-size:.66rem;
  color:var(--muted);margin-bottom:8px;
}}
.chunk-medal{{font-size:1rem;}}
.chunk-w{{
  margin-left:auto;background:var(--border);border-radius:4px;
  padding:2px 8px;color:var(--gold);font-size:.63rem;
}}
.chunk-text{{
  font-family:var(--mono);font-size:.74rem;color:#b2afc0;
  line-height:1.65;white-space:pre-wrap;word-break:break-word;
  max-height:175px;overflow-y:auto;
}}

/* ── Footer ──────────────────────────────────────────── */
footer{{
  margin-top:80px;padding-top:22px;
  border-top:1px solid var(--border);
  text-align:center;font-size:.76rem;color:var(--muted);
}}
footer b{{color:var(--text);}}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:var(--surface2);}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:99px;}}
</style>
</head>
<body>
<div class="wrap">

<header class="site-header">
  <p class="overtitle">University AI Research · YouTube Age Classification · File 4 of 4</p>
  <h1>XAI Attention<br>Inspection Report</h1>
  <p class="intro">
    Chunk-level self-attention weights extracted from the Hierarchical RoBERTa model.
    Each card reveals which segments of the transcript most strongly influenced
    the classification decision, providing transparency for academic review.
  </p>
  <div class="meta-row">
    <span class="mpill"><b>Generated</b> {ts}</span>
    <span class="mpill"><b>Samples shown</b> {len(results)}</span>
    <span class="mpill"><b>Correct</b> {n_correct} / {len(results)}</span>
    <span class="mpill"><b>Chunking</b> 512 tok, 256 stride</span>
    <span class="mpill"><b>Aggregation</b> Self-Attention Pooling</span>
  </div>
  <div class="stat-row">
    <div class="stat-box">
      <div class="stat-val" style="color:var(--general)">G</div>
      <div class="stat-lbl">General</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:var(--teen)">T</div>
      <div class="stat-lbl">Teen</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:var(--mature)">M</div>
      <div class="stat-lbl">Mature</div>
    </div>
  </div>
</header>

<h2 class="sec-title">Prediction Explanations ({len(results)} samples)</h2>

{cards_html}

<footer>
  <p>
    <b>Hierarchical RoBERTa</b> · Self-Attention Pooling over
    sliding-window chunks (512 tok / 256 stride) ·
    Classes: General / Teen / Mature
  </p>
  <p style="margin-top:6px">
    Attention weights re-normalised after padding-chunk removal.
    Dashed line in attention bars = uniform-baseline (1/N chunks).
  </p>
</footer>

</div>
</body>
</html>"""

    out = Path(cfg.reports_dir) / "attention_report.html"
    out.write_text(html, encoding="utf-8")
    log.info(f"  HTML report → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(cfg: EvalConfig) -> None:
    """End-to-end pipeline: metrics → plots → XAI → HTML."""

    for d in [cfg.plots_dir, cfg.reports_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("\n╔" + "═" * 62 + "╗")
    print("║" + "  EVALUATE & EXPLAIN  —  YouTube Age Classifier".center(62) + "║")
    print("║" + "  File 4 of 4".center(62) + "║")
    print("╚" + "═" * 62 + "╝\n")

    # ── Artefacts ─────────────────────────────────────────────────────────
    log.info("Loading tokenizer …")
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)

    log.info("Loading model from checkpoint …")
    model = load_model(cfg)

    # ── 1. Inference ──────────────────────────────────────────────────────
    log.info("Running full test-set inference …")
    y_true, y_pred, y_probs = run_inference(model, cfg)

    # ── 2. Classification report ──────────────────────────────────────────
    report_str = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, digits=4
    )
    div = "═" * 60
    print(f"\n{div}\n  CLASSIFICATION REPORT\n{div}\n{report_str}")
    rpt = Path(cfg.reports_dir) / "classification_report.txt"
    rpt.write_text(f"Generated: {datetime.now().isoformat()}\n\n{report_str}")
    log.info(f"  Classification report → {rpt}")

    # ── 3. Evaluation plots ───────────────────────────────────────────────
    log.info("Generating evaluation plots …")
    plot_confusion_matrix(y_true, y_pred, cfg)
    plot_roc_curves(y_true, y_probs, cfg)
    plot_pr_curves(y_true, y_probs, cfg)
    plot_confidence_histogram(y_true, y_pred, y_probs, cfg)
    plot_per_class_metrics(y_true, y_pred, cfg)

    # ── 4. Error analysis ─────────────────────────────────────────────────
    log.info("Running error analysis …")
    run_error_analysis(y_true, y_pred, y_probs, cfg)

    # ── 5. Gather XAI transcripts ─────────────────────────────────────────
    log.info("Gathering transcripts for XAI explanations …")
    sample_pairs: List[Tuple[str, Optional[str]]] = []

    if Path(cfg.csv_path).exists():
        df_all = pd.read_csv(cfg.csv_path).dropna(subset=["transcript", "Age_Label"])
        # Two examples per class for a balanced, representative report
        for label in LABEL_NAMES:
            sub   = df_all[df_all["Age_Label"] == label]
            picks = sub.sample(min(2, len(sub)), random_state=42)
            for _, row in picks.iterrows():
                sample_pairs.append((str(row["transcript"]), label))
        sample_pairs = sample_pairs[: cfg.n_explain_samples]
    else:
        log.warning("CSV not found — using synthetic demo transcripts.")
        sample_pairs = [
            (
                "Welcome everyone! Today we explore how volcanoes form. "
                "This geology tutorial covers plate tectonics and famous eruptions. "
                "Great for science students. Subscribe for more educational content!",
                "General",
            ),
            (
                "Oh my god chat this Warzone drop was INSANE. Landed Caldera, "
                "thirsted two squads, clutched a 1v3 with zero ammo. My hands were "
                "shaking lol. Drop a like if you want the full ranked grind series. LFG!",
                "Teen",
            ),
            (
                "Content advisory: this video contains explicit language and discussions "
                "of substance abuse. We investigate the underground drug trade with "
                "interviews from former dealers. Viewer discretion is strongly advised.",
                "Mature",
            ),
        ]

    # ── 6. Run XAI ────────────────────────────────────────────────────────
    log.info("Running explain_prediction() …")
    xai_results: List[ExplanationResult] = []

    for text, true_lbl in sample_pairs:
        res = explain_prediction(text, model, tokenizer, cfg, true_label=true_lbl)
        xai_results.append(res)

        ok   = (true_lbl is None) or (res.predicted_label == true_lbl)
        icon = "✅" if ok else "❌"
        print(
            f"  {icon}  Pred={res.predicted_label:8s}  "
            f"conf={res.confidence:.1%}  "
            f"true={str(true_lbl):8s}  "
            f"chunks={res.num_chunks_used}"
        )
        top = res.top3_chunks[0]
        print(
            f"       Top chunk [{top['weight']:.4f}]: "
            + textwrap.shorten(top["text"], 88, placeholder="…")
        )

        attn_path = (
            Path(cfg.plots_dir) / f"attention_sample_{len(xai_results)}.png"
        )
        plot_attention_bar(res, attn_path, cfg)

    # ── 7. HTML report ────────────────────────────────────────────────────
    log.info("Generating HTML attention report …")
    html_path = generate_html_report(xai_results, cfg)

    # ── 8. JSON summary ───────────────────────────────────────────────────
    summary = {
        "generated_at":     datetime.now().isoformat(),
        "checkpoint":       cfg.checkpoint_path,
        "n_test_samples":   int(len(y_true)),
        "test_accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "test_f1_macro":    round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "test_f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "label_names":      LABEL_NAMES,
        "plots_saved": [
            "confusion_matrix.png", "roc_curves.png", "pr_curves.png",
            "confidence_histogram.png", "per_class_metrics.png",
        ],
    }
    sp = Path(cfg.reports_dir) / "eval_summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    log.info(f"  JSON summary → {sp}")

    # ── Final banner ──────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print("  ✅  EVALUATION COMPLETE")
    print(f"{'═'*62}")
    print(f"  Test Accuracy    : {summary['test_accuracy']:.4f}")
    print(f"  Test F1 Macro    : {summary['test_f1_macro']:.4f}")
    print(f"  Test F1 Weighted : {summary['test_f1_weighted']:.4f}")
    print(f"\n  Plots    → {cfg.plots_dir}/")
    print(f"  Reports  → {cfg.reports_dir}/")
    print(f"  HTML XAI → {html_path}")
    print(f"{'═'*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_evaluation(EvalConfig())
