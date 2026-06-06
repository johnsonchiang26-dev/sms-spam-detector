"""Evaluation: metric computation, inference over a loader, and error analysis.

``spam`` (label 1) is treated as the positive class throughout — for a spam
filter the costly mistakes are asymmetric, so precision/recall/F1 are reported
with respect to the spam class, not just overall accuracy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import ID2LABEL

SPAM_ID = 1  # positive class


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compute headline classification metrics (spam = positive class).

    Args:
        y_true: Gold integer labels.
        y_pred: Predicted integer labels.
        y_prob: Optional predicted P(spam); enables ROC-AUC.

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, ``f1`` (all for the
        spam class), ``f1_macro`` and — when ``y_prob`` is given — ``roc_auc``.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=SPAM_ID, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=SPAM_ID, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=SPAM_ID, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_prob)))
    return metrics


def predict_loader(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and collect labels, predictions and P(spam).

    Args:
        model: A :class:`~src.model.BertSMSClassifier` (or compatible) module.
        loader: A ``DataLoader`` yielding ``input_ids``/``attention_mask``/``labels``.
        device: Torch device to run inference on.

    Returns:
        ``(y_true, y_pred, y_prob)`` as numpy arrays, where ``y_prob`` is the
        softmax probability of the spam class.
    """
    import torch

    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, SPAM_ID]
            preds = logits.argmax(dim=-1)

            y_true.extend(batch["labels"].cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


def error_analysis(
    texts: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
):
    """Extract the model's mistakes (false positives & false negatives).

    Args:
        texts: Original (raw) messages aligned with the predictions.
        y_true: Gold labels.
        y_pred: Predicted labels.
        y_prob: Predicted P(spam) for each message.

    Returns:
        A pandas DataFrame of misclassified rows, sorted by descending model
        confidence in the (wrong) prediction, with columns: ``text``,
        ``true``, ``pred``, ``p_spam``, ``error_type`` (``FP``/``FN``).
    """
    import pandas as pd

    rows = []
    for text, t, p, prob in zip(texts, y_true, y_pred, y_prob):
        if t == p:
            continue
        # FP = ham predicted spam; FN = spam predicted ham.
        error_type = "FP" if p == SPAM_ID else "FN"
        confidence = prob if p == SPAM_ID else 1.0 - prob
        rows.append(
            {
                "text": text,
                "true": ID2LABEL[int(t)],
                "pred": ID2LABEL[int(p)],
                "p_spam": round(float(prob), 4),
                "confidence": round(float(confidence), 4),
                "error_type": error_type,
            }
        )

    df = pd.DataFrame(rows, columns=["text", "true", "pred", "p_spam", "confidence", "error_type"])
    if not df.empty:
        df = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    return df


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix", save_path: Optional[str] = None):
    """Plot (and optionally save) a 2x2 confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(curves: Dict[str, Tuple[Sequence[int], Sequence[float]]], save_path: Optional[str] = None):
    """Overlay ROC curves for one or more models.

    Args:
        curves: Mapping of model name -> ``(y_true, y_prob)``.
        save_path: Optional path to write the figure to.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Baseline vs BERT")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
