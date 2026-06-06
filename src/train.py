"""Training entry point: a hand-written fine-tuning loop (no HF Trainer).

Run with defaults from the YAML config::

    python -m src.train --config configs/default.yaml

Override any hyperparameter from the CLI::

    python -m src.train --epochs 3 --batch_size 32 --learning_rate 3e-5

The loop is written out explicitly (optimizer grouping, warmup schedule,
gradient clipping, per-epoch validation, F1-based early stopping, best-checkpoint
saving) rather than delegated to ``transformers.Trainer`` — the point of this
project is to demonstrate that the mechanics are understood, not hidden.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.utils import LABEL2ID, load_config, set_seed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataframe(data_path: str, phishing_path: str | None = None) -> pd.DataFrame:
    """Load the UCI corpus (+ optional phishing examples) into one DataFrame.

    Args:
        data_path: Path to the UCI ``SMSSpamCollection`` TSV (``label\\ttext``).
        phishing_path: Optional CSV with ``label,text`` columns of hand-written
            financial phishing / legitimate finance messages.

    Returns:
        DataFrame with columns ``label`` (str), ``text`` (str), ``label_id`` (int),
        ``source`` ("uci" | "phishing").
    """
    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    df["source"] = "uci"

    if phishing_path and os.path.exists(phishing_path):
        extra = pd.read_csv(phishing_path)
        extra = extra[["label", "text"]].copy()
        extra["source"] = "phishing"
        df = pd.concat([df, extra], ignore_index=True)

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(LABEL2ID)]
    df["label_id"] = df["label"].map(LABEL2ID)
    return df.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Stratified train/val/test split preserving the ham/spam ratio.

    The test ratio is inferred as ``1 - train_ratio - val_ratio``.

    Returns:
        Mapping ``{"train"|"val"|"test": (texts, label_ids)}``.
    """
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio
    # First carve off the test set, then split the remainder into train/val.
    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df["label_id"], random_state=seed
    )
    val_fraction = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val["label_id"],
        random_state=seed,
    )
    return {
        "train": (train["text"].tolist(), train["label_id"].tolist()),
        "val": (val["text"].tolist(), val["label_id"].tolist()),
        "test": (test["text"].tolist(), test["label_id"].tolist()),
    }


# ---------------------------------------------------------------------------
# Optimiser / device helpers
# ---------------------------------------------------------------------------
def build_optimizer(model, learning_rate: float, weight_decay: float):
    """AdamW with weight decay disabled for bias & LayerNorm parameters.

    Applying L2 decay to biases and LayerNorm gains hurts performance and is the
    standard BERT fine-tuning convention, so those parameters are split into a
    separate group with ``weight_decay = 0``.
    """
    from torch.optim import AdamW

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay_params if any(nd in name for nd in no_decay) else decay_params).append(param)

    grouped = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(grouped, lr=learning_rate)


def get_device():
    """Pick the best available device: CUDA > Apple MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, gradient_clip: float) -> float:
    """Run one training epoch; returns the mean training loss."""
    import torch
    from tqdm.auto import tqdm

    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def train(config: Dict) -> Dict:
    """End-to-end training given a resolved config dict.

    Returns a small summary dict with the best validation F1 and the held-out
    test metrics.
    """
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from src.dataset import create_dataloaders
    from src.evaluate import compute_metrics, predict_loader
    from src.model import BertSMSClassifier

    set_seed(config["seed"])
    device = get_device()
    print(f"[train] device = {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    df = load_dataframe(config["data_path"], config.get("phishing_path"))
    print(f"[train] loaded {len(df)} messages "
          f"({(df['label'] == 'spam').mean():.1%} spam)")
    splits = stratified_split(df, config["train_ratio"], config["val_ratio"], config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    loaders = create_dataloaders(
        splits,
        tokenizer,
        max_length=config["max_length"],
        batch_size=config["batch_size"],
    )

    # ── Model / optimiser / schedule ────────────────────────────────────────
    model = BertSMSClassifier(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = build_optimizer(model, config["learning_rate"], config["weight_decay"])
    total_steps = len(loaders["train"]) * config["epochs"]
    warmup_steps = int(config["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Epoch loop with F1 early stopping ───────────────────────────────────
    best_f1 = -1.0
    epochs_without_improvement = 0
    os.makedirs(config["output_dir"], exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, device, config["gradient_clip"]
        )
        y_true, y_pred, y_prob = predict_loader(model, loaders["val"], device)
        val_metrics = compute_metrics(y_true, y_pred, y_prob)
        print(
            f"[epoch {epoch}/{config['epochs']}] "
            f"train_loss={train_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            model.save_pretrained(config["output_dir"])
            tokenizer.save_pretrained(config["output_dir"])
            print(f"           ↳ new best val F1 = {best_f1:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["early_stopping_patience"]:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no val-F1 improvement in {config['early_stopping_patience']} epochs)")
                break

    # ── Final test evaluation with the best checkpoint ──────────────────────
    best_model = BertSMSClassifier.from_pretrained(config["output_dir"]).to(device)
    y_true, y_pred, y_prob = predict_loader(best_model, loaders["test"], device)
    test_metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\n[test] " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    return {"best_val_f1": best_f1, "test_metrics": test_metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for SMS spam detection.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    # Optional overrides — None means "use the config value".
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--data_path")
    parser.add_argument("--phishing_path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    # Apply any CLI overrides that were actually supplied.
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value
    train(config)


if __name__ == "__main__":
    main()
