"""PyTorch ``Dataset`` and ``DataLoader`` helpers for the SMS corpus.

The dataset performs tokenisation lazily inside ``__getitem__`` so that the full
corpus is never materialised as padded tensors in memory — only the batch
currently being consumed is tokenised. Cleaning (:func:`src.utils.clean_sms_text`)
is applied once up front in the constructor.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import clean_sms_text


class SMSDataset(Dataset):
    """A tokenising dataset of SMS messages.

    Args:
        texts: Raw SMS strings.
        labels: Integer labels aligned with ``texts`` (0 = ham, 1 = spam).
        tokenizer: A HuggingFace tokenizer (e.g. ``BertTokenizerFast``).
        max_length: Token truncation / padding length.
        clean: Whether to apply :func:`clean_sms_text` to each message. Defaults
            to ``True``; set ``False`` only for ablation experiments.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 128,
        clean: bool = True,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) length mismatch"
            )
        self.texts: List[str] = [
            clean_sms_text(t) if clean else str(t) for t in texts
        ]
        self.labels: List[int] = [int(l) for l in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # The tokenizer returns shape (1, max_length); squeeze the batch dim so
        # the default collate_fn can re-stack into (batch, max_length).
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_dataloaders(
    splits: Dict[str, Tuple[Sequence[str], Sequence[int]]],
    tokenizer,
    max_length: int = 128,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders from pre-split text/label pairs.

    Args:
        splits: Mapping of split name -> ``(texts, labels)``. Conventionally the
            keys are ``"train"``, ``"val"`` and ``"test"``.
        tokenizer: HuggingFace tokenizer shared across splits.
        max_length: Token truncation / padding length.
        batch_size: Mini-batch size (applied to every split).
        num_workers: DataLoader worker processes.

    Returns:
        Mapping of split name -> ``DataLoader``. The ``"train"`` loader is
        shuffled; all others preserve order so predictions line up with the
        original rows for error analysis.
    """
    loaders: Dict[str, DataLoader] = {}
    for name, (texts, labels) in splits.items():
        dataset = SMSDataset(texts, labels, tokenizer, max_length=max_length)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
