"""Model definition: a thin classification head on top of BERT.

Architecture
------------
    input_ids / attention_mask
        -> BERT encoder (bert-base-uncased: 12 layers, 768 hidden)
        -> pooled [CLS] representation
        -> Dropout(p)
        -> Linear(768 -> 2)
        -> logits {ham, spam}

We deliberately re-pool the ``[CLS]`` token ourselves (``last_hidden_state[:, 0]``)
rather than relying on BERT's built-in ``pooler_output``. The built-in pooler
applies an extra tanh-activated dense layer trained on the next-sentence-prediction
objective, which is not aligned with sentence classification; using the raw
``[CLS]`` hidden state is the more common and slightly cleaner choice for
fine-tuning.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class BertSMSClassifier(nn.Module):
    """BERT encoder + dropout + linear classification head.

    Args:
        model_name: HuggingFace encoder checkpoint.
        num_labels: Number of output classes (2 for ham/spam).
        dropout: Dropout probability applied to the [CLS] vector.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for bert-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Run a forward pass.

        Args:
            input_ids: ``(batch, seq_len)`` token ids.
            attention_mask: ``(batch, seq_len)`` 1/0 mask.
            labels: Optional ``(batch,)`` gold labels. If supplied, the
                cross-entropy loss is returned alongside the logits.

        Returns:
            ``(loss, logits)`` if ``labels`` is given, otherwise ``logits`` of
            shape ``(batch, num_labels)``.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]  # [CLS] token, (batch, 768)
        pooled = self.dropout(cls_vector)
        logits = self.classifier(pooled)

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits
        return logits

    def save_pretrained(self, output_dir: str) -> None:
        """Persist model weights and config to ``output_dir``.

        The encoder is saved via HuggingFace's own serialiser (so the tokenizer
        and config round-trip cleanly), and the classification head's state dict
        is saved alongside it.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)
        self.encoder.save_pretrained(output_dir)
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "num_labels": self.num_labels,
                "dropout_p": self.dropout.p,
            },
            os.path.join(output_dir, "head.pt"),
        )

    @classmethod
    def from_pretrained(cls, output_dir: str) -> "BertSMSClassifier":
        """Reload a model previously written by :meth:`save_pretrained`."""
        import os

        head = torch.load(
            os.path.join(output_dir, "head.pt"), map_location="cpu"
        )
        model = cls(
            model_name=output_dir,
            num_labels=head["num_labels"],
            dropout=head["dropout_p"],
        )
        model.classifier.load_state_dict(head["classifier"])
        return model
