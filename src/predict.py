"""Inference: load a fine-tuned checkpoint and classify new messages.

The :class:`SMSClassifier` wraps tokenizer + model + cleaning into a single
``predict()`` call. It also surfaces any raw URLs found in a message — the hand-off
point to the companion ``malicious-url-detector`` (TF-IDF word/char n-grams →
LinearSVC, 4-class: benign / phishing / malware / defacement) project:

    SMS spam detection  →  spam contains a URL  →  URL threat scan (4-class)

Usage::

    python -m src.predict --model_dir checkpoints "URGENT: verify your account at http://bit.ly/x"
    python -m src.predict --model_dir checkpoints --interactive
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional, Sequence

from src.utils import ID2LABEL, clean_sms_text

# Raw URL extractor (operates on the *original* message so the actual link is
# preserved for downstream scanning, not the [URL] placeholder).
_RAW_URL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9.-]*\.(?:com|net|org|info|biz|"
    r"co|io|ly|me|uk|cn|ru|xyz|top|link|click|tk|app|online|site|win|vip)"
    r"(?:/\S*)?)",
    flags=re.IGNORECASE,
)


def extract_urls(text: str) -> List[str]:
    """Return any raw URLs present in ``text`` (for the URL-scan hand-off)."""
    return _RAW_URL_RE.findall(text or "")


class SMSClassifier:
    """Load a fine-tuned checkpoint and classify SMS messages.

    Args:
        model_dir: Directory written by :meth:`BertSMSClassifier.save_pretrained`
            (contains the encoder, ``head.pt`` and the tokenizer).
        max_length: Token truncation length (should match training).
        device: Optional explicit device; auto-selected if omitted.
    """

    def __init__(self, model_dir: str = "checkpoints", max_length: int = 128, device=None) -> None:
        import torch
        from transformers import AutoTokenizer

        from src.model import BertSMSClassifier
        from src.train import get_device

        self.device = device or get_device()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = BertSMSClassifier.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self._torch = torch

    def predict(self, text: str) -> Dict:
        """Classify a single message.

        Returns:
            Dict with ``text`` (original), ``cleaned``, ``label`` (ham/spam),
            ``confidence`` (P of the predicted class), ``p_spam`` and ``urls``
            (raw URLs found — the bridge to the URL detector).
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> List[Dict]:
        """Classify a list of messages in a single forward pass."""
        torch = self._torch
        cleaned = [clean_sms_text(t) for t in texts]
        encoding = self.tokenizer(
            cleaned,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            probs = torch.softmax(logits, dim=-1)

        results: List[Dict] = []
        for original, clean, prob in zip(texts, cleaned, probs):
            p_spam = float(prob[1])
            pred_id = int(prob.argmax())
            results.append(
                {
                    "text": original,
                    "cleaned": clean,
                    "label": ID2LABEL[pred_id],
                    "confidence": round(float(prob[pred_id]), 4),
                    "p_spam": round(p_spam, 4),
                    "urls": extract_urls(original),
                }
            )
        return results


def _format_result(r: Dict) -> str:
    """Pretty one-block summary of a prediction for the CLI."""
    flag = "🚨 SPAM" if r["label"] == "spam" else "✅ HAM"
    lines = [
        f"{flag}  (p_spam={r['p_spam']:.4f}, confidence={r['confidence']:.4f})",
        f"  text: {r['text']}",
    ]
    if r["label"] == "spam" and r["urls"]:
        lines.append(f"  ⚠️  URLs detected → forward to malicious-url-detector: {r['urls']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Classify SMS messages with a fine-tuned BERT model.")
    parser.add_argument("message", nargs="*", help="Message(s) to classify.")
    parser.add_argument("--model_dir", default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--interactive", action="store_true", help="Read messages from stdin in a loop.")
    args = parser.parse_args(argv)

    clf = SMSClassifier(model_dir=args.model_dir, max_length=args.max_length)

    if args.interactive:
        print("Interactive mode — type a message and press Enter (Ctrl-D to quit).")
        try:
            while True:
                line = input("> ").strip()
                if line:
                    print(_format_result(clf.predict(line)))
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
        return

    if not args.message:
        parser.error("provide a message, or use --interactive")

    text = " ".join(args.message)
    print(_format_result(clf.predict(text)))


if __name__ == "__main__":
    main()
