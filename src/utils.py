"""Shared utilities: text cleaning, label maps, config loading, seeding.

The single most important function here is :func:`clean_sms_text`. It does **not**
strip spam signals out of a message — it *normalises* them into typed
placeholders (``[URL]``, ``[PHONE]``, ``[EMAIL]``, ``[MONEY]``). The presence of
a link or a money amount is itself a strong spam/phishing feature, so we want the
model to learn "this message contains a URL" as a stable token rather than
memorising thousands of unique, never-again-seen URLs.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Label maps — the single source of truth for class <-> id conversion.
# ---------------------------------------------------------------------------
LABEL2ID: Dict[str, int] = {"ham": 0, "spam": 1}
ID2LABEL: Dict[int, str] = {0: "ham", 1: "spam"}

# Placeholder tokens injected by clean_sms_text(). Exposed so callers/tests can
# reference them without hard-coding strings.
PLACEHOLDERS = ("[URL]", "[PHONE]", "[EMAIL]", "[MONEY]")

# ---------------------------------------------------------------------------
# Pre-compiled regexes. Order of application matters (see clean_sms_text):
# EMAIL before URL (an email's domain would otherwise be eaten by the bare-domain
# URL rule); MONEY before PHONE (both contain digits, and the currency symbol
# disambiguates money cleanly).
# ---------------------------------------------------------------------------

# user@host.tld
_EMAIL_RE = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")

# http(s)://..., www...., or a bare domain ending in a common TLD (+ optional path)
_URL_RE = re.compile(
    r"(?:https?://\S+"
    r"|www\.\S+"
    r"|\b[a-z0-9][a-z0-9.-]*\.(?:com|net|org|info|biz|co|io|ly|me|uk|cn|ru|"
    r"xyz|top|link|click|tk|app|online|site|win|vip)(?:/\S*)?)"
)

# £1,000  $200  €9.99  ¥500  100usd  50 dollars
_MONEY_RE = re.compile(
    r"[$£€¥]\s?\d[\d,]*(?:\.\d+)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:usd|gbp|eur|jpy|dollars?|pounds?|euros?)\b"
)

# +44 7700 900000, (212) 555-0199, 08001234567 — at least 7 phone-like chars
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{5,}\d(?!\w)")

# Runs of whitespace -> a single space
_WHITESPACE_RE = re.compile(r"\s+")


def clean_sms_text(text: str) -> str:
    """Normalise a raw SMS string for tokenisation.

    Pipeline (order is deliberate):

    1. lowercase
    2. emails   -> ``[EMAIL]``
    3. URLs     -> ``[URL]``
    4. money    -> ``[MONEY]``
    5. phones   -> ``[PHONE]``
    6. collapse repeated whitespace and strip

    Args:
        text: A raw SMS message (anything ``str()``-able; ``None`` becomes "").

    Returns:
        The cleaned message with spam signals replaced by typed placeholders.

    Example:
        >>> clean_sms_text("WINNER!! Claim your £1000 prize at http://bit.ly/x now")
        'winner!! claim your [MONEY] prize at [URL] now'
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _URL_RE.sub("[URL]", text)
    text = _MONEY_RE.sub("[MONEY]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    NumPy/PyTorch are imported lazily so that lightweight callers (e.g. the test
    suite, which only needs the regexes) do not pay the heavy import cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy always present in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML training config into a plain dict.

    Args:
        path: Path to a YAML file (e.g. ``configs/default.yaml``).

    Returns:
        The parsed configuration as a dictionary.
    """
    import yaml

    with open(Path(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
