"""Unit tests for text cleaning, label maps and URL extraction.

These tests are intentionally dependency-light: they import only ``src.utils``
and the pure-Python parts of ``src.predict`` (no torch/transformers/numpy), so
``pytest tests/`` runs fast and green without a model checkpoint or a GPU.
"""

import pytest

from src.predict import extract_urls
from src.utils import (
    ID2LABEL,
    LABEL2ID,
    PLACEHOLDERS,
    clean_sms_text,
)


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
def test_label_maps_are_consistent():
    assert LABEL2ID == {"ham": 0, "spam": 1}
    # ID2LABEL must be the exact inverse of LABEL2ID.
    assert ID2LABEL == {v: k for k, v in LABEL2ID.items()}
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label


# ---------------------------------------------------------------------------
# clean_sms_text — placeholder substitution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, placeholder",
    [
        ("Claim your prize at http://bit.ly/abc now", "[URL]"),
        ("Visit www.free-prize.com today", "[URL]"),
        ("Go to secure-login.xyz/verify immediately", "[URL]"),
        ("Call 08001234567 to claim", "[PHONE]"),
        ("Ring +44 7700 900000 now", "[PHONE]"),
        ("Email us at winner@lottery.com", "[EMAIL]"),
        ("You won £1000 cash", "[MONEY]"),
        ("Transfer of $4,800 pending", "[MONEY]"),
        ("Prize of €9.99 waiting", "[MONEY]"),
    ],
)
def test_clean_sms_text_inserts_expected_placeholder(raw, placeholder):
    cleaned = clean_sms_text(raw)
    assert placeholder in cleaned, f"{placeholder} missing in: {cleaned!r}"


def test_clean_sms_text_lowercases_and_collapses_whitespace():
    cleaned = clean_sms_text("FREE   entry    NOW")
    assert cleaned == "free entry now"


def test_clean_sms_text_replaces_not_removes():
    # The signal must be preserved as a typed token, not deleted.
    cleaned = clean_sms_text("win now: http://x.com")
    assert "[URL]" in cleaned
    assert "http" not in cleaned  # original URL text is gone...
    assert "win now" in cleaned   # ...but surrounding context stays.


def test_clean_sms_text_multiple_signals_in_one_message():
    raw = "WINNER! Claim £1000 at http://scam.tk or call 09061234567"
    cleaned = clean_sms_text(raw)
    for ph in ("[MONEY]", "[URL]", "[PHONE]"):
        assert ph in cleaned


def test_clean_sms_text_email_not_swallowed_by_url_rule():
    # An email's domain must become [EMAIL], not [URL].
    cleaned = clean_sms_text("contact billing@paypal.com for details")
    assert "[EMAIL]" in cleaned
    assert "[URL]" not in cleaned


@pytest.mark.parametrize("bad_input", [None, "", 12345, "   "])
def test_clean_sms_text_handles_edge_inputs(bad_input):
    # Should never raise, always return a (possibly empty) string.
    result = clean_sms_text(bad_input)
    assert isinstance(result, str)


def test_known_placeholders_constant():
    assert set(PLACEHOLDERS) == {"[URL]", "[PHONE]", "[EMAIL]", "[MONEY]"}


def test_ham_message_is_unchanged_except_case():
    cleaned = clean_sms_text("Ok lah see you later at the cafe")
    assert cleaned == "ok lah see you later at the cafe"
    for ph in PLACEHOLDERS:
        assert ph not in cleaned


# ---------------------------------------------------------------------------
# extract_urls — the bridge to malicious-url-detector
# ---------------------------------------------------------------------------
def test_extract_urls_finds_links():
    urls = extract_urls("verify at http://secure-verify.com/login or www.bank.tk now")
    assert any("secure-verify.com" in u for u in urls)
    assert any("bank.tk" in u for u in urls)


def test_extract_urls_empty_when_none():
    assert extract_urls("hey are we still on for lunch tomorrow?") == []


def test_extract_urls_handles_none():
    assert extract_urls(None) == []
