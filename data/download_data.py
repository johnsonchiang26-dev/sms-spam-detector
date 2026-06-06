"""Download & assemble the training corpus.

1. Fetch the UCI **SMS Spam Collection** (5,574 labelled messages) and extract
   the ``SMSSpamCollection`` TSV into ``data/raw/``.
2. Write ``data/phishing_examples.csv`` — a small, hand-authored set of
   **financial** phishing messages (OTP scams, KYC harvesting, trading-platform
   credential phishing) plus legitimate finance SMS. These connect the public
   academic corpus to the real attack patterns seen in production SMS traffic
   and to the companion ``malicious-url-detector`` project.

Run::

    python data/download_data.py
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
UCI_MEMBER = "SMSSpamCollection"  # the TSV file inside the zip
PHISHING_CSV = DATA_DIR / "phishing_examples.csv"


# ---------------------------------------------------------------------------
# Hand-authored financial phishing / legitimate-finance examples.
#
# Columns: (label, category, text). Spam rows model concrete attack families;
# ham rows are realistic legitimate finance/banking notifications so the model
# learns that "contains a URL / OTP / money amount" is not, by itself, spam.
# ---------------------------------------------------------------------------
PHISHING_EXAMPLES = [
    # ── OTP / verification-code phishing (spoofed verification URLs) ─────────
    ("spam", "otp_phishing",
     "Your account verification code is 884213. Confirm now or your account will be suspended: http://secure-verify-account.com/login"),
    ("spam", "otp_phishing",
     "ALERT: Someone tried to access your bank account. Verify it's you within 15 mins: https://bit.ly/3xVrfy or service is locked."),
    ("spam", "otp_phishing",
     "We detected a login from a new device. If this wasn't you, secure your account immediately: http://account-security-check.vip/verify"),

    # ── KYC / identity-harvesting phishing ──────────────────────────────────
    ("spam", "kyc_phishing",
     "Dear customer, your KYC is incomplete. Update your details within 24h to avoid account freeze: http://kyc-update-portal.online/form"),
    ("spam", "kyc_phishing",
     "Your debit card will be blocked today due to pending KYC. Complete verification here: www.bank-kyc-verify.tk/update"),

    # ── Trading-platform / crypto credential phishing ───────────────────────
    ("spam", "trading_phishing",
     "Your crypto withdrawal of $4,800 is pending approval. Authorise it now or it will be cancelled: http://wallet-confirm.app/auth"),
    ("spam", "trading_phishing",
     "Congratulations! You've been selected for our exclusive trading signal group with 300% returns. Join: https://t.me/joinx-invest"),
    ("spam", "trading_phishing",
     "Your investment account has a credit of £2,000 ready to claim. Verify your identity to release funds: http://claim-funds-now.click"),

    # ── SMS pumping / shortened redirect chains ─────────────────────────────
    ("spam", "sms_pumping",
     "FINAL NOTICE: Your parcel is held at customs. Pay the £1.99 release fee here: http://tinyurl.com/parcel-fee-uk"),
    ("spam", "sms_pumping",
     "You have 1 unclaimed reward worth $500 from your bank loyalty program. Claim before it expires: http://rewards-redeem.link/go"),

    # ── Legitimate finance / banking notifications (ham) ─────────────────────
    ("ham", "legit_finance",
     "Your OTP for the transaction of GBP 45.00 at TESCO is 552190. Do not share this code with anyone."),
    ("ham", "legit_finance",
     "Your salary of $3,200 has been credited to your account ending 4471. Available balance: $5,118.22."),
    ("ham", "legit_finance",
     "Reminder: your credit card payment of $150 is due on the 15th. Log in to the app to pay or set up autopay."),
    ("ham", "legit_finance",
     "Thank you. We received your payment of £29.99. Your next billing date is 06 Jul. - Your subscription team"),
    ("ham", "legit_finance",
     "A withdrawal of $200 was made from ATM at Main St. If this wasn't you, call the number on the back of your card."),
]


def download_uci(force: bool = False) -> Path:
    """Download and extract the UCI SMS Spam Collection TSV.

    Args:
        force: Re-download even if the file already exists.

    Returns:
        Path to the extracted ``SMSSpamCollection`` TSV.
    """
    import requests

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / UCI_MEMBER

    if target.exists() and not force:
        print(f"[data] {target} already exists — skipping download.")
        return target

    print(f"[data] downloading {UCI_URL} ...")
    resp = requests.get(UCI_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # The archive contains 'SMSSpamCollection' and a readme.
        with zf.open(UCI_MEMBER) as src, open(target, "wb") as dst:
            dst.write(src.read())
    print(f"[data] extracted -> {target}")
    return target


def write_phishing_examples(force: bool = False) -> Path:
    """Write the hand-authored financial phishing CSV (label, text, category)."""
    import csv

    if PHISHING_CSV.exists() and not force:
        print(f"[data] {PHISHING_CSV} already exists — skipping.")
        return PHISHING_CSV

    with open(PHISHING_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "text", "category"])
        for label, category, text in PHISHING_EXAMPLES:
            writer.writerow([label, text, category])
    n_spam = sum(1 for r in PHISHING_EXAMPLES if r[0] == "spam")
    n_ham = len(PHISHING_EXAMPLES) - n_spam
    print(f"[data] wrote {PHISHING_CSV} ({n_spam} phishing / {n_ham} legit finance).")
    return PHISHING_CSV


def main() -> None:
    # The phishing CSV is offline data, so always write it first — even if the
    # UCI download fails (e.g. no network), the curated examples are available.
    write_phishing_examples()
    try:
        path = download_uci()
        n_lines = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"[data] UCI corpus ready: {n_lines} messages at {path}")
    except Exception as exc:  # noqa: BLE001 - surface any network/parse error clearly
        print(f"[data] WARNING: could not download UCI corpus ({exc}).")
        print(f"[data] Manually download {UCI_URL} and extract "
              f"'{UCI_MEMBER}' into {RAW_DIR}/")


if __name__ == "__main__":
    main()
