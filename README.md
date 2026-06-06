# SMS Spam & Phishing Detection with BERT

Fine-tuning `bert-base-uncased` to classify SMS messages as **ham** (legitimate)
or **spam / phishing**, with a preprocessing pipeline tuned for real-world abuse
signals (URLs, OTP codes, money amounts) and a strong classical baseline for
comparison.

> **This is stage 1 of a two-model SMS security pipeline.** A message is first
> screened for spam/phishing here; when it is flagged **and** carries a URL, the
> raw link is escalated to the companion
> [`malicious-url-detector`](https://github.com/johnsonchiang26-dev/malicious-url-detector)
> (a character-level CNN) for link-level threat scoring.

```
   ┌──────────────────────────┐    spam + contains URL    ┌───────────────────────────┐
   │   SMS Spam Detection      │ ────────────────────────▶ │  Malicious URL Detection  │
   │   BERT  ·  this repo      │                           │  CharCNN  ·  companion    │
   └──────────────────────────┘                           └───────────────────────────┘
        ham / spam decision                                    benign / malicious URL
```

This mirrors how a production SMS abuse-detection system triages a message, then
escalates the embedded payload — the exact shape of problem I work on day to day
(see [Professional context](#professional-context)).

---

## Architecture

```
raw SMS
   │  clean_sms_text()
   │     URL   → [URL]      phone → [PHONE]
   │     email → [EMAIL]    money → [MONEY]   (+ lowercase, collapse whitespace)
   ▼
BertTokenizer (bert-base-uncased, max_length = 128)
   ▼
BERT encoder  ─  12 transformer layers, 768 hidden
   ▼
[CLS] token (768-d)
   ▼
Dropout(0.3)  →  Linear(768 → 2)
   ▼
softmax → { ham, spam }
```

**Design choice — normalise, don't delete.** The preprocessor *replaces* spam
signals with typed placeholders instead of stripping them, because the presence
of a link / phone / money amount is itself a strong feature, while the specific
value (a one-off URL) is noise the model should not memorise.

---

## Results

Held-out test set (20%, stratified). **Spam is the positive class** —
precision / recall / F1 are reported for spam.

| Model | Accuracy | Precision | Recall | F1 (spam) | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.975 | ~0.99 | ~0.83 | ~0.90 | ~0.98 |
| **BERT (fine-tuned)** | **~0.991** | **~0.98** | **~0.98** | **~0.981** | **~0.99** |

The headline win for BERT is **recall** — it catches the obfuscated / reworded
phishing that bag-of-words misses, which is the metric that matters most for a
security filter. Plots (`class_distribution`, `message_length_distribution`,
`confusion_matrices`, `roc_curves`) are regenerated into [`docs/`](docs/) by the
notebook. Full details in the [model card](docs/model_card.md).

---

## Quick start

```bash
# 1. clone & install
git clone https://github.com/johnsonchiang26-dev/sms-spam-detector.git
cd sms-spam-detector
pip install -r requirements.txt          # or: pip install -e .

# 2. download data (UCI corpus + curated financial-phishing examples)
python data/download_data.py

# 3a. run the full annotated walkthrough
jupyter notebook notebooks/03_bert_finetune.ipynb

# 3b. ...or train from the CLI
python -m src.train --config configs/default.yaml

# 4. classify a new message
python -m src.predict --model_dir checkpoints \
    "URGENT: verify your account at http://bit.ly/x or it will be locked"
```

Example inference output:

```
🚨 SPAM  (p_spam=0.9971, confidence=0.9971)
  text: URGENT: verify your account at http://bit.ly/x or it will be locked
  ⚠️  URLs detected → forward to malicious-url-detector: ['http://bit.ly/x']
```

Run the tests:

```bash
pytest tests/        # text-cleaning, label-map and URL-extraction unit tests
```

---

## Project structure

```
sms-spam-detector/
├── configs/default.yaml         # training hyperparameters (YAML)
├── data/download_data.py        # UCI SMS Spam Collection + curated phishing CSV
├── docs/
│   ├── model_card.md            # architecture, training, results, limitations
│   └── *.png                    # generated EDA / evaluation figures
├── notebooks/
│   └── 03_bert_finetune.ipynb   # full walkthrough (EDA → baseline → BERT → demo)
├── src/
│   ├── utils.py                 # clean_sms_text(), label maps, seeding, config
│   ├── dataset.py               # SMSDataset + create_dataloaders()
│   ├── model.py                 # BertSMSClassifier ([CLS] → dropout → linear)
│   ├── train.py                 # hand-written training loop + argparse CLI
│   ├── evaluate.py              # metrics, error_analysis(), plotting
│   └── predict.py               # SMSClassifier inference + CLI (URL hand-off)
├── tests/test_predict.py        # pytest
├── requirements.txt
└── setup.py
```

---

## How it's trained

A **hand-written fine-tuning loop** (deliberately *not* `transformers.Trainer`)
to keep every mechanism explicit:

- **AdamW**, LR `2e-5`, weight decay `0.01` (excluded from bias & LayerNorm)
- **Linear warmup** over 10% of steps, then linear decay
- **Gradient clipping** at max-norm `1.0`
- **Early stopping** on validation **F1** (patience `2`) — accuracy is misleading
  under ~13% class imbalance
- **Best-checkpoint saving** by validation F1

Data is split **70 / 10 / 20**, stratified by label.

---

## Dataset

| Source | Messages | Notes |
|---|---|---|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | 5,574 | ~87% ham / ~13% spam |
| Curated financial phishing | 15 | hand-authored OTP / KYC / trading-platform phishing + legit finance SMS |

The curated examples connect the academic corpus to **real attack families** —
OTP phishing with spoofed verification URLs, credential harvesting via randomized
subdomains, and SMS-pumping with URL-shortened redirect chains.

---

## Tech stack

- **PyTorch** — model & hand-written training loop
- **HuggingFace Transformers** — `bert-base-uncased` encoder + tokenizer
- **scikit-learn** — TF-IDF + Logistic Regression baseline, metrics
- **pandas / NumPy** — data handling
- **matplotlib / seaborn** — EDA & evaluation visualisations
- **PyYAML** — config; **pytest** — testing; **Jupyter** — walkthrough

---

## Professional context

I currently work as a **Data Analyst / Data Engineer at a fintech company**, where
I own the SMS / risk module: an ETL pipeline over **90M+ SMS delivery records**,
including OTP-verification-rate monitoring and an anomaly-detection alerting system
(ABCD severity tiering). The recurring attack patterns I deal with — OTP phishing
with spoofed verification URLs, credential stuffing with randomized subdomains, and
SMS-pumping with URL-shortened redirect chains — directly motivated this project
and its companion URL detector. It's a focused, end-to-end reproduction of the
**detect → escalate** pattern that abuse-detection teams (e.g. at Trend Micro)
build at scale.

---

## License

MIT (code). The UCI SMS Spam Collection is distributed under its own terms.
