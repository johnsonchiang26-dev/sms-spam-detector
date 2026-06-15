# Model Card — BERT SMS Spam & Phishing Detector

> A fine-tuned `bert-base-uncased` classifier that labels SMS messages as **ham**
> (legitimate) or **spam** (unsolicited / phishing). Built as the first stage of
> a two-model security pipeline (see *Pipeline context* below).

---

## Model details

| | |
|---|---|
| **Developer** | Johnson Chiang (personal portfolio project) |
| **Model type** | Transformer encoder + linear classification head |
| **Base model** | `bert-base-uncased` (12 layers, 768 hidden, 110M params) |
| **Head** | `[CLS]` → Dropout(0.3) → Linear(768 → 2) |
| **Task** | Binary text classification (ham / spam) |
| **Language** | English |
| **Framework** | PyTorch + HuggingFace `transformers` (hand-written training loop, no `Trainer`) |
| **License** | MIT (code); UCI dataset under its own terms |

### Architecture

```
raw SMS
   │  clean_sms_text()  (URL→[URL], phone→[PHONE], email→[EMAIL], money→[MONEY])
   ▼
BertTokenizer (max_length = 128)
   ▼
BERT encoder (12 × transformer blocks)
   ▼
[CLS] token  (768-d)
   ▼
Dropout(p = 0.3)
   ▼
Linear(768 → 2)
   ▼
logits → softmax → {ham, spam}
```

---

## Intended use

- **Primary use**: Educational / portfolio demonstration of fine-tuning BERT for
  short-text security classification, and screening SMS for spam/phishing in a
  research or prototype setting.
- **Pipeline context**: Designed to feed a second-stage **URL maliciousness
  detector** (`malicious-url-detector` — feature engineering + TF-IDF word/char
  n-grams → LinearSVC, 4-class: benign / phishing / malware / defacement). When
  this model flags a message as spam *and* a URL is present, the raw URL is
  handed off for deeper analysis:

  ```
  SMS spam detection  →  spam contains a URL  →  URL threat scan (4-class)
  ```

- **Out-of-scope use**: Not validated for production message blocking,
  non-English text, or as a sole authority for irreversible actions (e.g.
  permanently blocking a sender). See *Limitations*.

---

## Training data

| Source | Messages | Notes |
|---|---|---|
| **UCI SMS Spam Collection** | 5,574 | ~87% ham / ~13% spam; public academic corpus |
| **Curated financial phishing** | 15 | Hand-authored OTP/KYC/trading-platform phishing + legitimate finance SMS |

The curated examples bridge the academic corpus to **real attack families**
observed in production SMS traffic: OTP phishing with spoofed verification URLs,
credential harvesting via randomized subdomains, and SMS-pumping with
URL-shortened redirect chains.

### Preprocessing

`clean_sms_text()` normalizes — rather than removes — spam signals into typed
placeholders, because the *presence* of a link/phone/amount is itself predictive:

| Pattern | Placeholder |
|---|---|
| URLs (`http(s)://`, `www.`, bare domains) | `[URL]` |
| Phone numbers | `[PHONE]` |
| Email addresses | `[EMAIL]` |
| Money (`$ £ € ¥` + amounts) | `[MONEY]` |

Text is lowercased and whitespace-collapsed. Replacement (not deletion) prevents
the model from overfitting to specific never-again-seen URLs while still learning
"this message contains a URL."

### Splits

70 / 10 / 20 train / validation / test, **stratified** by label to preserve the
ham:spam ratio across all three splits.

---

## Training procedure

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight decay excluded from bias & LayerNorm) |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Schedule | Linear warmup (10% of steps) → linear decay |
| Batch size | 16 |
| Max epochs | 4 |
| Gradient clipping | max-norm 1.0 |
| Early stopping | on validation F1, patience = 2 |
| Loss | Cross-entropy |
| Seed | 42 |

The best checkpoint is selected by **validation F1 of the spam class** (not
accuracy), because accuracy is misleading under class imbalance.

---

## Evaluation

Metrics are reported on the held-out **test** split, with **spam as the positive
class**. A TF-IDF + Logistic Regression model is trained on the same splits as a
baseline.

### Expected results

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.975 | ~0.99 | ~0.83 | ~0.90 | ~0.98 |
| **BERT (this model)** | **~0.991** | **~0.98** | **~0.98** | **~0.981** | **~0.99** |

> Figures are representative of a typical fine-tuning run on this corpus and may
> vary by a few tenths of a percent with seed / hardware. The notebook
> regenerates the exact numbers and saves the confusion-matrix and ROC plots to
> `docs/`.

The headline gain of BERT over the baseline is **recall** — it catches the
obfuscated / reworded phishing messages that bag-of-words misses, which is the
metric that matters most for a security filter.

### Error analysis

False positives (legit blocked) and false negatives (spam missed) are extracted
with their confidence scores in the notebook. Typical failure modes:

- **FN**: terse spam with no link/keyword ("call me", number-only messages).
- **FP**: legitimate marketing or contest-style ham with urgent language.

---

## Limitations

- **Small, single-domain dataset** (~5.6k messages); may not generalize to other
  carriers, regions, or message styles.
- **English only** — the tokenizer and corpus are English; performance on other
  languages is undefined.
- **Class imbalance** (~13% spam) — mitigated by stratification and F1-based
  selection, but rare spam sub-types are under-represented.
- **No adversarial testing** — not evaluated against deliberate evasion
  (homoglyphs, character spacing, zero-width chars). A motivated attacker can
  likely evade it.
- **Temporal drift** — spam tactics evolve; the model will degrade without
  periodic retraining.

---

## Ethical considerations

- **False positives have real cost**: wrongly flagging a legitimate message
  (e.g. a genuine bank OTP) could block important communication. The model
  should *assist* human/automated review, not unilaterally block messages.
- **Privacy**: SMS content is sensitive. This project trains only on public /
  synthetic data and stores no personal messages.
- **Dual-use**: spam-classification research can inform evasion. The intent here
  is defensive (filtering), consistent with anti-abuse / threat-detection work.

---

## How to reproduce

```bash
pip install -r requirements.txt
python data/download_data.py            # fetch UCI corpus + curated phishing CSV
python -m src.train --config configs/default.yaml
python -m src.predict --model_dir checkpoints "verify your account at http://bit.ly/x"
```

Or open `notebooks/03_bert_finetune.ipynb` for the full annotated walkthrough.
