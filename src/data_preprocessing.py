"""Data loading, cleaning and reusable text preprocessing pipeline."""

import os
import re
import string
import nltk

# Ensure required NLTK resources are available
for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "SMSSpamCollection")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "sms_spam_cleaned.csv")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Keep digits and common currency symbols that are meaningful spam indicators.
KEEP_CHARS = set("$£€") | set(string.digits)


def _ensure_str(text):
    if text is None:
        return ""
    return str(text)


def remove_html(text):
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text):
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_emails(text):
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_punctuation(text):
    # Keep letters, digits, whitespace and a few meaningful symbols.
    out = []
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch in KEEP_CHARS:
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def clean_text(text, lemmatize=True, remove_stopwords=True):
    text = _ensure_str(text)
    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    cleaned = " ".join(tokens)
    # handle empty after processing
    if not cleaned.strip():
        return ""
    return cleaned


def load_raw():
    rows = []
    with open(RAW_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            label, msg = parts
            rows.append({"label": label.strip(), "message": msg})
    import pandas as pd
    df = pd.DataFrame(rows)
    return df


def build_processed():
    import pandas as pd
    df = load_raw()
    # Missing values
    df["message"] = df["message"].fillna("")
    df["label"] = df["label"].fillna("")
    # Duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after} duplicate rows. Remaining: {after}")

    df["cleaned_text"] = df["message"].apply(clean_text)
    # handle empty messages: keep but mark
    df["is_empty_after_clean"] = df["cleaned_text"].apply(lambda x: 1 if str(x).strip() == "" else 0)

    # Encode label
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed dataset saved to {PROCESSED_PATH} (shape={df.shape})")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    build_processed()