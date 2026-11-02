"""TF-IDF feature extraction utilities."""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


def build_vectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )


def fit_transform(vectorizer, train_texts):
    return vectorizer.fit_transform(train_texts)


def transform(vectorizer, texts):
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path=VECTORIZER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path=VECTORIZER_PATH):
    return joblib.load(path)