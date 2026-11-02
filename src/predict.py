"""Inference helpers for spam prediction."""

import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import clean_text

MODELS = {
    "Naive Bayes": "naive_bayes.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Linear SVM": "svm.pkl",
}
DEFAULT_MODEL = "Linear SVM"


def load_artifacts(model_name=DEFAULT_MODEL):
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))
    model = joblib.load(os.path.join(BASE_DIR, "models", MODELS[model_name]))
    return vectorizer, model


def predict_text(text, model_name=DEFAULT_MODEL):
    vectorizer, model = load_artifacts(model_name)
    cleaned = clean_text(text)
    if cleaned.strip() == "":
        return {"label": "Ham", "confidence": 0.5, "cleaned": cleaned}
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    label = "Spam" if pred == 1 else "Ham"
    # confidence
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = float(proba[pred])
    elif hasattr(model, "decision_function"):
        import math
        score = float(model.decision_function(vec)[0])
        confidence = 1 / (1 + math.exp(-abs(score)))
    else:
        confidence = 1.0
    return {"label": label, "confidence": confidence, "cleaned": cleaned}


if __name__ == "__main__":
    import sys as _sys
    text = " ".join(_sys.argv[1:]) if len(_sys.argv) > 1 else "Free entry in 2 a weekly comp to win FA Cup final tkts"
    print(predict_text(text))