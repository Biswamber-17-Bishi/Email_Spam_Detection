"""Train and compare Naive Bayes, Logistic Regression and Linear SVM."""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import PROCESSED_PATH
from feature_extraction import build_vectorizer, save_vectorizer

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def main():
    df = pd.read_csv(PROCESSED_PATH)
    df = df.dropna(subset=["cleaned_text", "label_num"])
    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
    # drop rows where cleaned text empty
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

    X = df["cleaned_text"].values
    y = df["label_num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = build_vectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    save_vectorizer(vectorizer)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=2000),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        # LinearSVC decision; predict ok
        res = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0),
        }
        results.append(res)
        # save model
        safe = name.lower().replace(" ", "_")
        path = os.path.join(MODELS_DIR, f"{safe}.pkl")
        joblib.dump(model, path)
        print(f"Trained {name}: {res}")
    # fix filenames to match required: naive_bayes.pkl, logistic_regression.pkl, svm.pkl
    mapping = {
        "naive_bayes": "naive_bayes.pkl",
        "logistic_regression": "logistic_regression.pkl",
        "linear_svm": "svm.pkl",
    }
    for src_name, dst in mapping.items():
        src_path = os.path.join(MODELS_DIR, f"{src_name}.pkl")
        dst_path = os.path.join(MODELS_DIR, dst)
        if os.path.exists(src_path) and src_path != dst_path:
            os.replace(src_path, dst_path)

    # Save test data for evaluation
    import numpy as np
    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
    # store split indices via saving arrays
    import joblib as jb
    jb.dump((X_test, y_test, X_test_vec), os.path.join(BASE_DIR, "data", "processed", "test_split.pkl"))

    results_df = pd.DataFrame(results)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print("\nModel comparison saved.")
    print(results_df)


if __name__ == "__main__":
    import joblib
    main()