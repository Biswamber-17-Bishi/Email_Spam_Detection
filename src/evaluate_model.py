"""Evaluate saved models and generate plots/reports."""

import os
import sys
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "reports", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

MODEL_FILES = {
    "Naive Bayes": "naive_bayes.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Linear SVM": "svm.pkl",
}


def main():
    X_test, y_test, X_test_vec = joblib.load(os.path.join(BASE_DIR, "data", "processed", "test_split.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))

    rows = []
    for name, fname in MODEL_FILES.items():
        model = joblib.load(os.path.join(BASE_DIR, "models", fname))
        preds = model.predict(X_test_vec)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0),
        })
        # confusion matrix plot
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"))
        plt.close()

        with open(os.path.join(EVAL_DIR, f"classification_report_{name.replace(' ', '_').lower()}.txt"), "w") as f:
            f.write(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(BASE_DIR, "reports", "model_comparison.csv"), index=False)
    print(results_df)

    # bar chart comparison
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.title("Model Comparison")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "model_comparison.png"))
    plt.close()
    print("Plots saved to", EVAL_DIR)


if __name__ == "__main__":
    main()