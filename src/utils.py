import joblib
import numpy as np
import os
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path):
    """
    Load a model from the given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return joblib.load(model_path)


def predict(model, X):
    """
    Predict class labels.
    """
    return model.predict(X)


def predict_proba(model, X):
    """
    Predict class probabilities.
    """
    return model.predict_proba(X)[:, 1]


def evaluate_model(y_true, y_pred, y_proba):
    """
    Print evaluation metrics and return them as a dictionary.
    """
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ AUC Score: {auc:.4f}")
    print("✅ Confusion Matrix:\n", cm)
    print("✅ Classification Report:\n", cr)

    return {
        "accuracy": acc,
        "auc": auc,
        "confusion_matrix": cm,
        "classification_report": cr
    }


def plot_roc(y_true, y_proba, label='Model'):
    """
    Plot ROC Curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc_score(y_true, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix as heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
