"""
----
Helper functions for evaluating classification models, plotting results, and saving metrics.
----
Features:
    1) Compute classification scores (accuracy, precision, recall, ROC-AUC).
    2) Plot confusion matrix, ROC curve, PR curve, and training curve.
    3) Save metrics JSON and Keras model summary.
    4) One high-level function to evaluate, plot, save, and display results.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from src.config import FIGURES_DIR, METRICS_DIR

# Compute standard binary classification scores.
def compute_classification_scores(y_true, y_pred, y_prob) -> Dict[str, float]:
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    scores["roc_auc"] = auc(fpr, tpr)
    return scores

# Plot and save Keras training curve.
def plot_training_curve(history, save_path: Optional[Path] = None, overwrite: bool = True, show: bool = True):
    if save_path is None:
        save_path = FIGURES_DIR / "training_curve.png"
    if save_path.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.parent / f"{save_path.stem}_{timestamp}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return save_path

# Plot and save confusion matrix.
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path: Optional[Path] = None, overwrite: bool = True, show: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    if save_path is None:
        save_path = FIGURES_DIR / "confusion_matrix.png"
    if save_path.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.parent / f"{save_path.stem}_{timestamp}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Stable", "Pred Destab"],
                yticklabels=["True Stable", "True Destab"])
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return cm, save_path

# Plot and save ROC curve.
def plot_roc(y_true, y_prob, save_path: Optional[Path] = None, overwrite: bool = True, show: bool = True):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    if save_path is None:
        save_path = FIGURES_DIR / "roc_curve.png"
    if save_path.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.parent / f"{save_path.stem}_{timestamp}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return roc_auc, save_path

# Plot and save PR curve.
def plot_pr(y_true, y_prob, save_path: Optional[Path] = None, overwrite: bool = True, show: bool = True):
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    if save_path is None:
        save_path = FIGURES_DIR / "pr_curve.png"
    if save_path.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path.parent / f"{save_path.stem}_{timestamp}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(recalls, precisions, color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return save_path

# Save metrics dict to JSON file.
def save_metrics_json(metrics_dict: dict, filename: Optional[str] = None, overwrite: bool = True):
    if filename is None:
        filename = "nn_baseline.json"
    save_path = METRICS_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = METRICS_DIR / f"{save_path.stem}_{timestamp}.json"
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    return save_path

# Save Keras model summary to TXT file.
def save_model_summary(model, filename: Optional[str] = None, overwrite: bool = True, display: bool = True):
    if filename is None:
        filename = "model_summary.txt"
    save_path = METRICS_DIR / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = METRICS_DIR / f"{save_path.stem}_{timestamp}.txt"

    with open(save_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    if display:
        model.summary()
    return save_path

# Run high-level evaluation and return dict.
def evaluate_and_report_model(model, X_test, y_test, history=None,
                              overwrite: bool = True, show_plots: bool = True):
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    # Scores.
    scores = compute_classification_scores(y_test, y_pred, y_prob)

    # Plots.
    cm, cm_path = plot_confusion_matrix(y_test, y_pred, overwrite=overwrite, show=show_plots)
    roc_auc, roc_path = plot_roc(y_test, y_prob, overwrite=overwrite, show=show_plots)
    pr_path = plot_pr(y_test, y_prob, overwrite=overwrite, show=show_plots)
    if history is not None:
        train_curve_path = plot_training_curve(history, overwrite=overwrite, show=show_plots)
    else:
        train_curve_path = None

    # Save metrics and model summary.
    metrics_path = save_metrics_json(scores, overwrite=overwrite)
    model_summary_path = save_model_summary(model, overwrite=overwrite, display=show_plots)

    return {
        "scores": scores,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "figure_paths": {
            "confusion_matrix": cm_path,
            "roc_curve": roc_path,
            "pr_curve": pr_path,
            "training_curve": train_curve_path
        },
        "metrics_path": metrics_path,
        "model_summary_path": model_summary_path
    }