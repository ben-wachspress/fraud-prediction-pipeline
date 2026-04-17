"""Evaluation — PR-AUC, ROC-AUC, threshold optimization, confusion matrix, SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(y_test, proba, metric="f1")
    preds = (proba >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, preds),
        "threshold": threshold,
        "classification_report": classification_report(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def find_best_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if metric == "f1":
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return float(thresholds[best_idx])


def print_report(metrics: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key in ("pr_auc", "roc_auc", "f1", "threshold"):
        table.add_row(key.upper(), f"{metrics[key]:.4f}")

    console.print(table)
    console.print("\n[bold]Classification Report:[/bold]")
    console.print(metrics["classification_report"])
