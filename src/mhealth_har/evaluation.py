"""Metrics and evidence checks independent of TensorFlow."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import ACTIVITY_NAMES, write_json


def classification_metrics(truth, prediction) -> dict:
    """Compute every aggregate from one fixed prediction vector."""
    return {
        "Accuracy": float(accuracy_score(truth, prediction)),
        "Macro_Precision": float(
            precision_score(truth, prediction, average="macro", zero_division=0)
        ),
        "Macro_Recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "Macro_F1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
    }


def export_evaluation(truth, probabilities, activities, destination: str | Path) -> dict:
    """Save reproducible aggregate/per-class reports and a confusion matrix."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if probabilities.shape != (len(truth), len(activities)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability shape or values.")
    prediction = probabilities.argmax(axis=1)
    metrics = classification_metrics(truth, prediction)
    write_json(destination / "metrics.json", metrics)
    names = [ACTIVITY_NAMES[i - 1] for i in activities]
    report = classification_report(
        truth,
        prediction,
        labels=np.arange(len(activities)),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    write_json(destination / "classification_report.json", report)
    cm = confusion_matrix(truth, prediction, labels=np.arange(len(activities)))
    pd.DataFrame(cm, index=names, columns=names).to_csv(destination / "confusion_matrix_raw.csv")
    normalized = np.divide(
        cm,
        cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype=float),
        where=cm.sum(axis=1, keepdims=True) != 0,
    )
    pd.DataFrame(normalized, index=names, columns=names).to_csv(
        destination / "confusion_matrix_normalized.csv"
    )
    np.savez_compressed(
        destination / "predictions.npz",
        y=truth,
        probabilities=probabilities,
        activity_ids=activities,
    )
    return metrics


def audit_saved_results(evidence: str | Path) -> pd.DataFrame:
    """Recompute metrics from original integer confusion counts, not model inference."""
    rows = []
    for results_path in sorted(Path(evidence).glob("*/results.csv")):
        folder = results_path.parent
        cm_path = folder / "confusion_matrix_raw.csv"
        if not cm_path.exists():
            cm_path = folder / "confusion_matrix.csv"
        matrix = pd.read_csv(cm_path, index_col=0).to_numpy()
        if matrix.shape[0] != matrix.shape[1] or not np.equal(matrix, matrix.astype(int)).all():
            raise ValueError(f"Invalid confusion matrix: {cm_path}")
        truth, predicted = [], []
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                truth.extend([i] * int(matrix[i, j]))
                predicted.extend([j] * int(matrix[i, j]))
        metrics = classification_metrics(truth, predicted)
        recorded = pd.read_csv(results_path).set_index("Split").loc["Test"]
        for key, value in metrics.items():
            if not np.isclose(value, recorded[key], rtol=0, atol=1e-12):
                raise ValueError(f"Metric mismatch: {folder.name}/{key}")
        rows.append(
            {
                "Experiment": folder.name,
                "Classes": len(matrix),
                "Test_windows": int(matrix.sum()),
                **metrics,
            }
        )
    if not rows:
        raise ValueError("No saved results found.")
    return pd.DataFrame(rows)
