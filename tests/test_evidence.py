from pathlib import Path

import numpy as np
import pytest

from mhealth_har.evaluation import audit_saved_results, export_evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_all_original_metrics_reconcile():
    audited = audit_saved_results(ROOT / "evidence/original")
    assert len(audited) == 5
    hybrid = audited.set_index("Experiment").loc["cnn_multiscale_bilstm_11class"]
    assert hybrid.Classes == 12
    assert hybrid.Test_windows == 987
    assert hybrid.Accuracy == pytest.approx(917 / 987)


def test_export_uses_one_prediction_vector(tmp_path):
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    metrics = export_evaluation(np.array([0, 1, 1]), probabilities, [1, 2], tmp_path)
    assert metrics["Accuracy"] == pytest.approx(2 / 3)
    assert (tmp_path / "classification_report.json").exists()
    with np.load(tmp_path / "predictions.npz") as saved:
        np.testing.assert_array_equal(saved["probabilities"], probabilities)


def test_invalid_probabilities_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        export_evaluation([0, 1], np.array([[np.nan, 1.0]]), [1, 2], tmp_path)
