import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from mhealth_har.config import FEATURES
from mhealth_har.data import activity_runs, load_csv, load_split, prepare, split_runs, window_runs


def synthetic_frame():
    # Different offsets per run make validation/test contamination detectable.
    labels = np.tile(np.repeat(np.arange(1, 13), 160), 3)
    x = np.arange(len(labels) * 23, dtype=np.float32).reshape(-1, 23) / 100
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["activity"] = labels
    return frame


def test_whole_runs_are_disjoint_and_deterministic():
    runs = activity_runs(synthetic_frame().activity.to_numpy())
    a = split_runs(runs)
    pd.testing.assert_frame_equal(a, split_runs(runs))
    assert a.run_id.is_unique
    assert len(a) == len(runs)
    for activity in range(1, 13):
        assert set(a[a.activity == activity].split) == {"train", "validation", "test"}


def test_windows_respect_run_boundaries_and_overlap():
    labels = np.repeat([1, 2], 9)
    x = np.arange(18, dtype=np.float32)[:, None]
    scaler = StandardScaler().fit(x)
    windows, y, provenance = window_runs(x, labels, activity_runs(labels), scaler, 4, 2)
    assert windows.shape == (6, 4, 1)
    assert y.tolist() == [1, 1, 1, 2, 2, 2]
    assert provenance.start_row.tolist() == [0, 2, 4, 9, 11, 13]
    np.testing.assert_array_equal(windows[0, 2:], windows[1, :2])
    for row in provenance.itertuples():
        assert np.all(labels[row.start_row : row.end_row_exclusive] == row.activity)


def test_train_only_scaler_and_roundtrip(tmp_path):
    frame = synthetic_frame()
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    out = tmp_path / "prepared"
    config = {
        "data_path": str(source),
        "output_dir": str(out),
        "ratios": [0.7, 0.15, 0.15],
        "seed": 43,
        "window_size": 128,
        "overlap": 64,
    }
    summary = prepare(config)
    assert summary.Windows.sum() == 36
    assignment = pd.read_csv(out / "activity_run_split_assignment.csv")
    mask = np.zeros(len(frame), dtype=bool)
    for row in assignment[assignment.split == "train"].itertuples():
        mask[row.start_row : row.end_row_exclusive] = True
    with np.load(out / "scaler.npz") as saved:
        expected = frame[list(FEATURES)].to_numpy(dtype=np.float32)[mask].mean(0, dtype=np.float64)
        np.testing.assert_allclose(saved["mean"], expected)
        assert not np.allclose(saved["mean"], frame[list(FEATURES)].mean().to_numpy())
    x, y, activities = load_split(out / "test.npz")
    assert x.shape == (12, 128, 23)
    assert set(y) == set(range(12))
    assert activities == list(range(1, 13))
    x11, y11, activities11 = load_split(out / "test.npz", 2)
    assert len(x11) == 11 and set(y11) == set(range(11))
    assert activities11 == [1] + list(range(3, 13))
    with pytest.raises(FileExistsError):
        prepare(config)


@pytest.mark.parametrize("change", ["fractional", "infinite", "missing", "extra"])
def test_invalid_csv_is_rejected(tmp_path, change):
    frame = synthetic_frame()
    if change == "fractional":
        frame["activity"] = frame["activity"].astype(float)
        frame.loc[0, "activity"] = 1.5
    elif change == "infinite":
        frame.loc[0, "A"] = np.inf
    elif change == "missing":
        frame = frame.drop(columns="A")
    else:
        frame["subject"] = 1
    source = tmp_path / "invalid.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError):
        load_csv(source)


def test_insufficient_runs_and_invalid_window_settings():
    with pytest.raises(ValueError, match="three runs"):
        split_runs(activity_runs(np.repeat([1, 2], 10)))
    x = np.ones((20, 1))
    labels = np.ones(20)
    with pytest.raises(ValueError):
        window_runs(x, labels, activity_runs(labels), StandardScaler().fit(x), 4, 4)
