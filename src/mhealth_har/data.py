"""Activity-run splitting and train-only standardization, matching notebook cell 3.

A label run is a CSV row-order proxy, not a subject or a guaranteed session.
Saved NPZ labels remain 1..12; conversion to model indices happens on loading.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .config import FEATURES, write_json


def load_csv(path: str | Path) -> pd.DataFrame:
    """Validate the exact 23-channel CSV schema without reordering rows."""
    frame = pd.read_csv(path)
    if set(frame.columns) != set(FEATURES) | {"activity"}:
        raise ValueError("Expected exactly columns A..W and activity; do not pass raw UCI logs.")
    if frame.empty or not all(pd.api.types.is_numeric_dtype(d) for d in frame.dtypes):
        raise ValueError("Dataset must be nonempty and numeric.")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Missing or nonfinite data detected.")
    labels = frame.activity.to_numpy()
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Activity labels must be integers.")
    if not np.array_equal(np.unique(labels), np.arange(1, 13)):
        raise ValueError("Expected all original activities 1..12.")
    return frame


def activity_runs(labels: np.ndarray) -> pd.DataFrame:
    """Return half-open row intervals for contiguous equal-label runs."""
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Labels must be a nonempty one-dimensional array.")
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return pd.DataFrame(
        [
            {
                "run_id": i + 1,
                "start_row": int(start),
                "end_row_exclusive": int(end),
                "length": int(end - start),
                "activity": int(labels[start]),
            }
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]
    )


def split_runs(runs: pd.DataFrame, ratios=(0.70, 0.15, 0.15), seed=43) -> pd.DataFrame:
    """Shuffle runs per activity with the original rounding and remainder rule.

    Ratios target run counts, not row/window counts. Small numbers of runs cause
    the achieved split proportions to differ substantially from 70/15/15.
    """
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("Three positive split ratios must sum to one.")
    rng = np.random.default_rng(seed)
    result = runs.copy()
    result["split"] = ""
    for activity in sorted(runs.activity.unique()):
        indices = runs.index[runs.activity == activity].to_numpy()
        n = len(indices)
        if n < 3:
            raise ValueError(f"Activity {activity} needs at least three runs.")
        shuffled = indices[rng.permutation(n)]
        train = max(1, int(round(n * ratios[0])))
        validation = max(1, int(round(n * ratios[1])))
        while n - train - validation < 1:
            if train > validation and train > 1:
                train -= 1
            elif validation > 1:
                validation -= 1
            else:
                raise ValueError("Cannot allocate nonempty splits.")
        result.loc[shuffled[:train], "split"] = "train"
        result.loc[shuffled[train : train + validation], "split"] = "validation"
        result.loc[shuffled[train + validation :], "split"] = "test"
    return result


def window_runs(features, labels, runs, scaler, window_size=128, overlap=64):
    """Scale each run and emit windows with row/run provenance; never bridge runs."""
    if window_size <= 0 or not 0 <= overlap < window_size:
        raise ValueError("Require window_size > 0 and 0 <= overlap < window_size.")
    windows, targets, provenance = [], [], []
    for row in runs.sort_values("start_row").itertuples():
        start, end = row.start_row, row.end_row_exclusive
        if not np.all(labels[start:end] == row.activity):
            raise ValueError("Run contains mixed labels.")
        transformed = scaler.transform(features[start:end]).astype(np.float32)
        for offset in range(0, end - start - window_size + 1, window_size - overlap):
            windows.append(transformed[offset : offset + window_size])
            targets.append(row.activity)
            provenance.append(
                {
                    "run_id": row.run_id,
                    "start_row": start + offset,
                    "end_row_exclusive": start + offset + window_size,
                    "activity": row.activity,
                }
            )
    if not windows:
        raise ValueError("No windows: runs are shorter than window_size.")
    return np.stack(windows), np.asarray(targets, dtype=np.int64), pd.DataFrame(provenance)


def prepare(config: dict) -> pd.DataFrame:
    """Persist NPZ splits, safe scaler statistics, weights, and audit metadata."""
    source = Path(config["data_path"])
    destination = Path(config["output_dir"])
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {destination}")
    frame = load_csv(source)
    features = frame[list(FEATURES)].to_numpy(dtype=np.float32)
    labels = frame.activity.to_numpy(dtype=np.int64)
    runs = activity_runs(labels)
    assigned = split_runs(runs, tuple(config["ratios"]), config["seed"])
    train_rows = assigned[assigned.split == "train"].sort_values("start_row")
    scaler = StandardScaler().fit(
        np.concatenate(
            [features[r.start_row : r.end_row_exclusive] for r in train_rows.itertuples()]
        )
    )
    prepared, summaries = {}, []
    for split in ("train", "validation", "test"):
        selected = assigned[assigned.split == split]
        x, y, provenance = window_runs(
            features, labels, selected, scaler, config["window_size"], config["overlap"]
        )
        if not np.array_equal(np.unique(y), np.arange(1, 13)):
            raise ValueError(f"{split}: windowing removed at least one activity.")
        prepared[split] = (x, y, provenance)
        summaries.append(
            {
                "Dataset": split.title(),
                "Raw_Samples": int(selected.length.sum()),
                "Windows": len(y),
                "Window_Size": config["window_size"],
                "Overlap": config["overlap"],
                "Features_Per_Window": 23,
                "Activity_Runs": len(selected),
            }
        )
    destination.mkdir(parents=True, exist_ok=True)
    for split, (x, y, provenance) in prepared.items():
        np.savez_compressed(destination / f"{split}.npz", X=x, y=y)
        provenance.to_csv(destination / f"{split}_windows.csv", index=False)
    np.savez(
        destination / "scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        variance=scaler.var_,
        n_samples_seen=scaler.n_samples_seen_,
    )
    weights = compute_class_weight("balanced", classes=np.arange(1, 13), y=prepared["train"][1])
    write_json(
        destination / "class_weights.json", {str(i): float(w) for i, w in enumerate(weights)}
    )
    runs.to_csv(destination / "activity_runs.csv", index=False)
    assigned.sort_values(["activity", "run_id"]).to_csv(
        destination / "activity_run_split_assignment.csv", index=False
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(destination / "dataset_summary.csv", index=False)
    write_json(
        destination / "preprocessing_configuration.json",
        {
            **config,
            "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "features": list(FEATURES),
            "raw_labels": "1..12",
            "subject_independent": False,
            "duplicate_rows": int(frame.duplicated().sum()),
            "continuous_activity_runs": len(runs),
        },
    )
    return summary


def load_split(path: str | Path, removed_activity: int | None = None):
    """Load trusted local NPZ data and map original activity IDs to model indices."""
    with np.load(path, allow_pickle=False) as arrays:
        x, raw = arrays["X"].astype(np.float32), arrays["y"]
    if x.ndim != 3 or x.shape[1:] != (128, 23) or raw.ndim != 1 or len(x) != len(raw):
        raise ValueError("Expected X=(N,128,23), y=(N,).")
    if not np.isfinite(x).all() or not np.isin(raw, np.arange(1, 13)).all():
        raise ValueError("Invalid windows or activity IDs.")
    if removed_activity not in (None, 2):
        raise ValueError("Only the original 12-class or activity-2-removed task is supported.")
    activities = [i for i in range(1, 13) if i != removed_activity]
    mask = raw != removed_activity if removed_activity is not None else np.ones(len(raw), bool)
    x, raw = x[mask], raw[mask].astype(np.int64)
    if not np.array_equal(np.unique(raw), activities):
        raise ValueError("Every expected activity must be represented in the split.")
    indices = np.searchsorted(activities, raw)
    return x, indices, activities
