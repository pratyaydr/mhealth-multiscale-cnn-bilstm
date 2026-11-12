"""JSON configuration, shared schema, and portable path conventions."""

import json
from pathlib import Path

FEATURES = tuple("ABCDEFGHIJKLMNOPQRSTUVW")
ACTIVITY_NAMES = (
    "Standing still",
    "Sitting and relaxing",
    "Lying down",
    "Walking",
    "Climbing stairs",
    "Waist bends forward",
    "Frontal elevation of arms",
    "Knees bending",
    "Cycling",
    "Jogging",
    "Running",
    "Jump front & back",
)


def load_config(path: str | Path) -> dict:
    """Read a config; relative data/output paths resolve from the working directory."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: object) -> None:
    """Write human-readable JSON, rejecting nonfinite values."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
