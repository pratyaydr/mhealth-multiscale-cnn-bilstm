"""Train with the original per-experiment settings; test evaluation is separate."""

import importlib.metadata
import os
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from .config import write_json
from .data import load_split
from .models import build_model, compile_model


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Select the first strict maximum of validation Macro-F1, as in the notebook."""

    def __init__(self, x, y, path, batch_size):
        super().__init__()
        self.x, self.y, self.path, self.batch_size = x, y, path, batch_size
        self.best, self.best_epoch = -np.inf, None

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(self.x, batch_size=self.batch_size, verbose=0)
        score = float(f1_score(self.y, probabilities.argmax(1), average="macro", zero_division=0))
        if logs is not None:
            logs["val_macro_f1"] = score
        if score > self.best:
            self.best, self.best_epoch = score, epoch + 1
            self.model.save(self.path)


def train(config: dict) -> dict:
    """Train and checkpoint without opening the held-out test arrays."""
    destination = Path(config["output_dir"])
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Use a new output directory: {destination}")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])
    data = Path(config["data_dir"])
    x, y, activities = load_split(data / "train.npz", config["removed_activity"])
    x_val, y_val, _ = load_split(data / "validation.npz", config["removed_activity"])
    model = build_model(config)
    compile_model(model, config)
    if config["class_weight_mode"] == "balanced":
        weights = compute_class_weight("balanced", classes=np.arange(len(activities)), y=y)
        weights = dict(enumerate(map(float, weights)))
    elif config["class_weight_mode"] == "manual":
        weights = dict.fromkeys(range(len(activities)), 1.0)
        weights[len(activities) - 1] = config["manual_last_class_weight"]
    else:
        raise ValueError("Unknown class weight mode.")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", config)
    write_json(
        destination / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "keras")
            },
            "tensorflow": tf.__version__,
            "parameters": model.count_params(),
            "hash_seed_at_startup": os.environ.get("PYTHONHASHSEED"),
            "class_weights": weights,
            "activity_ids": activities,
        },
    )
    checkpoint = destination / "model.keras"
    macro = MacroF1Checkpoint(x_val, y_val, checkpoint, config["batch_size"])
    callbacks = [macro] if config["selection"] == "macro_f1" else []
    if config["selection"] not in ("macro_f1", "last_epoch"):
        raise ValueError("Unsupported checkpoint selection.")
    callbacks += [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config["lr_factor"],
            patience=config["lr_patience"],
            min_lr=config["min_lr"],
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=False,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        x,
        y,
        validation_data=(x_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    if config["selection"] == "last_epoch":
        model.save(checkpoint)
    frame = pd.DataFrame(history.history)
    frame.insert(0, "Epoch", np.arange(1, len(frame) + 1))
    frame.to_csv(destination / "training_history.csv", index=False)
    selection = {
        "selection": config["selection"],
        "epochs_executed": len(frame),
        "selected_epoch": macro.best_epoch if config["selection"] == "macro_f1" else len(frame),
    }
    write_json(destination / "selection.json", selection)
    return selection
