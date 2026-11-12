"""Validate the supplied local CSV/results ZIP without redistributing sensor arrays.

Only pass artifacts you trust. Optional checkpoint loading uses Keras safe mode.
"""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from mhealth_har.config import load_config
from mhealth_har.data import load_split, prepare
from mhealth_har.evaluation import classification_metrics

ROOT = Path(__file__).resolve().parents[1]
MAPPING = {
    "hybrid": "cnn_multiscale_bilstm_11class/best_cnn_multiscale_bilstm_12class.keras",
    "cnn": "cnn_only_12class/cnn_only_12class.keras",
    "lstm": "lstm_only_generalization_v1/best_lstm_only.keras",
    "bilstm": "bilstm_only_generalization_v1/best_bilstm_only.keras",
    "cnn11": "cnn_only_11class/cnn_only_11class.keras",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(args.results) as archive:
        temp = Path(directory)
        config = load_config(ROOT / "configs/preprocessing.json")
        config.update(data_path=str(Path(args.csv).resolve()), output_dir=str(temp / "rebuilt"))
        print(prepare(config).to_string(index=False))
        for split in ("train", "validation", "test"):
            original_path = temp / f"{split}.npz"
            original_path.write_bytes(archive.read(f"mHealth_preprocessed/{split}.npz"))
            with (
                np.load(original_path) as original,
                np.load(temp / "rebuilt" / f"{split}.npz") as rebuilt,
            ):
                for key in ("X", "y"):
                    if key == "y":
                        np.testing.assert_array_equal(original[key], rebuilt[key])
                    else:
                        np.testing.assert_allclose(original[key], rebuilt[key], rtol=0, atol=3e-6)
            print(
                f"{split}: labels match exactly; sensor windows match within absolute tolerance 3e-6"
            )
        if args.checkpoints:
            import tensorflow as tf

            from mhealth_har.models import build_model

            for name, relative in MAPPING.items():
                config = load_config(ROOT / f"configs/{name}.json")
                model_path = temp / "checkpoint.keras"
                model_path.write_bytes(archive.read(f"mHealth_preprocessed/{relative}"))
                model = tf.keras.models.load_model(model_path, compile=False)
                refactored = build_model(config)
                assert model.count_params() == refactored.count_params()
                refactored.set_weights(model.get_weights())
                x, y, _ = load_split(temp / "test.npz", config["removed_activity"])
                probabilities = np.concatenate(
                    [model(x[i : i + 32], training=False).numpy() for i in range(0, len(x), 32)]
                )
                np.testing.assert_allclose(
                    refactored(x[:32], training=False), probabilities[:32], atol=1e-6
                )
                metrics = classification_metrics(y, probabilities.argmax(1))
                recorded = pd.read_csv(
                    ROOT / "evidence/original" / Path(relative).parent / "results.csv"
                )
                test = recorded.set_index("Split").loc["Test"]
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, test[key], rtol=0, atol=1e-12)
                print(name, json.dumps(metrics))
                tf.keras.backend.clear_session()
    digest = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    print("CSV SHA256:", digest)


if __name__ == "__main__":
    main()
