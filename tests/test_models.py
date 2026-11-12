import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from mhealth_har.config import load_config  # noqa: E402
from mhealth_har.models import build_model, compile_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.tensorflow
@pytest.mark.parametrize(
    "name,parameters",
    [
        ("hybrid", 61404),
        ("cnn", 29916),
        ("lstm", 36588),
        ("bilstm", 89132),
        ("cnn11", 29867),
        ("bilstm11", 17995),
    ],
)
def test_architecture_shape_and_probabilities(name, parameters):
    tf.keras.backend.clear_session()
    model = build_model(load_config(ROOT / f"configs/{name}.json"))
    assert model.count_params() == parameters
    output = model(np.zeros((2, 128, 23), dtype=np.float32), training=False).numpy()
    assert output.shape == (2, 11 if name.endswith("11") else 12)
    np.testing.assert_allclose(output.sum(1), 1.0, atol=1e-6)


@pytest.mark.tensorflow
def test_training_step_and_model_roundtrip(tmp_path):
    config = load_config(ROOT / "configs/hybrid.json")
    model = build_model(config)
    compile_model(model, config)
    x = np.random.default_rng(43).normal(size=(4, 128, 23)).astype(np.float32)
    metrics = model.train_on_batch(x, np.arange(4), return_dict=True)
    assert np.isfinite(metrics["loss"])
    expected = model(x, training=False).numpy()
    model.save(tmp_path / "model.keras")
    loaded = tf.keras.models.load_model(tmp_path / "model.keras", compile=False)
    np.testing.assert_allclose(loaded(x, training=False), expected, atol=1e-6)
