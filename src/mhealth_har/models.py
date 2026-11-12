"""Notebook-equivalent Keras architectures; TensorFlow is an optional dependency."""

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


def build_model(config: dict) -> Model:
    """Build one recorded architecture without compiling or training it."""
    reg = regularizers.l2(config["l2"])
    kind = config["model"]
    inputs = layers.Input(shape=(128, 23), name="sensor_input")
    x = layers.GaussianNoise(config["input_noise"], name="input_noise")(inputs)
    x = layers.SpatialDropout1D(config["input_dropout"], name="input_feature_dropout")(x)
    if kind in ("hybrid", "cnn"):
        x = layers.Conv1D(48, 5, padding="same", kernel_regularizer=reg, name="cnn_block1_conv")(x)
        x = layers.BatchNormalization(name="cnn_block1_bn")(x)
        x = layers.Activation("relu", name="cnn_block1_activation")(x)
        x = layers.SpatialDropout1D(config["cnn_dropout"], name="cnn_block1_dropout")(x)
        x = layers.MaxPooling1D(2, name="cnn_block1_pool")(x)
        branches = []
        for name, size, dilation in (("short", 3, 1), ("long", 5, 2)):
            branch = layers.Conv1D(
                48,
                size,
                padding="same",
                dilation_rate=dilation,
                kernel_regularizer=reg,
                name=f"temporal_{name}_conv",
            )(x)
            branch = layers.BatchNormalization(name=f"temporal_{name}_bn")(branch)
            branches.append(layers.Activation("relu", name=f"temporal_{name}_activation")(branch))
        x = layers.Concatenate(name="multi_scale_concat")(branches)
        x = layers.SpatialDropout1D(config["temporal_dropout"], name="multi_scale_dropout")(x)
        if kind == "cnn":
            x = layers.GlobalAveragePooling1D(name="global_average_pooling")(x)
        else:
            x = layers.Bidirectional(
                layers.LSTM(
                    32,
                    dropout=config["lstm_dropout"],
                    recurrent_dropout=0.0,
                    kernel_regularizer=reg,
                    recurrent_regularizer=reg,
                    name="temporal_lstm",
                ),
                name="bidirectional_lstm",
            )(x)
        width = 48
    elif kind in ("lstm", "bilstm"):
        for i, units in enumerate((64, 32), start=1):
            recurrent = layers.LSTM(
                units,
                return_sequences=i == 1,
                dropout=config["lstm_dropout"],
                recurrent_dropout=0.0,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
                name=f"{'bilstm' if kind == 'bilstm' else 'lstm'}_{i}",
            )
            if kind == "bilstm":
                recurrent = layers.Bidirectional(recurrent, name=f"bidirectional_lstm_{i}")
            x = recurrent(x)
            prefix = "bilstm" if kind == "bilstm" else "lstm"
            x = layers.LayerNormalization(name=f"{prefix}_{i}_layer_norm")(x)
        width = 32
    elif kind == "bilstm_single":
        x = layers.Bidirectional(
            layers.LSTM(
                32,
                dropout=config["lstm_dropout"],
                recurrent_dropout=0.0,
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
                name="temporal_lstm",
            ),
            name="bidirectional_lstm",
        )(x)
        width = 48
    else:
        raise ValueError(f"Unknown model: {kind}")
    x = layers.Dense(width, activation="relu", kernel_regularizer=reg, name="dense_features")(x)
    x = layers.Dropout(config["dense_dropout"], name="classifier_dropout")(x)
    classes = 11 if config["removed_activity"] == 2 else 12
    outputs = layers.Dense(classes, activation="softmax", name="activity_output")(x)
    return Model(inputs, outputs, name=f"mhealth_{kind}_{classes}class")


def compile_model(model: Model, config: dict) -> None:
    """Use AdamW explicitly: never silently fall back to an unweighted optimizer."""
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config["learning_rate"], weight_decay=config["weight_decay"], clipnorm=1.0
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
