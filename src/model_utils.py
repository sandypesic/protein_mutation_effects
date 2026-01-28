import tensorflow as tf
from tensorflow.keras import layers, models

def build_nn(input_dim: int) -> tf.keras.Model:
    """
    ----
    Build a fully connected neural network for binary classification.
    ----
    Architecture:
        1) Input layer matching input_dim.
        2) Dense layer: 128 units, ReLU activation.
        3) Dropout layer: 30% (for regularization).
        4) Dense layer: 64 units, ReLU activation.
        5) Output layer: 1 unit, sigmoid activation (binary output).
    Args:
        input_dim (int): number of input features.
    Return:
        tf.keras.Model: compiled Keras model.
    """

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model