import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_nn(input_dim: int) -> tf.keras.Model:
    """
    Build a fully connected neural network for binary classification.

    Architecture:
        1) Input layer matching input_dim.
        2) Dense(128) + BatchNorm + ReLU + Dropout(0.3)
        3) Dense(64) + BatchNorm + ReLU + Dropout(0.3)
        4) Output: Dense(1, sigmoid)
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model


def get_callbacks(patience: int = 10) -> list:
    """
    Returns standard training callbacks:
        - EarlyStopping with best weight restoration
        - ReduceLROnPlateau for adaptive learning rate
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=patience,
            mode="max",
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=5,
            mode="max",
            min_lr=1e-6,
            verbose=1
        )
    ]