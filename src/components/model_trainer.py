from pathlib import Path
from datetime import datetime

import tensorflow as tf
import keras
from keras import layers, callbacks, optimizers, applications


class ModelTrainer:
    def __init__(
        self,
        image_size=(224, 224, 3),
        epochs=10,
        model_dir="models",
        log_dir="reports",
    ):
        self.image_size = image_size
        self.epochs = epochs
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_dir / "model_trainer.log", "a") as f:
            f.write(msg + "\n")

    def build_model(self, num_classes: int):
        self._log("Building EfficientNetB0 model (pure Keras)")

        base_model = applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=self.image_size,
        )
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self._log("Model compiled successfully")
        return model

    def train(self, train_ds, val_ds, class_names):
        self._log("Starting model training")

        model = self.build_model(len(class_names))

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=cb,
        )

        return model, history

    def save_model(self, model):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"image_classifier_{timestamp}"
        model.save(model_path)
        self._log(f"Model saved at {model_path.resolve()}")
        return model_path


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    transformer = DataTransformation()
    data = transformer.initiate_data_transformation()

    trainer = ModelTrainer(epochs=10)
    model, history = trainer.train(
        train_ds=data["train_dataset"],
        val_ds=data["test_dataset"],
        class_names=data["class_names"],
    )

    trainer.save_model(model)
