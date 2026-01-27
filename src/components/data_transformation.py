from pathlib import Path
from datetime import datetime

import tensorflow as tf
import keras
from keras import layers


class DataTransformation:
    def __init__(
        self,
        ingested_data_dir="data/ingested",
        image_size=(224, 224),
        batch_size=32,
        log_dir="reports",
    ):
        self.ingested_data_dir = Path(ingested_data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_dir / "data_transformation.log", "a") as f:
            f.write(msg + "\n")

    def initiate_data_transformation(self):
        self._log("Starting data transformation")

        train_dir = self.ingested_data_dir / "seg_train"
        test_dir = self.ingested_data_dir / "seg_test"

        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError("seg_train or seg_test directory missing")

        self._log(f"Train dir: {train_dir.resolve()}")
        self._log(f"Test dir: {test_dir.resolve()}")

        # 1️⃣ Load datasets (USING KERAS, NOT tf.keras)
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=True,
        )

        test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False,
        )

        class_names = train_ds.class_names
        self._log(f"Detected classes: {class_names}")

        # 2️⃣ Normalization layer
        normalization_layer = layers.Rescaling(1.0 / 255)

        # 3️⃣ Data augmentation (TRAIN ONLY)
        data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )

        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(normalization_layer(x)), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        test_ds = test_ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # 4️⃣ Performance optimizations
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        self._log("Data transformation completed successfully")

        return {
            "train_dataset": train_ds,
            "test_dataset": test_ds,
            "class_names": class_names,
            "num_classes": len(class_names),
        }


if __name__ == "__main__":
    DataTransformation().initiate_data_transformation()
