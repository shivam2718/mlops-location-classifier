from pathlib import Path
from PIL import Image
from datetime import datetime


class DataValidation:
    def __init__(self,
                 ingested_data_dir="data/ingested",
                 log_dir="reports"):
        self.ingested_data_dir = Path(ingested_data_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_dir / "data_validation.log", "a") as f:
            f.write(msg + "\n")

    def _get_class_folders(self, base_dir: Path):
        """
        Handles accidental nesting like seg_train/seg_train/
        """
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and subdirs[0].name == base_dir.name:
            base_dir = subdirs[0]
        return [d for d in base_dir.iterdir() if d.is_dir()]

    def _check_corrupted_images(self, class_dir: Path):
        corrupted = []
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    corrupted.append(img_path)
        return corrupted

    def initiate_data_validation(self):
        self._log("Starting data validation")

        train_dir = self.ingested_data_dir / "seg_train"
        test_dir = self.ingested_data_dir / "seg_test"

        self._log(f"Train directory: {train_dir.resolve()}")
        self._log(f"Test directory: {test_dir.resolve()}")

        # 1️⃣ Check directories exist
        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError("seg_train or seg_test directory missing")

        train_classes = self._get_class_folders(train_dir)
        test_classes = self._get_class_folders(test_dir)

        train_class_names = set(c.name for c in train_classes)
        test_class_names = set(c.name for c in test_classes)

        # 2️⃣ Check class consistency
        if train_class_names != test_class_names:
            raise ValueError(
                f"Class mismatch detected.\n"
                f"Train classes: {train_class_names}\n"
                f"Test classes: {test_class_names}"
            )

        self._log(f"Classes validated: {sorted(train_class_names)}")

        # 3️⃣ Check empty class folders
        for cls_dir in train_classes:
            images = list(cls_dir.glob("*"))
            if len(images) == 0:
                raise ValueError(f"Empty class folder detected: {cls_dir}")

        self._log("No empty class folders found")

        # 4️⃣ Check corrupted images
        total_corrupted = 0
        for cls_dir in train_classes:
            corrupted = self._check_corrupted_images(cls_dir)
            if corrupted:
                self._log(f"Corrupted images in {cls_dir.name}: {len(corrupted)}")
                total_corrupted += len(corrupted)

        if total_corrupted > 0:
            raise ValueError(f"Corrupted images detected: {total_corrupted}")

        self._log("No corrupted images detected")

        # 5️⃣ Train–test leakage check (file names)
        train_files = {
            img.name
            for cls in train_classes
            for img in cls.glob("*")
        }

        test_files = {
            img.name
            for cls in test_classes
            for img in cls.glob("*")
        }

        overlap = train_files.intersection(test_files)
        if overlap:
            raise ValueError(f"Train–test leakage detected: {len(overlap)} files")

        self._log("No train-test leakage detected")
        self._log("Data validation completed successfully")

        return True


if __name__ == "__main__":
    DataValidation().initiate_data_validation()
