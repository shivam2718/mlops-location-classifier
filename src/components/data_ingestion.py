import shutil
from pathlib import Path
from datetime import datetime


class DataIngestion:
    def __init__(self,
                 raw_data_dir="data/raw",
                 ingested_data_dir="data/ingested",
                 log_dir="reports"):
        self.raw_data_dir = Path(raw_data_dir)
        self.ingested_data_dir = Path(ingested_data_dir)
        self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_dir / "data_ingestion.log", "a") as f:
            f.write(msg + "\n")

    def _count_images(self, base_dir: Path):
     counts = {}
     total = 0

     for class_dir in base_dir.iterdir():
         if class_dir.is_dir():
             image_files = list(class_dir.glob("*.jpg")) + \
                          list(class_dir.glob("*.jpeg")) + \
                          list(class_dir.glob("*.png"))

             count = len(image_files)
             counts[class_dir.name] = count
             total += count

     return counts, total
 
     
    

    def initiate_data_ingestion(self):
        self._log("Starting data ingestion")

        # 1. Validate raw data exists
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_dir}")

        # 2. Remove old ingested data (clean run)
        if self.ingested_data_dir.exists():
            shutil.rmtree(self.ingested_data_dir)

        # 3. Copy raw data to ingested
        shutil.copytree(self.raw_data_dir, self.ingested_data_dir)
        self._log("Copied raw dataset to ingested directory")

        # 4. Log statistics
        train_dir = self.ingested_data_dir / "seg_train"
        test_dir = self.ingested_data_dir / "seg_test"

        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError("seg_train or seg_test missing in ingested data")

        train_counts, train_total = self._count_images(train_dir)
        test_counts, test_total = self._count_images(test_dir)

        self._log(f"TRAIN total images: {train_total}")
        for cls, cnt in train_counts.items():
            self._log(f"TRAIN | {cls}: {cnt}")

        self._log(f"TEST total images: {test_total}")
        for cls, cnt in test_counts.items():
            self._log(f"TEST  | {cls}: {cnt}")

        self._log("Data ingestion completed successfully")

        return {
            "train_dir": str(train_dir),
            "test_dir": str(test_dir),
            "pred_dir": str(self.ingested_data_dir / "seg_pred"),
            "train_distribution": train_counts,
            "test_distribution": test_counts
        }


if __name__ == "__main__":
    DataIngestion().initiate_data_ingestion()
