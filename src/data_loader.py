import pandas as pd
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

def main():
    print("🚀 Data loader started")

    if not os.path.exists(PROC_DIR):
        os.makedirs(PROC_DIR)

    # Auto-detect CSV files
    csv_files = list(RAW_DIR.glob("*.csv"))

    if len(csv_files) < 2:
        raise Exception("❌ CSV files not found in data/raw")

    train_file = [f for f in csv_files if "train" in f.name.lower()][0]
    test_file = [f for f in csv_files if "test" in f.name.lower()][0]

    print("📄 Train file:", train_file.name)
    print("📄 Test file :", test_file.name)

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train["DateTime"] = pd.to_datetime(train["DateTime"])
    test["DateTime"] = pd.to_datetime(test["DateTime"])

    train.sort_values(["Junction", "DateTime"], inplace=True)
    test.sort_values(["Junction", "DateTime"], inplace=True)

    train.to_csv(PROC_DIR / "train_cleaned.csv", index=False)
    test.to_csv(PROC_DIR / "test_cleaned.csv", index=False)

    print("✅ train_cleaned.csv & test_cleaned.csv created")
    print("🎉 Data loader finished successfully")

if __name__ == "__main__":
    main()
