import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"

def add_features(df):
    df["hour"] = df["DateTime"].dt.hour
    df["day"] = df["DateTime"].dt.day
    df["weekday"] = df["DateTime"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df

def main():
    print("🚀 Feature engineering started")

    train_path = PROC_DIR / "train_cleaned.csv"
    test_path = PROC_DIR / "test_cleaned.csv"

    print("📄 Reading:", train_path.name)
    print("📄 Reading:", test_path.name)

    train = pd.read_csv(train_path, parse_dates=["DateTime"])
    test = pd.read_csv(test_path, parse_dates=["DateTime"])

    train = add_features(train)
    test = add_features(test)

    train.to_csv(PROC_DIR / "features_train.csv", index=False)
    test.to_csv(PROC_DIR / "features_test.csv", index=False)

    print("✅ features_train.csv & features_test.csv created")
    print("🎉 Feature engineering finished")

if __name__ == "__main__":
    main()
