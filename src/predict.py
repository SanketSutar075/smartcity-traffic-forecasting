import pandas as pd
import joblib
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"

FEATURES = ["hour", "day", "weekday", "is_weekend"]

def main():
    print("🚀 Prediction started")

    # ensure outputs folder exists
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    test_path = PROC_DIR / "features_test.csv"
    print("📄 Reading:", test_path.name)

    test = pd.read_csv(test_path)

    results = []

    for j in sorted(test["Junction"].unique()):
        print(f"🔍 Predicting for Junction {j}")

        model_path = MODEL_DIR / f"model_junction_{j}.pkl"
        print("📦 Loading model:", model_path.name)

        model = joblib.load(model_path)

        sub = test[test["Junction"] == j].copy()
        sub["Vehicles"] = model.predict(sub[FEATURES])
        results.append(sub)

    final = pd.concat(results)

    out_file = OUT_DIR / "submission.csv"
    final[["ID", "Vehicles"]].to_csv(out_file, index=False)

    print("✅ submission.csv created at:", out_file)
    print("🎉 Prediction finished successfully")

if __name__ == "__main__":
    main()
