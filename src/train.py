import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["hour", "day", "weekday", "is_weekend"]

def main():
    print("🚀 Model training started")

    df = pd.read_csv(PROC_DIR / "features_train.csv")

    for j in sorted(df["Junction"].unique()):
        print(f"🔧 Training model for Junction {j}")
        sub = df[df["Junction"] == j]

        X = sub[FEATURES]
        y = sub["Vehicles"]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        model_path = MODEL_DIR / f"model_junction_{j}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Saved {model_path.name}")

    print("🎉 Model training finished")

if __name__ == "__main__":
    main()
