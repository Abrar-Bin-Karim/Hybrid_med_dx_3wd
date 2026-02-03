import json
import joblib
import numpy as np
from pathlib import Path

from src.load_wdbc import load_uci_wdbc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def main():
    X, y, ids = load_uci_wdbc("data/raw/wdbc.data")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_s, y_train)

    # Save artifacts
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")

    with open(ARTIFACT_DIR / "feature_names.json", "w") as f:
        json.dump(list(X.columns), f)

    print("Saved artifacts to /artifacts:")
    print("- model.joblib")
    print("- scaler.joblib")
    print("- feature_names.json")

if __name__ == "__main__":
    main()
