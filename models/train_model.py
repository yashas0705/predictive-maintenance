"""
Train a failure-prediction model on the AI4I 2020 dataset.
Produces:
  - models/model.pkl        (trained XGBoost classifier)
  - models/encoder.pkl      (LabelEncoder for machine 'Type')
  - models/feature_importance.csv
  - models/metrics.json
Run:  python train_model.py
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

DATA_PATH = "../data/ai4i2020.csv"

FEATURES = [
    "type_enc",
    "air_temp",
    "process_temp",
    "rpm",
    "torque",
    "tool_wear",
]
TARGET = "Machine failure"

RENAME_MAP = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=RENAME_MAP)

    le = LabelEncoder()
    df["type_enc"] = le.fit_transform(df["Type"])  # L=0/1, M, H

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class imbalance is real (~3.4% failure rate) -> weight positive class
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    print("Metrics:", metrics)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importances = pd.DataFrame(
        {"feature": FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv("feature_importance.csv", index=False)

    print("Saved model.pkl, encoder.pkl, metrics.json, feature_importance.csv")


if __name__ == "__main__":
    main()
