"""
train_model.py
---------------
Trains the AI model(s) used by the Student Performance Prediction System.

Two models are trained:
1. RandomForestRegressor -> predicts the numeric final exam score (0-100)
2. RandomForestClassifier -> predicts the performance category
   (Excellent / Good / Average / At Risk) directly, used for a secondary
   confidence check and for the dashboard's classification report.

Run this file AFTER generate_dataset.py.
It saves:
    model/student_performance_model.pkl   (regressor + scaler + metadata)
    model/feature_importance.json
    model/metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_performance_model.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance.json")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "previous_exam_score",
    "sleep_hours",
    "extracurricular_activities",
    "internet_access",
    "tutoring",
    "parental_education_level",
    "study_environment",
    "daily_screen_time",
]


def score_to_category(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "At Risk"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Run 'python model/generate_dataset.py' first."
        )

    df = pd.read_csv(DATA_PATH)
    df["performance_category"] = df["final_exam_score"].apply(score_to_category)

    X = df[FEATURES]
    y_reg = df["final_exam_score"]
    y_clf = df["performance_category"]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # Scale features (helps some models & keeps pipeline production-ready)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Regression model: predicts numeric score ----
    regressor = RandomForestRegressor(
        n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
    )
    regressor.fit(X_train_scaled, y_reg_train)
    y_pred = regressor.predict(X_test_scaled)

    mae = mean_absolute_error(y_reg_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_pred)))
    r2 = r2_score(y_reg_test, y_pred)

    # ---- Classification model: predicts performance category ----
    classifier = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
    )
    classifier.fit(X_train_scaled, y_clf_train)
    y_clf_pred = classifier.predict(X_test_scaled)
    acc = accuracy_score(y_clf_test, y_clf_pred)
    report = classification_report(y_clf_test, y_clf_pred, output_dict=True)

    # ---- Feature importance (from regressor) ----
    importances = dict(zip(FEATURES, regressor.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    # ---- Save everything ----
    joblib.dump(
        {
            "regressor": regressor,
            "classifier": classifier,
            "scaler": scaler,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importances, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2_score": round(r2, 4),
                "classification_accuracy": round(acc, 4),
                "classification_report": report,
                "train_size": len(X_train),
                "test_size": len(X_test),
            },
            f,
            indent=2,
        )

    print("=" * 55)
    print("MODEL TRAINING COMPLETE")
    print("=" * 55)
    print(f"Regression  -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
    print(f"Classification -> Accuracy: {acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print("Top 3 important features:")
    for k, v in list(importances.items())[:3]:
        print(f"   - {k}: {v:.3f}")


if __name__ == "__main__":
    main()
