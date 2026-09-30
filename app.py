"""
app.py
------
AI-Based Student Performance Prediction System
Flask web application entry point.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_performance_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "model", "feature_importance.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------
model_bundle = None
metrics = {}
feature_importance = {}


def load_artifacts():
    global model_bundle, metrics, feature_importance
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        with open(FEATURE_IMPORTANCE_PATH) as f:
            feature_importance = json.load(f)


load_artifacts()

FEATURE_LABELS = {
    "study_hours_per_day": "Study Hours / Day",
    "attendance_percentage": "Attendance %",
    "previous_exam_score": "Previous Exam Score",
    "sleep_hours": "Sleep Hours",
    "extracurricular_activities": "Extracurricular Activities",
    "internet_access": "Internet Access",
    "tutoring": "Tutoring",
    "parental_education_level": "Parental Education Level",
    "study_environment": "Study Environment",
    "daily_screen_time": "Daily Screen Time",
}


def score_to_category(score):
    if score >= 85:
        return "Excellent", "success"
    elif score >= 70:
        return "Good", "info"
    elif score >= 50:
        return "Average", "warning"
    else:
        return "At Risk", "danger"


def generate_recommendations(inputs, score):
    """Rule-based, human-readable study recommendations based on the
    input profile and predicted score. Keeps the system explainable."""
    tips = []
    if inputs["study_hours_per_day"] < 3:
        tips.append("Increase daily study time to at least 3-4 focused hours.")
    if inputs["attendance_percentage"] < 75:
        tips.append("Improve class attendance — it strongly correlates with performance.")
    if inputs["sleep_hours"] < 6:
        tips.append("Aim for 7-8 hours of sleep to improve focus and retention.")
    if inputs["daily_screen_time"] > 5:
        tips.append("Reduce non-study screen time; it is negatively impacting the score.")
    if inputs["tutoring"] == 0 and score < 60:
        tips.append("Consider additional tutoring support for weaker subjects.")
    if inputs["study_environment"] == 0:
        tips.append("Try studying in a quieter, distraction-free environment.")
    if not tips:
        tips.append("Great habits! Keep maintaining consistency to sustain this performance.")
    return tips


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            inputs = {
                "study_hours_per_day": float(request.form["study_hours_per_day"]),
                "attendance_percentage": float(request.form["attendance_percentage"]),
                "previous_exam_score": float(request.form["previous_exam_score"]),
                "sleep_hours": float(request.form["sleep_hours"]),
                "extracurricular_activities": int(request.form["extracurricular_activities"]),
                "internet_access": int(request.form["internet_access"]),
                "tutoring": int(request.form["tutoring"]),
                "parental_education_level": int(request.form["parental_education_level"]),
                "study_environment": int(request.form["study_environment"]),
                "daily_screen_time": float(request.form["daily_screen_time"]),
            }

            if model_bundle is None:
                raise RuntimeError(
                    "Model not found. Please run 'python model/train_model.py' first."
                )

            features = model_bundle["features"]
            scaler = model_bundle["scaler"]
            regressor = model_bundle["regressor"]
            classifier = model_bundle["classifier"]

            X = pd.DataFrame([[inputs[f] for f in features]], columns=features)
            X_scaled = scaler.transform(X)

            predicted_score = float(np.clip(regressor.predict(X_scaled)[0], 0, 100))
            predicted_category = classifier.predict(X_scaled)[0]
            category_probs = classifier.predict_proba(X_scaled)[0]
            confidence = float(max(category_probs) * 100)

            category, badge_color = score_to_category(predicted_score)
            pass_fail = "Pass" if predicted_score >= 40 else "Fail"
            recommendations = generate_recommendations(inputs, predicted_score)

            result = {
                "score": round(predicted_score, 1),
                "category": category,
                "model_category": predicted_category,
                "badge_color": badge_color,
                "pass_fail": pass_fail,
                "confidence": round(confidence, 1),
                "recommendations": recommendations,
                "inputs": inputs,
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("predict.html", result=result, labels=FEATURE_LABELS)


@app.route("/dashboard")
def dashboard():
    chart_data = {}
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        chart_data["score_distribution"] = np.histogram(
            df["final_exam_score"], bins=10, range=(0, 100)
        )[0].tolist()
        chart_data["avg_by_tutoring"] = (
            df.groupby("tutoring")["final_exam_score"].mean().round(1).tolist()
        )
        chart_data["avg_by_extracurricular"] = (
            df.groupby("extracurricular_activities")["final_exam_score"]
            .mean()
            .round(1)
            .tolist()
        )
        chart_data["dataset_size"] = len(df)

    return render_template(
        "dashboard.html",
        metrics=metrics,
        feature_importance=feature_importance,
        labels=FEATURE_LABELS,
        chart_data=chart_data,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint - useful for demoing the system programmatically."""
    if model_bundle is None:
        return jsonify({"error": "Model not trained yet."}), 503
    try:
        data = request.get_json(force=True)
        features = model_bundle["features"]
        X = pd.DataFrame([[float(data[f]) for f in features]], columns=features)
        X_scaled = model_bundle["scaler"].transform(X)
        score = float(np.clip(model_bundle["regressor"].predict(X_scaled)[0], 0, 100))
        category, _ = score_to_category(score)
        return jsonify({"predicted_score": round(score, 1), "category": category})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
