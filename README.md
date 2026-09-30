# EduPredict AI — AI-Based Student Performance Prediction System

A complete, professional, end-to-end machine learning web application that predicts a
student's final exam score and performance category from academic and lifestyle
indicators, built for a final-year academic project.

## Live Demo

Open the deployed app here:

https://performance-prediction-production.up.railway.app

## Features

- **Machine Learning pipeline**: synthetic dataset generator + RandomForestRegressor
  (score prediction) + RandomForestClassifier (category prediction), both built with
  scikit-learn.
- **Professional Flask web app** with 4 pages: Home, Predict, Dashboard, About.
- **Interactive prediction form** with an animated score ring, pass/fail verdict,
  model confidence, and personalized, rule-based study recommendations.
- **Analytics dashboard** with live Chart.js visualizations: feature importance,
  score distribution, tutoring impact, and a classification report table.
- **REST API endpoint** (`/api/predict`) for programmatic/JSON predictions —
  useful for demoing integrations during your viva/defense.
- Fully responsive, custom-designed UI (no generic template look).

## Project Structure

```
student-performance-prediction/
│
├── app.py                        # Flask application (routes & prediction logic)
├── requirements.txt               # Python dependencies
│
├── model/
│   ├── generate_dataset.py        # Creates the synthetic training dataset
│   ├── train_model.py             # Trains & saves the ML models
│   ├── student_performance_model.pkl   (generated after training)
│   ├── metrics.json                     (generated after training)
│   └── feature_importance.json          (generated after training)
│
├── data/
│   └── student_data.csv           (generated after running generate_dataset.py)
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── dashboard.html
│   └── about.html
│
└── static/
    ├── css/style.css
    └── js/script.js, dashboard.js
```

## How the Model Works

10 input features are used to predict the **final exam score (0–100)**:

| Feature | Description |
|---|---|
| study_hours_per_day | Average daily study time |
| attendance_percentage | Class attendance % |
| previous_exam_score | Score from the previous exam |
| sleep_hours | Average sleep per night |
| extracurricular_activities | Participates in extracurriculars (Yes/No) |
| internet_access | Has home internet access (Yes/No) |
| tutoring | Receives private tutoring (Yes/No) |
| parental_education_level | Highest parental education level |
| study_environment | Quiet or noisy study environment |
| daily_screen_time | Non-study screen time per day |

The predicted score is then mapped to a category:
`Excellent (≥85)` · `Good (≥70)` · `Average (≥50)` · `At Risk (<50)`,
and to a Pass/Fail verdict (Pass ≥ 40).

---

## ⚙️ Setup & Run Instructions

### 1. Requirements
- Python 3.9 or higher installed on your system
- pip (comes with Python)

### 2. Extract the project
Unzip `student-performance-prediction.zip` to a folder of your choice and open a
terminal / command prompt inside that folder.

### 3. Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Generate the dataset
```bash
python model/generate_dataset.py
```
This creates `data/student_data.csv` (1,500 synthetic student records).

### 6. Train the AI model
```bash
python model/train_model.py
```
This trains the Random Forest models and saves:
- `model/student_performance_model.pkl`
- `model/metrics.json`
- `model/feature_importance.json`

You only need to do steps 5 & 6 **once** (or whenever you want to retrain).

### 7. Run the web application
```bash
python app.py
```

You should see output similar to:
```
 * Running on http://127.0.0.1:5000
```

### 8. Open in your browser
Go to: **http://127.0.0.1:5000**

You'll land on the Home page. Use the navigation bar to try:
- **Predict** — enter a student profile and get an instant AI prediction
- **Dashboard** — view model accuracy, feature importance and dataset charts
- **About** — project overview, architecture, and tech stack (great for your report/demo)

To stop the server, press `CTRL + C` in the terminal.

---

## Retraining / Customizing

- To change dataset size or feature weighting, edit `model/generate_dataset.py`
  (see the `N` variable and the `final_score` formula) and re-run steps 5–6.
- To tune the model (e.g. number of trees, depth), edit the `RandomForestRegressor`
  / `RandomForestClassifier` parameters in `model/train_model.py`.
- All UI colors/fonts are defined as CSS variables at the top of
  `static/css/style.css` for easy re-theming.

## Suggested Report / Viva Talking Points

1. Problem statement & motivation (early identification of at-risk students).
2. Dataset design and feature selection rationale.
3. Model choice: why Random Forest (handles non-linear relationships, robust to
   noise, provides feature importance for explainability).
4. Evaluation metrics: R² score, MAE, RMSE (regression) and accuracy / precision /
   recall / F1 (classification) — all visible live on the Dashboard page.
5. System architecture (Data → Model → Flask API → UI) — see the About page.
6. Future scope: real institutional data, deep learning, LMS integration.

## Tech Stack

Python · Flask · scikit-learn · Pandas · NumPy · Chart.js · HTML5 · CSS3 · JavaScript

---

Built as a final-year academic project. Good luck with your submission! 🎓
