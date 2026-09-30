"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset of student records for the
AI-Based Student Performance Prediction System.

Run this file to create data/student_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 1500  # number of student records

# ---- Feature generation -----------------------------------------------
study_hours = np.round(np.random.normal(4, 1.8, N).clip(0, 10), 1)
attendance = np.round(np.random.normal(80, 12, N).clip(40, 100), 1)
previous_score = np.round(np.random.normal(65, 15, N).clip(20, 100), 1)
sleep_hours = np.round(np.random.normal(6.5, 1.3, N).clip(3, 10), 1)
extracurricular = np.random.choice([0, 1], N, p=[0.55, 0.45])
internet_access = np.random.choice([0, 1], N, p=[0.2, 0.8])
tutoring = np.random.choice([0, 1], N, p=[0.7, 0.3])
parental_education = np.random.choice(
    [0, 1, 2, 3], N, p=[0.25, 0.35, 0.25, 0.15]
)  # 0=High School,1=Diploma,2=Bachelor,3=Postgraduate
study_environment = np.random.choice([0, 1], N, p=[0.3, 0.7])  # 0=Noisy,1=Quiet
screen_time = np.round(np.random.normal(4, 2, N).clip(0, 12), 1)

# ---- Target generation (final exam score) ------------------------------
# Weighted formula that mimics real-world influence of each factor,
# plus random noise to keep it realistic (not a perfectly linear system).
final_score = (
    study_hours * 4.2
    + attendance * 0.35
    + previous_score * 0.30
    + sleep_hours * 1.1
    + extracurricular * 1.5
    + internet_access * 2.0
    + tutoring * 3.0
    + parental_education * 2.2
    + study_environment * 2.5
    - screen_time * 1.3
    + np.random.normal(0, 6, N)
)

final_score = np.round(final_score.clip(0, 100), 1)

df = pd.DataFrame({
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "previous_exam_score": previous_score,
    "sleep_hours": sleep_hours,
    "extracurricular_activities": extracurricular,
    "internet_access": internet_access,
    "tutoring": tutoring,
    "parental_education_level": parental_education,
    "study_environment": study_environment,
    "daily_screen_time": screen_time,
    "final_exam_score": final_score,
})

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "student_data.csv")
df.to_csv(out_path, index=False)

print(f"Dataset generated successfully -> {out_path}")
print(df.head())
print(f"\nTotal records: {len(df)}")
