"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset matching the Kaggle
"College Student Placement Factors Dataset" schema by Sahil Islam.

Columns:
  College_ID, IQ, Prev_Sem_Result, CGPA, Academic_Performance,
  Internship_Experience, Extra_Curricular_Score, Communication_Skills,
  Projects_Completed, Placement

Target: Placement (Yes/No)  →  ~65% placed, ~35% not placed
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 10000

# ── Feature generation ────────────────────────────────────────────────────────
college_ids = [f"COL{str(i+1).zfill(5)}" for i in range(N)]

iq                    = np.random.normal(110, 15, N).clip(70, 160).round(0).astype(int)
cgpa                  = np.random.normal(7.2, 1.2, N).clip(4.0, 10.0).round(2)
prev_sem              = (cgpa + np.random.normal(0, 0.5, N)).clip(4.0, 10.0).round(2)
academic_performance  = np.random.choice([1, 2, 3, 4, 5], N,
                                         p=[0.05, 0.10, 0.30, 0.35, 0.20])
internship_experience = np.random.choice([0, 1], N, p=[0.45, 0.55])
extra_curricular      = np.random.randint(0, 11, N)
communication_skills  = np.random.randint(1, 11, N)
projects_completed    = np.random.randint(0, 11, N)

# ── Placement probability (logistic function of features) ─────────────────────
# Intercept calibrated so ~65% placed, ~35% not placed
log_odds = (
    -12.5
    + 0.030 * iq
    + 0.40  * cgpa
    + 0.30  * prev_sem
    + 0.25  * academic_performance
    + 1.00  * internship_experience
    + 0.05  * extra_curricular
    + 0.08  * communication_skills
    + 0.06  * projects_completed
    + np.random.normal(0, 1.2, N)   # noise
)
prob_placed = 1 / (1 + np.exp(-log_odds))
placement_binary = (np.random.rand(N) < prob_placed).astype(int)
placement = np.where(placement_binary == 1, "Yes", "No")

# ── Build DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame({
    "College_ID"             : college_ids,
    "IQ"                     : iq,
    "Prev_Sem_Result"        : prev_sem,
    "CGPA"                   : cgpa,
    "Academic_Performance"   : academic_performance,
    "Internship_Experience"  : internship_experience,
    "Extra_Curricular_Score" : extra_curricular,
    "Communication_Skills"   : communication_skills,
    "Projects_Completed"     : projects_completed,
    "Placement"              : placement,
})

# ── Introduce ~2% missing values for realism ─────────────────────────────────
for col in ["IQ", "CGPA", "Communication_Skills", "Projects_Completed"]:
    mask = np.random.rand(N) < 0.02
    df.loc[mask, col] = np.nan

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "college_student_placement.csv")
df.to_csv(out_path, index=False)

placed     = (df["Placement"] == "Yes").sum()
not_placed = (df["Placement"] == "No").sum()
print(f"[OK] Dataset generated -> {out_path}")
print(f"     Rows: {N:,}  |  Placed: {placed:,} ({placed/N*100:.1f}%)  |  Not Placed: {not_placed:,} ({not_placed/N*100:.1f}%)")
print(f"     Columns: {list(df.columns)}")
