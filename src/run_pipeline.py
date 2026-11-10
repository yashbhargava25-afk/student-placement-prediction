"""
run_pipeline.py
---------------
Master script that executes the complete ML pipeline end-to-end.

Run from the project root:
    .venv\\Scripts\\python.exe src\\run_pipeline.py

This script:
  1. Loads and preprocesses the dataset          (Sachin)
  2. Applies feature engineering                  (Uttkarsh)
  3. Runs EDA and saves plots                     (Uttkarsh)
  4. Trains all models with regularization        (Vivek Ramen + Saket)
  5. Tunes hyperparameters via GridSearchCV       (Saket + Vivek Ramen)
  6. Evaluates all models on the test set         (Yash Bhargava)
  7. Saves final model and comparison artefacts   (Yash Bhargava)
"""

import sys
import os
import time

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import run_full_preprocessing
from feature_engineering import apply_feature_engineering, run_eda
from train_models import run_all_training
from deep_learning_model import run_deep_learning
from evaluate_models import run_evaluation

import numpy as np

def main():
    start = time.time()
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  STUDENT PLACEMENT PREDICTION SYSTEM — FULL PIPELINE    ║")
    print("║  B.Tech AI & ML End-Semester Project                    ║")
    print("║  Team: Sachin · Uttkarsh · Vivek Ramen · Saket · Yash  ║")
    print("╚" + "═" * 58 + "╝\n")

    # ── STEP 1: Data Preprocessing (Sachin) ──────────────────────────────
    (X_train, X_test, y_train, y_test,
     preprocessor, df_clean,
     numeric_features, categorical_features, target_col) = run_full_preprocessing()

    # ── STEP 2: Feature Engineering (Uttkarsh) ───────────────────────────
    df_fe = apply_feature_engineering(df_clean)

    # Refresh numeric features list after FE (new columns may have been added)
    fe_numeric = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    fe_numeric  = [c for c in fe_numeric if c != target_col]

    # Re-run preprocessing on the FE dataframe for a fresh split
    # (the pipeline in app.py will use the original preprocessor)
    # For training we use the original split to keep consistency
    # — derived features could be added as a pipeline step in production

    # ── STEP 3: EDA (Uttkarsh) ───────────────────────────────────────────
    run_eda(df_fe, fe_numeric, target_col)

    # ── STEP 4+5: Train + Tune Models (Vivek Ramen + Saket) ──────────────
    all_models, tuning_info = run_all_training(preprocessor, X_train, y_train)

    # ── STEP 5b: Deep Learning — MLP Neural Network ───────────────────────
    dl_models, dl_best_params, dl_cv_score = run_deep_learning(
        preprocessor, X_train, y_train, X_test, y_test
    )
    all_models.update(dl_models)

    # ── STEP 6+7: Evaluate + Compare (Yash Bhargava) ─────────────────────
    model_dict = {
        "LR Baseline" : all_models.get("LR_baseline"),
        "LR Tuned"    : all_models.get("LR_tuned"),
        "DT Baseline" : all_models.get("DT_baseline"),
        "DT Tuned"    : all_models.get("DT_tuned"),
        "RF Baseline" : all_models.get("RF_baseline"),
        "RF Tuned"    : all_models.get("RF_tuned"),
        "MLP Baseline": all_models.get("MLP Baseline"),
        "MLP Tuned"   : all_models.get("MLP Tuned"),
    }
    comparison_df, best_model = run_evaluation(model_dict, X_test, y_test)

    elapsed = time.time() - start
    print(f"\n{'═' * 60}")
    print(f"  Pipeline finished in {elapsed:.1f}s")
    print(f"  Best model : {best_model}")
    print(f"  All outputs in : outputs/  and  models/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
