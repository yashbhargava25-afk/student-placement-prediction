"""
predict.py
----------
Author (Primary): Yash Bhargava
Responsibility  : Inference — loading saved model and making predictions

This module:
  - Loads the final trained pipeline (preprocessor + model)
  - Accepts new student data as a dictionary or DataFrame
  - Returns predicted class and probability (if available)
  - Used by the Streamlit application (app.py)

CRITICAL: Input features must use the EXACT same column names used during
training. This is guaranteed because the same ColumnTransformer handles
encoding/scaling automatically via the saved pipeline.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model(model_name: str = "final_model"):
    """
    Load a saved pipeline from models/<model_name>.pkl.
    Raises FileNotFoundError with clear instructions if not found.
    """
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Model file not found: {path}\n"
            "  Please run the training pipeline first:\n"
            "    python src\\evaluate_models.py\n"
            "  or use the master pipeline script:\n"
            "    python src\\run_pipeline.py\n"
        )
    pipeline = joblib.load(path)
    return pipeline


def predict(student_data: dict, model_name: str = "final_model"):
    """
    Predict placement for a single student.
    
    Args:
        student_data : dict of {feature_name: value}
        model_name   : pkl filename (without .pkl extension)
    
    Returns:
        dict with keys:
          'predicted_class'  : 1 (Placed) or 0 (Not Placed)
          'label'            : "Placed" or "Not Placed"
          'probability'      : float ∈ [0,1] if model supports it, else None
          'confidence_note'  : reminder that this is an estimate
    """
    pipeline = load_model(model_name)
    df = pd.DataFrame([student_data])
    if "Internship_Experience" in df.columns:
        df["Internship_Experience"] = df["Internship_Experience"].apply(
            lambda v: "Yes" if (v > 0 if isinstance(v, (int, float)) else (str(v).strip().lower() in ["yes", "true", "y"] or (str(v).strip().isdigit() and int(str(v).strip()) > 0))) else "No"
        )

    predicted_class = int(pipeline.predict(df)[0])

    probability = None
    try:
        probability = float(pipeline.predict_proba(df)[0][1])
    except Exception:
        pass

    label = "Placed" if predicted_class == 1 else "Not Placed"

    return {
        "predicted_class": predicted_class,
        "label"          : label,
        "probability"    : probability,
        "confidence_note": (
            "⚠️  This prediction is an estimate based on historical data patterns. "
            "It is NOT a guarantee of actual placement outcome. Many real-world "
            "factors (interview performance, company requirements, economy) are "
            "not captured by this model."
        ),
    }


def predict_batch(df: pd.DataFrame, model_name: str = "final_model") -> pd.DataFrame:
    """
    Predict placement for a batch of students.
    
    Args:
        df : DataFrame with feature columns (no target column)
    
    Returns:
        df with added columns: 'predicted_class', 'predicted_label', 'probability'
    """
    pipeline = load_model(model_name)
    df = df.copy()
    if "Internship_Experience" in df.columns:
        df["Internship_Experience"] = df["Internship_Experience"].apply(
            lambda v: "Yes" if (v > 0 if isinstance(v, (int, float)) else (str(v).strip().lower() in ["yes", "true", "y"] or (str(v).strip().isdigit() and int(str(v).strip()) > 0))) else "No"
        )
    df["predicted_class"] = pipeline.predict(df)
    df["predicted_label"] = df["predicted_class"].map({1: "Placed", 0: "Not Placed"})
    try:
        df["probability"] = pipeline.predict_proba(df)[:, 1]
    except Exception:
        df["probability"] = None
    return df


if __name__ == "__main__":
    # Quick smoke-test with a sample student
    sample = {
        "Prev_Sem_Result"       : 8.5,
        "CGPA"                  : 8.2,
        "Academic_Performance"  : 3,
        "Internship_Experience" : 1,
        "Extra_Curricular_Score": 7,
        "Communication_Skills"  : 8,
        "Projects_Completed"    : 3,
    }
    result = predict(sample)
    print("\n── Sample Prediction ──────────────────────────")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
