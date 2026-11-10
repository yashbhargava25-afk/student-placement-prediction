"""
app.py — Streamlit Web Application
------------------------------------
Student Placement Prediction System
B.Tech AI & ML End-Semester Project

Author (Integration): Yash Bhargava
Team: Sachin · Uttkarsh · Vivek Ramen · Saket · Yash Bhargava

Run:
    .venv\\Scripts\\streamlit.exe run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── PATH SETUP ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
SRC_DIR    = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "outputs", "plots")
DATA_DIR   = os.path.join(BASE_DIR, "data")
sys.path.insert(0, SRC_DIR)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── STYLING ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252940);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3d4166;
        margin: 8px 0;
    }
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 20px;
        text-align: center;
    }
    .result-placed {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 12px 0;
    }
    .result-not-placed {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 12px 0;
    }
    .disclaimer {
        background: #2d2d2d;
        border-left: 4px solid #f39c12;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #bbb;
        margin-top: 16px;
    }
    .summary-card {
        background: rgba(26, 29, 45, 0.85);
        border: 1px solid #373b54;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 6px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .summary-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.88rem;
    }
    .summary-item:last-child {
        border-bottom: none;
    }
    .summary-label {
        color: #9aa0b6;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .summary-value {
        font-weight: 600;
        color: #f1f5f9;
    }
    .badge-yes {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(46, 204, 113, 0.35);
    }
    .badge-no {
        background: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(231, 76, 60, 0.35);
    }
    .stSlider > div { color: #ccc; }
</style>
""", unsafe_allow_html=True)


# ── CACHED LOADERS ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name: str = "final_model"):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_comparison():
    path = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── HELPER FUNCTION (defined here, before if/elif blocks) ──────────────────────
def build_feature_input(iq, prev_sem, cgpa, acad_perf, internship,
                        extra, comm, projects, selected_model):
    """
    Build the feature dict for inference.
    Column names are auto-detected from the saved model's ColumnTransformer.
    Falls back to a reasonable default mapping matching the Kaggle dataset.
    """
    pipe = load_model(selected_model)
    expected_cols = None
    if pipe is not None:
        try:
            ct = pipe.named_steps["preprocessor"]
            expected_cols = list(ct.feature_names_in_)
        except Exception:
            pass

    # Default mapping — matches known Kaggle dataset columns
    base_map = {
        "IQ"                     : iq,
        "Prev_Sem_Result"        : prev_sem,
        "CGPA"                   : cgpa,
        "Academic_Performance"   : acad_perf,
        "Internship_Experience"  : "Yes" if (internship > 0 if isinstance(internship, (int, float)) else str(internship).strip().lower() in ["yes", "true", "1"]) else "No",
        "Extra_Curricular_Score" : extra,
        "Communication_Skills"   : comm,
        "Projects_Completed"     : projects,
    }

    if expected_cols is None:
        return base_map

    # Map to actual expected columns (case-insensitive matching)
    result = {}
    lower_base = {k.lower(): v for k, v in base_map.items()}
    for col in expected_cols:
        result[col] = lower_base.get(col.lower(), 0)
    return result


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Placement Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🔮 Predict Placement", "📊 Model Performance", "📈 EDA Insights", "ℹ️ About Project"],
        label_visibility="collapsed"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PREDICT PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Placement":
    st.markdown("""
    <div class="hero-header">
        <h1 style="color:white; margin:0; font-size:2rem;">🎓 Student Placement Prediction System</h1>
        <p style="color:#e0e0ff; margin:8px 0 0 0;">Enter student attributes below to predict placement likelihood</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model("final_model")
    if model is None:
        st.warning(
            "⚠️  **Model not found.** Please run the training pipeline first:\n\n"
            "```\n.venv\\Scripts\\python.exe src\\run_pipeline.py\n```\n\n"
            "Then refresh this page."
        )
        st.stop()

    st.subheader("📝 Student Profile Input")
    st.caption("Adjust sliders or select a quick profile preset below:")

    # ── QUICK PRESETS ──────────────────────────────────────────────────────────
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    if "cgpa_val" not in st.session_state:
        st.session_state.cgpa_val = 7.5
        st.session_state.prev_sem_val = 7.5
        st.session_state.iq_val = 105
        st.session_state.acad_perf_val = 6
        st.session_state.internship_val = "Yes"
        st.session_state.projects_val = 2
        st.session_state.comm_val = 6
        st.session_state.extra_val = 5

    with preset_col1:
        if st.button("🌟 Preset: High Achiever", use_container_width=True):
            st.session_state.cgpa_val = 8.8
            st.session_state.prev_sem_val = 8.5
            st.session_state.iq_val = 115
            st.session_state.acad_perf_val = 9
            st.session_state.internship_val = "Yes"
            st.session_state.projects_val = 5
            st.session_state.comm_val = 9
            st.session_state.extra_val = 8
            st.rerun()

    with preset_col2:
        if st.button("⚖️ Preset: Balanced Average", use_container_width=True):
            st.session_state.cgpa_val = 7.5
            st.session_state.prev_sem_val = 7.5
            st.session_state.iq_val = 100
            st.session_state.acad_perf_val = 6
            st.session_state.internship_val = "Yes"
            st.session_state.projects_val = 2
            st.session_state.comm_val = 6
            st.session_state.extra_val = 5
            st.rerun()

    with preset_col3:
        if st.button("⚠️ Preset: Needs Support", use_container_width=True):
            st.session_state.cgpa_val = 6.2
            st.session_state.prev_sem_val = 5.5
            st.session_state.iq_val = 80
            st.session_state.acad_perf_val = 4
            st.session_state.internship_val = "No"
            st.session_state.projects_val = 1
            st.session_state.comm_val = 3
            st.session_state.extra_val = 3
            st.rerun()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📚 Academic Attributes**")
        cgpa = st.slider("CGPA", 4.0, 10.0, float(st.session_state.cgpa_val), 0.1,
                         help="Cumulative Grade Point Average (10-point scale)")
        prev_sem = st.slider("Previous Semester Result", 4.0, 10.0, float(st.session_state.prev_sem_val), 0.1,
                              help="Most recent semester GPA (10-point scale)")
        iq = st.slider("IQ Score", 40, 160, int(st.session_state.iq_val), 1,
                       help="Student IQ score (normal range: 85 - 115)")
        acad_perf = st.slider("Academic Performance Rating", 1, 10, int(st.session_state.acad_perf_val), 1,
                              help="Overall academic performance rating on a 1-10 scale")

    with col2:
        st.markdown("**💼 Employability Attributes**")
        intern_default_idx = 0 if st.session_state.internship_val == "Yes" else 1
        internship_choice = st.selectbox(
            "Internship Experience",
            options=["Yes", "No"],
            index=intern_default_idx,
            help="Has the student completed any internships?"
        )
        internship = 1 if internship_choice == "Yes" else 0
        projects = st.slider(
            "Projects Completed",
            min_value=0,
            max_value=5,
            value=int(st.session_state.projects_val),
            step=1,
            help="Number of completed technical projects (0 to 5)"
        )
        comm = st.slider("Communication Skills", 1, 10, int(st.session_state.comm_val), 1,
                         help="Verbal and written communication rating (1 to 10)")
        extra = st.slider("Extra-Curricular Score", 0, 10, int(st.session_state.extra_val), 1,
                          help="Participation rating in clubs, sports, and events (0 to 10)")

    with col3:
        st.markdown("**⚙️ Model Selection**")
        if os.path.exists(MODELS_DIR):
            available_models = [
                f.replace(".pkl", "") for f in os.listdir(MODELS_DIR)
                if f.endswith(".pkl") and os.path.isfile(os.path.join(MODELS_DIR, f))
            ]
        else:
            available_models = ["final_model"]

        default_idx = available_models.index("final_model") if "final_model" in available_models else 0
        selected_model = st.selectbox("Choose Model", available_models, index=default_idx)

        st.markdown("---")
        st.markdown("**📋 Profile Overview**")
        intern_badge = '<span class="badge-yes">Yes (Completed)</span>' if internship_choice == "Yes" else '<span class="badge-no">No (None)</span>'
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-item">
                <span class="summary-label">🎓 CGPA</span>
                <span class="summary-value">{cgpa:.1f} / 10</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">📈 Prev Sem</span>
                <span class="summary-value">{prev_sem:.1f} / 10</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">🧠 IQ Score</span>
                <span class="summary-value">{iq}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">⭐ Acad. Rating</span>
                <span class="summary-value">{acad_perf} / 10</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">💼 Internship</span>
                <span>{intern_badge}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">💻 Projects</span>
                <span class="summary-value">{int(projects)} Completed</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">🗣️ Comm. Skills</span>
                <span class="summary-value">{comm} / 10</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">🎯 Extra-Curr.</span>
                <span class="summary-value">{extra} / 10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Placement", use_container_width=True, type="primary")

    if predict_btn:
        feature_map = build_feature_input(
            iq=iq,
            prev_sem=prev_sem, cgpa=cgpa,
            acad_perf=acad_perf, internship=internship,
            extra=extra, comm=comm, projects=projects,
            selected_model=selected_model
        )
        with st.spinner("Running prediction…"):
            pipe = load_model(selected_model)
            if pipe is None:
                st.error(f"Model '{selected_model}' not found.")
                st.stop()

            input_df = pd.DataFrame([feature_map])
            try:
                pred_class = int(pipe.predict(input_df)[0])
            except Exception as e:
                st.error(
                    f"Prediction failed: {e}\n\n"
                    "This usually means input column names don't match training data. "
                    "Re-run the pipeline after downloading the dataset."
                )
                st.stop()

            label = "Placed ✅" if pred_class == 1 else "Not Placed ❌"
            prob  = None
            try:
                prob = float(pipe.predict_proba(input_df)[0][1])
            except Exception:
                pass

        # ── Result display ─────────────────────────────────────────────────
        css_class = "result-placed" if pred_class == 1 else "result-not-placed"
        st.markdown(f'<div class="{css_class}">Prediction: {label}</div>',
                    unsafe_allow_html=True)

        if prob is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Placement Probability", f"{prob:.1%}")
            with col_b:
                st.metric("Not-Placed Probability", f"{1 - prob:.1%}")
            st.progress(prob, text=f"Placement confidence: {prob:.1%}")

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> This prediction is an <em>estimate</em> based on
            historical dataset patterns. It is NOT a guarantee of actual placement.
            Many real-world factors (interview performance, company requirements, economy,
            soft skills) are not captured in this dataset. Use this as one data point —
            not as a definitive outcome.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Analysis")
    st.caption("All metrics computed on the held-out test set (never seen during training/tuning)")

    comparison_df = load_comparison()
    if comparison_df is not None:
        st.subheader("📋 Comparison Table")
        numeric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score"]
                        if c in comparison_df.columns]
        styled = comparison_df.style.background_gradient(
            subset=numeric_cols, cmap="RdYlGn", vmin=0, vmax=1
        ).format({c: "{:.4f}" for c in numeric_cols})
        st.dataframe(styled, use_container_width=True)

        st.subheader("📊 Visual Comparison")
        chart_df = comparison_df[["Model"] + numeric_cols].copy()
        for m in numeric_cols:
            chart_df[m] = pd.to_numeric(chart_df[m], errors="coerce")
        st.bar_chart(chart_df.set_index("Model")[numeric_cols], height=350)

        st.subheader("🖼️ Generated Plots")
        plot_files = {
            "Model Accuracy Comparison": "model_accuracy_comparison.png",
            "RF Feature Importance"    : "08_rf_feature_importance.png",
            "Model Comparison"         : "11_model_comparison.png",
            "Confusion Matrices"       : "09_confusion_matrices.png",
            "ROC Curves"               : "10_roc_curves.png",
            "Regularization Sweep (LR)": "07_lr_regularization_sweep.png",
            "MLP Architecture Compare" : "12_mlp_architecture_comparison.png",
            "MLP Loss Curve"           : "13_mlp_baseline_loss_curve.png",
            "MLP Alpha (L2) Sweep"     : "14_mlp_alpha_sweep.png",
        }
        for title, fname in plot_files.items():
            fpath = os.path.join(PLOTS_DIR, fname)
            if os.path.exists(fpath):
                with st.expander(f"📌 {title}", expanded=False):
                    st.image(fpath, use_container_width=True)
    else:
        st.info(
            "No comparison data found yet.\n\n"
            "Run the full pipeline first:\n"
            "```\n.venv\\Scripts\\python.exe src\\run_pipeline.py\n```"
        )

    st.subheader("📚 Metric Explanations")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Accuracy** = correct / total\n\nMisleading when classes are imbalanced.")
    with c2:
        st.info("**F1-Score** = harmonic mean of Precision & Recall\n\nBest single metric for imbalanced data.")
    with c3:
        st.info("**ROC-AUC** = area under the ROC curve\n\nAUC=0.5: random; AUC=1.0: perfect.")

    st.subheader("⚠️ False Positives vs False Negatives")
    c1, c2 = st.columns(2)
    with c1:
        st.error("**False Positive (FP)**\nModel says: *Placed* ✅\nReality: *Not Placed* ❌\n\nRisk: Student skips preparation.")
    with c2:
        st.warning("**False Negative (FN)**\nModel says: *Not Placed* ❌\nReality: *Placed* ✅\n\nRisk: Unnecessary intervention.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EDA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Insights":
    st.title("📈 Exploratory Data Analysis Insights")
    st.caption("Visualisations generated from the training dataset (Uttkarsh's contribution)")

    eda_plots = {
        "Placement Distribution"       : "placement_distribution.png",
        "CGPA Distribution"            : "cgpa_distribution.png",
        "IQ Distribution"              : "iq_distribution.png",
        "Communication Skills Analysis": "communication_skills_analysis.png",
        "Projects Completed Analysis"  : "projects_completed_analysis.png",
        "Class Distribution Overview"  : "01_class_distribution.png",
        "Feature Distributions (All)"  : "02_numerical_distributions.png",
        "Box Plots (Outlier Detection)": "03_boxplots.png",
        "Correlation Heatmap"          : "04_correlation_heatmap.png",
    }

    if os.path.exists(PLOTS_DIR):
        for fname in sorted(os.listdir(PLOTS_DIR)):
            if fname.startswith("05_") or fname.startswith("06_"):
                label = fname.replace(".png", "").replace("_", " ").title()
                eda_plots[label] = fname

    captions = {
        "01_class_distribution.png": "If one class dominates, accuracy is misleading — use F1-score.",
        "04_correlation_heatmap.png": "Correlation does not imply causation.",
        "03_boxplots.png": "Points beyond whiskers (1.5×IQR) are potential outliers.",
    }

    for title, fname in eda_plots.items():
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            st.subheader(f"📊 {title}")
            st.image(fpath, use_container_width=True)
            if fname in captions:
                st.caption(f"ℹ️ {captions[fname]}")
            st.markdown("---")
        else:
            st.info(f"Plot '{fname}' not generated yet. Run the pipeline to create it.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT PROJECT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## Student Placement Prediction System

    **Course**: B.Tech — Artificial Intelligence and Machine Learning  
    **Project Type**: End-Semester Machine Learning Project  

    ---

    ### 🎯 Objective
    Build a machine learning system that predicts whether a student is likely
    to be placed based on their academic and employability attributes.

    ### 📦 Dataset
    **College Student Placement Factors Dataset** by Sahil Islam (Kaggle)  
    ≈10,000 synthetic records with features including IQ, CGPA, internship
    experience, communication skills, projects, and extra-curricular scores.

    > **Note**: This is a *simulated* dataset. Predictions reflect patterns in
    > synthetic data, not real placement records.

    ---

    ### 👥 Team Contributions
    """)

    team = pd.DataFrame([
        {"Member": "Sachin",        "Primary Role": "Data Collection, Cleaning & Preprocessing",
         "Key Deliverable": "data_preprocessing.py"},
        {"Member": "Uttkarsh",      "Primary Role": "EDA, Visualisations & Feature Engineering",
         "Key Deliverable": "feature_engineering.py, 11+ plots"},
        {"Member": "Vivek Ramen",   "Primary Role": "Logistic Regression (L1/L2) & Decision Tree",
         "Key Deliverable": "train_models.py (LR + DT)"},
        {"Member": "Saket",         "Primary Role": "Random Forest, Ensemble & GridSearchCV",
         "Key Deliverable": "train_models.py (RF), tuning_summary.csv"},
        {"Member": "Yash Bhargava", "Primary Role": "Evaluation, Comparison, App & Docs",
         "Key Deliverable": "evaluate_models.py, app.py, docs/"},
    ])
    st.dataframe(team, use_container_width=True, hide_index=True)

    st.markdown("""
    ---

    ### 🛠️ Technology Stack
    | Tool | Purpose |
    |------|---------|
    | Python 3.14 | Core language |
    | Pandas / NumPy | Data manipulation |
    | Scikit-learn | ML models, pipelines, GridSearchCV |
    | Matplotlib / Seaborn | Visualisations |
    | Streamlit | Interactive web interface |
    | Joblib | Model serialisation |

    ---

    ### 📋 ML Components Implemented
    - ✅ Data cleaning and preprocessing pipeline (ColumnTransformer)
    - ✅ Exploratory Data Analysis (11+ plots)
    - ✅ Feature engineering (academic_score, skills_index)
    - ✅ Logistic Regression with L1 and L2 regularization
    - ✅ Decision Tree with complexity controls
    - ✅ Random Forest (ensemble learning)
    - ✅ GridSearchCV hyperparameter tuning (all 3 models)
    - ✅ Confusion matrix, ROC curves, F1-score, ROC-AUC
    - ✅ Comparative model analysis table and charts

    ---

    ### ⚠️ Important Limitations
    1. Synthetic dataset — may not generalise to real placement data.
    2. Model captures *correlation*, not *causation*.
    3. Placement depends heavily on interview performance, company requirements,
       economic conditions, and soft skills — none captured in this dataset.
    4. Never use this prediction as a definitive career decision.

    ---

    ### 📚 References
    1. Islam, S. *College Student Placement Factors Dataset*. Kaggle, 2024.
    2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
    3. Scikit-learn Documentation. https://scikit-learn.org/stable/user_guide.html
    """)
