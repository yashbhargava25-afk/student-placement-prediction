"""
train_models.py
---------------
Authors:
  Vivek Ramen — Logistic Regression (L1/L2 regularization) + Decision Tree
  Saket       — Random Forest, ensemble learning, GridSearchCV tuning

This module:
  - Defines all three classifiers with baseline configurations
  - Implements L1 and L2 regularization experiments for Logistic Regression
  - Implements tree complexity controls (max_depth, min_samples_leaf)
  - Runs GridSearchCV hyperparameter tuning on training data only
  - Saves the best model pipeline (preprocessor + model) via joblib
  - Generates a feature-importance plot for Random Forest

DESIGN NOTES:
  • All models are wrapped in sklearn Pipelines so that preprocessing
    (scaling/encoding) is always applied consistently.
  • GridSearchCV uses cv=5 (stratified by default for classifiers) and
    scoring='f1' to handle potential class imbalance fairly.
  • The test set is never seen during tuning — it is used only once, in
    evaluate_models.py.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_SEED = 42
CV_FOLDS    = 5
SCORING     = "f1"   # F1 chosen because it balances Precision/Recall
                     # and is robust when classes may be imbalanced.

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _make_pipeline(preprocessor, model):
    """Wrap preprocessor + model into a single sklearn Pipeline."""
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _save_model(pipeline: Pipeline, name: str) -> str:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(pipeline, path)
    print(f"  [saved] {name}.pkl  →  {path}")
    return path


def _save_plot(fig, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {filename}")
    return path


# ── LOGISTIC REGRESSION (Vivek Ramen) ─────────────────────────────────────────

def train_logistic_regression(preprocessor, X_train, y_train):
    """
    LOGISTIC REGRESSION
    ───────────────────
    Working principle:
      Logistic Regression models the log-odds of the positive class as a
      linear combination of input features. A sigmoid function maps this to
      a probability ∈ (0,1). The boundary is learned by minimising binary
      cross-entropy (log-loss) using gradient-based optimisation (lbfgs).

    Regularization:
      L2 (Ridge) — adds λ·‖w‖² to the loss; shrinks all coefficients toward
                   zero uniformly, preventing any single feature from dominating.
                   Good default choice; keeps all features.
      L1 (Lasso) — adds λ·‖w‖₁ to the loss; encourages sparse solutions by
                   driving irrelevant feature coefficients exactly to zero.
                   Acts as automatic feature selection.
      C = 1/λ: smaller C → stronger regularization; larger C → weaker.

    Limitation: Assumes a linear decision boundary. Will underfit if the
    true boundary is non-linear.
    """
    print("\n" + "─" * 55)
    print("  VIVEK RAMEN — LOGISTIC REGRESSION")
    print("─" * 55)

    results = {}

    # ── Baseline (L2, default C=1) ──────────────────────────────────────
    print("\n  [1] Baseline Logistic Regression (L2, C=1) …")
    lr_baseline = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=1000, random_state=RANDOM_SEED
    )
    pipe_baseline = _make_pipeline(preprocessor, lr_baseline)
    pipe_baseline.fit(X_train, y_train)
    results["LR_baseline"] = pipe_baseline
    _save_model(pipe_baseline, "lr_baseline")

    # ── L2 regularization sweep ──────────────────────────────────────────
    print("\n  [2] L2 Regularization sweep (C values) …")
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    l2_scores = {}
    for c in c_values:
        lr = LogisticRegression(
            penalty="l2", C=c, solver="lbfgs",
            max_iter=1000, random_state=RANDOM_SEED
        )
        pipe = _make_pipeline(preprocessor, lr)
        pipe.fit(X_train, y_train)
        train_acc = pipe.score(X_train, y_train)
        l2_scores[c] = train_acc
        print(f"    C={c:7.3f}  →  train accuracy: {train_acc:.4f}")
    results["LR_l2_sweep"] = l2_scores

    # ── L1 regularization (requires saga solver) ─────────────────────────
    print("\n  [3] L1 Regularization (saga solver) …")
    l1_scores = {}
    for c in c_values:
        lr = LogisticRegression(
            penalty="l1", C=c, solver="saga",
            max_iter=2000, random_state=RANDOM_SEED
        )
        pipe = _make_pipeline(preprocessor, lr)
        pipe.fit(X_train, y_train)
        train_acc = pipe.score(X_train, y_train)
        l1_scores[c] = train_acc
        print(f"    C={c:7.3f}  →  train accuracy: {train_acc:.4f}")
    results["LR_l1_sweep"] = l1_scores

    # ── Plot: Regularization strength vs accuracy ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(c_values, list(l2_scores.values()), "o-", color="#3498DB", label="L2 (Ridge)")
    ax.plot(c_values, list(l1_scores.values()), "s--", color="#E74C3C", label="L1 (Lasso)")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("Training Accuracy")
    ax.set_title("Logistic Regression: L1 vs L2 Regularization", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_plot(fig, "07_lr_regularization_sweep.png")

    return results


def tune_logistic_regression(preprocessor, X_train, y_train):
    """
    GridSearchCV hyperparameter tuning for Logistic Regression.
    Tuned on training data ONLY (cv=5 stratified folds).
    """
    print("\n  [4] GridSearchCV — Logistic Regression …")
    param_grid = [
        {"model__penalty": ["l2"], "model__C": [0.01, 0.1, 1.0, 10.0],
         "model__solver": ["lbfgs"]},
        {"model__penalty": ["l1"], "model__C": [0.01, 0.1, 1.0, 10.0],
         "model__solver": ["saga"]},
    ]
    pipe = _make_pipeline(preprocessor,
                          LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
    gs = GridSearchCV(pipe, param_grid, cv=CV_FOLDS, scoring=SCORING,
                      n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_train, y_train)

    print(f"    Best params : {gs.best_params_}")
    print(f"    Best CV F1  : {gs.best_score_:.4f}")
    _save_model(gs.best_estimator_, "lr_tuned")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ── DECISION TREE (Vivek Ramen) ────────────────────────────────────────────────

def train_decision_tree(preprocessor, X_train, y_train):
    """
    DECISION TREE
    ─────────────
    Working principle:
      A Decision Tree recursively partitions the feature space by choosing
      splits that maximise information gain (Gini impurity or entropy).
      Each leaf stores the majority class. No scaling required.

    Complexity control (overfitting prevention):
      max_depth         — limits tree depth; shallow trees generalise better.
      min_samples_leaf  — requires each leaf to have ≥ N samples; prevents
                          trees from memorising tiny subsets.
      min_samples_split — minimum samples needed to attempt a split.

    Limitation: Single trees are high-variance (overfit). Results can change
    drastically with small data changes → motivates Random Forest (ensemble).
    """
    print("\n" + "─" * 55)
    print("  VIVEK RAMEN — DECISION TREE")
    print("─" * 55)

    results = {}

    # ── Baseline (no depth limit) ─────────────────────────────────────────
    print("\n  [1] Baseline Decision Tree (unconstrained) …")
    dt_baseline = DecisionTreeClassifier(random_state=RANDOM_SEED)
    pipe_baseline = _make_pipeline(preprocessor, dt_baseline)
    pipe_baseline.fit(X_train, y_train)
    print(f"    Tree depth: {pipe_baseline.named_steps['model'].get_depth()}")
    results["DT_baseline"] = pipe_baseline
    _save_model(pipe_baseline, "dt_baseline")

    # ── Depth sweep for complexity control ────────────────────────────────
    print("\n  [2] Depth-constrained Decision Trees …")
    depths = [2, 3, 5, 7, 10, None]
    depth_scores = {}
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5,
                                    random_state=RANDOM_SEED)
        pipe = _make_pipeline(preprocessor, dt)
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_train, y_train)
        depth_scores[str(d)] = acc
        print(f"    max_depth={str(d):5s}  →  train accuracy: {acc:.4f}")
    results["DT_depth_sweep"] = depth_scores

    return results


def tune_decision_tree(preprocessor, X_train, y_train):
    """GridSearchCV for Decision Tree. Tuned on training data only."""
    print("\n  [3] GridSearchCV — Decision Tree …")
    param_grid = {
        "model__max_depth"        : [3, 5, 7, 10, None],
        "model__min_samples_split": [2, 10, 20],
        "model__min_samples_leaf" : [1, 5, 10],
        "model__criterion"        : ["gini", "entropy"],
    }
    pipe = _make_pipeline(preprocessor,
                          DecisionTreeClassifier(random_state=RANDOM_SEED))
    gs = GridSearchCV(pipe, param_grid, cv=CV_FOLDS, scoring=SCORING,
                      n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_train, y_train)

    print(f"    Best params : {gs.best_params_}")
    print(f"    Best CV F1  : {gs.best_score_:.4f}")
    _save_model(gs.best_estimator_, "dt_tuned")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ── RANDOM FOREST — ENSEMBLE LEARNING (Saket) ─────────────────────────────────

def train_random_forest(preprocessor, X_train, y_train):
    """
    RANDOM FOREST — ENSEMBLE LEARNING
    ───────────────────────────────────
    What is ensemble learning?
      Ensemble methods combine multiple weak learners to produce a stronger,
      more stable predictor. The key idea is that averaging the errors of
      many independent models reduces variance.

    How Random Forest works:
      1. Bootstrap sampling: Each tree is trained on a random sample (with
         replacement) of the training data — called bagging (Bootstrap AGGregating).
      2. Random feature selection: At each split, only a random subset of
         features (√p for classification) is considered. This de-correlates
         the trees, making the ensemble more diverse.
      3. Majority vote: Each tree predicts a class; the forest takes a vote.

    Why it outperforms a single Decision Tree:
      A single tree has high variance — small data changes cause large
      prediction changes. By averaging many de-correlated trees, variance
      averages out while bias stays low.

    Computational cost:
      O(n_estimators × N × log N × √p) per training. Inference is slower
      than a single tree but still practical.

    Limitations:
      Less interpretable than a single tree. Memory-intensive for large
      n_estimators. Feature importances do not imply causality.
    """
    print("\n" + "─" * 55)
    print("  SAKET — RANDOM FOREST (ENSEMBLE LEARNING)")
    print("─" * 55)

    results = {}

    # ── Baseline Random Forest ────────────────────────────────────────────
    print("\n  [1] Baseline Random Forest (100 trees) …")
    rf_baseline = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
    )
    pipe_baseline = _make_pipeline(preprocessor, rf_baseline)
    pipe_baseline.fit(X_train, y_train)
    results["RF_baseline"] = pipe_baseline
    _save_model(pipe_baseline, "rf_baseline")

    # ── Feature Importance Plot ───────────────────────────────────────────
    _plot_feature_importance(pipe_baseline, X_train)

    return results


def _plot_feature_importance(pipeline: Pipeline, X_train: pd.DataFrame):
    """
    Generate a horizontal bar chart of Random Forest feature importances.

    NOTE: Feature importance (Mean Decrease in Impurity) shows which
    features the model relied on most. It does NOT establish causality —
    a feature can rank highly due to correlation with a causal variable.
    """
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Retrieve feature names after transformation
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, "get_feature_names_out"):
                feature_names.extend(trans.get_feature_names_out(cols))
            elif hasattr(trans.steps[-1][1], "get_feature_names_out"):
                feature_names.extend(
                    trans.steps[-1][1].get_feature_names_out(cols)
                )
            else:
                feature_names.extend(cols if hasattr(cols, "__iter__") else [cols])

        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # top 20

        fig, ax = plt.subplots(figsize=(8, max(4, len(indices) * 0.35)))
        ax.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color="#3498DB", edgecolor="white"
        )
        ax.set_title("Random Forest — Feature Importances\n"
                     "(Mean Decrease in Impurity; does NOT imply causality)",
                     fontweight="bold")
        ax.set_xlabel("Importance")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, "08_rf_feature_importance.png")
        fig.savefig(path, bbox_inches="tight")
        # Also save as feature_importance_analysis.png
        alt_path = os.path.join(PLOTS_DIR, "feature_importance_analysis.png")
        fig.savefig(alt_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] 08_rf_feature_importance.png & feature_importance_analysis.png")

    except Exception as e:
        print(f"  [warn] Could not generate feature importance plot: {e}")


def tune_random_forest(preprocessor, X_train, y_train):
    """
    GridSearchCV for Random Forest. Tuned on training data only.
    Grid is deliberately modest to run on a standard student laptop.
    """
    print("\n  [2] GridSearchCV — Random Forest …")
    param_grid = {
        "model__n_estimators"  : [50, 100, 200],
        "model__max_depth"     : [None, 5, 10],
        "model__max_features"  : ["sqrt", "log2"],
        "model__min_samples_leaf": [1, 5],
    }
    pipe = _make_pipeline(preprocessor,
                          RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1))
    gs = GridSearchCV(pipe, param_grid, cv=CV_FOLDS, scoring=SCORING,
                      n_jobs=-1, verbose=1, refit=True)
    gs.fit(X_train, y_train)

    print(f"\n    Best params : {gs.best_params_}")
    print(f"    Best CV F1  : {gs.best_score_:.4f}")
    _save_model(gs.best_estimator_, "rf_tuned")

    # Save GridSearch results
    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(os.path.join(BASE_DIR, "outputs", "rf_gridsearch_results.csv"),
                      index=False)
    print("  [saved] rf_gridsearch_results.csv")

    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ── MAIN TRAINING ORCHESTRATOR ────────────────────────────────────────────────

def run_all_training(preprocessor, X_train, y_train):
    """
    Train and tune all three models.
    Returns a dict of all trained pipelines and tuning results.
    """
    print("\n" + "=" * 60)
    print("  MODEL TRAINING (VIVEK RAMEN + SAKET)")
    print("=" * 60)
    print(f"  Scoring metric : {SCORING}  |  CV folds : {CV_FOLDS}")
    print(f"  Training set   : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")

    all_results = {}

    # ── Logistic Regression ──────────────────────────────────────────────
    lr_results = train_logistic_regression(preprocessor, X_train, y_train)
    all_results.update(lr_results)
    lr_best, lr_params, lr_cv_score = tune_logistic_regression(
        preprocessor, X_train, y_train
    )
    all_results["LR_tuned"] = lr_best

    # ── Decision Tree ────────────────────────────────────────────────────
    dt_results = train_decision_tree(preprocessor, X_train, y_train)
    all_results.update(dt_results)
    dt_best, dt_params, dt_cv_score = tune_decision_tree(
        preprocessor, X_train, y_train
    )
    all_results["DT_tuned"] = dt_best

    # ── Random Forest ────────────────────────────────────────────────────
    rf_results = train_random_forest(preprocessor, X_train, y_train)
    all_results.update(rf_results)
    rf_best, rf_params, rf_cv_score = tune_random_forest(
        preprocessor, X_train, y_train
    )
    all_results["RF_tuned"] = rf_best

    # ── Save tuning summary ───────────────────────────────────────────────
    tuning_summary = pd.DataFrame([
        {"model": "Logistic Regression", "best_params": str(lr_params),
         "best_cv_f1": lr_cv_score},
        {"model": "Decision Tree",       "best_params": str(dt_params),
         "best_cv_f1": dt_cv_score},
        {"model": "Random Forest",       "best_params": str(rf_params),
         "best_cv_f1": rf_cv_score},
    ])
    out_path = os.path.join(BASE_DIR, "outputs", "tuning_summary.csv")
    tuning_summary.to_csv(out_path, index=False)
    print(f"\n  [saved] tuning_summary.csv → {out_path}")

    print("\n[✓] All models trained and saved.\n")
    return all_results, {
        "LR": (lr_best, lr_params, lr_cv_score),
        "DT": (dt_best, dt_params, dt_cv_score),
        "RF": (rf_best, rf_params, rf_cv_score),
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import run_full_preprocessing
    from feature_engineering import apply_feature_engineering

    (X_train, X_test, y_train, y_test,
     preprocessor, df_clean,
     numeric_features, categorical_features, target_col) = run_full_preprocessing()

    df_fe = apply_feature_engineering(df_clean)

    run_all_training(preprocessor, X_train, y_train)
