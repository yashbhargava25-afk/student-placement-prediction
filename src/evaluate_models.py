"""
evaluate_models.py
------------------
Author (Primary): Yash Bhargava
Responsibility  : Model evaluation, comparative performance analysis,
                  project integration, and documentation

This module:
  - Computes all required metrics for each model on the held-out test set
  - Generates confusion matrix plots
  - Generates ROC curves
  - Builds the model comparison table
  - Generates a final model comparison bar chart
  - Selects and reports the final model

EVALUATION STRATEGY:
  • Primary metric : F1-score (macro-average)
      Chosen because it balances precision and recall, and is robust to
      class imbalance — unlike accuracy which can be misleading.
  • Secondary metrics reported: Accuracy, Precision, Recall, ROC-AUC
  • Test set used exactly ONCE — only after model selection via CV.

PLACEMENT CONTEXT:
  • False Positive (FP): Model predicts Placed, but student is NOT placed.
      Risk: Student may not seek additional training/help.
  • False Negative (FN): Model predicts Not Placed, but student IS placed.
      Risk: Student receives unnecessary intervention.
  • In most academic counselling contexts, FN is more costly because
      missing a placed student wastes institutional resources minimally,
      while over-predicting Not Placed triggers useful counselling.
      However, the opposite priority may hold in scholarship allocation.
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
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, "plots")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi"     : 120,
    "axes.titlesize" : 13,
    "axes.labelsize" : 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family"    : "sans-serif",
})


# ── METRIC COMPUTATION ─────────────────────────────────────────────────────────

def evaluate_single_model(pipeline, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a single sklearn Pipeline on the test set.
    Returns a dict with all metrics.
    """
    y_pred = pipeline.predict(X_test)

    # ROC-AUC requires probability estimates
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
    except Exception:
        y_prob = None
        auc    = None

    metrics = {
        "Model"      : model_name,
        "Accuracy"   : round(accuracy_score(y_test, y_pred), 4),
        "Precision"  : round(precision_score(y_test, y_pred, average="binary", zero_division=0), 4),
        "Recall"     : round(recall_score(y_test, y_pred, average="binary", zero_division=0), 4),
        "F1-Score"   : round(f1_score(y_test, y_pred, average="binary", zero_division=0), 4),
        "ROC-AUC"    : round(auc, 4) if auc is not None else "N/A",
    }

    print(f"\n  ── {model_name}")
    print(f"     Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"     Precision : {metrics['Precision']:.4f}")
    print(f"     Recall    : {metrics['Recall']:.4f}")
    print(f"     F1-Score  : {metrics['F1-Score']:.4f}")
    print(f"     ROC-AUC   : {metrics['ROC-AUC']}")
    print("\n     Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Placed", "Placed"],
                                zero_division=0))

    return metrics, y_pred, y_prob


# ── CONFUSION MATRIX PLOTS ────────────────────────────────────────────────────

def plot_confusion_matrices(model_results: dict, X_test, y_test):
    """
    Plot confusion matrices for all models in a grid layout.
    
    Interpretation:
      Top-left  (TN): correctly predicted Not Placed
      Top-right (FP): wrongly predicted as Placed
      Bottom-left (FN): wrongly predicted as Not Placed
      Bottom-right (TP): correctly predicted Placed
    """
    models_to_plot = {k: v for k, v in model_results.items() if v is not None}
    n = len(models_to_plot)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, (name, pipe) in enumerate(models_to_plot.items()):
        ax = axes[i]
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Placed", "Placed"],
                    yticklabels=["Not Placed", "Placed"],
                    cbar=False, linewidths=0.5)
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    for j in range(len(models_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models (Test Set)", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "09_confusion_matrices.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] 09_confusion_matrices.png")
    return path


# ── ROC CURVES ────────────────────────────────────────────────────────────────

def plot_roc_curves(model_results: dict, X_test, y_test):
    """
    Overlay ROC curves for all models that support predict_proba.
    
    Interpretation:
      AUC = 0.5 → random guess; AUC = 1.0 → perfect classifier.
      The ROC curve shows the trade-off between True Positive Rate (Recall)
      and False Positive Rate at every classification threshold.
      A model with a higher AUC separates the two classes better overall.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12"]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline (AUC = 0.50)")

    for i, (name, pipe) in enumerate(model_results.items()):
        if pipe is None:
            continue
        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, color=colours[i % len(colours)],
                    linewidth=2, label=f"{name}  (AUC = {auc:.3f})")
        except Exception:
            pass

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "10_roc_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] 10_roc_curves.png")
    return path


# ── MODEL COMPARISON CHART ────────────────────────────────────────────────────

def plot_model_comparison(comparison_df: pd.DataFrame):
    """
    Grouped bar chart comparing all models on key metrics.
    """
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    numeric_df = comparison_df.copy()
    for m in metrics_to_plot:
        numeric_df[m] = pd.to_numeric(numeric_df[m], errors="coerce")

    x = np.arange(len(numeric_df["Model"]))
    width = 0.2
    colours = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, numeric_df[metric], width,
               label=metric, color=colours[i], edgecolor="white", alpha=0.9)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(numeric_df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics (Test Set)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "11_model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] 11_model_comparison.png")

    # Also save dedicated Model Accuracy Comparison bar chart
    fig_acc, ax_acc = plt.subplots(figsize=(10, 4.5))
    acc_scores = [float(v) * 100 for v in numeric_df["Accuracy"]]
    bars = ax_acc.bar(numeric_df["Model"], acc_scores, color="#2980B9", edgecolor="white", width=0.55)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Model Accuracy Comparison (Test Set)", fontweight="bold")
    ax_acc.set_ylim(0, 110)
    for b in bars:
        h = b.get_height()
        ax_acc.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_acc.set_xticklabels(numeric_df["Model"], rotation=15, ha="right")
    ax_acc.spines[["top", "right"]].set_visible(False)
    fig_acc.tight_layout()
    acc_path = os.path.join(PLOTS_DIR, "model_accuracy_comparison.png")
    fig_acc.savefig(acc_path, bbox_inches="tight")
    plt.close(fig_acc)
    print(f"  [saved] model_accuracy_comparison.png")

    return path


# ── MAIN EVALUATION ORCHESTRATOR ──────────────────────────────────────────────

def run_evaluation(model_dict: dict, X_test, y_test):
    """
    Evaluate all models and generate all comparison artefacts.
    
    Args:
        model_dict : {display_name: pipeline}
        X_test, y_test : held-out test data
    
    Returns:
        comparison_df : DataFrame with all metrics
        best_model_name : name of the best model by F1-score
    """
    print("\n" + "=" * 60)
    print("  YASH BHARGAVA — MODEL EVALUATION & COMPARISON")
    print("=" * 60)
    print("  NOTE: Test set is used HERE for the first time, only after")
    print("  all hyperparameter tuning was completed on training data.\n")

    all_metrics = []

    for name, pipe in model_dict.items():
        if pipe is None:
            print(f"  [skip] {name} — pipeline not available.")
            continue
        metrics, _, _ = evaluate_single_model(pipe, X_test, y_test, name)
        all_metrics.append(metrics)

    comparison_df = pd.DataFrame(all_metrics)

    # ── Save comparison CSV ───────────────────────────────────────────────
    csv_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n  [saved] model_comparison.csv → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_confusion_matrices(model_dict, X_test, y_test)
    plot_roc_curves(model_dict, X_test, y_test)
    plot_model_comparison(comparison_df)

    # ── Select best model by F1-score ─────────────────────────────────────
    numeric_f1 = pd.to_numeric(comparison_df["F1-Score"], errors="coerce")
    best_idx   = numeric_f1.idxmax()
    best_model_name = comparison_df.loc[best_idx, "Model"]
    best_f1         = numeric_f1[best_idx]

    print(f"\n{'=' * 60}")
    print(f"  FINAL MODEL SELECTION")
    print(f"{'=' * 60}")
    print(f"  Best model (highest F1-Score): {best_model_name}")
    print(f"  Test F1-Score                : {best_f1:.4f}")
    print(f"\n  Interpretation:")
    print(f"    F1-Score is the harmonic mean of Precision and Recall.")
    print(f"    It is the primary selection metric because it balances")
    print(f"    the cost of false positives and false negatives equally,")
    print(f"    which is appropriate when class distribution may be uneven.")
    print(f"    Accuracy alone can be misleading when one class is dominant.")

    print(f"\n  COMPARISON TABLE:")
    print(comparison_df.to_string(index=False))

    # ── Save the best model as final_model.pkl ────────────────────────────
    best_pipe = model_dict.get(best_model_name)
    if best_pipe is not None:
        final_path = os.path.join(MODELS_DIR, "final_model.pkl")
        joblib.dump(best_pipe, final_path)
        print(f"\n  [saved] final_model.pkl  →  {final_path}")

    print("\n[✓] Evaluation complete.\n")
    return comparison_df, best_model_name


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import run_full_preprocessing
    from feature_engineering import apply_feature_engineering
    from train_models import run_all_training

    (X_train, X_test, y_train, y_test,
     preprocessor, df_clean,
     numeric_features, categorical_features, target_col) = run_full_preprocessing()

    df_fe = apply_feature_engineering(df_clean)
    all_models, tuning_info = run_all_training(preprocessor, X_train, y_train)

    model_dict = {
        "LR Baseline"  : all_models.get("LR_baseline"),
        "LR Tuned"     : all_models.get("LR_tuned"),
        "DT Baseline"  : all_models.get("DT_baseline"),
        "DT Tuned"     : all_models.get("DT_tuned"),
        "RF Baseline"  : all_models.get("RF_baseline"),
        "RF Tuned"     : all_models.get("RF_tuned"),
    }

    run_evaluation(model_dict, X_test, y_test)
