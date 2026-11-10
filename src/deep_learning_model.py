"""
deep_learning_model.py
-----------------------
Author (Additional): All members — Deep Learning Extension

This module implements a Multi-Layer Perceptron (MLP) Neural Network using
scikit-learn's MLPClassifier, which is fully compatible with the existing
sklearn Pipeline infrastructure.

THEORETICAL BACKGROUND — NEURAL NETWORKS:
==========================================
A Multi-Layer Perceptron is a feedforward artificial neural network with:

  Input Layer  →  Hidden Layer(s)  →  Output Layer
  
Each neuron computes: z = W·x + b, then applies activation: a = f(z)

Components:
  • Neurons      : Basic computing units that apply a weighted sum + bias
  • Weights (W)  : Learnable parameters adjusted during training
  • Biases (b)   : Learnable offsets for each neuron
  • Activation   : Non-linear function applied to each neuron's output

Activation Functions used:
  • ReLU  (Hidden): max(0, z) — fast, avoids vanishing gradient, sparse
  • Logistic (Output): σ(z) = 1/(1+e^-z) — produces probabilities [0,1]

Training Algorithm:
  • Forward Pass : Input propagates through layers producing a prediction
  • Loss Function: Binary cross-entropy measures prediction error
  • Backpropagation: Gradient of loss w.r.t. each weight computed via chain rule
  • Optimizer    : Adam (Adaptive Moment Estimation) — adaptive learning rate

Regularization in Neural Networks:
  • L2 (alpha)   : Adds λ·‖W‖² to loss — penalises large weights, reduces overfitting
  • Early stopping: Stops training when validation loss stops improving

Why Neural Networks for Placement Prediction:
  • Can learn non-linear decision boundaries automatically
  • Discovers feature interactions without manual engineering
  • Scales well with more data

Limitations:
  • Less interpretable than LR/DT (black box)
  • Requires feature scaling (StandardScaler applied by pipeline)
  • More hyperparameters to tune
  • Prone to overfitting on small/synthetic datasets
  • Higher computational cost than LR

NOTE: TensorFlow/Keras requires Python ≤ 3.11 at the time of this project.
Since this project runs on Python 3.14, we use sklearn's MLPClassifier which
implements the same MLP mathematics and is fully pipeline-compatible.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score
)

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_SEED = 42
CV_FOLDS    = 5
SCORING     = "f1"

plt.rcParams.update({
    "figure.dpi"    : 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family"   : "sans-serif",
})


# ── HELPER ────────────────────────────────────────────────────────────────────
def _make_pipeline(preprocessor, model):
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _save_model(pipeline, name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(pipeline, path)
    print(f"  [saved] {name}.pkl")
    return path


def _save_plot(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {filename}")
    return path


# ── ARCHITECTURE CONFIGURATIONS ───────────────────────────────────────────────
"""
Network Architectures explored:

  Shallow  (1 hidden layer):  (64,)
    → Simple, less expressive, fast, low overfitting risk

  Standard (2 hidden layers): (128, 64)
    → Recommended for tabular data; can learn moderate non-linearities

  Deep     (3 hidden layers): (256, 128, 64)
    → More expressive; risk of overfitting on ~10K synthetic dataset

Architecture choice for tabular placement data:
  (128, 64) is a reasonable default — deep enough to capture non-linear
  patterns, not so deep that it overfits on a ~10K synthetic dataset.
"""

ARCHITECTURES = {
    "Shallow  (64)":       (64,),
    "Standard (128-64)":   (128, 64),
    "Deep     (256-128-64)": (256, 128, 64),
}


# ── BASELINE MLP ──────────────────────────────────────────────────────────────
def train_mlp_baseline(preprocessor, X_train, y_train):
    """
    Train a baseline MLP Neural Network.

    Architecture: 2 hidden layers — (128, 64) neurons
    Activation  : ReLU for hidden layers (fast convergence, no vanishing gradient)
    Solver      : Adam (adaptive learning rate, handles noisy gradients well)
    Alpha (L2)  : 0.0001 — mild weight regularization to prevent overfitting
    Max iter    : 500 — maximum training epochs
    """
    print("\n" + "─" * 55)
    print("  DEEP LEARNING — MLP NEURAL NETWORK (Baseline)")
    print("─" * 55)
    print("\n  Architecture : Input → 128 → 64 → Output (sigmoid)")
    print("  Activation   : ReLU (hidden), Logistic (output)")
    print("  Optimizer    : Adam")
    print("  L2 Alpha     : 0.0001")

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,          # L2 regularization strength
        batch_size="auto",
        learning_rate="adaptive",
        max_iter=500,
        early_stopping=True,   # stops when val loss plateaus
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    pipe = _make_pipeline(preprocessor, mlp)
    pipe.fit(X_train, y_train)

    n_iter = pipe.named_steps["model"].n_iter_
    print(f"\n  Training converged in {n_iter} iterations (early stopping active)")
    _save_model(pipe, "mlp_baseline")
    _plot_loss_curve(pipe, "mlp_baseline")
    return pipe


# ── ARCHITECTURE COMPARISON ───────────────────────────────────────────────────
def compare_architectures(preprocessor, X_train, y_train, X_test, y_test):
    """
    Train MLP with different architectures and compare training F1 scores.
    Helps visualise how network depth affects learning.
    """
    print("\n  [ARCH COMPARE] Shallow / Standard / Deep architectures …")
    results = {}
    for name, arch in ARCHITECTURES.items():
        mlp = MLPClassifier(
            hidden_layer_sizes=arch,
            activation="relu", solver="adam",
            alpha=0.0001, max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_SEED, verbose=False,
        )
        pipe = _make_pipeline(preprocessor, mlp)
        pipe.fit(X_train, y_train)
        train_f1 = f1_score(y_train, pipe.predict(X_train), zero_division=0)
        test_f1  = f1_score(y_test,  pipe.predict(X_test),  zero_division=0)
        results[name] = {"train_f1": train_f1, "test_f1": test_f1,
                         "n_iter": pipe.named_steps["model"].n_iter_}
        print(f"    {name:30s}  train F1={train_f1:.4f}  test F1={test_f1:.4f}")

    # Plot
    names      = list(results.keys())
    train_vals = [results[n]["train_f1"] for n in names]
    test_vals  = [results[n]["test_f1"]  for n in names]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, train_vals, w, color="#3498DB", label="Train F1", alpha=0.85)
    ax.bar(x + w/2, test_vals,  w, color="#E74C3C", label="Test F1",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-Score")
    ax.set_title("MLP Architecture Comparison: Train vs Test F1", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_plot(fig, "12_mlp_architecture_comparison.png")
    return results


# ── LOSS CURVE PLOT ───────────────────────────────────────────────────────────
def _plot_loss_curve(pipeline, name):
    """
    Plot the training (and validation) loss curve over iterations.
    
    Interpretation:
      Steadily decreasing loss = model is learning.
      Validation loss diverging from training loss = overfitting.
      Flat loss = convergence (early stopping fired).
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "loss_curve_"):
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.loss_curve_, color="#3498DB", linewidth=2, label="Training Loss")
    if hasattr(model, "validation_scores_") and model.validation_scores_ is not None:
        # validation_scores_ is accuracy — invert for loss proxy
        val_acc = model.validation_scores_
        ax.plot(
            [1 - v for v in val_acc],
            color="#E74C3C", linewidth=2, linestyle="--",
            label="Validation (1-Accuracy)"
        )
    ax.set_xlabel("Iteration (Epoch)")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("MLP Neural Network — Training Loss Curve\n"
                 "(Flat tail = early stopping fired)", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_plot(fig, f"13_{name}_loss_curve.png")


# ── REGULARIZATION: ALPHA SWEEP ───────────────────────────────────────────────
def sweep_regularization(preprocessor, X_train, y_train, X_test, y_test):
    """
    Sweep the L2 regularization parameter alpha for MLP.

    alpha (L2): penalises large weights → prevents overfitting.
    Small alpha → complex model (low bias, high variance).
    Large alpha → simpler model (high bias, low variance).
    """
    print("\n  [L2 ALPHA SWEEP] MLP regularization …")
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    train_scores, test_scores = [], []

    for a in alphas:
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            solver="adam", alpha=a, max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_SEED, verbose=False,
        )
        pipe = _make_pipeline(preprocessor, mlp)
        pipe.fit(X_train, y_train)
        train_scores.append(f1_score(y_train, pipe.predict(X_train), zero_division=0))
        test_scores.append(f1_score(y_test,   pipe.predict(X_test),  zero_division=0))
        print(f"    alpha={a:.5f}  train F1={train_scores[-1]:.4f}  test F1={test_scores[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas, train_scores, "o-", color="#3498DB", linewidth=2, label="Train F1")
    ax.plot(alphas, test_scores,  "s--", color="#E74C3C", linewidth=2, label="Test F1")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (L2 Regularization Strength)")
    ax.set_ylabel("F1-Score")
    ax.set_title("MLP Neural Network: L2 Regularization (Alpha) Sweep", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_plot(fig, "14_mlp_alpha_sweep.png")

    best_alpha = alphas[np.argmax(test_scores)]
    print(f"\n  Best alpha (by test F1): {best_alpha}")
    return best_alpha


# ── GRIDSEARCHCV TUNING ───────────────────────────────────────────────────────
def tune_mlp(preprocessor, X_train, y_train):
    """
    GridSearchCV for MLP.
    Grid is kept modest to run on a student laptop (5-10 min expected).

    Tuned parameters:
      hidden_layer_sizes : Network depth/width
      alpha              : L2 regularization strength
      learning_rate_init : Initial Adam step size
    """
    print("\n  [GridSearchCV] MLP Neural Network …")
    param_grid = {
        "model__hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)],
        "model__alpha"             : [1e-4, 1e-3, 1e-2],
        "model__learning_rate_init": [0.001, 0.005],
    }
    pipe = _make_pipeline(
        preprocessor,
        MLPClassifier(
            activation="relu", solver="adam",
            max_iter=500, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15,
            random_state=RANDOM_SEED, verbose=False,
        )
    )
    gs = GridSearchCV(pipe, param_grid, cv=CV_FOLDS, scoring=SCORING,
                      n_jobs=-1, verbose=1, refit=True)
    gs.fit(X_train, y_train)

    print(f"\n    Best params : {gs.best_params_}")
    print(f"    Best CV F1  : {gs.best_score_:.4f}")
    _save_model(gs.best_estimator_, "mlp_tuned")

    # Save GridSearch results
    pd.DataFrame(gs.cv_results_).to_csv(
        os.path.join(OUTPUTS_DIR, "mlp_gridsearch_results.csv"), index=False
    )
    print("  [saved] mlp_gridsearch_results.csv")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ── MAIN ORCHESTRATOR ─────────────────────────────────────────────────────────
def run_deep_learning(preprocessor, X_train, y_train, X_test, y_test):
    """
    Run the full deep learning pipeline:
      1. Baseline MLP
      2. Architecture comparison
      3. L2 regularization sweep
      4. GridSearchCV tuning
    
    Returns:
      dict of {"MLP Baseline": pipe, "MLP Tuned": pipe}
    """
    print("\n" + "=" * 60)
    print("  DEEP LEARNING — MULTI-LAYER PERCEPTRON (MLP)")
    print("=" * 60)
    print(f"""
  What is a Neural Network?
  ─────────────────────────
  A Multi-Layer Perceptron (MLP) is a fully connected feedforward
  neural network. It processes input features through stacked layers
  of neurons, each applying: output = activation(W·input + bias)

  Training uses backpropagation to compute gradients of the loss
  function with respect to every weight, then updates weights via
  the Adam optimiser.

  Why add MLP to this project?
  ─────────────────────────────
  • Captures non-linear decision boundaries that LR cannot
  • Learns feature interactions automatically
  • Provides a more modern, expressive model for comparison
  • Demonstrates knowledge of neural network concepts
    """)

    results = {}

    # 1. Baseline
    mlp_baseline = train_mlp_baseline(preprocessor, X_train, y_train)
    results["MLP Baseline"] = mlp_baseline

    # 2. Architecture comparison
    arch_results = compare_architectures(preprocessor, X_train, y_train, X_test, y_test)

    # 3. L2 regularization sweep
    best_alpha = sweep_regularization(preprocessor, X_train, y_train, X_test, y_test)

    # 4. GridSearchCV tuning
    mlp_tuned, best_params, cv_score = tune_mlp(preprocessor, X_train, y_train)
    results["MLP Tuned"] = mlp_tuned

    print("\n[✓] Deep Learning training complete.\n")
    return results, best_params, cv_score


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import run_full_preprocessing

    (X_train, X_test, y_train, y_test,
     preprocessor, df_clean,
     numeric_features, categorical_features, target_col) = run_full_preprocessing()

    dl_models, best_params, cv_score = run_deep_learning(
        preprocessor, X_train, y_train, X_test, y_test
    )

    print("\n── DL Model Test Results ───────────────────────────────")
    from sklearn.metrics import classification_report
    for name, pipe in dl_models.items():
        y_pred = pipe.predict(X_test)
        print(f"\n{name}:")
        print(classification_report(y_test, y_pred,
                                    target_names=["Not Placed", "Placed"]))
