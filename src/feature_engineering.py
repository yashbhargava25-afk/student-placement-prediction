"""
feature_engineering.py
-----------------------
Author (Primary): Uttkarsh
Responsibility  : EDA visualisations, feature engineering, and derived features

This module:
  - Creates derived academic and employability features
  - Provides an explanation for every transformation
  - Ensures all derivations use only feature columns (no target leakage)
  - Saves plots to outputs/plots/

DESIGN NOTES:
  • academic_score  = 0.6 × CGPA + 0.4 × Prev_Sem_Result
      Rationale: CGPA reflects cumulative performance; Prev_Sem_Result captures
      recent momentum. Weights are a reasonable heuristic — not a guaranteed
      improvement. Only created if both source columns exist.

  • skills_index = mean(communication_skills, extra_curricular_score,
                        projects_completed)
      Rationale: Combines complementary employability signals into one measure.
      Normalised to 0-1 range where data scales allow.
      Only created if at least one source column exists.

LIMITATION: Derived features collapse information. Their predictive value
must be verified empirically (via model evaluation), not assumed.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Possible column name variants (case-insensitive matching done at runtime)
CGPA_VARIANTS       = ["cgpa", "gpa", "cumulative_gpa"]
PREV_SEM_VARIANTS   = ["prev_sem_result", "prev_semester_result", "previous_sem", "prev_sem"]
COMM_VARIANTS       = ["communication_skills", "communication", "comm_skills"]
EXTRA_VARIANTS      = ["extra_curricular_score", "extracurricular_score", "extra_curricular", "extra_curricular_activities"]
PROJECTS_VARIANTS   = ["projects_completed", "projects", "num_projects", "no_of_projects"]
IQ_VARIANTS         = ["iq", "iq_score"]
INTERNSHIP_VARIANTS = ["internship_experience", "internship", "internships"]

plt.rcParams.update({
    "figure.dpi"        : 120,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "font.family"       : "sans-serif",
})

# ── COLUMN DETECTION ──────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, variants: list) -> str | None:
    """Return the first matching column name (case-insensitive), or None."""
    lower_map = {c.lower(): c for c in df.columns}
    for v in variants:
        if v.lower() in lower_map:
            return lower_map[v.lower()]
    return None


# ── DERIVED FEATURES ──────────────────────────────────────────────────────────

def derive_academic_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    academic_score = 0.6 × CGPA  +  0.4 × Prev_Sem_Result
    
    Both columns are normalised to [0,1] before combining so that
    differing scales don't distort the blend.
    Column is added only when BOTH source columns exist.
    """
    df = df.copy()
    cgpa_col     = _find_col(df, CGPA_VARIANTS)
    prev_sem_col = _find_col(df, PREV_SEM_VARIANTS)

    if cgpa_col and prev_sem_col:
        cgpa_norm = (df[cgpa_col] - df[cgpa_col].min()) / (df[cgpa_col].max() - df[cgpa_col].min() + 1e-9)
        prev_norm = (df[prev_sem_col] - df[prev_sem_col].min()) / (df[prev_sem_col].max() - df[prev_sem_col].min() + 1e-9)
        df["academic_score"] = 0.6 * cgpa_norm + 0.4 * prev_norm
        print(f"[FE] 'academic_score' derived from '{cgpa_col}' and '{prev_sem_col}'.")
    else:
        print(f"[FE] Skipped 'academic_score' — source columns not found "
              f"(CGPA: {cgpa_col}, Prev_Sem: {prev_sem_col}).")
    return df


def derive_skills_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    skills_index = normalised mean of available employability columns.
    
    Includes: communication_skills, extra_curricular_score, projects_completed.
    Each component is normalised to [0,1] before averaging.
    """
    df = df.copy()
    comm_col     = _find_col(df, COMM_VARIANTS)
    extra_col    = _find_col(df, EXTRA_VARIANTS)
    projects_col = _find_col(df, PROJECTS_VARIANTS)

    components = []
    labels     = []
    for col, name in [(comm_col, "comm"), (extra_col, "extra"), (projects_col, "projects")]:
        if col:
            norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
            components.append(norm)
            labels.append(name)

    if components:
        df["skills_index"] = pd.concat(components, axis=1).mean(axis=1)
        print(f"[FE] 'skills_index' derived from: {labels}.")
    else:
        print("[FE] Skipped 'skills_index' — no source columns found.")
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature-engineering steps and return the enriched DataFrame.
    Safe to call on the full cleaned DataFrame (before splitting).
    """
    print("\n" + "=" * 60)
    print("  UTTKARSH — FEATURE ENGINEERING")
    print("=" * 60)
    df = derive_academic_score(df)
    df = derive_skills_index(df)
    print(f"\n[FE] Final feature set ({df.shape[1]} columns): {list(df.columns)}\n")
    return df


# ── EDA VISUALISATIONS ────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {filename}")
    return path


def plot_class_distribution(df: pd.DataFrame, target_col: str) -> str:
    """
    Bar chart of placement class distribution.
    
    Interpretation: Shows whether the dataset is balanced. Significant
    imbalance means accuracy alone is misleading — F1-score is preferred.
    """
    counts = df[target_col].value_counts()
    labels = ["Not Placed (0)", "Placed (1)"]
    colours = ["#E74C3C", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colours, edgecolor="white", linewidth=1.2)
    ax.set_title("Placement Class Distribution", fontweight="bold")
    ax.set_ylabel("Number of Students")
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, "01_class_distribution.png")


def plot_numerical_distributions(df: pd.DataFrame, numeric_cols: list, target_col: str) -> str:
    """
    Histograms with KDE for each numerical feature, coloured by placement class.
    
    Interpretation: Helps spot which features separate placed from not-placed
    students. Wide separation indicates stronger predictive potential.
    """
    cols_to_plot = [c for c in numeric_cols if c != target_col][:12]  # cap at 12
    n = len(cols_to_plot)
    if n == 0:
        return ""

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    palette = {0: "#E74C3C", 1: "#2ECC71"}
    for i, col in enumerate(cols_to_plot):
        for cls in [0, 1]:
            subset = df[df[target_col] == cls][col].dropna()
            axes[i].hist(subset, bins=30, alpha=0.55, color=palette[cls],
                         label=["Not Placed", "Placed"][cls], density=True, edgecolor="none")
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=7)
        axes[i].spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions by Placement Class", fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "02_numerical_distributions.png")


def plot_boxplots(df: pd.DataFrame, numeric_cols: list, target_col: str) -> str:
    """
    Box plots of numerical features grouped by placement class.
    
    Interpretation: Boxes that don't overlap between classes suggest
    that the feature discriminates well. Whiskers indicate outlier extent.
    """
    cols_to_plot = [c for c in numeric_cols if c != target_col][:12]
    n = len(cols_to_plot)
    if n == 0:
        return ""

    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    palette = {0: "#E74C3C", 1: "#2ECC71"}
    for i, col in enumerate(cols_to_plot):
        data_plot = [
            df[df[target_col] == 0][col].dropna().values,
            df[df[target_col] == 1][col].dropna().values,
        ]
        try:
            bp = axes[i].boxplot(data_plot, patch_artist=True,
                                 tick_labels=["Not Placed", "Placed"],
                                 medianprops=dict(color="black", linewidth=2))
        except TypeError:
            bp = axes[i].boxplot(data_plot, patch_artist=True,
                                 labels=["Not Placed", "Placed"],
                                 medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], [palette[0], palette[1]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_title(col)
        axes[i].spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Box Plots: Numerical Features vs Placement", fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "03_boxplots.png")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list, target_col: str) -> str:
    """
    Pearson correlation heatmap for numerical features (including target).
    
    Interpretation: Shows linear relationships between features and the
    target. High absolute correlation with target = potentially useful
    predictor. High inter-feature correlation = possible multicollinearity
    (relevant for Logistic Regression).
    """
    cols = [c for c in numeric_cols if c in df.columns] + [target_col]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) - 2)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}, vmin=-1, vmax=1
    )
    ax.set_title("Correlation Heatmap (Numerical Features + Target)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_correlation_heatmap.png")


def plot_placement_by_feature(df: pd.DataFrame, target_col: str) -> list:
    """
    Bar/strip charts for key features vs. placement rate.
    Covers internship, communication, projects, IQ, CGPA if present.
    """
    saved = []
    categorical_features = {
        "internship": INTERNSHIP_VARIANTS,
    }
    continuous_features = {
        "iq"         : IQ_VARIANTS,
        "cgpa"       : CGPA_VARIANTS,
        "comm"       : COMM_VARIANTS,
        "extra"      : EXTRA_VARIANTS,
        "projects"   : PROJECTS_VARIANTS,
    }

    # ── Categorical bar: placement rate per category ──
    for label, variants in categorical_features.items():
        col = _find_col(df, variants)
        if col is None:
            continue
        rate = df.groupby(col)[target_col].mean().reset_index()
        rate.columns = [col, "placement_rate"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(rate[col].astype(str), rate["placement_rate"],
               color=["#3498DB", "#E74C3C", "#2ECC71"][:len(rate)], edgecolor="white")
        ax.set_title(f"Placement Rate by {col}", fontweight="bold")
        ax.set_ylabel("Placement Rate")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        saved.append(_save(fig, f"05_placement_by_{label}.png"))

    # ── Continuous: grouped mean placement rate ──
    for label, variants in continuous_features.items():
        col = _find_col(df, variants)
        if col is None:
            continue
        # Bin into quartiles
        df_copy = df[[col, target_col]].copy()
        df_copy["bin"] = pd.qcut(df_copy[col], q=4, duplicates="drop",
                                 labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
        rate = df_copy.groupby("bin")[target_col].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        rate.plot(kind="bar", ax=ax, color="#3498DB", edgecolor="white", alpha=0.85)
        ax.set_title(f"Placement Rate by {col} Quartile", fontweight="bold")
        ax.set_ylabel("Placement Rate")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlabel(f"{col} Quartile")
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        saved.append(_save(fig, f"06_placement_by_{label}.png"))

    return saved


def plot_individual_eda_showcase(df: pd.DataFrame, target_col: str) -> list:
    """
    Generate individual high-resolution showcase charts matching
    standard EDA expectations:
      - Placement Distribution
      - CGPA Distribution
      - IQ Distribution
      - Communication Skills Analysis
      - Projects Completed Analysis
    """
    saved = []

    # 1. Placement Distribution (Pie + Bar)
    counts = df[target_col].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    colours = ["#E74C3C", "#2ECC71"]
    labels = ["Not Placed", "Placed"]
    ax1.pie(
        [counts.get(0, 0), counts.get(1, 0)],
        labels=labels,
        colors=colours,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    ax1.set_title("Placement Distribution (%)", fontweight="bold")

    bars = ax2.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colours, edgecolor="white", width=0.55)
    ax2.set_ylabel("Student Count")
    ax2.set_title("Student Placement Count", fontweight="bold")
    ax2.bar_label(bars, fmt="%d", padding=3)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    saved.append(_save(fig, "placement_distribution.png"))

    # 2. CGPA Distribution
    cgpa_col = _find_col(df, CGPA_VARIANTS)
    if cgpa_col:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cls, colr, lbl in [(0, "#E74C3C", "Not Placed"), (1, "#2ECC71", "Placed")]:
            subset = df[df[target_col] == cls][cgpa_col].dropna()
            ax.hist(subset, bins=25, alpha=0.6, color=colr, label=lbl, edgecolor="white", density=True)
            subset.plot(kind="kde", ax=ax, color=colr, linewidth=2)
        ax.set_title("CGPA Distribution by Placement Outcome", fontweight="bold")
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Density")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        saved.append(_save(fig, "cgpa_distribution.png"))

    # 3. IQ Distribution
    iq_col = _find_col(df, IQ_VARIANTS)
    if iq_col:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cls, colr, lbl in [(0, "#E74C3C", "Not Placed"), (1, "#2ECC71", "Placed")]:
            subset = df[df[target_col] == cls][iq_col].dropna()
            ax.hist(subset, bins=25, alpha=0.6, color=colr, label=lbl, edgecolor="white", density=True)
            subset.plot(kind="kde", ax=ax, color=colr, linewidth=2)
        ax.set_title("IQ Distribution by Placement Outcome", fontweight="bold")
        ax.set_xlabel("IQ Score")
        ax.set_ylabel("Density")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        saved.append(_save(fig, "iq_distribution.png"))

    # 4. Communication Skills Analysis
    comm_col = _find_col(df, COMM_VARIANTS)
    if comm_col:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        # Counts per score
        comm_counts = df.groupby([comm_col, target_col]).size().unstack(fill_value=0)
        comm_counts.plot(kind="bar", stacked=False, ax=ax1, color=["#E74C3C", "#2ECC71"], edgecolor="white")
        ax1.set_title("Communication Skills vs Placement Count", fontweight="bold")
        ax1.set_xlabel("Communication Skills Rating")
        ax1.set_ylabel("Number of Students")
        ax1.legend(["Not Placed", "Placed"])
        ax1.spines[["top", "right"]].set_visible(False)

        # Placement rate by score
        rate = df.groupby(comm_col)[target_col].mean()
        rate.plot(kind="line", marker="o", color="#2980B9", linewidth=2.5, markersize=7, ax=ax2)
        ax2.set_title("Placement Rate by Communication Skills", fontweight="bold")
        ax2.set_xlabel("Communication Skills Rating")
        ax2.set_ylabel("Placement Probability")
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        saved.append(_save(fig, "communication_skills_analysis.png"))

    # 5. Projects Completed Analysis
    proj_col = _find_col(df, PROJECTS_VARIANTS)
    if proj_col:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        proj_counts = df.groupby([proj_col, target_col]).size().unstack(fill_value=0)
        proj_counts.plot(kind="bar", stacked=False, ax=ax1, color=["#E74C3C", "#2ECC71"], edgecolor="white")
        ax1.set_title("Projects Completed vs Placement Count", fontweight="bold")
        ax1.set_xlabel("Number of Projects Completed")
        ax1.set_ylabel("Student Count")
        ax1.legend(["Not Placed", "Placed"])
        ax1.spines[["top", "right"]].set_visible(False)

        rate = df.groupby(proj_col)[target_col].mean()
        bars = ax2.bar(rate.index.astype(str), rate.values, color="#3498DB", edgecolor="white", width=0.55)
        ax2.set_title("Placement Rate by Projects Completed", fontweight="bold")
        ax2.set_xlabel("Projects Completed")
        ax2.set_ylabel("Placement Rate")
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        for b in bars:
            height = b.get_height()
            ax2.annotate(f"{height:.1%}",
                         xy=(b.get_x() + b.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)
        ax2.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        saved.append(_save(fig, "projects_completed_analysis.png"))

    return saved


def run_eda(df: pd.DataFrame, numeric_cols: list, target_col: str) -> None:
    """
    Run the full EDA suite and save all plots to outputs/plots/.
    """
    print("\n" + "=" * 60)
    print("  UTTKARSH — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"  Saving plots to: {PLOTS_DIR}\n")

    plot_class_distribution(df, target_col)
    plot_numerical_distributions(df, numeric_cols, target_col)
    plot_boxplots(df, numeric_cols, target_col)
    plot_correlation_heatmap(df, numeric_cols, target_col)
    plot_placement_by_feature(df, target_col)
    plot_individual_eda_showcase(df, target_col)

    print("\n[✓] EDA complete — all plots saved.\n")


if __name__ == "__main__":
    from data_preprocessing import run_full_preprocessing
    (X_train, X_test, y_train, y_test,
     preprocessor, df_clean,
     numeric_features, categorical_features, target_col) = run_full_preprocessing()

    df_fe = apply_feature_engineering(df_clean)

    # Re-identify numeric cols after FE
    new_numeric = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    new_numeric = [c for c in new_numeric if c != target_col]

    run_eda(df_fe, new_numeric, target_col)
