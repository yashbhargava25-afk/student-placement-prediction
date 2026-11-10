# Team Contributions — Student Placement Prediction System

## Project Information
- **Title**: Student Placement Prediction System
- **Course**: B.Tech Artificial Intelligence and Machine Learning
- **Project Type**: End-Semester ML Project
- **Date**: October 2026

---

## Team Member Contributions

### 1. Sachin — Data Collection, Cleaning & Preprocessing
**Primary Module**: `src/data_preprocessing.py`

| Task | Description |
|------|-------------|
| Dataset acquisition | Downloaded and verified the Kaggle dataset (Sahil Islam, College Student Placement Factors) |
| Schema inspection | Examined columns, dtypes, shape, and sample records |
| Missing value handling | Detected and imputed NaN values (median for numeric, mode for categorical) |
| Duplicate detection | Identified and removed duplicate rows |
| Outlier identification | Applied IQR-based outlier detection (reported, not removed — tree models are robust) |
| Target encoding | Mapped Placement (Yes/No) → (1/0) with robust string matching |
| ID column removal | Dropped College_ID as a non-predictive identifier |
| Feature encoding | Integrated OneHotEncoder for categorical columns via ColumnTransformer |
| Feature scaling | Applied StandardScaler for numerical columns (prevents LR bias toward large-scale features) |
| Train/test split | Stratified 80/20 split (ensures class balance in both sets) |
| Pipeline design | Built sklearn ColumnTransformer + Pipeline to prevent data leakage |

**Key Design Decision**: All transformers are fitted on *training data only*, then applied to the test set. This prevents any information from the test set leaking into the model, which would artificially inflate evaluation metrics.

---

### 2. Uttkarsh — EDA, Visualisations & Feature Engineering
**Primary Module**: `src/feature_engineering.py`

| Task | Description |
|------|-------------|
| Class distribution plot | Bar chart showing Placed vs Not-Placed counts |
| Feature distribution plots | Histograms with class-conditional overlay (12 features) |
| Box plots | IQR-based outlier visualisation grouped by placement class |
| Correlation heatmap | Pearson correlation matrix for all numerical features |
| Placement-by-feature charts | Placement rate quartile charts for IQ, CGPA, communication, etc. |
| Academic score derivation | `academic_score = 0.6 × CGPA_norm + 0.4 × Prev_Sem_norm` |
| Skills index derivation | `skills_index = mean(comm_norm, extra_norm, projects_norm)` |
| Feature justification | Documented rationale and limitations for each derived feature |
| Plot management | All plots saved to `outputs/plots/` for reproducibility |

**Key Design Decision**: Derived features use normalised components (0–1 range) before combining to prevent scale bias. Both features include explicit limitation statements: they collapse information and their predictive value must be verified empirically.

---

### 3. Vivek Ramen — Logistic Regression & Decision Tree
**Primary Module**: `src/train_models.py` (LR + DT sections)

| Task | Description |
|------|-------------|
| LR baseline training | L2 regularization, C=1, lbfgs solver |
| L2 regularization sweep | Trained LR for C ∈ {0.001, 0.01, 0.1, 1, 10, 100} with L2 penalty |
| L1 regularization sweep | Trained LR for the same C values with L1 penalty (saga solver) |
| Regularization plot | L1 vs L2 accuracy-vs-C curve |
| LR GridSearchCV | Tuned penalty (L1/L2) and C simultaneously using 5-fold CV, F1 scoring |
| DT baseline training | Unconstrained Decision Tree |
| DT depth sweep | Trained DT for max_depth ∈ {2, 3, 5, 7, 10, None} |
| DT GridSearchCV | Tuned max_depth, min_samples_split, min_samples_leaf, criterion |
| Model documentation | Explained working principles, regularization theory, complexity controls |

**Key Theoretical Contributions**:
- L1 (Lasso): Encourages sparse coefficients; some become exactly zero → implicit feature selection.
- L2 (Ridge): Shrinks all coefficients proportionally; no coefficient goes exactly to zero.
- C = 1/λ: Smaller C = stronger regularization = simpler model.
- Decision Tree overfitting: An unconstrained tree can memorise training data. `max_depth` and `min_samples_leaf` prevent this.

---

### 4. Saket — Random Forest, Ensemble Learning & Hyperparameter Tuning
**Primary Module**: `src/train_models.py` (RF section)

| Task | Description |
|------|-------------|
| RF baseline training | 100 trees, default parameters |
| Feature importance plot | Horizontal bar chart of Mean Decrease in Impurity |
| RF GridSearchCV | Tuned n_estimators, max_depth, max_features, min_samples_leaf |
| GridSearch results export | Saved full CV results to `rf_gridsearch_results.csv` |
| Tuning summary | Combined tuning summary for all 3 models |
| Ensemble theory documentation | Bagging, bootstrap sampling, random feature selection, majority vote |

**Key Theoretical Contributions**:
- **Bagging**: Each tree trained on a bootstrap sample (random subset with replacement) → reduces variance.
- **Random feature selection**: At each split, only √p features are considered → de-correlates trees → more diverse ensemble.
- **Majority vote**: Class with most votes across all trees is the final prediction.
- **Why RF > single DT**: Averaging many de-correlated trees reduces variance while bias remains similar.
- **Feature importance caveat**: MDI importance does NOT imply causality. High importance may reflect correlation with another causal feature.

---

### 5. Yash Bhargava — Evaluation, Comparison, Integration & Documentation
**Primary Modules**: `src/evaluate_models.py`, `app.py`, `docs/`

| Task | Description |
|------|-------------|
| Metric computation | Accuracy, Precision, Recall, F1-Score, ROC-AUC for all 6 model variants |
| Confusion matrix plots | Grid of confusion matrices for all models |
| ROC curve overlay | ROC curves for all models with AUC annotations |
| Model comparison chart | Grouped bar chart of all metrics × all models |
| model_comparison.csv | Test-set metric table saved to outputs/ |
| Final model selection | Best model selected by F1-score after CV tuning |
| Streamlit app | 4-page web application: Prediction, Performance, EDA, About |
| App integration | Ensures consistent preprocessing between training and inference |
| Project integration | run_pipeline.py orchestrates all modules in order |
| Documentation | README.md, methodology.md, team_contributions.md, viva_questions.md |

**Key Design Decision**: The test set is used exactly *once* — in evaluate_models.py — only after all hyperparameter tuning decisions have been made using cross-validation on the training data. This prevents test-set leakage and ensures unbiased performance estimates.

---

## Viva Responsibility Matrix

| Topic | Primary Explainer | All Members Should Know |
|-------|-------------------|------------------------|
| Dataset and features | Sachin | Yes |
| Missing value strategy | Sachin | Yes |
| EDA plots interpretation | Uttkarsh | Yes |
| Feature engineering rationale | Uttkarsh | Yes |
| Logistic Regression math | Vivek Ramen | Basic concept |
| L1 vs L2 regularization | Vivek Ramen | Yes |
| Decision Tree splits | Vivek Ramen | Yes |
| Random Forest mechanics | Saket | Yes |
| GridSearchCV workflow | Saket | Yes |
| Ensemble learning theory | Saket | Yes |
| F1-score vs Accuracy | Yash Bhargava | Yes |
| Confusion matrix FP/FN | Yash Bhargava | Yes |
| ROC-AUC interpretation | Yash Bhargava | Yes |
| Overall pipeline flow | Yash Bhargava | Yes |
