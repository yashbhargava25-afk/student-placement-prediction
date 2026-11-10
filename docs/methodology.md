# Project Methodology
## Student Placement Prediction System

---

## Abstract

This project develops a machine learning system for predicting student campus placement outcomes using academic and employability attributes. Three classification algorithms — Logistic Regression, Decision Tree, and Random Forest — are trained and evaluated on the College Student Placement Factors Dataset (Sahil Islam, Kaggle, ~10,000 synthetic records). The methodology encompasses data cleaning, exploratory data analysis, feature engineering, regularization, hyperparameter tuning with cross-validation, ensemble learning, and comparative performance analysis across multiple evaluation metrics.

---

## 1. Introduction

Campus placement is a critical outcome for both students and educational institutions. Identifying students at risk of not being placed enables institutions to provide targeted training and counselling before recruitment season. This project applies supervised machine learning to predict placement status from a student's academic and employability profile.

---

## 2. Problem Statement

Manual analysis of student records cannot scale to thousands of students or capture non-linear relationships between features and placement outcomes. This project formulates placement prediction as a binary classification problem and builds interpretable, evaluated ML models to support data-driven counselling decisions.

---

## 3. Objectives

1. Analyse the dataset to identify academic and employability factors associated with placement.
2. Build and compare Logistic Regression, Decision Tree, and Random Forest classifiers.
3. Apply L1/L2 regularization to reduce overfitting in Logistic Regression.
4. Use GridSearchCV with cross-validation to tune all models without test-set leakage.
5. Evaluate models using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.
6. Select the best model and deploy it in a simple Streamlit web application.

---

## 4. Dataset Description

| Attribute | Value |
|-----------|-------|
| Source | Kaggle — Sahil Islam |
| Type | Synthetic (simulated) |
| Records | ~10,000 |
| Target | Placement (Yes=1, No=0) |
| Key features | IQ, CGPA, Prev_Sem_Result, Academic_Performance, Internship_Experience, Extra_Curricular_Score, Communication_Skills, Projects_Completed |
| Dropped | College_ID (non-predictive identifier) |

**Dataset URL**: https://www.kaggle.com/datasets/sahilislam007/college-student-placement-factors-dataset

---

## 5. Exploratory Data Analysis

EDA was conducted before any model training to understand data structure and feature-target relationships:

- **Class distribution**: Verified the balance of Placed vs Not Placed records.
- **Feature distributions**: Histogram plots with class-conditional overlay revealed which features separate the two classes.
- **Box plots**: Identified potential outliers; tree-based models are robust to them.
- **Correlation heatmap**: Revealed linear relationships between features and the target, and potential multicollinearity.
- **Placement-rate plots**: Quartile-grouped charts showed how placement rate varies across IQ, CGPA, communication skills, and internship experience.

All plots are saved to `outputs/plots/` for reproducibility.

---

## 6. Feature Engineering

Two derived features were created:

### academic_score
```
academic_score = 0.6 × CGPA_norm + 0.4 × Prev_Sem_Result_norm
```
Combines cumulative performance (CGPA) and recent momentum (Prev_Sem_Result). Both inputs normalised to [0,1] before blending. Created only when both source columns exist.

**Limitation**: The 0.6/0.4 weighting is a heuristic; actual optimal weights may differ.

### skills_index
```
skills_index = mean(comm_norm, extra_norm, projects_norm)
```
Aggregates communication skills, extra-curricular score, and projects completed into a composite employability measure.

**Limitation**: Equal weighting assumes all three components contribute equally to employability.

### Other Transformations
- **OneHotEncoding**: Categorical features encoded as binary indicator columns.
- **StandardScaler**: Numerical features normalised to zero mean and unit variance (required by Logistic Regression).
- **College_ID dropped**: Identifier with no predictive value.

All transformations are applied within an sklearn Pipeline fitted only on training data to prevent leakage.

---

## 7. Model Training

### 7.1 Logistic Regression
*Vivek Ramen's contribution*

Logistic Regression models the probability of placement as `P(y=1|x) = σ(w·x + b)` using the sigmoid function. Binary cross-entropy is minimised via gradient descent.

**L2 Regularization**: Adds `λ·‖w‖²` to the loss — shrinks all weights uniformly.  
**L1 Regularization**: Adds `λ·‖w‖₁` — can drive irrelevant feature weights to exactly zero (automatic feature selection).  
**Hyperparameter C**: Inverse of regularization strength (C = 1/λ).

Solvers: `lbfgs` for L2; `saga` for L1.

### 7.2 Decision Tree
*Vivek Ramen's contribution*

Recursively partitions the feature space using Gini impurity or information gain. Complexity controlled via:
- `max_depth`: caps tree depth
- `min_samples_leaf`: requires minimum samples per leaf
- `min_samples_split`: minimum samples before attempting a split

Limitation: High variance — small data changes produce very different trees.

### 7.3 Random Forest (Ensemble)
*Saket's contribution*

Random Forest combines N trees via:
1. **Bagging**: Each tree trained on a bootstrap sample.
2. **Random feature selection**: √p features considered at each split.
3. **Majority vote**: Final class predicted by the most trees.

Why it outperforms a single tree: Averaging de-correlated trees reduces variance without significantly increasing bias.

---

## 8. Hyperparameter Tuning

**Method**: GridSearchCV with 5-fold stratified cross-validation.  
**Scoring**: F1-score (balances Precision and Recall; robust to imbalance).  
**Data scope**: Training data only — test set not used during tuning.

| Model | Tuned Parameters |
|-------|-----------------|
| Logistic Regression | C, penalty (L1/L2), solver |
| Decision Tree | max_depth, min_samples_split, min_samples_leaf, criterion |
| Random Forest | n_estimators, max_depth, max_features, min_samples_leaf |

Best parameters saved to `outputs/tuning_summary.csv`.

---

## 9. Performance Evaluation

*Yash Bhargava's contribution*

**Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
**Test set used exactly once** — only after all tuning decisions were finalised.

**Key metric choice**: F1-score is the primary selection criterion because it penalises models that sacrifice either Precision or Recall, making it appropriate for potentially imbalanced placement datasets.

**Generated artefacts**:
- Confusion matrices for all models
- ROC curve overlay
- Model comparison bar chart
- `outputs/model_comparison.csv`

---

## 10. Experimental Results

> **Note**: Actual metric values are generated by running `src/run_pipeline.py`. 
> The table below shows the structure; fill in values after execution.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| LR Baseline | [run to get] | — | — | — | — |
| LR Tuned | — | — | — | — | — |
| DT Baseline | — | — | — | — | — |
| DT Tuned | — | — | — | — | — |
| RF Baseline | — | — | — | — | — |
| RF Tuned | — | — | — | — | — |

---

## 11. Limitations

1. **Synthetic data**: Results may not generalise to real placement outcomes.
2. **Missing features**: Interview performance, company preferences, and economic conditions are absent.
3. **Heuristic feature weights**: academic_score and skills_index use non-data-derived weights.
4. **Correlation ≠ causation**: Feature importance plots show model reliance, not causal effects.
5. **Threshold fixed at 0.5**: Default threshold may not be optimal for all use cases.

---

## 12. Conclusion

This project demonstrates a complete supervised ML pipeline for binary classification, covering all required academic components: EDA, feature engineering, classification, regularization, hyperparameter tuning, ensemble learning, and comparative evaluation. The Streamlit app makes predictions accessible without ML expertise, while clearly communicating prediction uncertainty.

---

## 13. Future Scope

- Use real placement data from institutional records (with appropriate privacy safeguards).
- Apply SHAP values for more interpretable feature importance.
- Experiment with gradient boosting models (XGBoost, LightGBM).
- Add SMOTE oversampling if class imbalance is significant.
- Deploy on a cloud platform (Streamlit Cloud, Heroku) for broader access.
- Build a feedback loop to retrain as new placement data becomes available.

---

## 14. References

1. Islam, S. *College Student Placement Factors Dataset*. Kaggle, 2024. https://www.kaggle.com/datasets/sahilislam007/college-student-placement-factors-dataset
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
3. Scikit-learn Developers. *User Guide*. https://scikit-learn.org/stable/user_guide.html
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
5. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *JRSS-B*, 58(1), 267–288.
