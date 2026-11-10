# Viva Questions & Answers
## Student Placement Prediction System

> Organised by team member responsibility. Every member should be able to answer ALL questions about the overall project — specialisation is for depth, not exclusion.

---

## 📌 SECTION A — General ML & Project Overview
*(All members must know these)*

**Q1. What is the problem you are solving?**  
A: We are predicting whether a student will be placed or not placed based on academic and employability attributes such as CGPA, IQ, internship experience, communication skills, and projects completed. This is a binary classification problem.

**Q2. What is binary classification?**  
A: Binary classification is a supervised machine learning task where the model learns to assign one of exactly two labels to each input. Here, the labels are Placed (1) and Not Placed (0).

**Q3. Why is machine learning useful for placement prediction?**  
A: Manual analysis of thousands of student records is time-consuming and subjective. ML models can identify non-obvious patterns in historical data, quantify the relative importance of different factors, and produce consistent, reproducible predictions.

**Q4. What dataset did you use?**  
A: The College Student Placement Factors Dataset created by Sahil Islam, published on Kaggle. It is a synthetic (simulated) dataset of approximately 10,000 student records. Key features include IQ, CGPA, Prev_Sem_Result, communication skills, internship experience, extra-curricular score, and projects completed. The target variable is Placement (Yes/No).

**Q5. What does "synthetic dataset" mean, and why does it matter?**  
A: Synthetic data is generated algorithmically rather than collected from real students. It is useful for demonstration and learning projects but may not capture the full complexity of real-world placement patterns. Our results should be interpreted as proof-of-concept, not real predictions.

**Q6. What is the target variable?**  
A: The Placement column, which indicates whether a student was placed (Yes/1) or not placed (No/0).

**Q7. What is supervised learning?**  
A: Supervised learning trains a model on labelled examples — pairs of (input features, known output label). The model learns a mapping from inputs to outputs by minimising prediction error on the training examples.

---

## 📌 SECTION B — Sachin's Topics: Data Preprocessing

**Q8. What is data preprocessing and why is it necessary?**  
A: Raw data often contains missing values, duplicates, inconsistent formatting, and mixed data types. Preprocessing transforms raw data into a clean, consistent format that machine learning algorithms can process correctly. Poor preprocessing leads to unreliable models.

**Q9. How did you handle missing values?**  
A: We detected missing values per column. Numerical columns were imputed with the column median (robust to outliers, unlike mean). Categorical columns were imputed with the mode (most frequent value). We chose imputation over deletion to preserve sample size.

**Q10. Why use median instead of mean for imputation?**  
A: The median is not affected by extreme values (outliers). If a column contains a few very large values, the mean is pulled upward, but the median stays representative of the centre. This makes median imputation more robust.

**Q11. What is data leakage, and how did you prevent it?**  
A: Data leakage occurs when information from outside the training set (e.g., from the test set or the target variable) is used during model training, artificially inflating evaluation metrics. We prevented it by:
- Fitting all transformers (StandardScaler, OneHotEncoder) on training data ONLY, then applying them to the test set.
- Never using the target variable as an input feature.
- Using GridSearchCV only on training data; the test set was touched only once for final evaluation.

**Q12. What is a ColumnTransformer in scikit-learn?**  
A: ColumnTransformer applies different preprocessing steps to different subsets of columns simultaneously. In our project, it applies StandardScaler to numerical columns and OneHotEncoder to categorical columns, then concatenates the results. This ensures consistent preprocessing and is reusable at inference time.

**Q13. What is stratified sampling in train/test split?**  
A: Stratified sampling ensures that the proportion of each class (Placed/Not Placed) is approximately the same in both the training and test sets as in the full dataset. Without stratification, the test set could accidentally contain mostly one class, making evaluation unreliable.

**Q14. Why did you drop the College_ID column?**  
A: College_ID is an identifier, not a predictive feature. Including it would cause the model to learn spurious correlations between arbitrary IDs and placement outcomes. Removing irrelevant identifier columns is standard practice.

**Q15. What is OneHotEncoding?**  
A: OneHotEncoding converts categorical variables into binary (0/1) indicator columns, one per unique category. This allows models that require numerical input (like Logistic Regression) to process categorical features without imposing an artificial ordinal relationship.

**Q16. Why is StandardScaler important for Logistic Regression but not Decision Trees?**  
A: Logistic Regression uses gradient-based optimisation that is sensitive to the scale of features. A feature with values in the thousands dominates features with values in the range 0–1. StandardScaler normalises all features to zero mean and unit variance, making optimisation stable. Decision Trees split on feature thresholds and are inherently scale-invariant, so scaling doesn't affect them.

**Q17. What is the IQR method for outlier detection?**  
A: IQR = Q3 - Q1 (the interquartile range). Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are flagged as outliers. We reported outliers but did not remove them because tree-based models are robust to outliers, and Logistic Regression handles them via StandardScaler.

---

## 📌 SECTION C — Uttkarsh's Topics: EDA & Feature Engineering

**Q18. What is Exploratory Data Analysis?**  
A: EDA is the process of examining a dataset to understand its structure, distributions, relationships, and anomalies before building models. It uses summary statistics and visualisations to reveal patterns that inform modelling decisions.

**Q19. What plots did you create and what did each show?**  
A: 
1. **Class distribution bar chart** — shows balance/imbalance of Placed vs Not Placed.
2. **Feature distribution histograms** — shows the shape of each feature distribution, split by class, to see if the feature discriminates between classes.
3. **Box plots** — visualises IQR and outliers for each feature, grouped by placement class.
4. **Correlation heatmap** — shows Pearson correlations between all numerical features and the target, revealing linear relationships and multicollinearity.
5. **Placement-rate-by-feature charts** — shows placement rate across quartiles or categories of key features.

**Q20. What is a correlation heatmap?**  
A: It is a colour-coded matrix where each cell shows the Pearson correlation coefficient between two variables. Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation). We use it to identify which features correlate with the target (good predictors) and which features correlate with each other (multicollinearity risk for LR).

**Q21. Does correlation imply causation?**  
A: No. Correlation only means two variables move together statistically. Causation requires experimental evidence (randomised controlled trials). For example, CGPA and placement may correlate, but a third factor (student motivation) could cause both.

**Q22. What is the academic_score feature you derived?**  
A: `academic_score = 0.6 × CGPA_normalised + 0.4 × Prev_Sem_normalised`. It combines cumulative academic performance (CGPA) with recent momentum (previous semester result). We weight CGPA more (0.6) because it reflects a longer track record. Both inputs are normalised to [0,1] first so that different original scales don't distort the blend. Limitation: the 0.6/0.4 weights are a heuristic, not derived from data.

**Q23. What is the skills_index feature?**  
A: `skills_index = mean(communication_skills_norm, extra_curricular_score_norm, projects_completed_norm)`. It aggregates three complementary employability signals into one index. Normalising each component before averaging prevents scale dominance. Limitation: equal weighting may not reflect their true relative importance.

**Q24. Why might a derived feature not improve model performance?**  
A: Derived features can be redundant (they add no new information), noisy (the transformation introduces error), or poorly weighted (arbitrary formulas). The model may already extract the signal from the original features. Feature usefulness must be verified empirically, not assumed.

**Q25. What does class imbalance mean, and why does it matter?**  
A: Class imbalance means one class has many more samples than the other. In an extreme case (e.g., 90% placed, 10% not placed), a model that always predicts "Placed" achieves 90% accuracy without learning anything. F1-score accounts for this better because it penalises models that ignore the minority class.

---

## 📌 SECTION D — Vivek Ramen's Topics: Logistic Regression & Decision Tree

**Q26. How does Logistic Regression work?**  
A: Logistic Regression models the probability of the positive class as:
`P(y=1|x) = σ(w·x + b)`, where σ is the sigmoid function: `σ(z) = 1 / (1 + e^(-z))`.
The model learns weights `w` by minimising binary cross-entropy (log-loss) using gradient descent (or lbfgs). The decision boundary is `P = 0.5`, equivalent to `w·x + b = 0`.

**Q27. What is binary cross-entropy loss?**  
A: It measures how far the model's predicted probabilities are from the true labels:
`Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`.
When the true label is 1 and the model predicts a probability close to 1, the loss is low. Incorrect or uncertain predictions are penalised strongly.

**Q28. What is L2 regularization, and when should you use it?**  
A: L2 (Ridge) adds the term `λ·Σwᵢ²` to the loss function. This penalises large weights, shrinking all coefficients toward zero uniformly. The result is a model that generalises better because no single feature dominates. L2 is the default choice when you want to keep all features but prevent overfitting.

**Q29. What is L1 regularization, and when should you use it?**  
A: L1 (Lasso) adds `λ·Σ|wᵢ|` to the loss. Unlike L2, L1 can drive some coefficients exactly to zero, effectively removing those features from the model. This makes L1 useful for automatic feature selection when you suspect many features are irrelevant.

**Q30. What is the hyperparameter C in Logistic Regression?**  
A: `C = 1/λ`, where λ is the regularization strength. A smaller C means stronger regularization (simpler model, possible underfitting). A larger C means weaker regularization (complex model, possible overfitting). We select C using GridSearchCV to find the value that generalises best.

**Q31. What solver did you use for L1 regularization, and why?**  
A: The `saga` solver, because it supports L1 penalty. The default `lbfgs` solver only supports L2. `saga` is a stochastic gradient descent variant that converges well for large datasets.

**Q32. How does a Decision Tree make predictions?**  
A: A Decision Tree recursively partitions the feature space by selecting the feature and threshold that maximises information gain (or minimises Gini impurity) at each node. New data points traverse the tree from root to leaf, where the majority class becomes the prediction.

**Q33. What is Gini impurity?**  
A: Gini impurity measures the probability of incorrectly classifying a randomly chosen element: `Gini = 1 - Σpᵢ²`. A pure node (all one class) has Gini = 0. A maximally impure node (equal classes) has Gini = 0.5 for binary classification.

**Q34. What is information gain (entropy)?**  
A: Entropy = `-Σpᵢ·log₂(pᵢ)`. Information gain is the reduction in entropy from a split. The tree selects splits that maximise information gain — meaning the split creates the most "pure" child nodes.

**Q35. What is overfitting in a Decision Tree?**  
A: An unconstrained Decision Tree can grow until every leaf contains a single training sample, achieving 100% training accuracy but failing on unseen data. This is overfitting — the model memorised the training data rather than learning generalisable patterns.

**Q36. How did you control Decision Tree complexity?**  
A: We used:
- `max_depth`: limits the maximum tree depth. Shallow trees are less likely to overfit.
- `min_samples_leaf`: requires at least N samples per leaf. Prevents trees from creating tiny, specialised leaves.
- `min_samples_split`: minimum samples needed before a split is attempted.
These were tuned using GridSearchCV.

---

## 📌 SECTION E — Saket's Topics: Random Forest & Hyperparameter Tuning

**Q37. What is ensemble learning?**  
A: Ensemble learning combines multiple models (weak learners) to produce a stronger, more stable predictor. The core idea is that averaging the errors of many independent models reduces overall error — even if individual models are imperfect.

**Q38. How does Random Forest work?**  
A: Random Forest uses two sources of randomness:
1. **Bagging (Bootstrap AGGregating)**: Each tree is trained on a random bootstrap sample of the training data (sampling with replacement). Different trees see slightly different training subsets.
2. **Random feature selection**: At each split, only a random subset of features (typically √p for classification) is considered. This de-correlates the trees.
Final predictions are made by majority vote across all trees.

**Q39. Why is Random Forest better than a single Decision Tree?**  
A: A single Decision Tree has high variance — small changes in training data cause very different trees. By averaging many de-correlated trees, variance is reduced without increasing bias significantly. This is the bias-variance trade-off in action.

**Q40. What is the bias-variance trade-off?**  
A: 
- **Bias**: Error from wrong assumptions in the model (e.g., assuming linear boundary when it's non-linear). High bias → underfitting.
- **Variance**: Error from sensitivity to small training data changes. High variance → overfitting.
- The trade-off: simple models have high bias, low variance; complex models have low bias, high variance. The goal is to find the sweet spot. Ensembles like Random Forest reduce variance without significantly increasing bias.

**Q41. What does n_estimators control in Random Forest?**  
A: `n_estimators` is the number of trees in the forest. More trees → more stable predictions, but slower training and inference. Performance generally improves with more trees up to a point of diminishing returns (often 100–500 trees).

**Q42. What is GridSearchCV?**  
A: GridSearchCV exhaustively searches over a specified grid of hyperparameter values. For each combination, it uses k-fold cross-validation on the training data to estimate performance. The combination with the best CV score is selected as the optimal set of hyperparameters.

**Q43. What is k-fold cross-validation?**  
A: The training data is split into k equal folds. The model is trained on k-1 folds and validated on the remaining 1 fold. This is repeated k times (once per fold). The final CV score is the mean performance across all k validation sets. We use k=5. This gives a more reliable estimate of generalisation than a single validation split.

**Q44. Why use F1-score as the GridSearchCV scoring metric?**  
A: F1-score is the harmonic mean of Precision and Recall. It penalises models that ignore one metric at the expense of the other. When classes might be imbalanced (more placed than not-placed or vice versa), F1-score is more informative than accuracy. We use F1 to select hyperparameters that balance both types of errors.

**Q45. Did you use the test set during GridSearchCV?**  
A: No. GridSearchCV was applied only to the training data using cross-validation. The test set was kept completely separate and used only once — in evaluate_models.py — for final performance reporting. Using the test set for tuning would constitute test-set leakage and produce optimistic, unreliable metrics.

**Q46. What is feature importance in Random Forest?**  
A: Feature importance (Mean Decrease in Impurity, MDI) measures how much each feature reduces Gini impurity across all splits and all trees on average. Features used in higher-level splits that affect more samples get higher importance scores. Caveat: MDI can be biased toward high-cardinality features and does not imply causality.

**Q47. What are the limitations of Random Forest?**  
A: 
- Less interpretable than a single tree (black-box ensemble).
- Higher computational cost (O(n_trees × N × log N × √p)).
- More memory usage than a single model.
- Feature importances don't capture interaction effects well.
- Slow for real-time inference when n_trees is very large.

---

## 📌 SECTION F — Yash Bhargava's Topics: Evaluation & Metrics

**Q48. What is accuracy, and when is it insufficient?**  
A: Accuracy = (TP + TN) / (TP + TN + FP + FN). It measures the overall fraction of correct predictions. It is insufficient when classes are imbalanced. For example, if 80% of students are placed, always predicting "Placed" gives 80% accuracy — without any learning.

**Q49. What is Precision?**  
A: Precision = TP / (TP + FP). Of all students the model predicted as Placed, what fraction were actually placed? High precision means few false positives — the model rarely cries "placed" incorrectly. Relevant when false positives are costly.

**Q50. What is Recall (Sensitivity)?**  
A: Recall = TP / (TP + FN). Of all students who were actually placed, what fraction did the model correctly identify? High recall means few false negatives. Relevant when missing a positive case is costly.

**Q51. What is the F1-score?**  
A: F1 = 2 × (Precision × Recall) / (Precision + Recall). It is the harmonic mean of Precision and Recall. Unlike arithmetic mean, harmonic mean punishes models that are good on one metric but very poor on the other. F1 = 1 is perfect; F1 = 0 is worst.

**Q52. What is a confusion matrix?**  
A: A 2×2 table showing:
- **TP** (True Positive): correctly predicted Placed
- **TN** (True Negative): correctly predicted Not Placed
- **FP** (False Positive): predicted Placed, but actually Not Placed
- **FN** (False Negative): predicted Not Placed, but actually Placed

**Q53. In the context of placement prediction, which error is more costly — FP or FN?**  
A: It depends on the use case:
- **FP (model says Placed, actually Not Placed)**: Student skips training/preparation. Risk: student is unprepared.
- **FN (model says Not Placed, actually Placed)**: Student receives unnecessary counselling/intervention. Risk: wasted resources, but less harmful.
In academic counselling, FN is generally less harmful. However, in scholarship allocation, FP could unfairly deny funding. The model should be transparent about its error rate.

**Q54. What is ROC-AUC?**  
A: The ROC (Receiver Operating Characteristic) curve plots True Positive Rate (Recall) against False Positive Rate at every possible classification threshold. AUC (Area Under the Curve) is a single number summarising the curve:
- AUC = 0.5: no better than random guessing.
- AUC = 1.0: perfect separation.
- AUC > 0.7: generally considered acceptable.
ROC-AUC is useful for comparing models and is threshold-independent.

**Q55. Which model did you select as the final model, and why?**  
A: [After running the pipeline] The model with the highest F1-score on the held-out test set. F1-score was chosen as the primary selection criterion because it balances Precision and Recall and is robust to class imbalance. The specific model and its metrics are reported in `outputs/model_comparison.csv`. *(Fill in actual values after running the pipeline.)*

**Q56. How does the Streamlit app make predictions?**  
A: The app loads the saved `final_model.pkl` file, which contains the complete sklearn Pipeline (preprocessor + model). When the user inputs student attributes and clicks Predict, the app constructs a one-row DataFrame with the same column names used during training, passes it through the pipeline (which applies scaling/encoding automatically), and displays the predicted class and probability.

**Q57. Why is it important that the app uses the same preprocessing as training?**  
A: If the app applied different scaling, encoding, or feature names than what was used during training, the input data would be in a different format than what the model expects, leading to incorrect predictions. Using the saved Pipeline object guarantees identical preprocessing at inference time.

---

## 📌 SECTION G — Overfitting, Underfitting & Model Selection

**Q58. What is overfitting?**  
A: Overfitting occurs when a model performs very well on training data but poorly on unseen data. The model has memorised training-specific noise and patterns rather than learning generalisable rules. Signs: training accuracy >> test accuracy.

**Q59. What is underfitting?**  
A: Underfitting occurs when a model is too simple to capture the underlying patterns in the data. Signs: both training and test accuracy are low.

**Q60. How do you detect overfitting in your experiments?**  
A: Compare training accuracy with test accuracy. A large gap (e.g., training accuracy = 98%, test accuracy = 72%) indicates overfitting. We addressed overfitting via regularization (Logistic Regression) and depth constraints + GridSearchCV (Decision Tree, Random Forest).

**Q61. What is the purpose of the validation set vs the test set?**  
A: 
- **Validation set** (created by cross-validation inside GridSearchCV): Used to tune hyperparameters during model selection. The model "sees" validation data indirectly via CV scores.
- **Test set**: Held completely separate. Used only once at the end to estimate real-world performance. Provides an unbiased estimate because the model never saw this data during training or tuning.

**Q62. Can you guarantee that the selected model will perform exactly as measured on the test set in production?**  
A: No. Test-set performance is an estimate. In production, performance may differ because:
- The new data distribution may differ from the training data (distribution shift).
- The synthetic dataset may not represent real students.
- Features important in reality (interview skills, company preferences) are absent from the dataset.

---

## 📌 SECTION H — Practical Code Questions

**Q63. If a new student's data has a feature not seen during training, what happens?**  
A: The OneHotEncoder in the Pipeline uses `handle_unknown='ignore'`, which replaces unseen categories with a zero vector. StandardScaler can't handle new features directly — the input must have the same columns as training. The app validates input format before prediction.

**Q64. What is joblib.dump / joblib.load?**  
A: joblib is a serialisation library optimised for large numpy arrays. `joblib.dump(pipeline, 'model.pkl')` saves the entire trained Pipeline (preprocessor + model) to disk. `joblib.load('model.pkl')` restores it. This allows us to use the trained model in the Streamlit app without retraining.

**Q65. Why use sklearn Pipelines rather than transforming data manually?**  
A: sklearn Pipelines chain preprocessing steps with the model into a single object. Benefits:
- Prevents leakage: the pipeline's fit() calls only transform training data, and transform() applies the fitted transformers to test data.
- Reproducibility: the entire workflow is one object.
- Deployment: the saved pipeline includes preprocessing — inference requires no additional code.

**Q66. What would happen if you used n_jobs=-1 in GridSearchCV on a machine with limited RAM?**  
A: n_jobs=-1 uses all CPU cores. For large parameter grids, this can cause memory exhaustion because each parallel job holds a copy of the training data and model. On a student laptop, reducing n_jobs to 2 or 4 can prevent crashes at the cost of longer runtime.

**Q67. Is the Kaggle dataset real or synthetic, and does that affect the project?**  
A: The dataset is synthetic (simulated). This means:
- We can use it freely without privacy concerns.
- The patterns in the data are generated by an algorithm, so they may be "too clean" compared to real data.
- Our model's performance on this dataset may be higher than on real placement data.
- The project demonstrates the ML workflow correctly regardless of data origin.

---

## 📌 SECTION I — Deep Learning: MLP Neural Network
*(All members should know the basics; explain in context of the project)*

**Q68. What is a Neural Network (Multi-Layer Perceptron)?**  
A: An MLP is a stack of fully connected layers of neurons. Data flows from the input layer through one or more hidden layers to an output layer. Each neuron computes `z = W·x + b` (weighted sum plus bias) and then applies a non-linear activation function `a = f(z)`. The network learns by adjusting weights to minimise a loss function using backpropagation and gradient descent.

**Q69. What is a neuron and what does it compute?**  
A: A neuron receives multiple inputs, multiplies each by a weight, sums them with a bias, and passes the result through an activation function:
```
output = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```
The activation function introduces non-linearity, allowing the network to model complex patterns that linear models like Logistic Regression cannot.

**Q70. What activation functions did you use, and why?**  
A:
- **ReLU (Rectified Linear Unit)** for hidden layers: `f(z) = max(0, z)`. Fast to compute, avoids the vanishing gradient problem (gradients don't shrink to zero for positive inputs), and produces sparse activations.
- **Logistic (Sigmoid)** for the output layer: `f(z) = 1/(1+e⁻ᶻ)`. Maps the output to [0,1], giving a probability of placement. The decision boundary is at 0.5.

**Q71. What is backpropagation?**  
A: Backpropagation is the algorithm used to compute the gradient of the loss function with respect to every weight in the network using the chain rule of calculus. Starting from the output layer and working backward:
1. Compute the error (loss) at the output.
2. Propagate the gradient backward through each layer.
3. Each weight's gradient tells us how much changing that weight would reduce the loss.
4. The optimiser (Adam) uses these gradients to update the weights.

**Q72. What is the Adam optimiser?**  
A: Adam (Adaptive Moment Estimation) is a gradient descent optimiser that maintains:
- **First moment** (momentum): exponential moving average of past gradients.
- **Second moment** (adaptive learning rate): exponential moving average of squared gradients.
This allows Adam to adapt the learning rate per-parameter and handle noisy gradients efficiently. It converges faster than basic stochastic gradient descent.

**Q73. What is a loss function, and which one did you use for the MLP?**  
A: The loss function measures how wrong the model's predictions are. For binary classification, we use **binary cross-entropy**:
```
Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
Where `y` is the true label (0 or 1) and `ŷ` is the predicted probability. Low loss = accurate predictions. Backpropagation minimises this loss.

**Q74. What is early stopping, and why did you use it?**  
A: Early stopping monitors validation loss (or accuracy) during training. If the validation metric does not improve for `n_iter_no_change=15` consecutive iterations, training stops. This:
- Prevents overfitting (model doesn't keep memorising training data beyond the optimal point).
- Reduces unnecessary computation.
- Acts as an implicit regularization technique.

**Q75. What does the training loss curve tell you?**  
A: The loss curve plots cross-entropy loss against training iterations (epochs):
- **Steadily decreasing** = model is learning.
- **Flattening** at the end = convergence (early stopping fired).
- **Validation loss diverging** from training loss = overfitting (model memorising training data).
- **Loss not decreasing at all** = learning rate too low, or model too simple.

**Q76. What is the alpha parameter in sklearn's MLPClassifier?**  
A: `alpha` controls L2 regularization strength — exactly analogous to `1/C` in Logistic Regression. It adds `α·‖W‖²` to the loss function, penalising large weights:
- **Small alpha** (e.g., 1e-5): weak regularization → complex model → possible overfitting.
- **Large alpha** (e.g., 1.0): strong regularization → simpler model → possible underfitting.
We tuned alpha via GridSearchCV on the training data.

**Q77. Why does the MLP require feature scaling but a Decision Tree does not?**  
A: MLP uses gradient descent to optimise weights. If features have very different scales (e.g., IQ ≈ 100, CGPA ≈ 8), the gradient landscape is elongated and convergence is slow or unstable. StandardScaler normalises all features to zero mean and unit variance, making the gradient landscape more spherical and convergence faster. Decision Trees split on thresholds and are unaffected by scale — the relative ordering of values is what matters, not magnitude.

**Q78. What architectures did you compare and which worked best?**  
A: We compared:
- **Shallow (64,)**: 1 hidden layer — fast, simple, lowest capacity.
- **Standard (128, 64)**: 2 hidden layers — our recommended choice for tabular data.
- **Deep (256, 128, 64)**: 3 hidden layers — most expressive, highest overfitting risk.

For a ~10K synthetic tabular dataset, (128, 64) typically strikes the best bias-variance trade-off. Deeper networks tend to overfit on small/synthetic datasets.

**Q79. What are the advantages of MLP over Logistic Regression?**  
A:
- MLP can learn **non-linear decision boundaries**; LR can only model linear ones.
- MLP automatically learns **feature interactions** (cross-effects between variables).
- With enough data, MLP can be more expressive and achieve better accuracy.

**Q80. What are the disadvantages of MLP compared to Random Forest?**  
A:
- **Less interpretable**: MLP is a black box — we can't easily explain why it made a specific prediction. RF provides feature importance.
- **Needs scaling**: MLP requires StandardScaler; RF is scale-invariant.
- **More hyperparameters**: architecture, alpha, learning rate, batch size, etc.
- **Slower to train**: especially for many layers and large grids.
- **Overfitting risk**: higher on small/synthetic datasets without careful regularization.

**Q81. Why did you use sklearn's MLPClassifier instead of TensorFlow/Keras?**  
A: TensorFlow and Keras require Python ≤ 3.11 at the time of this project. Our environment uses Python 3.14 (the latest version), which is not yet supported by TensorFlow. sklearn's MLPClassifier implements the same MLP mathematics (backpropagation, Adam, ReLU, cross-entropy) and integrates seamlessly with the existing sklearn Pipeline infrastructure — the same preprocessing, GridSearchCV, and evaluation code works unchanged.

**Q82. Is MLP a "deep learning" model?**  
A: Yes, technically. Any neural network with more than one hidden layer qualifies as a deep learning model. Our (128, 64) architecture has 2 hidden layers. In industrial practice, "deep learning" usually refers to much deeper networks (10+ layers) applied to images or text. For tabular placement prediction, 2–3 layers is appropriate and is still correctly classified as a deep learning approach. Mentioning this distinction demonstrates understanding during viva.

