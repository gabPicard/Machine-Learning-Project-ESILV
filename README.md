Credit Default Prediction — Machine Learning Project

This project develops a supervised learning pipeline to predict credit default risk using demographic information and longitudinal credit history. The objective is to build a reliable and interpretable model capable of identifying clients with past delinquency.

1. Data and Target Construction

Two datasets are used:

Application data: demographic, socio-economic, and financial variables (income, employment duration, education, household structure, etc.).

Credit history: monthly credit status for each client.

The binary target variable is defined as:

Default = 1 if the client has any STATUS in {1, 2, 3, 4, 5}.
Default = 0 otherwise.

Credit histories are aggregated into client-level features (number of late payments, maximum delinquency, history length, etc.) before merging with the application data.

2. Preprocessing Pipeline

Missing values: median for numerical features, mode for categorical features

Encoding: one-hot encoding for nominal variables, label encoding for binary variables

Scaling: applied to models sensitive to feature magnitude

Class imbalance: compared under baseline, undersampling, SMOTE, and hybrid strategies

3. Modelling and Evaluation

Models trained:

Logistic Regression

Random Forest

Gradient Boosting / XGBoost

Support Vector Machine

K-Nearest Neighbors

Evaluation metrics:

F1-score (primary)

Precision and Recall

ROC-AUC

Confusion Matrix

Tree-based models generally achieve the most stable performance, particularly when combined with sampling techniques.

4. Model Interpretation

Interpretability is based on:

Permutation feature importance

SHAP values

Key predictors include delinquency severity, employment duration, and income.

5. Conclusion

This project delivers a complete and reproducible credit-scoring pipeline covering data cleaning, feature engineering, class imbalance management, model comparison, and interpretability. The methodology aligns with industry standards in credit risk modelling and demonstrates the applicability of machine learning to default prediction