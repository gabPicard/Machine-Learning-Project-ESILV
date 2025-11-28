Credit Default Prediction - Machine Learning Project

This project develops a supervised learning pipeline to predict credit default risk using demographic information and longitudinal credit history. The objective is to build a reliable and interpretable model capable of identifying clients with past delinquency.


1. Data and Target Construction

Two datasets are used:

Application data: demographic, socio-economic, and financial variables (income, employment duration, education, household structure, etc.).

Credit history: monthly credit status for each client.

The binary target variable is defined as:

> Default = 1 if the client has any STATUS in {1, 2, 3, 4, 5}.
> Default = 0 otherwise.

Credit histories are aggregated into client-level features (number of late payments, maximum delinquency, history length, etc.) before merging with the application data.


2. Preprocessing Pipeline

* Missing values: median for numerical features, mode for categorical features
* Encoding: one-hot encoding for nominal variables, label encoding for binary variables
* Scaling: applied to models sensitive to feature magnitude
* Class imbalance: compared under baseline, undersampling, SMOTE, and hybrid strategies


3. Modelling and Evaluation

Models trained:

* Logistic Regression
* Random Forest
* Gradient Boosting / XGBoost
* Support Vector Machine
* K-Nearest Neighbors

Evaluation metrics:

* F1-score (primary)
* Precision and Recall
* ROC-AUC
* Confusion Matrix

Tree-based models generally achieve the most stable performance, particularly when combined with sampling techniques.

4. Model Interpretation

Interpretability is based on:

* Permutation feature importance
* SHAP values

Key predictors include delinquency severity, employment duration, and income.


5. Conclusion

This project delivers a complete and reproducible credit-scoring pipeline covering data cleaning, feature engineering, class imbalance management, model comparison, and interpretability. The methodology aligns with industry standards in credit risk modelling and demonstrates the applicability of machine learning to default prediction.


6. Data Description and Source

This project uses the *Home Credit Default Risk* dataset, a publicly available real-world credit-scoring benchmark commonly used in academic and industrial machine learning applications.

Link :
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

The project relies on two core tables:

application_record.csv

Contains one entry per client with demographic and socio-economic information. Examples include:

* AMT_INCOME_TOTAL — yearly income
* CNT_CHILDREN — number of dependents
* DAYS_BIRTH — age in negative days
* DAYS_EMPLOYED — employment duration
* NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS — categorical variables
* OCCUPATION_TYPE — job category (contains many missing values)

credit_record.csv

A longitudinal dataset detailing monthly credit repayment behavior for each client.
Relevant fields:

* STATUS — monthly repayment status (0, 1–5 delinquency codes, C for closed, X for unknown)
* MONTHS_BALANCE — historical offset in months
* ID — client identifier

The combination of demographic and historical credit-behavior features makes this dataset ideally suited for credit scoring, default prediction, and general risk modeling tasks.


7. Business Case and Motivation

Credit default prediction is a fundamental problem in finance risk analytics and regulation and it directly aligns with the competencies of Financial Engineering master’s students.

Its relevance includes:

* Risk Management: Banks use credit-scoring models to evaluate borrower reliability, estimate expected losses, and determine lending conditions.
* Regulatory Compliance: Frameworks like Basel II/III require quantitative models to compute capital requirements based on probabilities of default (PD).
* Portfolio Decisions: Understanding borrower credit quality is essential for securitization, pricing of credit products, and stress-testing.
* Fintech & Lending Platforms: Automated credit scoring is central to online lending, microfinance, and consumer risk assessment.

For us as Financial Engineering students, this project offers an opportunity to:

* Apply machine learning to a real financial dataset
* Explore risk modelling techniques used in the industry
* Understand class imbalance, a key challenge in fraud detection and credit default prediction
* Practice model evaluation aligned with business impact (recall vs precision trade-offs)
* Develop a skill set directly applicable to future roles in banking, asset management, fintech, or quantitative risk

The project is therefore both academically relevant and professionally valuable.


8. Obstacles Encountered So Far (Project Progress Evaluation)

8.1 Target Construction from Longitudinal Credit Data

The credit history contains multiple months per client with varied status codes (0–5, C, X).
Aggregating these into a single binary indicator required decisions on:

* which statuses truly indicate delinquency,
* how to interpret ambiguous codes (X, C),
* how to handle clients with very short histories,
* how to merge histories with application data.

This step required careful reasoning because different aggregation strategies lead to very different class distributions and model performance.


8.2 Managing Severe Class Imbalance

Default cases represent a small minority, causing:

* high accuracy but zero recall in baseline models,
* unstable performance depending on resampling strategy,
* need for SMOTE, oversampling, and class-weighted methods early in the process.

Balancing data without causing overfitting was a recurring challenge.


8.3 Preprocessing Heterogeneous Data

The dataset includes:

* high-cardinality categorical variables,
* numerical features with extreme values (e.g., DAYS_EMPLOYED outliers),
* features with 30–40% missing values.

Building clean and reusable preprocessing pipelines required several iterations.


8.4 Computational Limitations

Certain models (ex: SVM with RBF kernel) became computationally heavy when combined with oversampling or large one-hot-encoded matrices.
This forced trade-offs between performance exploration and runtime constraints.


8.5 Early Model Instability

Different models and resampling strategies produced:

* high variance in recall,
* unstable F1-scores,
* models that detected almost no defaulters,
* others that detected many defaulters with low precision.

Finding a baseline model that behaved consistently took more time than expected.


8.6 Handling Missing Values

Features like OCCUPATION_TYPE contain large proportions of missing data.
Decisions had to be made on:

* whether to drop or impute,
* how to integrate imputation into pipelines,
* how missingness might itself be predictive.


These challenges reflect normal progress at this stage of the project.
The next milestones include expanding feature engineering, performing systematic hyperparameter tuning, validating stability with cross-validation, and deploying interpretability tools such as SHAP.

9. Work Completed So Far vs Planned Work

Completed so far:

Exploration of application and credit datasets
Construction of a working baseline target from STATUS codes
Preliminary preprocessing pipeline (encoding, scaling, basic handling of missing values)
Baseline models: Logistic Regression, Decision Tree, Random Forest, SVM
First trials on handling class imbalance (SMOTE, oversampling, class weights)
Early evaluation using F1-score, precision, recall, ROC-AUC

Planned for the next milestone:

Improved target engineering (better aggregation of credit histories)
Systematic hyperparameter tuning (GridSearch/RandomSearch)
Addition of Gradient Boosting models and KNN
Implementation of interpretability methods (SHAP, permutation importance)
Cleanup of missing values strategy (medians/modes integrated in the pipeline)
More robust cross-validation (K-Fold or StratifiedKFold)
Stability checks and reduction of overfitting
