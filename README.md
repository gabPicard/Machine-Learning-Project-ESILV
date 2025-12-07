Credit Default Prediction
Course: Machine Learning – ESILV (Master 1 Financial Engineering) Authors: Gabriel Picard, Thibault Pelou, Hugo Picard Date: December 2025

Project Overview
This project aims to build a supervised machine learning pipeline to predict credit default risk. In the context of financial risk management, accurately estimating the Probability of Default (PD) is critical for loan pricing, regulatory compliance (Basel II/III), and minimizing portfolio losses.

We used the Home Credit Default Risk dataset to classify clients based on their application data and longitudinal credit history. The primary challenge of this project was handling the significant class imbalance—defaulters represent only about 12% of the dataset—while maintaining a model that is both predictive and interpretable.

Data Construction and Preprocessing
The analysis is based on two sources: demographic/socio-economic data (application_record.csv) and monthly repayment history (credit_record.csv).

We constructed the target variable by aggregating client history. A client was flagged as a "defaulter" (1) if they had any payment past due by more than 30 days (Status 1-5) during the observation window. Otherwise, they were labeled as a "good" client (0).

Our preprocessing strategy focused on data integrity and reproducibility:

Data Cleaning: We identified and treated anomalies, such as the placeholder value 365243 in the employment duration column, and calculated age from birth days for better readability.

Imputation: To prevent data leakage, we built a pipeline using SimpleImputer. Numerical gaps were filled with medians, and categorical gaps with the most frequent value.

Encoding & Scaling: We used One-Hot Encoding for categorical variables and Standard Scaling for numerical features to ensure compatibility with distance-based algorithms like k-NN and SVM.

Modeling Approach
We benchmarked several algorithms, ranging from linear models to ensemble methods. Given the imbalanced nature of the target, we moved beyond simple accuracy and focused on F1-score and ROC-AUC. We also paid close attention to Recall, as missing a potential defaulter is costly for a financial institution.

The models implemented include:

Logistic Regression: Used as a baseline, both with and without PCA dimensionality reduction.

Support Vector Machines (SVM): Tested with an RBF kernel.

Tree-based Ensembles: Decision Trees, Random Forest, and Gradient Boosting.

Voting Classifier: A soft-voting ensemble combining our top-performing estimators.

To address the class imbalance, we experimented with class_weight='balanced' parameters and integrated resampling techniques (Random Over-Sampling and SMOTE) directly into the cross-validation pipelines.

Results and Discussion
Our experiments showed that tree-based models significantly outperformed linear ones. The Logistic Regression baseline struggled to capture non-linear relationships in the socio-economic data.

The Random Forest model (tuned via GridSearchCV) and the Voting Ensemble yielded the best results. While the baseline accuracy was high due to the majority class, the Random Forest achieved a much more balanced F1-score and a ROC-AUC around 0.78.

We found that while resampling techniques like SMOTE improved the model's ability to detect defaulters (higher Recall), they often did so at the expense of Precision. The class_weight='balanced' approach in Random Forest provided the most stable trade-off.

Repository Structure
Project-ML-V2.ipynb: The main Jupyter Notebook containing the full analysis, from EDA to final model evaluation.

ML_REPORT.pdf: The detailed project report covering the business case, methodology, and in-depth result analysis.

data/: Directory containing the source CSV files (not included in the repo due to size, available on Kaggle).

Requirements
The project requires Python 3.8+ and the following libraries:

pandas

numpy

scikit-learn

imbalanced-learn (for SMOTE and pipelines)

matplotlib / seaborn (for visualization)