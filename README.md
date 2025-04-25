 """# Bank Customer Churn Prediction

## Overview
This project predicts customer churn for a bank using machine learning techniques. It analyzes a dataset of 10,000 customers to identify key factors influencing churn (e.g., Age, Balance) and builds predictive models to classify customers as likely to churn or not. The project demonstrates data analysis, visualization, preprocessing, and modeling, suitable for academic evaluation or portfolio展示.

## Dataset
- **Source**: `dataset.csv` (preprocessed, 10,000 rows, 12 columns).
- **Features**:
  - Numerical: `CreditScore`, `Age`, `Tenure`, `Balance`, `Num Of Products`, `Estimated Salary` (scaled).
  - Binary/Encoded: `Has Credit Card`, `Is Active Member`, `Geography_Germany`, `Geography_Spain`, `Gender_Male`.
  - Target: `Churn` (0: Non-Churned, 1: Churned, ~20% churn rate).
- **Notes**: The dataset is preprocessed (no `CustomerId`, `Surname`; `Geography`, `Gender` are one-hot encoded).

## Methodology
1. **Data Analysis**:
   - Used Pandas to compute churn distribution, summary statistics, and correlations.
   - Key findings: Churned customers are older, have higher balances; `Age` and `Geography_Germany` correlate with churn.
2. **Visualizations**:
   - Plots: Churn distribution, correlation heatmap, Age distribution, Balance boxplot, feature importance, confusion matrices (RandomForest, XGBoost).
3. **Preprocessing**:
   - Skipped encoding (already done in dataset).
   - Scaled numerical features using `StandardScaler`.
   - Applied Random Over-Sampling (ROS) to balance classes (~7,963 per class).
4. **Modeling**:
   - **RandomForestClassifier**: Tuned with `RandomizedSearchCV` (best parameters: `n_estimators=200`, `max_depth=30`, etc.).
   - **XGBoostClassifier**: Baseline model with `eval_metric='logloss'`.
   - Evaluated using accuracy, precision, recall, F1-score, and confusion matrices.
5. **Feature Importance**:
   - Identified `Age`, `Balance`, `Is Active Member` as top predictors.

## Results
- **RandomForest**:
  - Accuracy: 94%
  - Recall (Churn): 97%
  - F1-Score (Churn): 94%
  - Confusion Matrix: [[1502, 131], [45, 1508]]
- **XGBoost**:
  - Accuracy: 90%
  - Recall (Churn): 93%
  - F1-Score (Churn): 90%
  - Confusion Matrix: [[1411, 222], [108, 1445]]
- **Conclusion**: RandomForest outperforms XGBoost, with higher accuracy and recall, making it ideal for identifying at-risk customers.

## Setup
### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `xgboost`, `python-pptx`
- Jupyter Notebook or Google Colab

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bank-customer-churn-prediction.git
   cd bank-customer-churn-prediction
