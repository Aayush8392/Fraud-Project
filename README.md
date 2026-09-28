# Vehicle Insurance Fraud Detection

Exploratory data analysis, data cleaning, and predictive modeling on a real vehicle insurance claims dataset, built to detect fraudulent claims.

## Dataset

**Vehicle Insurance Claim Fraud Detection** (also known as the "carclaims" dataset), 15,420 real vehicle insurance claims.

- Source: [Kaggle - shivamb/vehicle-claim-fraud-detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- 33 columns covering policy details, vehicle details, claimant details, and accident/claim details
- Target column: `FraudFound_P` (1 = fraudulent, 0 = legitimate) - 923 of 15,420 claims (5.99%) are fraudulent

The dataset file (`fraud_oracle.csv`) is included in this repo.

## What this project does

1. **Data quality audit** - goes beyond a basic null check to find masked missing values (e.g. `Age == 0`), inconsistent text-bin formatting, and a cross-column inconsistency between `VehicleCategory` and `PolicyType` affecting 32.3% of rows.
2. **Data cleaning** - fixes each issue found, with reasoning and before/after row counts for every step.
3. **Exploratory data analysis** - fraud rate broken down by the features that matter most (`Fault`, `VehicleCategory`, `VehiclePrice`, `PastNumberOfClaims`), plus a correlation heatmap.
4. **Feature engineering** - three features derived directly from EDA findings (`is_policyholder_fault`, `is_extreme_vehicle_price`, `fault_and_no_history`).
5. **Modeling** - Logistic Regression, Random Forest, and XGBoost, all using class weighting to handle the 5.99% fraud rate, plus one SMOTE-based comparison.
6. **Evaluation** - ROC-AUC as the primary metric (chosen over accuracy, which is misleading at this level of class imbalance), with a full precision/recall/F1 breakdown and decision threshold analysis.
7. **Model interpretation** - Logistic Regression coefficients and Random Forest feature importance, cross-checked against each other.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.921 | 0.277 | 0.195 | 0.229 | **0.835** |
| Random Forest | 0.939 | 0.400 | 0.032 | 0.060 | 0.833 |
| Logistic Regression | 0.645 | 0.130 | 0.865 | 0.226 | 0.798 |
| Logistic Regression (SMOTE) | 0.666 | 0.134 | 0.832 | 0.230 | 0.794 |

XGBoost performs best by ROC-AUC. All three model families land well above the 0.5 random-guessing baseline, confirming the dataset has genuine, learnable fraud signal.

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
jupyter notebook fraud_detection_analysis.ipynb
```

The notebook reproduces the full pipeline end to end: data loading, cleaning, EDA, feature engineering, modeling, and evaluation, with explanations and charts inline between cells.

`xgboost` and `imbalanced-learn` are optional - if either isn't installed, the notebook falls back automatically (`HistGradientBoostingClassifier` in place of XGBoost, and the SMOTE comparison is skipped).

## Files

- `fraud_detection_analysis.ipynb` - the full analysis pipeline
- `fraud_oracle.csv` - the dataset
- `README.md` - this file
