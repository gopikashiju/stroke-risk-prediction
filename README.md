# Stroke Risk Prediction

A machine learning project analyzing patient health data to identify key stroke risk factors and predict stroke likelihood, with a focus on handling severe class imbalance in healthcare data.

## Overview

Stroke prediction is a classic imbalanced classification problem — only ~5% of patients in most datasets have experienced a stroke, which makes naive models misleading (a model that always predicts "no stroke" looks 95% accurate while being clinically useless). This project works through the full pipeline: cleaning, exploration, imbalance-aware modeling, and honest evaluation.

## Dataset

5,000+ patient records with features including age, gender, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status.

## Key Findings

- **Stroke incidence:** 4.9% of patients — a strongly imbalanced target
- **Age** is the strongest predictor; stroke cases cluster sharply after age 50–60
- **Hypertension** raises stroke rate ~3.3x (3.97% → 13.25%)
- **Heart disease** raises stroke rate ~4x (4.18% → 17.03%)

## Approach

1. **Data cleaning** — missing value imputation, duplicate removal, outlier detection (IQR method)
2. **Feature engineering** — one-hot encoding, label encoding, standard scaling
3. **EDA** — distribution analysis and group comparisons to surface risk factors
4. **Modeling** — trained and compared three classifiers:
   - Logistic Regression (`class_weight='balanced'`)
   - Decision Tree (`class_weight='balanced'`)
   - Random Forest (with and without SMOTE oversampling)
5. **Evaluation** — prioritized **recall** over accuracy, since missing an actual stroke case is far costlier than a false alarm in a healthcare context

## Results

| Model | Stroke Recall | Stroke Precision |
|---|---|---|
| **Logistic Regression** | **0.80** | 0.14 |
| Random Forest + SMOTE | 0.22 | 0.12 |
| Decision Tree | 0.16 | 0.21 |
| Random Forest (no correction) | 0.00 | 0.00 |

## Conclusion

Logistic Regression with balanced class weighting substantially outperformed tree-based models at catching actual stroke cases, even after applying SMOTE to Random Forest. This suggests that for severely imbalanced healthcare data, a well-weighted linear model can be more reliable than more complex ensemble methods — a useful reminder that model complexity isn't always the answer to imbalance problems.

**Next steps:** threshold tuning to improve precision, and testing gradient-boosted models (XGBoost/LightGBM) with imbalance-aware loss functions.

## Tools

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · Matplotlib
