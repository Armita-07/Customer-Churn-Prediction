# Customer Churn Prediction — E-Commerce ML Classification Study

> **Co-authored research published at IEEE** — Performance Evaluation of Classification Models for E-Commerce Customer Churn Prediction  
> *School of Computer Engineering, KIIT Deemed to be University*

---

## Overview

This project presents a comparative evaluation of **9 machine learning classification models** for predicting customer churn in e-commerce platforms. The goal is to identify which customers are likely to disengage — enabling businesses to deploy proactive retention strategies before revenue is lost.

The study combines rigorous data preprocessing, domain-driven feature engineering, class imbalance handling (SMOTE), and systematic model evaluation across standard classification metrics.

**Key result: LightGBM achieved the best performance — 92.3% accuracy and F1-score of 0.903.**

---

## Models Evaluated

| Model | Accuracy | F1 Score | Recall | Precision |
|---|---|---|---|---|
| **LightGBM** ✅ | **0.923** | **0.903** | **0.890** | **0.921** |
| XGBoost | 0.922 | 0.902 | 0.888 | 0.922 |
| Gradient Boosting | 0.913 | 0.892 | 0.885 | 0.901 |
| Random Forest | 0.906 | 0.882 | 0.874 | 0.893 |
| Extra Trees | 0.877 | 0.840 | 0.821 | 0.871 |
| Decision Tree | 0.835 | 0.805 | 0.813 | 0.798 |
| AdaBoost | 0.818 | 0.787 | 0.801 | 0.778 |
| KNN | 0.741 | 0.723 | 0.770 | 0.724 |
| Logistic Regression | 0.728 | 0.701 | 0.732 | 0.697 |

---

## Dataset

- **Source:** [E-Commerce Customer Behavior Dataset — Kaggle](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset)
- Customer-level data covering demographics, transaction history, behavioral patterns, and after-sale engagement
- Binary target variable: Churned vs. Not Churned

---

## Methodology

### 1. Exploratory Data Analysis
- Checked for duplicates, null values, and class imbalance
- Visualized churn distribution and feature correlations

### 2. Feature Engineering
Six domain-driven features were engineered to capture behavioral and financial signals beyond raw attributes:

| Feature | Description |
|---|---|
| `engagement_total` | Combined engagement activity score |
| `high_engagement_flag` | Binary flag for high-engagement customers |
| `returns_per_review` | Return rate relative to review activity |
| `high_support_calls` | Flag for customers with elevated support interactions |
| `ltv_to_credit_ratio` | Customer lifetime value relative to credit usage |
| `lifetime_bucket` | Segmentation of customers by tenure |

### 3. Data Preprocessing
- Missing values imputed using mean (numeric) and mode (categorical)
- Feature scaling applied
- Categorical encoding for model compatibility
- **80/20 stratified train-test split** to preserve class distribution

### 4. Class Imbalance Handling
- **SMOTE (Synthetic Minority Oversampling Technique)** applied during training to address imbalance between churned and non-churned classes

### 5. Model Training & Evaluation
All 9 models trained and evaluated on:
- Accuracy, F1-Score, Recall, Precision
- Cross-validation for robustness
- ROC curve analysis
- Confusion matrix for false negative minimization

### 6. Hyperparameter Tuning
Best-performing model (LightGBM) further optimized through hyperparameter tuning and validated via confusion matrix analysis.

---

## Key Findings

- **Ensemble tree-based models consistently outperform** linear and instance-based classifiers for churn prediction
- **LightGBM's leaf-wise growth** makes it superior to level-wise boosting for this use case — faster convergence, better accuracy
- **Feature engineering was critical** — behavioral and financial derived features were more informative than raw attributes alone
- **Churn is strongly influenced by:** service usage patterns, cart abandonment behavior, engagement intensity, and customer lifetime value
- Minimizing **false negatives** (churners predicted as retained) is the most business-critical optimization target

---

## Tech Stack

```
Language:     Python
Environment:  Google Colab
Libraries:    Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, Imbalanced-learn (SMOTE)
Hardware:     Intel Core i7 (12th Gen), 16GB RAM
```

---

## Repository Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction.ipynb   # Full notebook — EDA, feature engineering, model training, evaluation
├── README.md                         # Project documentation
└── requirements.txt                  # Dependencies
```

---

## Research Publication

This project is part of a peer-reviewed study accepted for IEEE publication:

> **"Performance Evaluation of Classification Models for E-Commerce Customer Churn Prediction"**  
> Dhruv Saxena, **Armita Patro**, Diya Agarwal, Jay Prakash Singh, Sonal Jain, Mahendra Kumar Gourisaria  
> School of Computer Engineering, KIIT Deemed to be University, Bhubaneswar

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
imbalanced-learn
matplotlib
seaborn
```

---

## Author

**Armita Patro**  
B.Tech CSE, KIIT University  
[LinkedIn](https://linkedin.com/in/armitapatro) · [GitHub](https://github.com/Armita-07) · armitapatro77@gmail.com
