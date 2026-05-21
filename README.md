# 💳 Credit Card Fraud Detection

A complete machine learning pipeline for detecting fraudulent credit card transactions using Decision Trees and Ensemble Learning techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Key Concepts](#key-concepts)

---

## Overview

This project builds a full end-to-end ML model to classify credit card transactions as **Normal (0)** or **Fraud (1)**. The dataset is highly imbalanced (only ~0.17% fraud), so we use **SMOTE** to handle this before training.

The notebook covers everything from data exploration to saving the best model — using Decision Trees as the foundation and multiple Ensemble methods on top.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Rows:** 284,807 transactions
- **Features:** 30 (V1–V28 are PCA-transformed, plus `Amount` and `Time`)
- **Target:** `Class` — 0 = Normal, 1 = Fraud
- **Fraud rate:** ~0.17% (highly imbalanced)

> ⚠️ The dataset file is listed in `.gitignore` and must be downloaded separately from Kaggle.

---

## Project Structure

```
credit-card-fraud-detection/
│
├── CREDIT_CARD_FRAUD_DETECTION_COMPLETE.ipynb   # Main notebook
├── creditcard.csv                               # Dataset (download from Kaggle)
├── fraud_model.pkl                              # Saved best model (generated after running)
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 3. Download the dataset

Go to [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), download `creditcard.csv`, and place it in the project root folder.

### 4. Launch the notebook

```bash
jupyter notebook CREDIT_CARD_FRAUD_DETECTION_COMPLETE.ipynb
```

---

## Usage

Run all cells **top to bottom** in the notebook. Each section is clearly labelled.

```
1. Import Libraries
2. Load Dataset
3. Explore the Data (EDA)
4. Preprocessing  →  scale features, apply SMOTE
5. Helper Function  →  evaluate() used for every model
6. Logistic Regression  (baseline)
7. Decision Tree  →  basic, pruned, best depth, feature importance
8. Ensemble Methods  →  7 different techniques
9. ROC Curve Comparison
10. Model Comparison Table
11. Save Best Model
```

The best model is automatically saved as `fraud_model.pkl` at the end.

---

## Models Used

### Baseline
| Model | Description |
|---|---|
| Logistic Regression | Simple linear baseline |

### Decision Tree
| Model | Description |
|---|---|
| Decision Tree (Basic) | Unpruned full tree |
| Decision Tree (Pruned) | Limited depth to reduce overfitting |
| Decision Tree (Best Depth) | Depth chosen via cross-validation |

### Ensemble Learning
| Model | Type | Description |
|---|---|---|
| Bagging | Bagging | Many DTs on random data subsets |
| Random Forest | Bagging | Many DTs on random features |
| Extra Trees | Bagging | Like RF but with fully random splits |
| AdaBoost | Boosting | Each tree focuses on previous errors |
| Gradient Boosting | Boosting | Trees correct the full ensemble's errors |
| Voting (Soft) | Voting | Averages predicted probabilities |
| Voting (Hard) | Voting | Majority vote across models |
| Stacking | Stacking | Base models feed a meta Logistic Regression |

---

## Results

After running the notebook, a comparison table is printed with these metrics for every model:

| Metric | Description |
|---|---|
| **ROC-AUC** | Area under ROC curve — main metric |
| **F1 Score** | Balance of precision and recall |
| **Precision** | Of predicted frauds, how many were real? |
| **Recall** | Of real frauds, how many did we catch? |

> The best model is highlighted automatically and saved to disk.

---

## Key Concepts

**Why SMOTE?**
The dataset has ~492 fraud cases out of 284,807 transactions. Without balancing, models just predict "Normal" for everything. SMOTE creates synthetic fraud samples so the model learns both classes properly.

**Why Ensemble Methods?**
A single Decision Tree overfits easily. Ensemble methods combine many trees to get more stable, accurate predictions:
- **Bagging** reduces variance (overfitting)
- **Boosting** reduces bias (underfitting)
- **Voting/Stacking** combines the strengths of different model types

**Why ROC-AUC over Accuracy?**
With only 0.17% fraud, a model that always predicts "Normal" gets 99.83% accuracy — but catches zero fraud. ROC-AUC and F1 Score give a much more honest view of performance.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```

---

## License

This project is open source and available under the [MIT License](LICENSE).
