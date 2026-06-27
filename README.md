# Credit Card Fraud Detection

Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This project implements a machine learning pipeline to detect fraudulent credit card transactions using an `XGBoost` classifier. Due to the highly imbalanced nature of fraud datasets, the model is specifically optimized to prioritize detecting minority fraud cases while maintaining reasonable precision.

## Project Structure

The workflow follows a rigorous data science process:

### 1. Data Splitting & Stratification
The data is partitioned using a **70-20-10** strategy to ensure robust training, tuning, and evaluation:
* **70% Training:** Used for model weight optimization.
* **20% Validation:** Used for hyperparameter tuning and monitoring early stopping.
* **10% Test:** A "held-out" set for final unbiased performance assessment.
* **Stratification:** Applied to the target variable (`Class`) to ensure the minority fraud cases are proportionally represented in all subsets.

### 2. Model Training & Regularization
The pipeline compares two XGBoost models:
* **Unregularized Model (Baseline):** Serves as the initial performance benchmark.
* **Regularized Model:** Implements L1 (`reg_alpha=0.1`) and L2 (`reg_lambda=1.0`) regularization to reduce complexity and prevent overfitting to noise.
* **Class Imbalance Handling:** Both models use `scale_pos_weight` to compensate for the skew in the dataset, ensuring the model does not ignore the rare positive (fraud) cases.

### 3. Quantitative Evaluation
The primary evaluation metric used is the **F1-Score**. In fraud detection, accuracy is misleading because a model can achieve high accuracy by simply predicting "not fraud" for every transaction. The F1-Score provides a better balance between **Precision** (avoiding false alarms) and **Recall** (ensuring fraud is detected).

### 4. Diagnostics & Interpretability
The project includes tools for model transparency:
* **Loss Curves:** Visualizes convergence during training; divergence between training and validation loss indicates overfitting.
* **Precision-Recall Curve:** Demonstrates the trade-off between sensitivity and specificity.
* **Feature Importance:** Identifies the top 10 most influential variables that the model uses to distinguish fraudulent from legitimate transactions.

---

## Technical Summary

| Component | Strategy / Tool |
| :--- | :--- |
| **Model** | `XGBClassifier` |
| **Objective** | `binary:logistic` |
| **Strategy** | 70/20/10 Stratified Split |
| **Key Metric** | F1-Score |

---

*Note: This project contains a complete pipeline intended for educational and analytical purposes regarding credit card fraud detection methodologies.*
