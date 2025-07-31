# 🧠 Fraud Detection - End-to-End Project (1M+ Records)

This is a complete, professional-grade machine learning and deep learning pipeline for detecting fraudulent transactions. The dataset used contains over **1 million real-world records**, and the problem is **highly imbalanced** — fraudulent cases form only a small fraction of the data.

The goal is to build a high-performance fraud detection system that generalizes well and can be easily integrated into real-world applications.

---

## 🔍 Problem Statement

Fraudulent transactions are rare, making fraud detection a challenging classification task due to the severe **class imbalance**. A naive classifier might predict "no fraud" 99% of the time and still achieve high accuracy, which is misleading.

This project focuses on building smart models that:
- Handle **imbalanced datasets**
- Detect fraud cases with **high precision and recall**
- Combine the strengths of traditional ML models and deep learning

---

## 📊 Dataset Overview

- 📦 Over **1 million transactions**
- ⚠️ Highly **imbalanced** (fraud cases ≪ non-fraud)
- 🧹 Includes both **categorical and numerical features**
- 💡 Data anonymized (as is common in financial datasets)

---

## 🧪 Workflow Summary (From A to Z)

### ✅ 1. Data Preprocessing
- Null value treatment
- Label encoding for categorical columns
- Scaling numeric features using `StandardScaler`
- Feature selection based on correlation and importance
- Outlier detection (IQR + Z-Score)
- Saving processed data for reuse

### ⚖️ 2. Handling Class Imbalance
- **SMOTE Oversampling**
- **Class Weights** in model training
- Evaluation with **Recall**, **F1**, **PR-AUC**, not just accuracy

### 📈 3. Exploratory Data Analysis (EDA)
- Fraud vs. Non-fraud distribution
- Feature-wise fraud patterns
- Correlation heatmaps
- Distribution plots, boxplots

---

## 🧠 Models Trained

| Model                 | Technique                        | Notes                                      |
|----------------------|----------------------------------|--------------------------------------------|
| Random Forest Classifier | GridSearchCV + Class Weight      | Interpretable, strong baseline             |
| LightGBM             | Early Stopping + Tuning          | Fast, scalable boosting model              |
| Keras DNN            | Dropout + BatchNorm + Adam       | Trained on SMOTE-balanced data             |
| Ensemble Voting      | Soft Voting (RFC + LGBM )   | Best overall performance                   |

All models were carefully tuned and saved for inference.

---

## 📊 Evaluation Metrics

- **Confusion Matrix**
- **ROC-AUC Score**
- **Precision-Recall Curve**
- **Classification Report**
- **Calibration Curve (for probabilistic confidence)**

> Models were evaluated using stratified splits and multiple metrics, especially **F1-Score** and **Recall**, to ensure fairness on imbalanced data.

---

## 🔬 Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Imbalanced-learn (SMOTE)
- LightGBM
- Keras / TensorFlow (DNN)
- Joblib (model saving)
- Jupyter Notebooks

---

## 🗂️ Project Structure

