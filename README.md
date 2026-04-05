# bank-churn-ann-classifier

An Artificial Neural Network (ANN) built with TensorFlow/Keras to predict whether a bank customer will churn (exit), trained on the classic **Churn Modelling** dataset (10,000 records).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Results](#results)
- [Prediction Example](#prediction-example)
- [Tech Stack](#tech-stack)

---

## Overview

This project builds a binary classification model to identify customers likely to leave a bank. It covers end-to-end steps: data exploration, preprocessing (encoding + scaling), ANN training with regularization, evaluation, and single-sample inference.

---

## Dataset

**File:** `Churn_Modelling.csv`  
**Rows:** 10,000 | **Columns:** 14

| Feature | Type | Description |
|---|---|---|
| CreditScore | Numeric | Customer credit score |
| Geography | Categorical | France / Spain / Germany |
| Gender | Binary | Male / Female |
| Age | Numeric | Customer age |
| Tenure | Numeric | Years with the bank |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of bank products used |
| HasCrCard | Binary | Has credit card (1/0) |
| IsActiveMember | Binary | Active member (1/0) |
| EstimatedSalary | Numeric | Estimated annual salary |
| **Exited** | **Target** | **Churned (1) or not (0)** |

> Columns `RowNumber`, `CustomerId`, and `Surname` are dropped as non-predictive.

---

## Project Structure

```
bank-churn-ann-classifier/
│
├── ANN_model.ipynb          # Main Jupyter notebook (EDA → training → inference)
├── Churn_Modelling.csv      # Dataset
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/bank-churn-ann-classifier.git
cd bank-churn-ann-classifier

pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
tensorflow
keras
streamlit
pyngrok
```

---

## Usage

Open and run the notebook end-to-end:

```bash
jupyter notebook ANN_model.ipynb
```

Or run on **Google Colab** (GPU: T4 recommended for faster training).

---

## Model Architecture

A Keras `Sequential` ANN with dropout and L2 regularization to prevent overfitting:

```
Input (12 features after preprocessing)
   │
Dense(128, ReLU)
Dropout(0.3)
   │
Dense(64, ReLU) + L2(0.001)
Dropout(0.3)
   │
Dense(32, ReLU)
Dropout(0.2)
   │
Dense(1, Sigmoid)   ← Binary output (churn probability)
```

**Total Parameters:** ~12,033

---

## Preprocessing Pipeline

Built with `sklearn.compose.ColumnTransformer`:

| Step | Transformer | Applied To |
|---|---|---|
| OneHotEncoder | `Geography` (3 categories → 3 binary columns) | Categorical |
| StandardScaler | All numeric features | Numeric |

Gender is label-encoded to binary (0/1) before the pipeline.

---

## Results

The model outputs a churn probability between 0 and 1. A threshold of **0.5** is used for the final binary classification:

- `probability > 0.5` → **Customer likely to churn**
- `probability ≤ 0.5` → **Customer not likely to churn**

---

## Prediction Example

```python
input_data = {
    'creditscore': 619,
    'geography': 'France',
    'gender': 1,        # 1=Male, 0=Female
    'age': 42,
    'tenure': 2,
    'balance': 0.00,
    'numofproducts': 1,
    'hascrcard': 1,
    'isactivemember': 1,
    'estimatedsalary': 101348.88
}

# After preprocessing and model inference:
# Predicted probability: 0.386
# Predicted class: 0
# → "The customer is not likely to churn."
```

---

## Tech Stack

- **Python 3.12**
- **TensorFlow / Keras** — ANN model
- **scikit-learn** — Preprocessing pipeline
- **Pandas / NumPy** — Data manipulation
- **Google Colab** — Training environment (GPU: T4)
- **Streamlit + pyngrok** — (Optional) interactive web demo

---

## License

MIT License. Feel free to use, modify, and distribute.
