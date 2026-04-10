# bank-churn-ann-classifier

An Artificial Neural Network (ANN) built with TensorFlow/Keras to predict whether a bank customer will churn (exit), trained on the classic Churn Modelling dataset (10,000 records). The notebook also includes a full **GridSearchCV hyperparameter tuning** pipeline using `scikeras` to find the optimal architecture.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Prediction Example](#prediction-example)
- [Saved Artifacts](#saved-artifacts)
- [Tech Stack](#tech-stack)

---

## Overview

This project builds a binary classification model to identify customers likely to leave a bank. It covers end-to-end steps:

1. Data loading & exploration
2. Preprocessing (label encoding + OneHotEncoding + StandardScaler)
3. ANN training with Dropout and L2 regularization
4. TensorBoard integration for training visualization
5. Early stopping to prevent overfitting
6. GridSearchCV hyperparameter tuning over neurons, layers, and epochs
7. Final model evaluation and single-sample inference

---

## Dataset

**Source:** Loaded directly from [krishnaik06's GitHub](https://raw.githubusercontent.com/krishnaik06/ANN-CLassification-Churn/refs/heads/main/Churn_Modelling.csv)
**File:** `Churn_Modelling.csv` — Rows: 10,000 | Columns: 14

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
| Exited | **Target** | Churned (1) or not (0) |

> Columns `RowNumber`, `CustomerId`, and `Surname` are dropped as non-predictive. All column names are lowercased for consistency.

---

## Project Structure

```
bank-churn-ann-classifier/
│
├── ANN_model.ipynb          # Main notebook (EDA → training → tuning → inference)
├── Churn_Modelling.csv      # Dataset (or loaded from URL in notebook)
├── preprocessor.joblib      # Saved sklearn ColumnTransformer
├── model.h5                 # Saved base ANN model
├── final_model.h5           # Saved best model after GridSearchCV
├── logs/                    # TensorBoard logs
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

**`requirements.txt`**

```
pandas
numpy
scikit-learn
tensorflow==2.15.0
keras
scikeras==0.12.0
joblib
matplotlib
streamlit
pyngrok
```

---

## Usage

Open and run the notebook end-to-end:

```bash
jupyter notebook ANN_model.ipynb
```

Or run on **Google Colab** (GPU: T4 recommended for faster training, especially for GridSearchCV).

To view TensorBoard training logs:

```bash
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

---

## Model Architecture

A Keras `Sequential` ANN with Dropout and L2 regularization to prevent overfitting:

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

- **Optimizer:** Adam (`learning_rate=0.001`)
- **Loss:** Binary Crossentropy
- **Callbacks:** EarlyStopping (`patience=5`, monitors `val_loss`) + TensorBoard
- **Max Epochs:** 150 (early stopping kicks in earlier)
- **Total Parameters:** ~12,033

---

## Preprocessing Pipeline

Built with `sklearn.compose.ColumnTransformer` and saved via `joblib`:

| Step | Transformer | Applied To |
|---|---|---|
| Label Encoding | `map({'Male': 0, 'Female': 1})` | `gender` |
| OneHotEncoder | `OneHotEncoder()` | `geography` (→ 3 binary columns) |
| StandardScaler | `StandardScaler()` | All numeric features |

The fitted preprocessor is saved as `preprocessor.joblib` for reuse during inference without re-fitting.

---

## Hyperparameter Tuning

The notebook also includes a **GridSearchCV** section using `scikeras.wrappers.KerasClassifier` to find the best combination of:

| Parameter | Values Searched |
|---|---|
| `model__neurons` | 16, 32, 64, 128 |
| `model__layers` | 1, 2, 3 |
| `epochs` | 50, 100 |

```python
param_grid = {
    'model__neurons': [16, 32, 64, 128],
    'model__layers': [1, 2, 3],
    'epochs': [50, 100]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=1)
grid_res = grid.fit(X_train, y_train)

print("Best:", grid_res.best_score_, grid_res.best_params_)
```

The best model is re-trained with the optimal parameters and saved as `final_model.h5`. Performance is evaluated with a **confusion matrix** and **accuracy score** on the test set.

---

## Results

The model outputs a churn probability between 0 and 1. A threshold of **0.5** is used for binary classification:

- `probability > 0.5` → Customer **likely to churn**
- `probability ≤ 0.5` → Customer **not likely to churn**

Model performance is visualized via:
- TensorBoard (training vs. validation loss/accuracy curves)
- Confusion matrix (`sklearn.metrics.ConfusionMatrixDisplay`)
- Test accuracy score

---

## Prediction Example

```python
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

preprocessor = joblib.load('preprocessor.joblib')
model = load_model('model.h5')   # or 'final_model.h5' for the tuned version

input_data = {
    'creditscore': 619,
    'geography': 'France',
    'gender': 1,          # 1=Male, 0=Female
    'age': 42,
    'tenure': 2,
    'balance': 0.00,
    'numofproducts': 1,
    'hascrcard': 1,
    'isactivemember': 1,
    'estimatedsalary': 101348.88
}

df_input = pd.DataFrame([input_data])
X_input = preprocessor.transform(df_input)

y_pred_prob = model.predict(X_input)
y_pred_class = (y_pred_prob > 0.5).astype(int)

print("Predicted probability:", y_pred_prob[0][0])   # e.g. 0.386
print("Predicted class:", y_pred_class[0][0])         # e.g. 0
# → "The customer is not likely to churn."
```

---

## Saved Artifacts

| File | Description |
|---|---|
| `preprocessor.joblib` | Fitted `ColumnTransformer` (OHE + Scaler) |
| `model.h5` | Base ANN trained with Dropout + L2 regularization |
| `final_model.h5` | Best ANN from GridSearchCV hyperparameter tuning |
| `logs/fit/` | TensorBoard training logs |

---

## Tech Stack

- **Python 3.12**
- **TensorFlow 2.15 / Keras** — ANN model
- **scikeras 0.12** — Keras + scikit-learn wrapper for GridSearchCV
- **scikit-learn** — Preprocessing pipeline, train-test split, GridSearchCV, metrics
- **Pandas / NumPy** — Data manipulation
- **Matplotlib** — Confusion matrix visualization
- **joblib** — Model/preprocessor serialization
- **Google Colab** — Training environment (GPU: T4)
- **Streamlit + pyngrok** — (Optional) interactive web demo

