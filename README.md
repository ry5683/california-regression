# 🏡 California Housing Price Predictor

This project builds and compares three regression models — Linear Regression, Decision Tree Regressor, and Random Forest Regressor — to predict median housing prices in California using census data. It demonstrates core machine learning practices including data cleaning, model evaluation, and visual inspection of predictions.

---

## 📊 Features

- 📥 Load & clean real-world housing data
- 🔁 Train/test data splitting
- 🧠 Train three models: Linear Regression, Decision Tree Regressor, Random Forest Regressor
- 📏 Evaluate models using RMSE (Root Mean Squared Error), and R² Score
- 📈 Visualize actual vs predicted prices
- 🧱 Modular project structure (production-ready layout)

## 🔄 Updates

- Added StandardScaler to normalize features for better performance on linear models.
- Now prints both RMSE and R² score for Linear and Tree models.

---

## 📉 Evaluation Metrics

| Model            | RMSE ↓ | R² Score ↑ |
|------------------|--------|------------|
| Linear Regression| 0.75   | 0.58       |
| Decision Tree    | 0.71   | 0.62       |
| Random Forest    | 0.51   | 0.80       |

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/ry5683/california-regression.git
cd california-regression</file>