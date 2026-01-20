# Supervised Learning – Regression & Classification Project

This project includes two supervised machine-learning tasks, optimized and fixed based on an academic audit report to ensure methodological correctness and prevent data leakage.

1. **Regression:** Predicting a country’s GDP based on COVID-19 and economic indicators.  
2. **Classification:** Predicting whether a customer will churn based on behavioral and service-related data.

---

## 📘 Part 1 – GDP Regression

### ✔ Goal  
Predict the **GDP** of countries using COVID-19 statistics and economic indicators.

### ✔ Steps Performed  
- **Data Cleaning:** Removed unnecessary columns and handled missing values (CPI imputation).
- **Aggregation:** Combined data per country to ensure a balanced dataset.
- **Methodology Fix:** Implemented `StandardScaler` and `PolynomialFeatures` correctly (fitting only on Train set) to prevent **Data Leakage**.
- **Analysis:** Correlation heatmap and Beta Coefficients interpretation.
- **Models Trained:** Linear Regression, RidgeCV, LassoCV, and Polynomial Regression.

### ✔ Saved Artifacts  
- `final_regression_model.joblib`  
- `final_regression_scaler.joblib`  

---

## 📘 Part 2 – Customer Churn Classification

### ✔ Goal  
Predict whether a customer will **churn** (Yes/No).

### ✔ Steps Performed  
- **Data Preprocessing:** Handled duplicates and missing labels in the Churn column.
- **EDA:** Correlation analysis and Pairplots for feature distribution.
- **Optimization:** Used `GridSearchCV` for hyperparameter tuning across all models.
- **Models Trained:** Logistic Regression, KNN, **Support Vector Machine (SVM)**, and Random Forest.
- **Evaluation:** Confusion Matrices and comparison of Accuracy, Recall, and F1-Score.

### ✔ Best Model  
**Random Forest Classifier** (or the model with the highest F1-score in your specific run).

### ✔ Saved Artifacts  
- `final_classification_model.joblib`  
- `final_classification_scaler.joblib`

---

## 📈 Results Summary
The project now follows a strict machine learning pipeline, ensuring that all preprocessing steps are isolated to the training data, resulting in more reliable and generalizable models.

---

## 🚀 How to Run
1. Clone the repository.
2. Ensure you have the data files (`Covid19_With_GDP_Values.csv` and `customer_churn_dataset.csv`) in the same folder.
3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
