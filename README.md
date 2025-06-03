# Final-Year-Capstone-Project

# Optimizing Customer Retention: A Machine Learning Approach to Predicting E-Commerce Churn

## Project Overview
Customer churn is a major challenge in e-commerce, affecting both revenue and long-term growth. This capstone project applies classical machine learning and deep learning techniques to predict churn using structured behavioral data. The project covers data exploration, preprocessing, model training and evaluation, and statistical testing to determine the most effective predictive approach.

**Project Summary Video:** [Watch here](https://youtu.be/O0uSK3xY_tE)

---

## Dataset Overview
- **Source:** Proprietary e-commerce dataset  
- **Size:** 5,630 customer records  
- **Features:** 20 variables, including:
  - Tenure  
  - Order Count  
  - Days Since Last Order  
  - Cashback Amount  
  - Time Spent on App  
  - Marital Status  
  - Complaint Indicator  
  - Satisfaction Score  
  - Distance from Warehouse  
  - Coupon Usage  
- **Target Variable:** `Churn` (0 = Retained, 1 = Churned)

---

## Objectives
- Identify behavioral patterns associated with churn  
- Handle missing data using imputation techniques  
- Engineer and encode features for modeling  
- Train and evaluate a variety of machine learning models  
- Compare classical models with deep learning models (CNN, LSTM)  
- Use Matthews Correlation Coefficient (MCC) for evaluation  
- Perform paired t-tests to assess statistical significance

---

## Technologies Used
- **Language:** Python  
- **Libraries:**
  - `pandas`, `numpy` (Data Processing)
  - `seaborn`, `matplotlib`, `missingno` (Visualization)
  - `scikit-learn`, `xgboost`, `lightgbm` (Machine Learning)
  - `TensorFlow`, `Keras` (Deep Learning)
  - `joblib` (Model Persistence)
- **Environment:** Google Colab / Jupyter Notebook

---

## Exploratory Data Analysis (EDA)
- Churn rate: ~17% (indicating class imbalance)  
- Higher churn among single users, dissatisfied customers, and those with complaints  
- Key indicators include recency, cashback behavior, and low order counts  
- Outliers retained for tree-based models due to robustness

---

## Data Preprocessing
- **Missing Values:** Handled using `IterativeImputer` (MAR assumption)  
- **Encoding:** `OneHotEncoder` within a `ColumnTransformer`  
- **Scaling:** Applied `StandardScaler` to numerical features  
- **Pipelines:** Used to integrate all preprocessing steps and avoid data leakage

---

## Models Trained

### Classical Machine Learning Models
- Decision Tree  
- K-Nearest Neighbors  
- Gradient Boosting  
- Random Forest  
- AdaBoost  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Gaussian Naive Bayes  
- XGBoost  
- LightGBM  

### Deep Learning Models
- **CNN (1D):** Applied convolution and max pooling to identify feature patterns  
- **LSTM:** Used to model sequential dependencies in tabular data

---

## Evaluation Metrics
- Accuracy  
- F1 Score  
- Matthews Correlation Coefficient (MCC) — prioritized due to class imbalance

---

## Best Performing Model
- **XGBoost** delivered the highest MCC and overall best performance  
- CNN and LSTM performed well but did not outperform XGBoost  
- Final model saved as: `best_model.joblib`  
- Model results saved in: `model_performance_with_time.csv`

---

## Statistical Testing
- Paired t-tests conducted between XGBoost and LightGBM across multiple folds  
- Metrics tested: Accuracy, F1 Score, MCC  
- Result: XGBoost significantly outperformed LightGBM (*p* < 0.05)

---

## How to Run the Project

1. Clone this repository or open in Google Colab  
2. Install required libraries:
    ```bash
    pip install scikit-learn xgboost lightgbm tensorflow missingno
    ```
3. Update the dataset path in the script if necessary  
4. Run the main script:
    ```bash
    python capstone_project.py
    ```

---

## Key Insights
- Tree-based models like XGBoost are highly effective for structured churn data  
- CNN and LSTM can be adapted to tabular data and perform competitively  
- Key churn predictors include tenure, complaint behavior, and satisfaction score  
- Pipelines help ensure reproducibility and reduce data leakage  
- MCC is a better metric than accuracy for imbalanced classification

---

## Future Work

1. **Feature Engineering**
   - Include temporal trends (e.g., monthly purchase frequency)
   - Use customer segmentation and RFM scoring

2. **Model Explainability**
   - Integrate SHAP to interpret model predictions

3. **Deployment**
   - Convert model into REST API using Flask or FastAPI for real-time predictions

4. **Ensemble Learning**
   - Use soft voting or stacking to combine top-performing models

5. **Monitoring**
   - Track model drift and retrain as new data becomes available

6. **Cost-Sensitive Learning**
   - Incorporate the cost of false negatives (e.g., lost customers)

---

## Author

**Godfred Kogkane**  
Final Year Student – BSc. Management Information Systems  
Ashesi University, Ghana
