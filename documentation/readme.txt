Optimizing Customer Retention: A Machine Learning Approach to Predicting E-commerce Churn
Project Overview
Customer churn is a critical issue in e-commerce, directly affecting revenue and long-term sustainability. This capstone project applies both classical machine learning and deep learning techniques to predict customer churn using structured behavioral data. The project includes data exploration, preprocessing, model training and evaluation, and a statistical comparison of model performance to identify the most effective predictive approach.

Dataset Description
• Source: Proprietary e-commerce dataset
• Records: 5,630 customers
• Features: 20 features including:
o Tenure
o Order Count
o Day Since Last Order
o Cashback Amount
o Hour Spent on App
o Marital Status
o Complain Indicator
o Satisfaction Score
o Distance from Warehouse to Home
o Coupon Usage
• Target Variable: Churn (0 = Retained, 1 = Churned)

Objectives
• Explore patterns in customer behavior linked to churn
• Handle missing data appropriately using imputation
• Engineer features and encode categorical variables for modeling
• Train and evaluate a range of classification models
• Compare traditional ML models with deep learning approaches (CNN, LSTM)
• Use Matthews Correlation Coefficient (MCC) to assess performance
• Statistically test performance differences using paired t-tests

Technologies Used
• Language: Python
• Libraries:
o Data Processing: pandas, numpy
o Visualization: seaborn, matplotlib, missingno
o Machine Learning: scikit-learn, xgboost, lightgbm
o Deep Learning: TensorFlow, Keras
o Model Persistence: joblib
• Environment: Google Colab / Jupyter Notebook

Exploratory Data Analysis (EDA)
• The churn rate is approximately 17%, indicating a class imbalance.
• Churn is higher among customers who are single, have lodged complaints, or have low satisfaction scores.
• Visualizations reveal patterns in recency, order count, and cashback behavior.
• Several numeric features have outliers, but these are retained for tree-based models due to their robustness.

Data Preprocessing
• Missing Values: Handled using IterativeImputer, based on the identification of Missing at Random (MAR) patterns.
• Categorical Encoding: Used OneHotEncoder within a ColumnTransformer.
• Scaling: Applied StandardScaler to numerical features.
• Pipelines: All preprocessing steps integrated into pipelines to prevent data leakage and streamline training.

Models Trained
Classical Machine Learning Models
• Decision Tree
• K-Nearest Neighbors
• Gradient Boosting
• Random Forest
• AdaBoost
• Logistic Regression
• Support Vector Machine (SVM)
• Gaussian Naive Bayes
• XGBoost
• LightGBM
Deep Learning Models
• Convolutional Neural Network (CNN): Used 1D convolution and max-pooling to extract feature patterns.
• Long Short-Term Memory (LSTM): Applied for sequential modeling of structured data.

Evaluation Metrics
• Accuracy
• F1 Score
• Matthews Correlation Coefficient (MCC) — prioritized due to imbalanced classes

Best Performing Model
• XGBoost achieved the highest MCC score and overall best performance.
• Deep learning models (CNN and LSTM) also performed well but did not outperform XGBoost.
• Final model saved as: best_model.joblib
• Performance metrics for all models saved in: model_performance_with_time.csv

Statistical Testing
To ensure the difference in performance between models is statistically significant:
• A paired t-test was conducted between XGBoost and LightGBM across multiple folds.
• Metrics tested: Accuracy, F1 Score, MCC
• XGBoost significantly outperformed LightGBM (p-values < 0.05)

How to Run the Project
1. Clone this repository or upload the files to Google Colab.
2. Install required libraries:
pip install scikit-learn xgboost lightgbm tensorflow missingno
3. Update the dataset path in the script if needed.
4. Run the script:
python capstone_project.py

Key Insights
• Tree-based ensemble models, particularly XGBoost, are highly effective for structured churn data.
• CNN and LSTM can perform competitively even on tabular data when reshaped appropriately.
• Feature patterns such as tenure, complaint behavior, and satisfaction scores are strong indicators of churn.
• Pipelines and imputation strategies are critical for building reproducible and clean models.
• Matthews Correlation Coefficient is a superior metric for class imbalance compared to simple accuracy.

Future Work
1. Feature Engineering:
o Incorporate temporal trends such as purchase frequency over time.
o Introduce customer segmentation or RFM (Recency, Frequency, Monetary) scores.
2. Model Explainability:
o Integrate SHAP (SHapley Additive exPlanations) to interpret model predictions at a feature level.
o Provide decision reasoning for churn predictions to stakeholders.
3. Real-Time Prediction System:
o Convert the model into a REST API using Flask or FastAPI for live predictions on customer data.
4. Ensemble Stacking:
o Combine predictions from top-performing models using ensemble techniques like soft voting or stacking.
5. Production Readiness:
o Monitor model drift over time.
o Automate retraining using updated customer data.
6. Cost-Sensitive Learning:
o Account for the business cost of misclassification, especially false negatives (lost customers).

Author
Godfred Kogkane
Final Year Student – Management Information Systems
Ashesi University, Ghana
