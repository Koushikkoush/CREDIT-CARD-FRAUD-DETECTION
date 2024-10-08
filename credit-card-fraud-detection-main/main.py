import os
from src.preprocessing import load_data, preprocess_data
from src.train_model import train_logistic_regression, train_decision_tree, train_random_forest
from src.evaluation import evaluate_model
from src.save_model import save_model

# File paths
train_path = 'C:/Users/kaush/OneDrive/Desktop/CODSOFT-main/credit-card-fraud-detection-main/credit-card-fraud-detection-main/data/fraudTrain.csv'
test_path = 'C:/Users/kaush/OneDrive/Desktop/CODSOFT-main/credit-card-fraud-detection-main/credit-card-fraud-detection-main/data/fraudTest.csv'

# Load and preprocess data
train_data, test_data = load_data(train_path, test_path)
X_train, X_val, y_train, y_val = preprocess_data(train_data)

# Create models directory if it doesn't exist
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Train Logistic Regression Model
lr_model = train_logistic_regression(X_train, y_train)
evaluate_model(lr_model, X_val, y_val)
save_model(lr_model, f'{models_dir}/credit_fraud_lr_model.pkl')

# Train Decision Tree Model
dt_model = train_decision_tree(X_train, y_train)
evaluate_model(dt_model, X_val, y_val)
save_model(dt_model, f'{models_dir}/credit_fraud_dt_model.pkl')

# Train Random Forest Model
rf_model = train_random_forest(X_train, y_train)
evaluate_model(rf_model, X_val, y_val)
save_model(rf_model, f'{models_dir}/credit_fraud_rf_model.pkl')
