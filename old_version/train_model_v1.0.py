import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
import joblib
import os
import time

# Configuration
model_version = 'v0.1'
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data', model_version)
model_dir = os.path.join(base_dir, 'models', model_version)
responses_path = os.path.join(data_dir, 'responses.csv')
rf_model_path = os.path.join(model_dir, 'career_predictor_model.pkl')

# Time tracking function
def log_time(start_time, message):
    end_time = time.time()
    duration = end_time - start_time
    print(f"{message} - Time taken: {duration:.2f} seconds.")
    return end_time

start_time = time.time()

# Dataset Loading
responses_df = pd.read_csv(responses_path)
X = responses_df.drop(columns=["Career_Code"])
y = responses_df["Career_Code"]
start_time = log_time(start_time, f"Dataset loaded. Total size: {len(responses_df)} samples.")

# Configuration
n_estimators = 300
max_depth = 20
n_iter = 20
test_size = 0.2
start_time = log_time(start_time, f"Configuration set: n_estimators={n_estimators}, max_depth={max_depth}, n_iter={n_iter}, test_size={test_size}.")

# Dataset Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
start_time = log_time(start_time, "Dataset split into training and testing sets.")

# Hyperparameter Grid
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

# Random Forest Tuning
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=n_iter,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
rf_random.fit(X_train, y_train)
start_time = log_time(start_time, "Hyperparameter tuning completed.")

# Best Parameters
best_rf = rf_random.best_estimator_
start_time = log_time(start_time, f"Best Random Forest parameters: {rf_random.best_params_}")

# Model Evaluation
y_pred = best_rf.predict(X_test)
start_time = log_time(start_time, "Model evaluation completed.")
start_time = log_time(start_time, f"Classification report:\n{classification_report(y_test, y_pred, zero_division=0)}")

# Save Model
joblib.dump(best_rf, rf_model_path)
start_time = log_time(start_time, f"Optimized Random Forest model saved at {rf_model_path}.")
