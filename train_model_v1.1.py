import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
import time
import logging

# Configuration
model_version = 'v1.2'
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data', model_version)
model_dir = os.path.join(base_dir, 'models', model_version)
log_file_path = os.path.join(model_dir, 'training.log')
responses_path = os.path.join(data_dir, 'responses.csv')
rf_model_path = os.path.join(model_dir, 'career_predictor_model.pkl')

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_time(start_time, message):
    """Logs time and prints message to console and log file."""
    end_time = time.time()
    duration = end_time - start_time
    log_message = f"{message} - Time taken: {duration:.2f} seconds."
    print(log_message)
    logging.info(log_message)
    return end_time

start_time = time.time()
logging.info("=== Training Process Started ===")

try:
    # Dataset Loading
    responses_df = pd.read_csv(responses_path)
    X = responses_df.drop(columns=["Career_Code"])
    y = responses_df["Career_Code"]
    start_time = log_time(start_time, f"Dataset loaded. Total size: {len(responses_df)} samples.")

    # Handle Class Imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    start_time = log_time(start_time, "Class imbalance handled using SMOTE.")

    # Dataset Splitting
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    start_time = log_time(start_time, "Dataset split into training and testing sets.")

    # Expanded Hyperparameter Grid
    param_grid = {
        'n_estimators': [200, 500, 1000],
        'max_depth': [10, 20, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # Feature Scaling
    scaler = StandardScaler()

    # Random Forest Model
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('rf', rf)
    ])

    # Randomized Search
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=50,  # More iterations for better parameter exploration
        cv=5,  # 5-fold cross-validation for robustness
        verbose=3,
        random_state=42,
        n_jobs=-1
    )

    # Hyperparameter Tuning
    rf_random.fit(X_train, y_train)
    start_time = log_time(start_time, "Hyperparameter tuning completed.")

    # Best Parameters
    best_rf = rf_random.best_estimator_
    start_time = log_time(start_time, f"Best Random Forest parameters: {rf_random.best_params_}")

    # Model Evaluation
    y_pred = best_rf.predict(X_test)
    classification_report_str = classification_report(y_test, y_pred, zero_division=0)
    start_time = log_time(start_time, "Model evaluation completed.")
    logging.info(f"Classification report:\n{classification_report_str}")
    print(f"Classification report:\n{classification_report_str}")

    # Save Model
    joblib.dump(best_rf, rf_model_path)
    start_time = log_time(start_time, f"Optimized Random Forest model saved at {rf_model_path}.")

    logging.info("=== Training Process Completed Successfully ===")

except Exception as e:
    error_message = f"An error occurred: {e}"
    logging.error(error_message, exc_info=True)
    print(error_message)
