# src/train.py

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pymongo import MongoClient
import pandas as pd
from src.preprocess import preprocess_data
import joblib
import os

# Set MLflow tracking URI to the MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Set or create MLflow experiment
mlflow.set_experiment("WindTurbine_Models")

def load_data_from_mongodb():
    """
    Loads data from MongoDB collection into a pandas DataFrame.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["windturbine"]
    collection = db["Wind Turbine"]
    data = pd.DataFrame(list(collection.find()))
    return data

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{model_name} trained with RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name.lower()}.pkl"
        joblib.dump(model, model_path)
        print(f"{model_name} saved locally at {model_path}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log the model itself
        mlflow.sklearn.log_model(model, model_name)

def train_model():
    # Load and preprocess data
    data = load_data_from_mongodb()
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column='Wind_Speed')

    # Define models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
        "LinearRegression": LinearRegression()
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name)

if __name__ == "__main__":
    train_model()
