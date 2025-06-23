# src/train.py

import mlflow

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pymongo import MongoClient
import pandas as pd
from src.preprocess import preprocess_data
import joblib
import os

def load_data_from_mongodb():
    """
    Loads data from MongoDB collection into a pandas DataFrame.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["windturbine"]
    collection = db["Wind Turbine"]
    data = pd.DataFrame(list(collection.find()))
    return data

def train_model():
    # Set MLflow tracking URI to the mlflow server URL
    mlflow.set_tracking_uri("http://localhost:5000")

    # Load and preprocess data
    data = load_data_from_mongodb()
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column='Wind_Speed')

    with mlflow.start_run():
        # Define and train model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, artifact_path="random_forest_model", 
                                 signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
                                 input_example=X_train[:5])

        print(f"Model logged with RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    # Save model locally for testing
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")
    print("Model saved locally at models/random_forest.pkl")

if __name__ == "__main__":
    train_model()
