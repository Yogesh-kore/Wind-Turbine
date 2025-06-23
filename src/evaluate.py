import mlflow

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymongo import MongoClient
import joblib
from src.preprocess import preprocess_data
import mlflow.sklearn
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

def evaluate_model(model, model_name="Model"):
    """
    Evaluates the model using test data, prints metrics, and generates plots.
    """
    # Load and preprocess test data
    df = load_data_from_mongodb()
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column='Wind_Speed')

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nEvaluation Results for {model_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.2f}")

    # Visualization - Actual vs Predicted
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Wind Speed")
    plt.ylabel("Predicted Wind Speed")
    plt.title(f"Actual vs Predicted Wind Speed - {model_name}")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/actual_vs_predicted_{model_name}.png")
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True)
    plt.title(f"Residuals Distribution - {model_name}")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"outputs/residuals_distribution_{model_name}.png")
    plt.show()

if __name__ == "__main__":
    # Set MLflow tracking URI to the mlflow server URL
    mlflow.set_tracking_uri("http://localhost:5000")

    # List of models to evaluate: can be local files or mlflow run URIs
    models_to_evaluate = [
        {"name": "RandomForest_Local", "type": "local", "path": "models/random_forest.pkl"},
        # Add more models here, e.g.:
        {"name": "RandomForest_Run", "type": "mlflow", "run_id": "36e1026033b44d68ab3eb1c91dd7bd58", "artifact_path": "random_forest_model"},
    ]

    for model_info in models_to_evaluate:
        if model_info["type"] == "local":
            model = joblib.load(model_info["path"])
            evaluate_model(model, model_info["name"])
        elif model_info["type"] == "mlflow":
            model_uri = f"runs:/{model_info['run_id']}/{model_info['artifact_path']}"
            model = mlflow.sklearn.load_model(model_uri)
            evaluate_model(model, model_info["name"])
