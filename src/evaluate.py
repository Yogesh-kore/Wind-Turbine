import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymongo import MongoClient
import joblib
from src.preprocess import preprocess_data
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
    # List of models to evaluate: only local files now
    models_to_evaluate = [
        {"name": "RandomForest", "type": "local", "path": "models/randomforest.pkl"},
        {"name": "XGBoost", "type": "local", "path": "models/xgboost.pkl"},
        {"name": "LinearRegression", "type": "local", "path": "models/linearregression.pkl"},
    ]

    for model_info in models_to_evaluate:
        if model_info["type"] == "local":
            model = joblib.load(model_info["path"])
            evaluate_model(model, model_info["name"])
