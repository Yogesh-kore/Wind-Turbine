import pytest
import os
import joblib
import pandas as pd
from src.ingest import data as ingest_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def load_data_from_mongodb():
    """
    Loads data from MongoDB collection into a pandas DataFrame.
    """
    return ingest_data

def test_load_data():
    data = load_data_from_mongodb()
    assert data is not None
    assert not data.empty

def test_preprocess_data():
    data = load_data_from_mongodb()
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column='Wind_Speed')
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_preprocess_data_missing_target():
    # Test preprocess_data raises ValueError if target column missing
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    with pytest.raises(ValueError):
        preprocess_data(df, target_column='Wind_Speed')

def test_train_and_save_all_models():
    # Run training for all models
    train_model()
    # Check if all model files exist
    assert os.path.exists("models/randomforest.pkl")
    assert os.path.exists("models/linearregression.pkl")
    assert os.path.exists("models/xgboost.pkl")

def test_evaluate_all_models():
    # Load and evaluate all models
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter errors

    rf_model = joblib.load("models/randomforest.pkl")
    evaluate_model(rf_model, model_name="RandomForest")

    lr_model = joblib.load("models/linearregression.pkl")
    evaluate_model(lr_model, model_name="LinearRegression")

    xgb_model = joblib.load("models/xgboost.pkl")
    evaluate_model(xgb_model, model_name="XGBoost")

def test_train_main_block():
    # Test the main block in train.py by calling train_model
    train_model()

def test_train_and_evaluate_model_function():
    from sklearn.linear_model import LinearRegression
    import numpy as np
    from src.train import train_and_evaluate_model
    # Create dummy data
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([2, 4, 6, 8])
    X_test = np.array([[5], [6]])
    y_test = np.array([10, 12])
    model = LinearRegression()
    # Call the function (should not raise exceptions)
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name="TestLR")
