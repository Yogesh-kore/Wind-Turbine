import pytest
import os
import joblib
from src.ingest import data as ingest_data

def load_data_from_mongodb():
    """
    Loads data from MongoDB collection into a pandas DataFrame.
    """
    return ingest_data

from src.ingest import client, db, collection
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

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

def test_train_and_save_model():
    # Run training
    train_model()
    # Check if model file exists
    assert os.path.exists("models/random_forest.pkl")

def test_evaluate_model():
    # Load model
    model = joblib.load("models/random_forest.pkl")
    # Evaluate model (should not raise exceptions)
    evaluate_model(model, model_name="TestModel")

if __name__ == "__main__":
    pytest.main()
