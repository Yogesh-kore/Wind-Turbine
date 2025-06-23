# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column='Wind_Speed', limit_rows=1000):
    """
    Preprocesses the wind turbine dataset.

    Parameters:
    - df (DataFrame): Raw data loaded from MongoDB or CSV.
    - target_column (str): Column to be predicted (default is 'Wind_Speed').
    - limit_rows (int): Number of rows to use for training/testing.

    Returns:
    - X_train_scaled, X_test_scaled, y_train, y_test
    """

    # Limit number of rows
    df = df.head(limit_rows)

    # Drop unnecessary or redundant columns
    drop_cols = [
        'TurbineName',              # Categorical, not encoded
        '_id',                      # MongoDB index (if exists)
        'Wind_Speed_winsorized',    # Processed versions of target
        'Wind_Speed_capped'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Drop rows with missing values
    df = df.dropna()

    # Ensure target exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
