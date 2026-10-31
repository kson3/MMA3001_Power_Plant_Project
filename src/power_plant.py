"""Functions for Combined Cycle Power Plant output prediction."""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Input variables used to predict electrical power output
FEATURES = ["AT", "V", "AP", "RH"]


def load_and_clean_data(file_path):
    """Load the Excel dataset and remove duplicate rows."""
    data = pd.read_excel(file_path, sheet_name=0)
    clean_data = data.drop_duplicates().copy()
    return clean_data


def split_data(clean_data):
    """Split the dataset into 70% training, 15% validation and 15% test."""
    X = clean_data[FEATURES]
    y = clean_data["PE"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_models(X_train, y_train):
    """Train linear regression and depth-8 decision tree models."""
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    tree_model = DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    )
    tree_model.fit(X_train, y_train)

    return linear_model, tree_model


def evaluate_predictions(y_true, y_pred):
    """Calculate MAE, RMSE and R2 for predicted power output."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE (MW)": mae,
        "RMSE (MW)": rmse,
        "R2": r2
    }
