
"""Combined Cycle Power Plant electrical power output prediction.

This module contains functions for data preparation, regression model
training, and prediction evaluation.

Inputs: AT, V, AP, RH
Output: PE (net electrical power output in MW)

The models used are Linear Regression and Decision Tree Regression.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = ["AT", "V", "AP", "RH"]


def load_and_clean_data(file_path):
    """Load the power plant dataset and remove duplicate rows.

    Args:
        file_path: Path to the original Excel dataset.

    Returns:
        A pandas DataFrame containing the cleaned dataset.
        The columns are AT, V, AP, RH, and PE.

    Notes:
        Only exact duplicate rows are removed.
        """
    data = pd.read_excel(file_path, sheet_name=0)
    clean_data = data.drop_duplicates().copy()
    return clean_data


def split_data(clean_data):
    """Split the cleaned dataset into training, validation, and test sets.

    Args:
        clean_data: DataFrame containing the four input features
            and the target variable PE.

    Returns:
        Six datasets in the following order:
        X_train, X_val, X_test, y_train, y_val, y_test.

    Notes:
        The split is approximately 70% training, 15% validation,
        and 15% test. A random_state of 42 ensures reproducibility.
        """
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
    """Train Linear Regression and Decision Tree Regression models.

    Args:
        X_train: Training input data containing AT, V, AP, and RH.
        y_train: Training target values of electrical power output
            in MW.

    Returns:
        Two fitted regression models:
        linear_model and tree_model.

    Notes:
        The decision tree uses max_depth=8 and random_state=42.
        The tree depth was selected using validation results.
        """
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    tree_model = DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    )
    tree_model.fit(X_train, y_train)

    return linear_model, tree_model


def evaluate_predictions(y_true, y_pred):
    """Evaluate predicted electrical power output.

    Args:
        y_true: Actual electrical power output values in MW.
        y_pred: Predicted electrical power output values in MW.

    Returns:
        A dictionary containing:
        MAE (MW): Mean Absolute Error.
        RMSE (MW): Root Mean Squared Error.
        R2: Coefficient of Determination (dimensionless).
        """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE (MW)": mae,
        "RMSE (MW)": rmse,
        "R2": r2
    }
