# Unit tests for the power plant prediction project

import numpy as np
import pytest

from src.power_plant import (
    load_and_clean_data,
    split_data,
    train_models,
    evaluate_predictions
)


# Test 1 - Data loading and cleaning
def test_load_and_clean_data():

    data = load_and_clean_data("data/Folds5x2_pp.xlsx")

    assert len(data) == 9527
    assert data.duplicated().sum() == 0
    assert data.isnull().sum().sum() == 0


# Test 2 - Data splitting
def test_split_data():

    data = load_and_clean_data("data/Folds5x2_pp.xlsx")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    assert len(X_train) == 6668
    assert len(X_val) == 1429
    assert len(X_test) == 1430

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    # Check that the three sets do not overlap
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)


# Test 3 - Model training
def test_train_models():

    data = load_and_clean_data("data/Folds5x2_pp.xlsx")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    linear_model, tree_model = train_models(X_train, y_train)

    assert len(linear_model.predict(X_val)) == len(X_val)
    assert len(tree_model.predict(X_val)) == len(X_val)
    assert tree_model.max_depth == 8


# Test 4 - Evaluation metrics
def test_evaluate_predictions():

    actual = np.array([430, 450])
    predicted = np.array([432, 446])

    results = evaluate_predictions(actual, predicted)

    assert results["MAE (MW)"] == pytest.approx(3.0)
    assert results["RMSE (MW)"] == pytest.approx(np.sqrt(10))
    assert results["R2"] == pytest.approx(0.9)
