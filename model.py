"""
Resampling, Model Selection and Regularization with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import pandas as pd
from sklearn.datasets import load_diabetes


def load_data():
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data          # pandas DataFrame, shape (442, 10)
    y = diabetes.target        # pandas Series
    return X, y


def describe_data(X, y):
    return {
        'n': X.shape[0],
        'p': X.shape[1],
        'features': list(X.columns),
        'y_mean': round(float(y.mean()), 2),
    }

# Step 2 - train_test
from sklearn.model_selection import train_test_split


def train_test(X, y, test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test

# Step 3 - validation_set_curve (not yet solved)
# TODO: implement

# Step 4 - cv_mse (not yet solved)
# TODO: implement

# Step 5 - cv_spread_by_k (not yet solved)
# TODO: implement

# Step 6 - bootstrap_coefficients (not yet solved)
# TODO: implement

# Step 7 - stepwise_path (not yet solved)
# TODO: implement

# Step 8 - score_path (not yet solved)
# TODO: implement

# Step 9 - one_se_rule (not yet solved)
# TODO: implement

# Step 10 - ridge_path (not yet solved)
# TODO: implement

# Step 11 - cv_curve (not yet solved)
# TODO: implement

# Step 12 - lasso_path (not yet solved)
# TODO: implement

# Step 13 - pcr_model (not yet solved)
# TODO: implement

# Step 14 - pls_model (not yet solved)
# TODO: implement

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

