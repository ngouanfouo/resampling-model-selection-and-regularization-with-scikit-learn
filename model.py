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

# Step 3 - validation_set_curve
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def poly_model(degree):
    """Return a pipeline of PolynomialFeatures(degree) followed by LinearRegression."""
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


def validation_set_curve(X, y, feature, degrees, random_state):
    """
    Estimate validation MSE for each polynomial degree using a single 50/50 split.

    Parameters
    ----------
    X : DataFrame
    y : Series
    feature : str, the single feature column to use
    degrees : list of int
    random_state : int, seed for train_test_split

    Returns
    -------
    dict {degree: validation MSE rounded to 1 decimal}
    """
    X_feat = X[[feature]]
    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y, test_size=0.5, random_state=random_state
    )

    results = {}
    for degree in degrees:
        model = poly_model(degree)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        results[degree] = round(mse, 1)

    return results


def curve_spread(X, y, feature, degrees, seeds):
    """
    For each degree, compute the spread (max - min) of validation MSE across seeds.

    Parameters
    ----------
    X : DataFrame
    y : Series
    feature : str
    degrees : list of int
    seeds : list of int, random seeds

    Returns
    -------
    dict {degree: spread rounded to 1 decimal}
    """
    mse_dict = {degree: [] for degree in degrees}

    for seed in seeds:
        curve = validation_set_curve(X, y, feature, degrees, random_state=seed)
        for degree in degrees:
            mse_dict[degree].append(curve[degree])

    spread = {}
    for degree in degrees:
        spread[degree] = round(max(mse_dict[degree]) - min(mse_dict[degree]), 1)

    return spread

# Step 4 - cv_mse
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score


def cv_mse(estimator, X, y, k=5, random_state=0):
    """
    k-fold cross-validated MSE.

    Returns
    -------
    (mean, se) : tuple of floats
        mean of the fold MSEs, and the standard error
        = std(fold MSEs, ddof=1) / sqrt(k),
        both rounded to 2 decimals.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    neg_mse = cross_val_score(estimator, X, y, cv=kf, scoring='neg_mean_squared_error')
    mses = -neg_mse  # convert to positive MSEs

    mean = float(np.mean(mses))
    se = float(np.std(mses, ddof=1) / np.sqrt(k))

    return round(mean, 2), round(se, 2)


def loocv_mse(estimator, X, y):
    """
    Leave-one-out cross-validated MSE.

    Each fold holds out a single sample, so the cross-validated score
    with 'neg_mean_squared_error' gives the per-point squared error.
    The mean over all points is the LOOCV MSE.

    Returns
    -------
    float rounded to 2 decimals.
    """
    loo = LeaveOneOut()
    neg_mse = cross_val_score(estimator, X, y, cv=loo, scoring='neg_mean_squared_error')
    mses = -neg_mse  # squared errors, one per held-out point

    return round(float(np.mean(mses)), 2)

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

