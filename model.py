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

# Step 5 - cv_spread_by_k
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

# Assuming cv_mse and loocv_mse are defined earlier (from the previous step)
# def cv_mse(estimator, X, y, k=5, random_state=0): ...
# def loocv_mse(estimator, X, y): ...


def cv_spread_by_k(estimator, X, y, ks, seeds):
    """
    For each k, compute the k-fold MSE estimate across multiple seeds
    and report the mean and standard deviation (ddof=1) over seeds.

    Returns
    -------
    dict {k: (mean_over_seeds, std_over_seeds)} both rounded to 1 decimal.
    """
    result = {}
    for k in ks:
        estimates = []
        for seed in seeds:
            mean_mse, _ = cv_mse(estimator, X, y, k=k, random_state=seed)
            estimates.append(mean_mse)
        result[k] = (
            round(float(np.mean(estimates)), 1),
            round(float(np.std(estimates, ddof=1)), 1),
        )
    return result


def compare_with_loocv(estimator, X, y, ks, seeds):
    """
    Compare LOOCV MSE against the k-fold spread across seeds.

    Returns
    -------
    dict with 'loocv' (float) and 'kfold' (the spread dict from cv_spread_by_k).
    """
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds),
    }

# Step 6 - bootstrap_coefficients
import numpy as np
from sklearn.linear_model import LinearRegression


def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    """
    Fit LinearRegression on n_boot bootstrap resamples of the rows.

    Returns
    -------
    coefs : ndarray of shape (n_boot, p)
        Each row is the coefficient vector from one bootstrap replicate.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n, p = X_arr.shape

    rng = np.random.default_rng(random_state)
    coefs = np.empty((n_boot, p), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = LinearRegression()
        model.fit(X_arr[idx], y_arr[idx])
        coefs[b] = model.coef_

    return coefs


def bootstrap_se(coefs):
    """
    Per-coefficient bootstrap standard error (std over replicates, ddof=1).

    Returns
    -------
    ndarray of shape (p,) rounded to 2 decimals.
    """
    coefs = np.asarray(coefs, dtype=float)
    se = np.std(coefs, axis=0, ddof=1)
    return np.round(se, 2)


def ols_standard_errors(X, y):
    """
    Textbook OLS standard errors for the slope coefficients.

    Uses A = [1, X], sigma2 = RSS / (n - p - 1),
    SE = sqrt(sigma2 * diag((A^T A)^-1)), dropping the intercept entry.

    Returns
    -------
    ndarray of shape (p,) rounded to 2 decimals.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n, p = X_arr.shape

    # Design matrix with intercept column
    A = np.hstack([np.ones((n, 1)), X_arr])

    # Fit via least squares
    beta, *_ = np.linalg.lstsq(A, y_arr, rcond=None)

    # Residuals and residual variance
    residuals = y_arr - A @ beta
    rss = float(residuals @ residuals)
    sigma2 = rss / (n - p - 1)

    # Covariance of beta_hat: sigma2 * (A^T A)^-1
    AtA_inv = np.linalg.inv(A.T @ A)
    var_beta = sigma2 * np.diag(AtA_inv)
    se = np.sqrt(var_beta)

    # Drop the intercept entry
    return np.round(se[1:], 2)

# Step 7 - stepwise_path
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression


def select_features(X, y, k, direction, cv):
    """
    Run SequentialFeatureSelector with LinearRegression and negative MSE scoring.

    Parameters
    ----------
    X : DataFrame
    y : Series
    k : int, number of features to select
    direction : 'forward' or 'backward'
    cv : cross-validation splitter

    Returns
    -------
    list of selected column names (in X.columns order).
    """
    selector = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=k,
        direction=direction,
        cv=cv,
        scoring='neg_mean_squared_error',
    )
    selector.fit(X, y)
    support = selector.get_support()
    return list(X.columns[support])


def stepwise_path(X, y, direction, cv):
    """
    Build the stepwise selection path for k = 1, ..., p.

    SequentialFeatureSelector cannot select all p features, so k = p
    is added manually as the full column list.

    Returns
    -------
    dict {k: list of feature names}
    """
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)
    path[p] = list(X.columns)
    return path

# Step 8 - score_path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def score_path(X, y, path, cv):
    """
    Cross-validate LinearRegression on each subset along the path.

    Parameters
    ----------
    X : DataFrame
    y : Series
    path : dict {k: list of feature names}
    cv : cross-validation splitter (e.g. KFold)

    Returns
    -------
    (sizes, means, ses) : three lists in increasing k
        sizes : list of subset sizes
        means : mean cross-validated MSE per size, rounded to 1 decimal
        ses   : standard error = std(ddof=1) / sqrt(n_folds), rounded to 1 decimal
    """
    sizes = sorted(path.keys())
    means = []
    ses = []

    for k in sizes:
        cols = path[k]
        scores = cross_val_score(
            LinearRegression(),
            X[cols],
            y,
            cv=cv,
            scoring='neg_mean_squared_error',
        )
        mses = -scores  # positive MSE per fold
        n_folds = len(mses)

        means.append(round(float(np.mean(mses)), 1))
        ses.append(round(float(np.std(mses, ddof=1) / np.sqrt(n_folds)), 1))

    return sizes, means, ses


def best_size(sizes, means):
    """
    Return the subset size with the smallest mean cross-validated MSE.
    """
    i = int(np.argmin(means))
    return sizes[i]

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

