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

# Step 9 - one_se_rule
import numpy as np


def one_se_rule(values, means, ses, prefer='smaller'):
    """
    Apply the one-standard-error rule.

    Find the index i_min of the smallest mean, set
        threshold = means[i_min] + ses[i_min],
    and among all values whose mean is at most the threshold, return
    the smallest value if prefer='smaller', else the largest.

    Parameters
    ----------
    values : list-like of candidate values (e.g. subset sizes or penalties)
    means  : list-like of mean CV scores, aligned with values
    ses    : list-like of standard errors, aligned with values
    prefer : 'smaller' or 'larger' — which direction is considered simpler

    Returns
    -------
    The chosen value.
    """
    values = list(values)
    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)

    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]

    candidates = [values[i] for i in range(len(values)) if means[i] <= threshold]

    return min(candidates) if prefer == 'smaller' else max(candidates)


def choose_subset(X, y, direction, cv):
    """
    Run the stepwise path, score it, and report both the raw best size
    and the one-SE-rule size (preferring smaller).

    Returns
    -------
    (size_min, size_1se, features_1se)
    """
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)

    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    features_1se = path[size_1se]

    return size_min, size_1se, features_1se

# Step 10 - ridge_path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ridge_model(alpha):
    """StandardScaler followed by Ridge(alpha)."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def ridge_path(X, y, alphas):
    """
    Fit a ridge pipeline for each alpha and stack the coefficient vectors.

    Returns
    -------
    coefs : ndarray of shape (len(alphas), p)
    """
    coefs = []
    for alpha in alphas:
        model = ridge_model(alpha)
        model.fit(X, y)
        # Last step in the pipeline is the Ridge estimator
        coefs.append(model.named_steps['ridge'].coef_)
    return np.asarray(coefs)


def coef_norms(path):
    """
    L2 norm of each row of the coefficient matrix, rounded to 2 decimals.
    """
    return np.round(np.linalg.norm(path, axis=1), 2)

# Step 11 - cv_curve
import numpy as np
from sklearn.model_selection import cross_val_score


def cv_curve(make_model, X, y, values, cv):
    """
    Cross-validate a model builder over a grid of values.

    Parameters
    ----------
    make_model : callable, value -> estimator
    X, y : data
    values : iterable of hyperparameter values
    cv : cross-validation splitter

    Returns
    -------
    (means, ses) : two lists rounded to 1 decimal
        means : mean cross-validated MSE per value
        ses   : standard error = std(ddof=1) / sqrt(n_folds) per value
    """
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        scores = cross_val_score(
            model, X, y, cv=cv, scoring='neg_mean_squared_error'
        )
        mses = -scores  # positive MSE per fold
        n_folds = len(mses)

        means.append(round(float(np.mean(mses)), 1))
        ses.append(round(float(np.std(mses, ddof=1) / np.sqrt(n_folds)), 1))

    return means, ses


def choose_penalty(make_model, X, y, values, cv):
    """
    Choose a penalty using the one-SE rule, preferring larger values.

    Returns
    -------
    (value_min, value_1se)
        value_min : the value with the smallest mean CV MSE
        value_1se : the largest value within one SE of the best
    """
    values = list(values)
    means, ses = cv_curve(make_model, X, y, values, cv)

    i_min = int(np.argmin(means))
    value_min = values[i_min]

    value_1se = one_se_rule(values, means, ses, prefer='larger')

    return value_min, value_1se

# Step 12 - lasso_path
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def lasso_model(alpha):
    """StandardScaler followed by Lasso(alpha)."""
    return make_pipeline(
        StandardScaler(),
        Lasso(alpha=alpha, max_iter=20000),
    )


def lasso_path(X, y, alphas):
    """
    Fit a lasso pipeline for each alpha and stack the coefficient vectors.

    Returns
    -------
    coefs : ndarray of shape (len(alphas), p)
    """
    coefs = []
    for alpha in alphas:
        model = lasso_model(alpha)
        model.fit(X, y)
        coefs.append(model.named_steps['lasso'].coef_)
    return np.asarray(coefs)


def nonzero_features(coef, names, tol=1e-8):
    """
    Return the names whose absolute coefficient exceeds tol, in input order.
    """
    coef = np.asarray(coef)
    return [name for name, c in zip(names, coef) if abs(c) > tol]


def lasso_cv(X, y, cv):
    """
    Fit StandardScaler + LassoCV and report the chosen alpha and selected features.

    Returns
    -------
    (alpha, selected)
        alpha    : LassoCV's alpha_ rounded to 4 decimals
        selected : list of feature names with nonzero coefficients
    """
    model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, max_iter=20000, random_state=0),
    )
    model.fit(X, y)

    lasso = model.named_steps['lassocv']
    alpha = round(float(lasso.alpha_), 4)
    selected = nonzero_features(lasso.coef_, list(X.columns))

    return alpha, selected

# Step 13 - pcr_model
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# cv_curve is assumed to be defined from the previous step:
# def cv_curve(make_model, X, y, values, cv): ...


def pcr_model(n_components):
    """StandardScaler -> PCA(n_components) -> LinearRegression."""
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        LinearRegression(),
    )


def explained_variance(X):
    """
    Fit StandardScaler then full PCA on X and return the cumulative
    explained-variance ratio, rounded to 3 decimals.
    """
    model = make_pipeline(StandardScaler(), PCA())
    model.fit(X)
    pca = model.named_steps['pca']
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return np.round(cumulative, 3)


def pcr_curve(X, y, cv):
    """
    Cross-validate PCR over 1..p components.

    Returns
    -------
    (components, means, ses)
        components : list [1, 2, ..., p]
        means      : mean CV MSE per component count, rounded to 1 decimal
        ses        : standard error per component count, rounded to 1 decimal
    """
    p = X.shape[1]
    components = list(range(1, p + 1))
    means, ses = cv_curve(pcr_model, X, y, components, cv)
    return components, means, ses

# Step 14 - pls_model
import numpy as np
from sklearn.cross_decomposition import PLSRegression

# cv_curve and one_se_rule are assumed defined from earlier steps.


def pls_model(n_components):
    """
    PLSRegression with internal scaling (no separate StandardScaler needed).
    """
    return PLSRegression(n_components=n_components, scale=True)


def pls_curve(X, y, cv):
    """
    Cross-validate PLS over 1..p components.

    Returns
    -------
    (components, means, ses)
        components : list [1, 2, ..., p]
        means      : mean CV MSE per component count, rounded to 1 decimal
        ses        : standard error per component count, rounded to 1 decimal
    """
    p = X.shape[1]
    components = list(range(1, p + 1))
    means, ses = cv_curve(pls_model, X, y, components, cv)
    return components, means, ses


def pls_predict(model, X):
    """
    Return PLSRegression predictions as a 1-D array.

    PLSRegression.predict returns a 2-D column matrix of shape (n_samples, 1);
    ravel() flattens it to the 1-D form the rest of the pipeline expects.
    """
    return np.asarray(model.predict(X)).ravel()


def best_components(components, means, ses):
    """
    Pick the best component count and the one-SE-rule component count.

    Returns
    -------
    (m_min, m_1se)
        m_min : component count with the smallest mean CV MSE
        m_1se : smallest component count within one SE of the best
                (one-SE rule with prefer='smaller')
    """
    components = list(components)
    means_arr = np.asarray(means, dtype=float)
    i_min = int(np.argmin(means_arr))
    m_min = components[i_min]

    m_1se = one_se_rule(components, means, ses, prefer='smaller')

    return m_min, m_1se

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

