"""
Resampling, Model Selection and Regularization with Scikit-Learn scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Resampling, Model Selection and Regularization with scikit-learn (ISL, chapters 5 and 6).

Story: load the diabetes data and hold out a test set; watch the validation-set
estimate move with the split, then replace it with k-fold and leave-one-out
cross-validation; get coefficient standard errors from the bootstrap and check them
against the formula; run forward stepwise selection and pick a size with the
one-standard-error rule; trace ridge and lasso paths and pick penalties the same
way; build PCR and PLS pipelines and pick component counts; finally open the test
set once and compare every method in one table.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def main() -> None:
    X, y = load_data()
    info = describe_data(X, y)
    print(f"diabetes data: n={info['n']} p={info['p']} features={info['features']} mean progression {info['y_mean']}")
    X_tr, X_te, y_tr, y_te = train_test(X, y)
    cv = KFold(5, shuffle=True, random_state=0)
    print(f"train {len(X_tr)} / test {len(X_te)} (the test set is opened once, at the end)")

    # ---- 1. One split lies ----
    degrees = [1, 2, 3, 4]
    curves = [validation_set_curve(X_tr, y_tr, 'bmi', degrees, s) for s in range(3)]
    print("\nvalidation-set MSE of poly(bmi) by degree, three different 50/50 splits:")
    for s, c in enumerate(curves):
        print(f"  seed {s}: " + "  ".join(f"d{d}={c[d]:.0f}" for d in degrees) + f"  -> best degree {min(c, key=c.get)}")
    spread = curve_spread(X_tr, y_tr, 'bmi', degrees, seeds=range(8))
    print("  range across 8 splits: " + "  ".join(f"d{d}={spread[d]:.0f}" for d in degrees))

    # ---- 2. Cross-validation and the bootstrap ----
    r = compare_with_loocv(LinearRegression(), X_tr, y_tr, ks=[2, 5, 10], seeds=range(6))
    print("\nfull linear model, CV estimate of MSE (mean over 6 fold assignments, sd across them):")
    for k, (m, sd) in r['kfold'].items():
        print(f"  {k:2d}-fold: {m:.0f} (sd {sd:.0f})")
    print(f"  LOOCV : {r['loocv']:.0f} (no randomness)")
    coefs = bootstrap_coefficients(X_tr, y_tr, n_boot=300)
    se_b, se_f = bootstrap_se(coefs), ols_standard_errors(X_tr, y_tr)
    print("bootstrap vs formula standard errors: " + ", ".join(f"{n} {b:.0f}/{f:.0f}" for n, b, f in zip(X.columns, se_b, se_f)))

    # ---- 3. Subset selection ----
    size_min, size_1se, feats = choose_subset(X_tr, y_tr, 'forward', cv)
    print(f"\nforward stepwise: CV minimum at {size_min} features; one-SE rule keeps {size_1se}: {feats}")

    # ---- 4. Ridge and the lasso ----
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    means, ses = cv_curve(ridge_model, X_tr, y_tr, alphas, cv)
    a_min, a_1se = choose_penalty(ridge_model, X_tr, y_tr, alphas, cv)
    print("\nridge CV curve: " + "  ".join(f"a={a:g}:{m:.0f}" for a, m in zip(alphas, means)))
    print(f"  minimum at alpha={a_min:g}, one-SE rule picks alpha={a_1se:g}; coefficient norms along the path: {coef_norms(ridge_path(X_tr, y_tr, alphas)).tolist()}")
    lasso_alphas = [0.01, 0.5, 2.0, 5.0, 20.0]
    counts = [len(nonzero_features(row, list(X.columns))) for row in lasso_path(X_tr, y_tr, lasso_alphas)]
    alpha_l, selected = lasso_cv(X_tr, y_tr, cv)
    print("lasso nonzero coefficients along the path: " + "  ".join(f"a={a:g}:{c}" for a, c in zip(lasso_alphas, counts)))
    print(f"  LassoCV alpha={alpha_l}: keeps {len(selected)} features {selected}")

    # ---- 5. PCR and PLS ----
    ev = explained_variance(X_tr)
    comps, pm, ps = pcr_curve(X_tr, y_tr, cv)
    _, lm, ls = pls_curve(X_tr, y_tr, cv)
    print("\ncomponents:      " + " ".join(f"{c:5d}" for c in comps))
    print("cum. variance:   " + " ".join(f"{v:5.2f}" for v in ev))
    print("PCR CV MSE:      " + " ".join(f"{m:5.0f}" for m in pm))
    print("PLS CV MSE:      " + " ".join(f"{m:5.0f}" for m in lm))
    print(f"one-SE component counts: PCR {best_components(comps, pm, ps)[1]}, PLS {best_components(comps, lm, ls)[1]}")

    # ---- 6. One honest table ----
    models = fit_all(X_tr, y_tr, cv, alphas)
    report = test_report(models, X_te, y_te)
    print("\ntest set, opened once:")
    for line in format_table(report):
        print("  " + line)
    print(f"best on the test set: {best_method(report)}; the regularized models sit within a few units of each other, "
          f"and the sparsest one pays a little accuracy for using only {report['forward_1se']['n_features']} features")


if __name__ == "__main__":
    main()

