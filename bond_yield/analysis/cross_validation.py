
from bond_yield.interpolators.baseInterpolator import BaseInterpolator

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def cross_validate_single_dataframe(df, interpolator_class, n_splits=5, random_state=42):
    """
    Conduct k-fold cross-validation on a single DataFrame.

    Parameters:
    - df: DataFrame to perform cross-validation on.
    - interpolator_class: Class inheriting from BaseInterpolator.
    - n_splits: Number of folds for k-fold cross-validation.
    - random_state: Seed for reproducible random splits.

    Returns:
    - tuple: (Mean squared error from cross-validation, number of samples).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    mse_metrics = []

    X, y = [], []
    for tenor in df.columns:
        for rating in df.index:
            X.append([rating, tenor])
            y.append(df.loc[rating, tenor])
    X = np.array(X)
    y = np.array(y)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        interpolator = interpolator_class()
        interpolator.fit(X_train, y_train)
        y_pred = interpolator.interpolate(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_metrics.append(mse)

    return np.mean(mse_metrics), len(y)


def perform_cross_validation(interpolator_class, date_dataframes, n_splits=5, random_state=42):
    """
    Perform k-fold cross-validation across multiple DataFrames, computing a weighted average MSE.

    Parameters:
    - interpolator_class: Class inheriting from BaseInterpolator.
    - date_dataframes: Dictionary of date keys and DataFrame values for cross-validation.
    - n_splits: Number of folds for k-fold cross-validation.
    - random_state: Seed for reproducible random splits.

    Returns:
    - float: Weighted mean squared error averaged over all dates.
    """
    total_mse = 0
    total_samples = 0

    for date, df in date_dataframes.items():
        mse, samples = cross_validate_single_dataframe(df, interpolator_class, n_splits, random_state)
        total_mse += mse * samples
        total_samples += samples

    if total_samples > 0:
        weighted_mse = total_mse / total_samples
        print(f"Overall Weighted Cross-Validation MSE: {weighted_mse:.4f}")
        return weighted_mse
    else:
        raise ValueError("No data available for cross-validation.")


