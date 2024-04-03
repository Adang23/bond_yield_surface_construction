import numpy as np

from sklearn.model_selection import KFold
from bond_yield.data_processing.loader import load_bond_yields
from bond_yield.interpolators.thin_plate_spline import ThinPlateSplineInterpolator
from bond_yield.interpolators.baseInterpolator import BaseInterpolator


# Ensure your BaseInterpolator has fit and interpolate abstract methods properly defined

# Function to calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def perform_cross_validation(interpolator_class, data_path, n_splits=5, random_state=42):
    """
    Perform k-fold cross-validation on a given interpolator.

    Parameters:
    - interpolator_class: A class that inherits from BaseInterpolator.
    - data_path: str, path to the dataset to be loaded.
    - n_splits: int, number of folds for the cross-validation.
    - random_state: int, seed for random number generator for reproducible results.
    """
    assert issubclass(interpolator_class, BaseInterpolator), "Interpolator must inherit from BaseInterpolator"

    dataframes_list = load_bond_yields(data_path)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    aggregated_rmse_metrics = []

    for df in dataframes_list:
        X = df[['X1', 'X2']].values  # Adjust column names as needed
        y = df['y'].values

        date_rmse_metrics = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            interpolator = interpolator_class()
            interpolator.fit(X_train, y_train)

            y_pred = interpolator.interpolate(X_test)
            rmse = calculate_rmse(y_test, y_pred)
            date_rmse_metrics.append(rmse)

        aggregated_rmse_metrics.append(np.mean(date_rmse_metrics))

    overall_rmse = np.mean(aggregated_rmse_metrics)
    print(f"Overall Cross-Validation RMSE: {overall_rmse:.4f}")


# Example usage
if __name__ == "__main__":
    perform_cross_validation(ThinPlateSplineInterpolator, './sample_historical_bond_yields.csv')
