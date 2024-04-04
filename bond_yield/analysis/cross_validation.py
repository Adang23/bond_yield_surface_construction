import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from bond_yield.data_processing.loader import load_bond_yields
from bond_yield.interpolators.thin_plate_spline import ThinPlateSplineInterpolator
from bond_yield.interpolators.baseInterpolator import BaseInterpolator


# Function to calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def perform_cross_validation(interpolator_class, csv_path, yaml_path, n_splits=5, random_state=42):
    """
    Perform k-fold cross-validation on a given interpolator, adapted for data structure from loader.py.

    Parameters:
    - interpolator_class: Class inheriting from BaseInterpolator.
    - csv_path: Path to CSV file containing bond yields.
    - yaml_path: Path to YAML file containing rating scores.
    - n_splits: Number of folds for k-fold cross-validation.
    - random_state: Seed for reproducible random splits.
    """
    assert issubclass(interpolator_class, BaseInterpolator), "Interpolator must inherit from BaseInterpolator"

    date_dataframes = load_bond_yields(csv_path, yaml_path)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    aggregated_rmse_metrics = []

    # Iterate through each date's DataFrame in the dictionary
    for date, df in date_dataframes.items():
        # Flatten DataFrame to a suitable format (Rating, Tenor) -> Yield
        X, y = [], []
        for tenor in df.columns:
            for rating in df.index:
                X.append([rating, tenor])
                y.append(df.loc[rating, tenor])
        X = np.array(X)
        y = np.array(y)

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


# Example usage, assuming the correct paths are provided
if __name__ == "__main__":
    csv_file_path = './sample_historical_bond_yields.csv'
    yaml_file_path = './rating_scores.yaml'
    perform_cross_validation(ThinPlateSplineInterpolator, csv_file_path, yaml_file_path)
