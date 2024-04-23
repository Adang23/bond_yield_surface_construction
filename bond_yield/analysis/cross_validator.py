import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from bond_yield.interpolators.baseInterpolator import BaseInterpolator


class CrossValidator:
    def __init__(self, interpolator_class, date_dataframes, rating_converter_cls=None, tenor_converter_cls=None,
                 n_splits=5, random_state=42):
        """
        Initialize the cross-validation class with necessary components and configurations.

        Parameters:
        - interpolator_class (class): A class inheriting from BaseInterpolator.
        - date_dataframes (dict): Dictionary with dates as keys and bond yield DataFrames as values.
        - rating_converter_cls (class, optional): Class for converting ratings, may require DataFrame initialization.
        - tenor_converter_cls (class, optional): Class for converting tenors, may require DataFrame initialization.
        - n_splits (int): Number of folds for k-fold cross-validation.
        - random_state (int): Seed for reproducible random splits.
        """
        self.interpolator_class = interpolator_class
        self.date_dataframes = date_dataframes
        self.rating_converter_cls = rating_converter_cls
        self.tenor_converter_cls = tenor_converter_cls
        self.n_splits = n_splits
        self.random_state = random_state

    def cross_validate_single_dataframe(self, df):
        """
        Conduct k-fold cross-validation on a single DataFrame.

        Parameters:
        - df (DataFrame): DataFrame to perform cross-validation on.

        Returns:
        - tuple: (Mean squared error from cross-validation, number of samples).
        """
        if self.rating_converter_cls:
            rating_converter = self.rating_converter_cls(df)  # Assume the converter initializes with the df if needed
            df.index = df.index.map(rating_converter.get_rating_scale)

        if self.tenor_converter_cls:
            tenor_converter = self.tenor_converter_cls(df)  # Assume the converter initializes with the df if needed
            df.columns = df.columns.map(tenor_converter.get_tenor_scale)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
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

            interpolator = self.interpolator_class()
            interpolator.fit(X_train, y_train)
            y_pred = interpolator.interpolate(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_metrics.append(mse)

        return np.mean(mse_metrics), len(y)

    def perform_cross_validation(self):
        """
        Perform k-fold cross-validation across multiple DataFrames, computing a weighted average MSE.

        Returns:
        - float: Weighted mean squared error averaged over all dates.
        """
        total_mse = 0
        total_samples = 0

        for date, df in self.date_dataframes.items():
            mse, samples = self.cross_validate_single_dataframe(df)
            total_mse += mse * samples
            total_samples += samples

        if total_samples > 0:
            weighted_mse = total_mse / total_samples
            print(f"Overall Weighted Cross-Validation MSE: {weighted_mse:.4f}")
            return weighted_mse
        else:
            raise ValueError("No data available for cross-validation.")
