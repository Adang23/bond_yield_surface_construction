import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class BaseCrossValidator:
    def __init__(self, interpolator_class, date_dataframes, n_splits=5, random_state=42):
        self.interpolator_class = interpolator_class
        self.date_dataframes = date_dataframes
        self.n_splits = n_splits
        self.random_state = random_state

    def prepare_dataframes(self):
        # Default implementation does nothing.
        pass

    def cross_validate_single_dataframe(self, df):
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
        self.prepare_dataframes()

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

class CrossValidator(BaseCrossValidator):
    def prepare_dataframes(self):
        # Implement specific logic if necessary.
        pass

class CrossValidatorBySlope(BaseCrossValidator):
    def __init__(self, interpolator_class, date_dataframes, rating_converter, n_splits=5, random_state=42):
        super().__init__(interpolator_class, date_dataframes, n_splits, random_state)
        self.rating_converter = rating_converter

    def prepare_dataframes(self):
        for date, df in self.date_dataframes.items():
            self.rating_converter.strategy.bond_yield_df = df
            self.rating_converter.optimize_ratings()
            scale = self.rating_converter.get_rating_scale()
            df.index = df.index.map(lambda x: scale.get(x, 0))

