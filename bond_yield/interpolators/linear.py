import numpy as np
from scipy.interpolate import interp1d
from bond_yield.interpolators.baseInterpolator import BaseInterpolator


class LinearInterpolator(BaseInterpolator):
    def __init__(self):
        super().__init__()
        self.interpolators = None

    def fit(self, X, y):
        """
        Fit the LinearInterpolator model to the data.

        Parameters:
        - X: 2D array-like, shape (n_samples, 2)
          Training data, where n_samples is the number of samples.
          The interpolation is performed based on the second coordinate.
        - y: array-like, shape (n_samples,)
          Target values.
        """
        # Ensure data is sorted by the second coordinate for each unique first coordinate
        unique_coords = np.unique(X[:, 0])
        self.interpolators = {}

        for coord in unique_coords:
            indices = np.where(X[:, 0] == coord)
            sorted_indices = np.argsort(X[indices][:, 1])
            x_vals = X[indices][:, 1][sorted_indices]
            y_vals = y[indices][sorted_indices]

            # Create linear interpolator with flat extrapolation
            self.interpolators[coord] = interp1d(x_vals, y_vals, kind='linear', bounds_error=False,
                                                 fill_value="extrapolate")

    def interpolate(self, X):
        """
        Predict using the linear interpolator.

        Parameters:
        - X: 2D array-like, shape (n_samples, 2)
          Samples for which to predict the target values, based on the second coordinate.

        Returns:
        - y: array, shape (n_samples,)
          Predicted target values.
        """
        y_pred = np.zeros(X.shape[0])

        for i, (x1, x2) in enumerate(X):
            if x1 in self.interpolators:
                y_pred[i] = self.interpolators[x1](x2)
            else:
                # Handle case where x1 is not in the interpolators (e.g., use nearest or zero)
                y_pred[i] = 0  # This could be replaced with more sophisticated handling

        return y_pred
