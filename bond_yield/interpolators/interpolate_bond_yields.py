from pathlib import Path
import pandas as pd
import numpy as np
from bond_yield.interpolators.baseInterpolator import BaseInterpolator
from bond_yield.interpolators.thin_plate_spline import ThinPlateSplineInterpolator

class MatrixInterpolator:
    def __init__(self, interpolator):
        assert issubclass(type(interpolator), BaseInterpolator), "Interpolator must be a subclass of BaseInterpolator"
        self.interpolator = interpolator

    def fit_interpolate(self, df):
        """
        Fit the interpolator to available data and interpolate missing values only.

        Parameters:
        - df: DataFrame, DataFrame where index is rating and columns are tenors.

        Returns:
        - df_filled: DataFrame, DataFrame with missing values filled, original data retained.
        """
        df.index = df.index.astype(str)
        mask = ~df.isna()  # Mask of available (non-NaN) data

        # Prepare data for fitting
        X_train = np.array([(i, j) for i in df.index for j in df.columns if mask.at[i, j]])
        y_train = np.array([df.at[i, j] for i in df.index for j in df.columns if mask.at[i, j]])

        # Fit the interpolator to the available data
        self.interpolator.fit(X_train, y_train)

        # Prepare to predict only the missing values
        X_pred = np.array([(i, j) for i in df.index for j in df.columns if not mask.at[i, j]])

        if len(X_pred) > 0:  # Check if there are any missing values to predict
            predictions = self.interpolator.interpolate(X_pred)

            # Fill only the missing values in the DataFrame
            for (i, j), pred in zip(X_pred, predictions):
                df.at[i, j] = pred

        return df

if __name__ == "__main__":
    # Load the DataFrame
    csv_url = Path('../tests/single_date_yield.csv')
    df = pd.read_csv(csv_url, index_col=0)

    # Initialize interpolator (example using a dummy subclass of BaseInterpolator)
    interpolator = ThinPlateSplineInterpolator()  # Replace with your actual interpolator class

    # Initialize matrix interpolator
    matrix_interpolator = MatrixInterpolator(interpolator)

    # Fit and interpolate the matrix
    df_filled = matrix_interpolator.fit_interpolate(df)

    # Print or save the filled DataFrame
    print(df_filled)
