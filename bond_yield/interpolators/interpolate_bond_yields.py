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
        Fit the interpolator to available data and interpolate missing values.

        Parameters:
        - df: DataFrame, DataFrame where index is rating and columns are tenors.

        Returns:
        - df_filled: DataFrame, DataFrame with missing values filled.
        """
        # Mask to identify non-NaN entries (available data)
        df = df.apply(pd.to_numeric, errors='coerce')
        mask = ~df.isna()

        # Prepare data for fitting
        X_train = np.array([(i, j) for i in df.index for j in df.columns if mask.at[i, j]])
        y_train = np.array([df.at[i, j] for i in df.index for j in df.columns if mask.at[i, j]])

        X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)

        # Fit the interpolator
        self.interpolator.fit(X_train, y_train)

        # Prepare to predict the entire matrix
        X_pred = np.array([(i, j) for i in df.index for j in df.columns])
        X_pred = np.array(X_pred, dtype=float)

        # Predict using the interpolator
        predictions = self.interpolator.interpolate(X_pred)

        # Reshape predictions back to DataFrame format
        df_filled = pd.DataFrame(predictions.reshape(df.shape), index=df.index, columns=df.columns)

        return df_filled



if __name__ == "__main__":
    # Load the DataFrame
    csv_url = Path('../tests/single_date_yield.csv')
    df = pd.read_csv(csv_url, index_col=0)

    # Initialize interpolator (example using a dummy subclass of BaseInterpolator)
    interpolator = ThinPlateSplineInterpolator()  # You should replace this with your actual interpolator class

    # Initialize matrix interpolator
    matrix_interpolator = MatrixInterpolator(interpolator)

    # Fit and interpolate the matrix
    df_filled = matrix_interpolator.fit_interpolate(df)

    # Print or save the filled DataFrame
    print(df_filled)
