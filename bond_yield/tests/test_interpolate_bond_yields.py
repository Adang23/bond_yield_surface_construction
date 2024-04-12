import numpy as np
import pandas as pd
from bond_yield.interpolators.interpolate_bond_yields import MatrixInterpolator
from bond_yield.interpolators.thin_plate_spline import ThinPlateSplineInterpolator

def test_interpolate_complete_matrix():
    # Create a complete matrix without NaNs to test basic functionality
    data = {
        365: [0.02, 0.022, 0.025],
        720: [0.023, 0.025, 0.027],
        1080: [0.026, 0.028, 0.03]
    }
    df = pd.DataFrame(data, index=[1, 2, 3])

    interpolator = ThinPlateSplineInterpolator()  # Or another interpolator that you have implemented
    matrix_interpolator = MatrixInterpolator(interpolator)

    result_df = matrix_interpolator.fit_interpolate(df)

    # Check if the result DataFrame matches the original DataFrame
    pd.testing.assert_frame_equal(df, result_df, atol=0.1)

def test_interpolate_missing_values():
    # Create a matrix with missing values
    data = {
        365: [np.nan, 0.022, 0.025],
        720: [0.023, np.nan, 0.027],
        1080: [0.026, 0.028, np.nan]
    }
    df = pd.DataFrame(data, index=[1, 2, 3])

    interpolator = ThinPlateSplineInterpolator()  # Ensure this interpolator can handle NaN values
    matrix_interpolator = MatrixInterpolator(interpolator)

    result_df = matrix_interpolator.fit_interpolate(df)

    # Check that no cell in the result DataFrame is NaN
    assert not result_df.isnull().values.any(), "Resulting DataFrame should not contain NaNs"
