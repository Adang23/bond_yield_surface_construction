import numpy as np
from bond_yield.interpolators.thin_plate_spline import ThinPlateSplineInterpolator

def test_2d_fit_and_interpolate():
    # Generate a grid of points in 2D space
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))  # Z is the value to predict, based on the (X, Y) positions

    # Flatten the grids for fitting into the interpolator
    points = np.vstack((X.ravel(), Y.ravel())).T
    values = Z.ravel()

    # Initialize the ThinPlateSplineInterpolator and fit it
    tps_interpolator = ThinPlateSplineInterpolator()
    tps_interpolator.fit(points, values)

    # Interpolate at new points (for simplicity, use the same points)
    Z_predicted = tps_interpolator.interpolate(points)

    # Reshape for comparison
    Z_predicted_reshaped = Z_predicted.reshape(X.shape)

    # Assert that the predicted values are close to the actual function values
    assert np.allclose(Z_predicted_reshaped, Z, atol=2e-1), "The interpolated values should closely match the actual function values."

