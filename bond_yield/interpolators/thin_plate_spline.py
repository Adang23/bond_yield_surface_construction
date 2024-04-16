import numpy as np
from .baseInterpolator import BaseInterpolator

class ThinPlateSplineInterpolator(BaseInterpolator):
    """ This is an interpolator for 2-D array only"""
    def __init__(self, lambda_val=0.1):
        """
        Initializes the ThinPlateSplineInterpolator with a regularization parameter.

        Parameters:
        - lambda_val: Regularization parameter for smoothing.
        """
        self.lambda_val = lambda_val
        self.w = None  # Non-affine coefficients
        self.b = None  # Affine coefficients
        self.X_training = None  # Training points
        self.N = None  # Matrix for affine part

    def compute_green_function(self, xr, xc):
        """
        Computes the Green's function for two points.
        """
        r = np.linalg.norm(xr - xc, axis=1)
        # Avoid log(0) by adding a small value
        #return 1 / np.pi * r ** 2 * np.log(r + 1e-10)
        return  r ** 2 * np.log(r + 1e-10)

    def construct_M(self, points):
        """
        Constructs the matrix M from the input points using the Green's function.
        """
        n = points.shape[0]
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = self.compute_green_function(points[i, np.newaxis], points[j, np.newaxis])
        return M

    def construct_N(self, points):
        """
        Constructs the matrix N for the affine part of the TPS model.
        """
        N = np.hstack((np.ones((points.shape[0], 1)), points))
        return N

    def fit(self, X, Y):
        """
        Fits the interpolator to the given points and values.
        """
        self.X_training = X.astype(float)
        y = Y.astype(float).reshape(-1, 1)
        M = self.construct_M(self.X_training)
        N = self.construct_N(self.X_training)
        self.N = N

        # Apply regularization
        M_lambda_I = M + self.lambda_val * np.eye(M.shape[0])

        # Solve for b and then for a
        b = np.linalg.inv(N.T @ np.linalg.inv(M_lambda_I) @ N) @ N.T @ np.linalg.inv(M_lambda_I) @ y
        w = np.linalg.inv(M_lambda_I) @ (y - N @ b)

        self.w = w
        self.b = b

    def interpolate(self, X):
        """
        Interpolates the value at a new point x using the fitted model.
        """
        X = np.atleast_2d(X).astype(float)
        green_values = np.array([self.compute_green_function(x_i, X) for x_i in self.X_training])
        non_affine_part = green_values.T @ self.w
        extended_X = np.hstack((np.ones((X.shape[0], 1)),X))
        affine_part = extended_X @ self.b
        return non_affine_part + affine_part[0]

