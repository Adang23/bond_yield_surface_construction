import numpy as np


class ThinPlateSplineInterpolator:
    def __init__(self, lambda_val=0.1):
        """
        Initializes the ThinPlateSplineInterpolator with a regularization parameter.

        Parameters:
        - lambda_val: Regularization parameter for smoothing.
        """
        self.lambda_val = lambda_val
        self.a = None  # Non-affine coefficients
        self.b = None  # Affine coefficients
        self.xi = None  # Training points
        self.N = None  # Matrix for affine part

    def compute_green_function(self, xr, xc):
        """
        Computes the Green's function for two points.
        """
        r = np.linalg.norm(xr - xc, axis=1)
        # Avoid log(0) by adding a small value
        return 1 / np.pi * r ** 2 * np.log(r + 1e-10)

    def construct_M(self, points):
        """
        Constructs the matrix M from the input points using the Green's function.
        """
        n = points.shape[0]
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i, j] = 0
                else:
                    M[i, j] = self.compute_green_function(points[i, np.newaxis], points[j, np.newaxis])
        return M

    def construct_N(self, points):
        """
        Constructs the matrix N for the affine part of the TPS model.
        """
        N = np.hstack((np.ones((points.shape[0], 1)), points))
        return N

    def fit(self, xi, yi):
        """
        Fits the interpolator to the given points and values.
        """
        self.xi = xi
        y = yi.reshape(-1, 1)
        M = self.construct_M(xi)
        N = self.construct_N(xi)
        self.N = N

        # Apply regularization
        M_lambda_I = M + self.lambda_val * np.eye(M.shape[0])

        # Solve for b and then for a
        b = np.linalg.inv(N.T @ np.linalg.inv(M_lambda_I) @ N) @ N.T @ np.linalg.inv(M_lambda_I) @ y
        a = np.linalg.inv(M_lambda_I) @ (y - N @ b)

        self.a = a
        self.b = b

    def interpolate(self, x):
        """
        Interpolates the value at a new point x using the fitted model.
        """
        x = np.atleast_2d(x)
        green_values = np.array([self.compute_green_function(x_i, x) for x_i in self.xi])
        non_affine_part = green_values.T @ self.a
        affine_part = self.N @ self.b
        return non_affine_part + affine_part[0]

