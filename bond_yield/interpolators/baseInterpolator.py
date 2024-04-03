from abc import ABC, abstractmethod
import numpy as np

class BaseInterpolator(ABC):
    """
    Abstract base class for interpolation methods.

    This class provides the framework for fitting an interpolator to data
    and using it to predict new values. Subclasses must implement the
    fit and interpolate methods.
    """

    def __init__(self):
        """
        Initializer for the BaseInterpolator class.
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the interpolator to the data.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
          Training data, where n_samples is the number of samples
          and n_features is the number of features.
        - y: array-like, shape (n_samples,)
          Target values.
        """
        pass

    @abstractmethod
    def interpolate(self, X):
        """
        Predict using the interpolator.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
          Samples for which to predict the target values.

        Returns:
        - y: array, shape (n_samples,)
          Predicted target values.
        """
        pass
