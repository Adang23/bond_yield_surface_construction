
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

from bond_yield.data_processing.rating_converter import BaseRatingConverter

class ObjectiveStrategy(ABC):
    def __init__(self, bond_yield_df=None):
        self.bond_yield_df = bond_yield_df

    def calculate_slope_difference(self, rating_values):
        total_difference = 0
        for j, tenor in enumerate(self.bond_yield_df.columns):
            for i in range(len(rating_values) - 2):
                if self.is_valid_data(i, j, rating_values):
                    k_ij, k_i1j = self.compute_slopes(i, j, rating_values)
                    total_difference += self.calculate(k_ij, k_i1j)
        return total_difference

    def is_valid_data(self, i, j, rating_values):
        # Check if the required bond yield data points are non-NaN and the rating differences are non-zero
        y_values = self.bond_yield_df.iloc[i:i+3, j]
        r_values = rating_values[i:i+3]
        return all(pd.notna(y_values)) and all(r_values[k+1] != r_values[k] for k in range(2))

    def compute_slopes(self, i, j, rating_values):
        y_i, y_i1, y_i2 = self.bond_yield_df.iloc[i:i+3, j]
        r_i, r_i1, r_i2 = rating_values[i:i+3]
        k_ij = (y_i1 - y_i) / (r_i1 - r_i)
        k_i1j = (y_i2 - y_i1) / (r_i2 - r_i1)
        return k_ij, k_i1j

    @abstractmethod
    def calculate(self, k_ij, k_i1j):
        pass

class AbsoluteDifferenceStrategy(ObjectiveStrategy):
    def calculate(self, k_ij, k_i1j):
        return abs(k_ij - k_i1j)

class SquaredDifferenceStrategy(ObjectiveStrategy):
    def calculate(self, k_ij, k_i1j):
        return (k_ij - k_i1j) ** 2


from scipy.optimize import minimize, Bounds

class SlopeMinimizingRatingConverter(BaseRatingConverter):
    def __init__(self, initial_ratings, strategy):
        self.ratings = initial_ratings
        self.strategy = strategy

    def convert(self, rating):
        return self.ratings.get(rating, 0)

    def optimize_ratings(self, bounds=None, constraints=None):
        initial_values = list(self.ratings.values())
        rating_labels = list(self.ratings.keys())

        options = {'method': 'SLSQP'}
        if bounds:
            options['bounds'] = Bounds(*zip(*bounds))
        if constraints:
            options['constraints'] = constraints

        result = minimize(self.strategy.calculate_slope_difference, initial_values, **options)

        if result.success:
            optimized_values = result.x
            self.ratings = dict(zip(rating_labels, optimized_values))
        else:
            print("Optimization failed:", result.message)
