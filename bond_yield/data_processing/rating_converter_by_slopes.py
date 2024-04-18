
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

from bond_yield.data_processing.rating_converter import BaseRatingConverter

class ObjectiveStrategy(ABC):
    @abstractmethod
    def calculate(self, slope_difference):
        pass

class AbsoluteDifferenceStrategy(ObjectiveStrategy):
    def calculate(self, slope_difference):
        return abs(slope_difference)

class SquaredDifferenceStrategy(ObjectiveStrategy):
    def calculate(self, slope_difference):
        return slope_difference ** 2


from scipy.optimize import minimize, Bounds


class SlopeMinimizingRatingConverter(BaseRatingConverter):
    def __init__(self, bond_yield_df, initial_ratings, strategy):
        self.bond_yield_df = bond_yield_df
        self.ratings = initial_ratings  # dict mapping rating labels to initial numerical values
        self.strategy = strategy

    def convert(self, rating):
        """
        Retrieve the optimized numerical rating for a given label.
        Assumes `optimize_ratings` has been successfully called to optimize and update `self.ratings`.
        Parameters:
        rating (str): The rating label.
        Returns:
        float: The numerical rating value.
        """
        return self.ratings.get(rating, 0)

    def optimize_ratings(self, bounds = []):
        initial_values = list(self.ratings.values())
        rating_labels = list(self.ratings.keys())


        # Constraint function for adjacent elements
        def adj_constraint(rating_values):
            return [rating_values[i] - rating_values[i - 1] - 0.02 for i in range(1, len(rating_values))]

        # Additional constraints for adjacent ratings differences and last two elements
        constraints = {
            'type': 'ineq',  # Inequality means that it is required to be non-negative
            'fun': adj_constraint
        }

        def objective(rating_values):
            total_slope_difference = 0
            for j, tenor in enumerate(self.bond_yield_df.columns):
                for i in range(len(rating_values) - 2):
                    y_i2 = self.bond_yield_df.iloc[i + 2, j]
                    y_i1 = self.bond_yield_df.iloc[i + 1, j]
                    r_i2 = rating_values[i + 2]
                    r_i1 = rating_values[i + 1]

                    if pd.notna(y_i2) and pd.notna(y_i1) and r_i2 != r_i1:
                        k_ij = (y_i2 - y_i1) / (r_i2 - r_i1)
                        total_slope_difference += self.strategy.calculate(k_ij)

            return total_slope_difference

        # Optimization with bounds and constraints
        result = minimize(objective, initial_values, method='SLSQP', bounds=Bounds(*zip(*bounds)),
                          constraints=constraints)

        if result.success:
            optimized_values = result.x
            self.ratings = dict(zip(rating_labels, optimized_values))
        else:
            print("Optimization failed:", result.message)
