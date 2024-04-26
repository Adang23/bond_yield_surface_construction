from abc import ABC, abstractmethod
import pandas as pd
from scipy.optimize import minimize

class BaseTenorConverter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def scale_tenor(self, tenor):
        """
        Scale a tenor value according to some rule defined in the subclass.

        Parameters:
        tenor (float): The original tenor value.

        Returns:
        float: The scaled tenor value.
        """
        pass

    @abstractmethod
    def get_tenor_scales(self):
        pass


class IdentityTenorConverter(BaseTenorConverter):
    def scale_tenor(self, tenor):
        """
        Return the tenor value unchanged.

        Parameters:
        tenor (float): The original tenor value.

        Returns:
        float: The same tenor value as input, unchanged.
        """
        return tenor


class TenorBasedObjectiveStrategy(ABC):
    def __init__(self, bond_yield_df=None):
        self.bond_yield_df = bond_yield_df

    def calculate_slope_difference(self, tenor_values):
        total_difference = 0
        for i, rating in enumerate(self.bond_yield_df.index):
            for j in range(len(tenor_values) - 2):
                if self.is_valid_data(i, j, tenor_values):
                    k_j, k_j1 = self.compute_slopes(i, j, tenor_values)
                    total_difference += self.calculate(k_j, k_j1)
        return total_difference

    def is_valid_data(self, i, j, tenor_values):
        # Check if the required bond yield data points are non-NaN and the tenor differences are non-zero
        y_values = self.bond_yield_df.iloc[i, j:j+3]
        t_values = tenor_values[j:j+3]
        return all(pd.notna(y_values)) and all(t_values[k+1] != t_values[k] for k in range(2))

    def compute_slopes(self, i, j, tenor_values):
        y_j, y_j1, y_j2 = self.bond_yield_df.iloc[i, j:j+3]
        t_j, t_j1, t_j2 = tenor_values[j:j+3]
        k_j = (y_j1 - y_j) / (t_j1 - t_j)
        k_j1 = (y_j2 - y_j1) / (t_j2 - t_j1)
        return k_j, k_j1

    @abstractmethod
    def calculate(self, k_j, k_j1):
        pass

# Implement the specific strategies for absolute and squared differences
class TenorAbsoluteDifferenceStrategy(TenorBasedObjectiveStrategy):
    def calculate(self, k_j, k_j1):
        return abs(k_j - k_j1)

class TenorSquaredDifferenceStrategy(TenorBasedObjectiveStrategy):
    def calculate(self, k_j, k_j1):
        return (k_j - k_j1) ** 2

class TenorMinimizingTenorConverter(BaseTenorConverter):
    def __init__(self, initial_tenor_values, strategy):
        self.tenor_values = initial_tenor_values
        self.strategy = strategy

    def convert(self, tenor):
        return self.tenor_values.get(tenor, 0)

    def optimize_tenors(self, bounds=None, constraints=None):
        initial_values = list(self.tenor_values.values())
        tenor_labels = list(self.tenor_values.keys())

        options = {'method': 'SLSQP'}
        if bounds:
            options['bounds'] = bounds
        if constraints:
            options['constraints'] = constraints

        result = minimize(self.strategy.calculate_slope_difference, initial_values, **options)

        if result.success:
            optimized_values = result.x
            self.tenor_values = dict(zip(tenor_labels, optimized_values))
        else:
            print("Optimization failed:", result.message)

    def get_tenor_scale(self):
        return self.tenor_values


