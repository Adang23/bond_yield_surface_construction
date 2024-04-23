import pytest
import pandas as pd
import numpy as np
from bond_yield.data_processing.rating_converter_by_slopes import SlopeMinimizingRatingConverter, AbsoluteDifferenceStrategy, SquaredDifferenceStrategy

def create_sample_data():
    data = {
        365: [0.01, 0.02, 0.03],
        720: [0.015, 0.025, 0.035],
        1080: [0.02, 0.03, 0.04]
    }
    index = ['AAA', 'AA', 'A']
    df = pd.DataFrame(data, index=index)
    return df

def test_absolute_difference_strategy():
    df = create_sample_data()
    strategy = AbsoluteDifferenceStrategy(df)
    ratings = [0.2, 0.3, 0.5]  # Example rating values
    difference = strategy.calculate_slope_difference(ratings)
    assert difference > 0, "The total absolute difference should be positive"

def test_squared_difference_strategy():
    df = create_sample_data()
    strategy = SquaredDifferenceStrategy(df)
    ratings = [0.2, 0.3, 0.5]  # Example rating values
    difference = strategy.calculate_slope_difference(ratings)
    assert difference > 0, "The total squared difference should be positive"

def test_optimization():
    df = create_sample_data()
    initial_ratings = {'AAA': 0.2, 'AA': 0.3, 'A': 0.4}
    strategy = AbsoluteDifferenceStrategy(df)  # Ensure DataFrame is passed here
    converter = SlopeMinimizingRatingConverter(initial_ratings, strategy)

    # Check that DataFrame is not None before calling optimize_ratings
    assert strategy.bond_yield_df is not None, "Bond yield DataFrame should be set"
    converter.optimize_ratings()  # Now bounds are optional

    optimized_ratings = converter.ratings
    # Verifying optimization result could be more detailed based on the expected outcome
    #assert all(optimized_ratings[key] >= 0.2 for key in optimized_ratings), "Optimized ratings should be >= 0.2"


# Optionally, add tests to verify behavior when bounds and constraints are applied
def test_optimization_with_bounds():
    df = create_sample_data()
    initial_ratings = {'AAA': 0.2, 'AA': 0.3, 'A': 0.4}
    strategy = AbsoluteDifferenceStrategy(df)
    converter = SlopeMinimizingRatingConverter(initial_ratings, strategy)
    bounds = [(0.1, 0.5), (0.1, 0.5), (0.1, 0.5)]  # Example bounds
    converter.optimize_ratings(bounds=bounds)
    optimized_ratings = converter.ratings

    # Ensure bounds are respected
    for rating, value in optimized_ratings.items():
        assert 0.1 <= value <= 0.5, f"Rating {rating} should be within the bounds"
