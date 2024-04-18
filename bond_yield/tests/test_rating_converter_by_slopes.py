import pytest
import numpy as np
import pandas as pd
from bond_yield.data_processing.rating_converter_by_slopes import SlopeMinimizingRatingConverter, AbsoluteDifferenceStrategy

def test_slope_minimizing_rating_converter_constraints():
    # Setup: Define a simple DataFrame with predictable results
    data = {
        365: [0.01, 0.02, 0.03],  # Simple progressive increase
        720: [0.015, 0.025, 0.035],  # Same trend, slightly different values
        1080: [0.02, 0.03, 0.04]  # Same trend, higher baseline
    }
    index = ['AAA', 'AA', 'A']
    df = pd.DataFrame(data, index=index)

    # Initial ratings somewhat arbitrary but within the required range
    initial_ratings = {'AAA': 0.2, 'AA': 0.22, 'A': 0.25}

    # Initialize the rating converter with the DataFrame and an absolute difference strategy
    strategy = AbsoluteDifferenceStrategy()
    converter = SlopeMinimizingRatingConverter(df, initial_ratings, strategy)

    # Run optimization
    converter.optimize_ratings()

    # Fetch the optimized ratings
    optimized_ratings = [converter.convert(rating) for rating in index]
    print(optimized_ratings)

    # Test conditions
    assert optimized_ratings[0] > 0.15, "First rating must be greater than 0.15"
    assert optimized_ratings[-1] < 1, "Last rating must be less than 1"
    assert optimized_ratings[-1] > optimized_ratings[-2] + 0.2, "Difference between last two ratings must be at least 0.2"
    for i in range(1, len(optimized_ratings)):
        assert optimized_ratings[i] > optimized_ratings[i-1] + 0.02, f"Rating {index[i]} should be at least 0.02 greater than {index[i-1]}"

