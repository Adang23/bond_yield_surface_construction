import pytest

import pandas as pd

from bond_yield.data_processing.rating_converter import SimpleRatingConverter, YieldBasedRatingConverter

@pytest.fixture
def rating_converter():
    return SimpleRatingConverter()

def test_known_rating_conversion(rating_converter):
    # Test known ratings
    ratings = {'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4, 'A+': 5, 'A': 6, 'A-': 7, 'BBB': 8, 'BB': 9, 'B': 10, 'CCC': 11, 'CC': 12, 'C': 13, 'D': 14}
    for rating, expected in ratings.items():
        assert rating_converter.convert(rating) == expected, f"Failed to convert {rating}"

def test_unknown_rating_conversion(rating_converter):
    # Test an unknown rating
    assert rating_converter.convert("XXX") == 0, "Failed to return 0 for an unknown rating"

def test_yield_based_rating_converter():
    # Create a sample DataFrame
    data = {
        365: [0.02, 0.022, None],  # Include None to test NaN handling
        720: [0.023, 0.025, 0.027],
        1080: [0.026, 0.028, 0.03]
    }
    index = ['AAA', 'AA', 'A']
    df = pd.DataFrame(data, index=index)

    # Initialize the rating converter with the DataFrame
    converter = YieldBasedRatingConverter(df)

    # Test known ratings
    assert converter.convert('AAA') == pytest.approx((0.02 + 0.023 + 0.026) / 3), "Should calculate the average yield for 'AAA'"
    assert converter.convert('AA') == pytest.approx((0.022 + 0.025 + 0.028) / 3), "Should calculate the average yield for 'AA'"
    # Test handling of NaN (None) in input data
    assert converter.convert('A') == pytest.approx((0.027 + 0.03) / 2), "Should calculate the average yield for 'A' ignoring NaN"
    # Test a rating not in the DataFrame
    assert converter.convert('BBB') == 0, "Should return 0 for ratings not found"

def test_yield_based_rating_converter_with_all_nan():
    # DataFrame where one rating only contains NaN values
    data = {
        365: [0.02, None, None],
        720: [0.023, None, None],
        1080: [0.026, None, None]
    }
    index = ['AAA', 'AA', 'A']
    df = pd.DataFrame(data, index=index)

    converter = YieldBasedRatingConverter(df)
    # Test when all values for a rating are NaN
    assert converter.convert('AA') == 0, "Should return 0 when all tenor values are NaN"
