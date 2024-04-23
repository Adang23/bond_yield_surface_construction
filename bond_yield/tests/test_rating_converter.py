
import pytest
import yaml
from bond_yield.data_processing.rating_converter import SimpleRatingConverter, YieldBasedRatingConverter
import pandas as pd

def test_simple_rating_converter_with_dict():
    rating_map = {'AAA': 1, 'BBB': 2, 'CCC': 3}
    converter = SimpleRatingConverter(rating_map=rating_map)
    assert converter.convert('AAA') == 1
    assert converter.convert('BBB') == 2
    with pytest.raises(ValueError):
        converter.convert('ZZZ')

def test_simple_rating_converter_with_yaml(tmpdir):
    rating_map = {'AAA': 1, 'BBB': 2, 'CCC': 3}
    yaml_file = tmpdir.join("rating_map.yaml")
    yaml_file.write(yaml.dump(rating_map))
    converter = SimpleRatingConverter(yaml_path=str(yaml_file))
    assert converter.convert('AAA') == 1
    assert converter.convert('CCC') == 3
    with pytest.raises(ValueError):
        converter.convert('ZZZ')

def test_simple_rating_converter_without_map():
    with pytest.raises(ValueError):
        SimpleRatingConverter()

def test_yield_based_rating_converter():
    data = {'AAA': [2.5, 2.6, 2.7], 'BBB': [3.5, 3.6, 3.7]}
    df = pd.DataFrame(data).transpose()
    converter = YieldBasedRatingConverter(bond_yield_df=df)
    assert converter.convert('AAA') == pytest.approx(2.6)
    assert converter.convert('BBB') == pytest.approx(3.6)
    assert converter.convert('ZZZ') == 0

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

