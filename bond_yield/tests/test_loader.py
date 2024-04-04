# tests/test_loader.py

# tests/test_loader.py
import pathlib
from bond_yield.data_processing.loader import load_bond_yields


def test_load_bond_yields():
    # Define the base directory (this could be your project's root directory)
    base_dir = pathlib.Path(__file__).parent.parent

    # Define paths to the data and YAML files using the base directory
    csv_path = base_dir / 'tests' / 'sample_historical_bond_yields.csv'
    yaml_path = base_dir / 'tests' / 'rating_scores.yaml'

    # Convert Path objects to strings if the called functions expect string paths
    df_bond_yields = load_bond_yields(str(csv_path), str(yaml_path))

    print(df_bond_yields)


if __name__ == "__main__":
    test_load_bond_yields()
