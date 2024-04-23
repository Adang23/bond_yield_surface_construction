import pandas as pd

from bond_yield.data_processing.loader import load_bond_yields

def load_mdss_bond_yields(csv_path, yaml_path):
    date_dataframes = load_bond_yields(csv_path, yaml_path)

    return date_dataframes
