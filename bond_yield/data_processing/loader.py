import pandas as pd
import yaml
from pathlib import Path


def load_rating_scores(yaml_path):
    with open(yaml_path, 'r') as file:
        rating_scores = yaml.safe_load(file)['ratings']
    return rating_scores


def load_bond_yields(csv_path, yaml_path):
    rating_scores = load_rating_scores(yaml_path)
    df = pd.read_csv(csv_path, index_col=0)  # Assuming first column is the date

    date_dataframes = {}
    for date in df.index.unique():
        data_for_date = []
        for col in df.columns:
            rating, tenor = col.split("::")
            numeric_rating = rating_scores.get(rating, 0)  # Fallback to 0 if rating not found
            tenor_days = int(tenor)
            yield_value = df.at[date, col]
            data_for_date.append((numeric_rating, tenor_days, yield_value))

        # Create a DataFrame for the current date
        date_df = pd.DataFrame(data_for_date, columns=['Rating', 'Tenor', 'Yield'])
        date_df = date_df.pivot(index='Rating', columns='Tenor', values='Yield')
        date_dataframes[date] = date_df

    return date_dataframes




if __name__ == "__main__":
    csv_file_path = Path('../tests/sample_historical_bond_yields.csv').resolve()
    yaml_file_path = Path('../tests/rating_scores.yaml').resolve()

    bond_yields = load_bond_yields(csv_file_path, yaml_file_path)
    # Print or process loaded DataFrames as needed, for example:
    for date, df in bond_yields.items():
        print(f"Date: {date}\nDataFrame:\n{df}\n")

    # To access and print a DataFrame for a specific date (e.g., '2023-01-01'):
    specific_date = '2023-01-01'
    if specific_date in bond_yields:
        print(f"DataFrame for {specific_date}:\n{bond_yields[specific_date]}")
    else:
        print(f"No data available for {specific_date}")
