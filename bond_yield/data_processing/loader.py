import pandas as pd
from pathlib import Path

def load_bond_yields_single_dataframe(csv_path):
    """
    Load bond yields into a single DataFrame with the first column as the index (ratings)
    and the remaining columns as tenors (in days).

    Parameters:
    - csv_path: str, Path to the CSV file containing bond yields.

    Returns:
    - pd.DataFrame: DataFrame where the index is ratings from the first column and columns
                    are tenors in days.
    """
    # Read the bond yields CSV file, assuming the first column is the index (ratings)
    df = pd.read_csv(csv_path, index_col=0)

    return df


def load_bond_yields(csv_path):
    df = pd.read_csv(csv_path, index_col=0)  # Assuming first column is the date

    date_dataframes = {}
    for date in df.index.unique():
        data_for_date = []
        for col in df.columns:
            rating, tenor = col.split("::")
            tenor_days = int(tenor)
            yield_value = df.at[date, col]
            data_for_date.append((rating, tenor_days, yield_value))

        # Create a DataFrame for the current date
        date_df = pd.DataFrame(data_for_date, columns=['Rating', 'Tenor', 'Yield'])
        date_df = date_df.pivot(index='Rating', columns='Tenor', values='Yield')
        date_dataframes[date] = date_df

    return date_dataframes




if __name__ == "__main__":
    csv_file_path = Path('../tests/sample_historical_bond_yields.csv').resolve()

    bond_yields = load_bond_yields(csv_file_path)
    # Print or process loaded DataFrames as needed, for example:
    for date, df in bond_yields.items():
        print(f"Date: {date}\nDataFrame:\n{df}\n")

    # To access and print a DataFrame for a specific date (e.g., '2023-01-01'):
    specific_date = '2023-01-01'
    if specific_date in bond_yields:
        print(f"DataFrame for {specific_date}:\n{bond_yields[specific_date]}")
    else:
        print(f"No data available for {specific_date}")

    pass
