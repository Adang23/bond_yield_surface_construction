import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_fake_bond_yields(start_date, end_date):
    """
    Generates a DataFrame with fake bond yields for a range of dates, ratings, and tenors.

    Parameters:
    - start_date (str): The start date in 'YYYYMMDD' format.
    - end_date (str): The end date in 'YYYYMMDD' format.

    Returns:
    - A pandas DataFrame containing the generated data.
    """
    # Define ratings and tenors
    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
    tenors_days = [365 * year for year in range(1, 11)]  # 1 year to 10 years in days

    # Generate date range
    date_range = pd.date_range(start=datetime.strptime(start_date, '%Y%m%d'),
                               end=datetime.strptime(end_date, '%Y%m%d'))

    # Prepare column names
    column_names = [f'{rating}::{tenor}' for rating in ratings for tenor in tenors_days]

    # Generate random yields between 0.001 and 0.5
    data = np.random.uniform(low=0.001, high=0.5, size=(len(date_range), len(column_names)))

    # Create DataFrame
    df = pd.DataFrame(data, index=date_range, columns=column_names)
    return df


# Generate the fake bond yields data
df_bond_yields = generate_fake_bond_yields('20230101', '20230130')

# Specify the file path (adjust the path as needed for your project structure)
file_path = './sample_historical_bond_yields.csv'
# Save the DataFrame to a CSV file
df_bond_yields.to_csv(file_path)

print(f'Sample historical bond yields data has been saved to {file_path}')
