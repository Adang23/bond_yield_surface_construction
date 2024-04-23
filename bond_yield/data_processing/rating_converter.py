from abc import ABC, abstractmethod
import yaml

class BaseRatingConverter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def convert(self, rating):
        """
        Convert a rating to a numerical value.

        Parameters:
        - rating (str): The rating to convert.

        Returns:
        - int or float: The numerical value of the rating.
        """
        pass

class SimpleRatingConverter(BaseRatingConverter):
    def __init__(self, rating_map=None, yaml_path=None):
        """
        Initialize the converter with a rating map provided either as a dictionary or loaded from a YAML file.

        Parameters:
        - rating_map (dict, optional): A dictionary mapping rating labels to numerical values.
        - yaml_path (str, optional): Path to a YAML file containing the rating mappings.
        """
        super().__init__()
        if yaml_path:
            # Load the rating map from a YAML file
            with open(yaml_path, 'r') as file:
                self.rating_map = yaml.safe_load(file)
        elif rating_map:
            # Use the provided dictionary
            self.rating_map = rating_map
        else:
            raise ValueError("A rating map must be provided either as a dictionary or a YAML file path.")

    def convert(self, rating):
        """
        Convert a rating to its numerical value using the rating map.

        Parameters:
        - rating (str): The rating to convert.

        Returns:
        - int or float: The numerical value of the rating, or raises an error if not found.
        """
        if rating in self.rating_map:
            return self.rating_map[rating]
        else:
            raise ValueError(f"Rating '{rating}' not found in the rating map.")

class YieldBasedRatingConverter(BaseRatingConverter):
    def __init__(self, bond_yield_df):
        """
        Initialize the converter with a DataFrame of bond yields.

        Parameters:
        - bond_yield_df (pd.DataFrame): DataFrame indexed by ratings with columns as tenors,
                                        values are yields.
        """
        self.bond_yield_df = bond_yield_df

    def convert(self, rating):
        """
        Convert a rating based on the average yield across all tenors for that rating.

        Parameters:
        - rating (str): The rating to convert.

        Returns:
        - float: The average yield for the given rating, or 0 if the rating is not found.
        """
        if rating in self.bond_yield_df.index:
            # Calculate the average yield for the given rating
            average_yield = self.bond_yield_df.loc[rating].mean()
            return average_yield
        else:
            return 0  # Return 0 or perhaps NaN if the rating is not found