from abc import ABC, abstractmethod

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
    def __init__(self):
        super().__init__()
        self.rating_map = {
            'AAA': 1,
            'AA+': 2,
            'AA': 3,
            'AA-': 4,
            'A+': 5,
            'A': 6,
            'A-': 7,
            'BBB': 8,
            'BB': 9,
            'B': 10,
            'CCC': 11,
            'CC': 12,
            'C': 13,
            'D': 14
        }

    def convert(self, rating):
        return self.rating_map.get(rating, 0)  # Returns 0 if the rating is not found

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