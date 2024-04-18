from abc import ABC, abstractmethod

class BaseTenorScaler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def scale_tenor(self, tenor):
        """
        Scale a tenor value according to some rule defined in the subclass.

        Parameters:
        tenor (float): The original tenor value.

        Returns:
        float: The scaled tenor value.
        """
        pass


class IdentityTenorScaler(BaseTenorScaler):
    def scale_tenor(self, tenor):
        """
        Return the tenor value unchanged.

        Parameters:
        tenor (float): The original tenor value.

        Returns:
        float: The same tenor value as input, unchanged.
        """
        return tenor

