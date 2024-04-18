import pytest
from bond_yield.data_processing.tenor_converter import IdentityTenorScaler

@pytest.fixture
def tenor_scaler():
    return IdentityTenorScaler()

def test_identity_tenor_scaler(tenor_scaler):
    # Test various tenor values to ensure they are returned unchanged
    tenor_values = [1, 5.5, 10.1, 100, 0, -1, -5.5]
    for tenor in tenor_values:
        assert tenor_scaler.scale_tenor(tenor) == tenor, f"Tenor value {tenor} was not unchanged"
