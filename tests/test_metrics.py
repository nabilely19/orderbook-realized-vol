import numpy as np
from rv.metrics import rmspe


def test_rmspe_perfect():
    y = np.array([0.01, 0.02])
    assert rmspe(y, y) == 0.0


def test_rmspe_known_value():
    assert np.isclose(rmspe(np.array([1.0]), np.array([1.1])), 0.1)
