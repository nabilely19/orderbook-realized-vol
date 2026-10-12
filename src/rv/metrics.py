import numpy as np


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared percentage error: sqrt(mean(((y - yhat) / y) ** 2))."""
    raise NotImplementedError
