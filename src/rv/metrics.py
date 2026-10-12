import numpy as np


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared percentage error: sqrt(mean(((y - yhat) / y) ** 2))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)))
