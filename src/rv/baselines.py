import numpy as np
import pandas as pd


def naive(features: pd.DataFrame) -> np.ndarray:
    """Predict next-window RV as the current-window RV (column ``rv``)."""
    return features["rv"].to_numpy()


def har_rv(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Linear regression of target on RV over sub-windows, fitted on train only."""
    raise NotImplementedError


def kalman_local_level(series: pd.Series) -> pd.Series:
    """Local-level filter on log RV; one-step-ahead predictions only (no smoothing: smoothing uses the future)."""
    raise NotImplementedError
