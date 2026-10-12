import numpy as np
import pandas as pd


def random_folds(time_ids: np.ndarray, k: int = 5, seed: int = 0):
    """K-fold split grouped by time_id (all stocks of a time_id in the same fold)."""
    raise NotImplementedError


def reconstruct_time_order(prices: pd.DataFrame) -> np.ndarray:
    """Recover chronological order of time_ids (e.g. nearest-neighbour chain on the cross-section of price levels).
    Return time_ids sorted chronologically."""
    raise NotImplementedError


def walk_forward_folds(ordered_time_ids: np.ndarray, n_folds: int = 5, gap: int = 0):
    """Expanding-window splits: train on earlier time_ids, test on later ones, with an optional gap."""
    raise NotImplementedError
