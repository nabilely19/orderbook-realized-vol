import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def random_folds(time_ids: np.ndarray, k: int = 5, seed: int = 0):
    """K-fold split grouped by time_id (all stocks of a time_id in the same fold).
    Yields (train_time_ids, test_time_ids)."""
    ids = np.random.default_rng(seed).permutation(np.unique(time_ids))
    for test in np.array_split(ids, k):
        yield np.setdiff1d(ids, test), test


def reconstruct_time_order(prices: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Recover chronological order of time_ids (e.g. nearest-neighbour chain on the cross-section of price levels).
    Return time_ids sorted chronologically.

    ``prices``: one row per time_id, one column per stock, real price levels (see ``features.tick_price``).
    Windows close in time have similar prices across all stocks, so a 1-D t-SNE embedding of the
    standardised log prices lays them out along the time axis. The direction (forward or backward)
    is not identifiable from prices alone: see ``orient``."""
    x = np.log(prices)
    x = x.fillna(x.mean())
    x = ((x - x.mean()) / x.std()).to_numpy()
    embedding = TSNE(n_components=1, perplexity=30, random_state=seed).fit_transform(x)[:, 0]
    return prices.index.to_numpy()[np.argsort(embedding)]


def orient(ordered_time_ids: np.ndarray, rv: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Flip the order if it runs backwards in time.

    ``target`` is the RV of the 10 minutes right after each window, so it should track the RV of
    the following window more closely than the RV of the preceding one. ``rv`` and ``target``
    have one row per time_id and one column per stock."""
    rv, target = rv.loc[ordered_time_ids].to_numpy(), target.loc[ordered_time_ids].to_numpy()
    err_next = np.nanmean(np.abs(np.log(target[:-1] / rv[1:])))
    err_prev = np.nanmean(np.abs(np.log(target[1:] / rv[:-1])))
    return ordered_time_ids if err_next <= err_prev else ordered_time_ids[::-1]


def walk_forward_folds(ordered_time_ids: np.ndarray, n_folds: int = 5, gap: int = 0):
    """Expanding-window splits: train on earlier time_ids, test on later ones, with an optional gap.
    The ordered ids are cut into n_folds + 1 blocks; fold i trains on blocks 0..i and tests on block i+1.
    Yields (train_time_ids, test_time_ids)."""
    blocks = np.array_split(np.asarray(ordered_time_ids), n_folds + 1)
    for i in range(n_folds):
        train = np.concatenate(blocks[: i + 1])
        yield (train[:-gap] if gap else train), blocks[i + 1]
