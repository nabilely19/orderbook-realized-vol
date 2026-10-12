"""Every model under every validation scheme. Used by scripts/run_cv.py and the notebook."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rv.baselines import (
    BOOK_TRADE_FEATURES, NEIGHBOUR_FEATURES, har_rv, kalman_local_level, lightgbm, naive, rmspe_scale,
)
from rv.cv import orient, random_folds, reconstruct_time_order, walk_forward_folds
from rv.features import neighbour_features
from rv.metrics import rmspe

SCHEMES = ["random", "walk_forward", "walk_forward_legal"]
MODELS = ["naive", "har_rv", "kalman", "lightgbm_no_neighbours", "lightgbm"]


def add_market_rv(features: pd.DataFrame) -> pd.DataFrame:
    """Mean RV across stocks in the same window: known at prediction time, so legal."""
    return features.assign(market_rv=features.groupby("time_id")["rv"].transform("mean"))


def time_order(features: pd.DataFrame) -> np.ndarray:
    """Reconstructed chronological order of time_ids, oriented forwards."""
    pivot = lambda col: features.pivot(index="time_id", columns="stock_id", values=col)
    return orient(reconstruct_time_order(pivot("tick_price")), pivot("rv"), pivot("target"))


def _kalman_stock(series: pd.Series, train_ids, test_ids, target: pd.Series) -> pd.Series:
    pred = kalman_local_level(series, fit_index=train_ids)
    train = series.index.isin(train_ids)
    c = rmspe_scale(target[train].to_numpy(), pred[train].to_numpy())
    return c * pred[series.index.isin(test_ids)]


def kalman(features: pd.DataFrame, order: np.ndarray, train_ids, test_ids, n_jobs: int = -1) -> pd.Series:
    """Per-stock Kalman forecasts for the test rows, indexed like ``features``."""
    stocks = [g.set_index("time_id").reindex(order).dropna(subset=["rv"]) for _, g in features.groupby("stock_id")]
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_kalman_stock)(g["rv"], train_ids, test_ids, g["target"]) for g in stocks
    )
    preds = pd.concat(
        [p.rename("pred").rename_axis("time_id").reset_index().assign(stock_id=g["stock_id"].iloc[0]) for p, g in zip(preds, stocks)]
    )
    keys = features[["stock_id", "time_id"]].reset_index()
    return keys.merge(preds, on=["stock_id", "time_id"]).set_index("index")["pred"]


def folds(scheme: str, features: pd.DataFrame, order: np.ndarray):
    if scheme == "random":
        return random_folds(features["time_id"].unique(), k=5)
    return walk_forward_folds(order, n_folds=5)


def run(features: pd.DataFrame, order: np.ndarray, models=MODELS, schemes=SCHEMES, log=print) -> pd.DataFrame:
    """RMSPE of every model under every scheme, pooled over the test rows of all folds.

    random              random 5-fold over time_id, neighbour features from all time_ids
    walk_forward        train on the past, test on the future; neighbours still from all time_ids
    walk_forward_legal  walk-forward, neighbours restricted to earlier time_ids (production-legal)"""
    features = add_market_rv(features)
    nn = {
        "all": neighbour_features(features),
        "legal": neighbour_features(features, order),
    }
    results = {}
    for scheme in schemes:
        f = features.merge(nn["legal" if scheme == "walk_forward_legal" else "all"], on=["stock_id", "time_id"])
        preds = {m: pd.Series(np.nan, index=f.index) for m in models}
        for i, (train_ids, test_ids) in enumerate(folds(scheme, f, order)):
            train, test = f[f["time_id"].isin(train_ids)], f[f["time_id"].isin(test_ids)]
            for m in models:
                if m == "naive":
                    p = naive(test)
                elif m == "har_rv":
                    p = har_rv(train, test)
                elif m == "kalman":
                    p = kalman(f, order, train_ids, test_ids).reindex(test.index).to_numpy()
                elif m == "lightgbm_no_neighbours":
                    p = lightgbm(train, test, BOOK_TRADE_FEATURES)
                elif m == "lightgbm":
                    p = lightgbm(train, test, BOOK_TRADE_FEATURES + NEIGHBOUR_FEATURES)
                preds[m].loc[test.index] = p
            log(f"{scheme}: fold {i + 1} done")
        tested = preds[models[0]].notna()
        results[scheme] = {m: rmspe(f.loc[tested, "target"].to_numpy(), preds[m][tested].to_numpy()) for m in models}
    return pd.DataFrame(results).loc[models]
