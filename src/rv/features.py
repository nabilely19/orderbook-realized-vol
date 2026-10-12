import warnings

import numpy as np
import pandas as pd


def wap(book: pd.DataFrame, level: int = 1) -> pd.Series:
    """Weighted average price at a given book level (bid price weighted by ask size, and vice versa)."""
    bid_p, ask_p = book[f"bid_price{level}"], book[f"ask_price{level}"]
    bid_s, ask_s = book[f"bid_size{level}"], book[f"ask_size{level}"]
    return (bid_p * ask_s + ask_p * bid_s) / (bid_s + ask_s)


def log_return(price: pd.Series) -> pd.Series:
    """Log return between consecutive updates. Must not cross time_id boundaries.

    Call per time_id, e.g. ``book.groupby("time_id")["wap"].transform(log_return)``.
    The first update of each window has no predecessor and returns NaN."""
    return np.log(price).diff()


def realized_volatility(returns: pd.Series) -> float:
    """sqrt(sum r^2). NaNs (first update of a window) are ignored."""
    return float(np.sqrt(np.nansum(np.square(returns))))


def book_features(book: pd.DataFrame) -> pd.DataFrame:
    """One row per (stock_id, time_id): RV over the full window and sub-windows
    (seconds_in_bucket >= 300, 450, 540), spread, depth imbalance, number of updates.

    ``book`` holds a single stock, sorted by (time_id, seconds_in_bucket), with a ``stock_id`` column."""
    sec = book["seconds_in_bucket"]
    g = book["time_id"]
    r2 = np.log(wap(book)).groupby(g).diff() ** 2
    r2_level2 = np.log(wap(book, 2)).groupby(g).diff() ** 2
    bid_depth = book["bid_size1"] + book["bid_size2"]
    ask_depth = book["ask_size1"] + book["ask_size2"]
    df = pd.DataFrame({
        "time_id": g,
        "rv": r2,
        "rv_300": r2.where(sec >= 300),
        "rv_450": r2.where(sec >= 450),
        "rv_540": r2.where(sec >= 540),
        "rv_level2": r2_level2,
        "spread": book["ask_price1"] / book["bid_price1"] - 1,
        "depth": bid_depth + ask_depth,
        "imbalance": ((bid_depth - ask_depth) / (bid_depth + ask_depth)).abs(),
    })
    out = df.groupby("time_id").agg(
        rv=("rv", "sum"), rv_300=("rv_300", "sum"), rv_450=("rv_450", "sum"),
        rv_540=("rv_540", "sum"), rv_level2=("rv_level2", "sum"),
        spread=("spread", "mean"), depth=("depth", "mean"), imbalance=("imbalance", "mean"),
        n_updates=("spread", "size"),
    )
    rv_cols = ["rv", "rv_300", "rv_450", "rv_540", "rv_level2"]
    out[rv_cols] = np.sqrt(out[rv_cols])
    out["tick_price"] = tick_price(book)
    out.insert(0, "stock_id", book["stock_id"].iloc[0])
    return out.reset_index()


def tick_price(book: pd.DataFrame) -> pd.Series:
    """Real price level per time_id, recovered from the tick size.

    Prices are normalised to ~1 within each window, but the tick is $0.01 in real money,
    so the smallest gap between distinct normalised prices is 0.01 / real_price."""
    prices = book.melt(id_vars="time_id", value_vars=["bid_price1", "ask_price1", "bid_price2", "ask_price2"])
    prices = prices.drop_duplicates(["time_id", "value"]).sort_values(["time_id", "value"])
    gaps = prices.groupby("time_id")["value"].diff()
    tick = gaps[gaps > 1e-7].groupby(prices["time_id"]).min()
    return (0.01 / tick).rename("tick_price")


def trade_features(trade: pd.DataFrame) -> pd.DataFrame:
    """One row per (stock_id, time_id): traded volume, number of trades, RV of trade prices.

    ``trade`` holds a single stock, sorted by (time_id, seconds_in_bucket), with a ``stock_id`` column."""
    r2 = np.log(trade["price"]).groupby(trade["time_id"]).diff() ** 2
    df = pd.DataFrame({"time_id": trade["time_id"], "r2": r2, "size": trade["size"], "orders": trade["order_count"]})
    out = df.groupby("time_id").agg(
        trade_rv=("r2", "sum"), volume=("size", "sum"), n_trades=("size", "size"), n_orders=("orders", "sum"),
    )
    out["trade_rv"] = np.sqrt(out["trade_rv"])
    out.insert(0, "stock_id", trade["stock_id"].iloc[0])
    return out.reset_index()


def neighbour_features(features: pd.DataFrame, ordered_time_ids: np.ndarray | None = None, k: int = 10) -> pd.DataFrame:
    """Average RV over the k time_ids whose cross-section of price levels is closest.

    Time_ids with similar price levels across all stocks are close in time, so with no
    restriction the neighbours include the windows right after the current one: their RV
    is close to the target. This is look-ahead that top competition solutions exploited.
    With ``ordered_time_ids`` given, neighbours are restricted to earlier time_ids
    (production-legal). Returns one row per (stock_id, time_id)."""
    prices = np.log(features.pivot(index="time_id", columns="stock_id", values="tick_price"))
    prices = (prices - prices.mean()) / prices.std()
    prices = prices.fillna(0.0)
    x = prices.to_numpy()
    sq = (x ** 2).sum(axis=1)
    dist = sq[:, None] + sq[None, :] - 2 * x @ x.T
    np.fill_diagonal(dist, np.inf)
    if ordered_time_ids is not None:
        pos = pd.Series(np.arange(len(ordered_time_ids)), index=ordered_time_ids)[prices.index].to_numpy()
        dist[pos[None, :] >= pos[:, None]] = np.inf
    k_eff = np.minimum(k, np.isfinite(dist).sum(axis=1))
    nn = np.argsort(dist, axis=1)[:, :k]
    valid = np.arange(k)[None, :] < k_eff[:, None]

    rv = features.pivot(index="time_id", columns="stock_id", values="rv").loc[prices.index].to_numpy()
    nn_rv = np.where(valid[:, :, None], rv[nn], np.nan)          # (time_id, k, stock)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # all-NaN rows: no earlier neighbour
        nn_rv_mean = np.nanmean(nn_rv, axis=1)                    # (time_id, stock)
        nn_market_rv = np.nanmean(nn_rv_mean, axis=1)             # (time_id,)
    n_time, n_stock = nn_rv_mean.shape
    return pd.DataFrame({
        "stock_id": np.tile(prices.columns.to_numpy(), n_time),
        "time_id": np.repeat(prices.index.to_numpy(), n_stock),
        "nn_rv": nn_rv_mean.ravel(),
        "nn_market_rv": np.repeat(nn_market_rv, n_stock),
    })
