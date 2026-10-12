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
    (seconds_in_bucket >= 300, 450), spread, depth imbalance, number of updates."""
    raise NotImplementedError


def trade_features(trade: pd.DataFrame) -> pd.DataFrame:
    """One row per (stock_id, time_id): traded volume, number of trades, RV of trade prices."""
    raise NotImplementedError
