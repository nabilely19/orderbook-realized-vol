import numpy as np
import pandas as pd


def wap(book: pd.DataFrame, level: int = 1) -> pd.Series:
    """Weighted average price at a given book level (bid price weighted by ask size, and vice versa)."""
    raise NotImplementedError


def log_return(price: pd.Series) -> pd.Series:
    """Log return between consecutive updates. Must not cross time_id boundaries."""
    raise NotImplementedError


def realized_volatility(returns: pd.Series) -> float:
    """sqrt(sum r^2)."""
    raise NotImplementedError


def book_features(book: pd.DataFrame) -> pd.DataFrame:
    """One row per (stock_id, time_id): RV over the full window and sub-windows
    (seconds_in_bucket >= 300, 450), spread, depth imbalance, number of updates."""
    raise NotImplementedError


def trade_features(trade: pd.DataFrame) -> pd.DataFrame:
    """One row per (stock_id, time_id): traded volume, number of trades, RV of trade prices."""
    raise NotImplementedError
