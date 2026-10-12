import numpy as np
import pandas as pd
from rv.features import log_return, realized_volatility, wap


def test_wap_weights_by_opposite_size():
    book = pd.DataFrame({"bid_price1": [99.0], "ask_price1": [101.0], "bid_size1": [3.0], "ask_size1": [1.0]})
    # Large bid size pushes WAP towards the ask: (99*1 + 101*3) / 4 = 100.5
    assert np.isclose(wap(book).iloc[0], 100.5)


def test_log_return_does_not_cross_time_id():
    book = pd.DataFrame({"time_id": [5, 5, 11, 11], "wap": [1.0, 2.0, 4.0, 4.0]})
    r = book.groupby("time_id")["wap"].transform(log_return)
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[2])
    assert np.isclose(r.iloc[1], np.log(2.0))
    assert r.iloc[3] == 0.0


def test_realized_volatility_ignores_nan():
    assert np.isclose(realized_volatility(pd.Series([np.nan, 0.3, -0.4])), 0.5)
