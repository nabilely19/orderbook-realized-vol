import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.structural import UnobservedComponents

HAR_FEATURES = ["rv", "rv_300", "rv_540"]           # last 10, 5 and 1 minute(s)
BOOK_TRADE_FEATURES = [
    "rv", "rv_300", "rv_450", "rv_540", "rv_level2", "spread", "depth", "imbalance", "n_updates",
    "trade_rv", "volume", "n_trades", "n_orders", "market_rv",
]
NEIGHBOUR_FEATURES = ["nn_rv", "nn_market_rv"]


def naive(features: pd.DataFrame) -> np.ndarray:
    """Predict next-window RV as the current-window RV (column ``rv``)."""
    return features["rv"].to_numpy()


def har_rv(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Linear regression of target on RV over sub-windows, fitted on train only.
    Weighted by 1/y^2, so least squares minimises the squared percentage error (RMSPE)."""
    model = LinearRegression().fit(train[HAR_FEATURES], train["target"], sample_weight=train["target"] ** -2)
    return model.predict(test[HAR_FEATURES])


def kalman_local_level(series: pd.Series, fit_index=None) -> pd.Series:
    """Local-level filter on log RV; one-step-ahead predictions only (no smoothing: smoothing uses the future).

    ``series``: RV of one stock, indexed by time_id in chronological order. The noise variances are
    estimated on ``fit_index`` only (default: all), then the filter runs over the whole series:
    the filtered level at t uses observations up to t and is the forecast for t + 1.
    Returns exp(level), a forecast of the next window's RV."""
    y = np.log(series.to_numpy())
    y_fit = np.where(series.index.isin(fit_index), y, np.nan) if fit_index is not None else y
    model = UnobservedComponents(y_fit, "llevel")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = model.fit(disp=False).params
    level = UnobservedComponents(y, "llevel").filter(params).filtered_state[0]
    return pd.Series(np.exp(level), index=series.index)


def rmspe_scale(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Constant c minimising RMSPE(y, c * y_pred); fitted on train, it corrects a level bias."""
    ratio = y_pred / y_true
    return float(ratio.sum() / (ratio ** 2).sum())


LGB_PARAMS = dict(
    objective="rmse", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0,
)


def lightgbm(train: pd.DataFrame, test: pd.DataFrame, features: list[str], num_rounds: int = 500) -> np.ndarray:
    """Gradient boosting on book/trade features, RMSPE-consistent.

    Trees cannot extrapolate, so instead of the RV level (which drifts between volatility regimes)
    the model predicts the ratio target / rv. With weight (rv / target)^2, squared error on the ratio
    equals squared percentage error on the target. Fixed number of rounds: early stopping on the
    test fold would leak."""
    cols = ["stock_id", *features]
    ratio = train["target"] / train["rv"]
    data = lgb.Dataset(train[cols], ratio, weight=ratio ** -2, categorical_feature=["stock_id"])
    model = lgb.train(LGB_PARAMS, data, num_boost_round=num_rounds)
    return model.predict(test[cols]) * test["rv"].to_numpy()
