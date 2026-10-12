from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from rv.features import book_features, trade_features


def _stock_features(data_dir: Path, split: str, stock_id: int) -> pd.DataFrame:
    book = pd.read_parquet(data_dir / f"book_{split}.parquet" / f"stock_id={stock_id}")
    trade = pd.read_parquet(data_dir / f"trade_{split}.parquet" / f"stock_id={stock_id}")
    book = book.sort_values(["time_id", "seconds_in_bucket"]).assign(stock_id=stock_id)
    trade = trade.sort_values(["time_id", "seconds_in_bucket"]).assign(stock_id=stock_id)
    return book_features(book).merge(trade_features(trade), on=["stock_id", "time_id"], how="left")


def load_features(data_dir: str | Path = "data", split: str = "train", n_jobs: int = -1) -> pd.DataFrame:
    """Book + trade features for every (stock_id, time_id) of a split, joined to its csv.
    Cached to ``data/features_{split}.parquet`` (building it reads ~2.5 GB of parquet)."""
    data_dir = Path(data_dir)
    cache = data_dir / f"features_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = pd.read_csv(data_dir / f"{split}.csv")
    stock_ids = sorted(rows["stock_id"].unique())
    parts = Parallel(n_jobs=n_jobs)(delayed(_stock_features)(data_dir, split, s) for s in stock_ids)
    features = rows.merge(pd.concat(parts, ignore_index=True), on=["stock_id", "time_id"], how="left")
    features.to_parquet(cache, index=False)
    return features
