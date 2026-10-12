"""Is a difference in RMSPE between two models real? Fold-by-fold scores and a block bootstrap over time."""
import numpy as np
import pandas as pd

from rv.metrics import rmspe


def fold_table(preds: pd.DataFrame, models) -> pd.DataFrame:
    """RMSPE of each model on each fold's test rows."""
    return preds.groupby("fold").apply(
        lambda g: pd.Series({m: rmspe(g["target"].to_numpy(), g[m].to_numpy()) for m in models}),
        include_groups=False,
    )


def block_bootstrap_diff(
    preds: pd.DataFrame, order: np.ndarray, model_a: str, model_b: str,
    block_size: int = 20, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Bootstrap confidence interval for RMSPE(model_b) - RMSPE(model_a); positive means model_a is better.

    Errors are correlated across stocks within a window and across nearby windows (volatility clusters),
    so resampling rows would understate the uncertainty. Instead, tested time_ids are put in
    reconstructed chronological order, cut into blocks of ``block_size`` consecutive windows (all stocks
    included), and whole blocks are resampled with replacement."""
    sq = lambda m: ((preds["target"] - preds[m]) / preds["target"]) ** 2
    per_window = pd.DataFrame({"a": sq(model_a), "b": sq(model_b), "n": 1.0, "time_id": preds["time_id"]})
    per_window = per_window.groupby("time_id").sum()
    tested = [t for t in order if t in per_window.index]
    per_window = per_window.loc[tested].to_numpy()
    n_blocks = int(np.ceil(len(per_window) / block_size))
    blocks = np.add.reduceat(per_window, np.arange(0, len(per_window), block_size))   # (n_blocks, [a, b, n])

    def diff(sums):
        return np.sqrt(sums[..., 1] / sums[..., 2]) - np.sqrt(sums[..., 0] / sums[..., 2])

    idx = np.random.default_rng(seed).integers(0, n_blocks, size=(n_boot, n_blocks))
    boot = diff(blocks[idx].sum(axis=1))
    return {
        "diff": float(diff(blocks.sum(axis=0))),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_a_not_better": float((boot <= 0).mean()),
        "n_blocks": n_blocks,
    }
