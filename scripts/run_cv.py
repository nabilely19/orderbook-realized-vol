"""Entry point: build features, run every model under every CV scheme, write reports/results.csv,
reports/folds.csv (walk-forward, legal features, fold by fold) and reports/significance.csv."""
from pathlib import Path

import pandas as pd

from rv.data import load_features
from rv.experiment import MODELS, predictions, summarise, time_order
from rv.stats import block_bootstrap_diff, fold_table

# (better model, baseline, scheme): is the first really better than the second?
COMPARISONS = [
    ("lightgbm_no_neighbours", "har_rv", "walk_forward_legal"),
    ("har_rv", "naive", "walk_forward_legal"),
    ("lightgbm", "lightgbm_no_neighbours", "walk_forward"),
    ("lightgbm", "lightgbm_no_neighbours", "walk_forward_legal"),
]


def significance(preds: dict, order) -> pd.DataFrame:
    rows = []
    for a, b, scheme in COMPARISONS:
        for block_size in (20, 100):
            res = block_bootstrap_diff(preds[scheme], order, a, b, block_size=block_size)
            rows.append({"model": a, "vs": b, "scheme": scheme, "block_size": block_size, **res})
    return pd.DataFrame(rows)


def main():
    features = load_features("data")
    order = time_order(features)
    preds = predictions(features, order)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    results = summarise(preds)
    folds = fold_table(preds["walk_forward_legal"], MODELS)
    sig = significance(preds, order)
    results.round(4).to_csv(reports / "results.csv")
    folds.round(4).to_csv(reports / "folds.csv")
    sig.round(4).to_csv(reports / "significance.csv", index=False)
    print(results.round(4).to_string(), folds.round(4).to_string(), sig.round(4).to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
