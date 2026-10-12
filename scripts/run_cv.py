"""Entry point: build features, run every model under every CV scheme, write reports/results.csv."""
from pathlib import Path

from rv.data import load_features
from rv.experiment import run, time_order


def main():
    features = load_features("data")
    order = time_order(features)
    results = run(features, order)
    Path("reports").mkdir(exist_ok=True)
    results.round(4).to_csv("reports/results.csv")
    print(results.round(4).to_string())


if __name__ == "__main__":
    main()
