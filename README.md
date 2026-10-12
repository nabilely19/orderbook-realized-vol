# Realized volatility from order-book data: how much of the edge is look-ahead?

**Question.** Forecast the next 10-minute realized volatility of 112 stocks from limit-order-book snapshots
(Optiver Realized Volatility Prediction dataset), and measure how much of the top competition scores came
from information that would not be available in production (reconstructed `time_id` ordering, cross-time
neighbour features).

## Data
Kaggle competition `optiver-realized-volatility-prediction`. Accept the competition rules on Kaggle, then:
```
kaggle competitions download -c optiver-realized-volatility-prediction -p data/
unzip data/optiver-realized-volatility-prediction.zip -d data/
```
`data/` is git-ignored.

## Definitions
- WAP = (bid_price1 * ask_size1 + ask_price1 * bid_size1) / (bid_size1 + ask_size1)
- log return r_t = log(WAP_t / WAP_{t-1}) over consecutive book updates
- realized volatility RV = sqrt(sum r_t^2)
- metric RMSPE = sqrt(mean(((y - yhat) / y)^2))

## Validation design
1. Random K-fold over `time_id` (what most public notebooks report).
2. Walk-forward over reconstructed `time_id` order (train on past, test on future).
3. Walk-forward with "production-legal" features only (no neighbour features built from future time_ids).

Gap 1 -> 2 = optimism from random CV. Gap 2 -> 3 = share of edge that is not tradeable.

## Models
| Model | Description |
|---|---|
| Naive | RV of the observed 10-minute window |
| HAR-RV | Linear regression on RV over sub-windows (last 1, 5, 10 min) |
| Kalman | Local-level state-space model on log RV |
| LightGBM | Book + trade features, sample weight 1/y^2 (RMSPE-consistent) |

## Results
| Model | Random CV | Walk-forward | Walk-forward, legal features |
|---|---|---|---|
| Naive | | | |
| HAR-RV | | | |
| Kalman | | | |
| LightGBM | | | |

## What did not work
_To fill._

## Layout
```
src/rv/        features, metrics, baselines, cv
scripts/       run_cv.py entry point
notebooks/     EDA only, no logic
reports/       figures and tables
tests/         unit tests on metrics and features
```

## References
- Optiver competition 1st place write-up (nyanp), Kaggle discussion 274970
- michaelpoluektov/orvp: competition model vs production-legal model
- Corsi (2009), A simple approximate long-memory model of realized volatility (HAR-RV)
