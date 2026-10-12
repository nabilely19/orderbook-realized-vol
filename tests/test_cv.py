import numpy as np
import pandas as pd
from rv.cv import orient, random_folds, reconstruct_time_order, walk_forward_folds


def test_random_folds_partition_time_ids():
    ids = np.arange(100)
    tests = [test for train, test in random_folds(ids, k=5)]
    assert sorted(np.concatenate(tests)) == list(ids)
    for train, test in random_folds(ids, k=5):
        assert not set(train) & set(test)


def test_walk_forward_trains_only_on_the_past():
    ids = np.arange(60)
    folds = list(walk_forward_folds(ids, n_folds=5, gap=2))
    assert len(folds) == 5
    for train, test in folds:
        assert train.max() < test.min() - 2


def test_reconstruct_time_order_on_random_walk():
    rng = np.random.default_rng(0)
    n = 300
    log_prices = np.cumsum(rng.normal(0, 0.01, size=(n, 20)), axis=0)
    shuffled_ids = rng.permutation(n)                     # time_id -> true position
    prices = pd.DataFrame(np.exp(log_prices[shuffled_ids]), index=np.arange(n))
    order = reconstruct_time_order(prices)
    true_pos = shuffled_ids[order]
    # Recovered order matches the truth up to direction.
    assert abs(np.corrcoef(true_pos, np.arange(n))[0, 1]) > 0.95


def test_orient_flips_backwards_order():
    rv = pd.DataFrame({0: [1.0, 2.0, 4.0, 8.0]})
    target = pd.DataFrame({0: [2.0, 4.0, 8.0, 16.0]})     # target[t] == rv[t + 1]
    ids = np.array([3, 2, 1, 0])
    assert list(orient(ids, rv, target)) == [0, 1, 2, 3]


def test_block_bootstrap_detects_a_clear_gap_and_not_a_null_one():
    from rv.stats import block_bootstrap_diff
    rng = np.random.default_rng(0)
    n = 400
    target = rng.uniform(1, 2, n)
    preds = pd.DataFrame({
        "time_id": np.arange(n), "target": target,
        "good": target * (1 + rng.normal(0, 0.1, n)),
        "bad": target * (1 + rng.normal(0, 0.3, n)),
        "bad_twin": target * (1 + rng.normal(0, 0.3, n)),
    })
    clear = block_bootstrap_diff(preds, np.arange(n), "good", "bad", block_size=10)
    assert clear["ci_low"] > 0
    null = block_bootstrap_diff(preds, np.arange(n), "bad", "bad_twin", block_size=10)
    assert null["ci_low"] < 0 < null["ci_high"]
