"""
test_analytics.py — Unit tests for the framework-free analytics layer.

Run:  python -m pytest -q      (or)   python test_analytics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import analytics as A
from data_gen import generate_transactions


def _toy() -> pd.DataFrame:
    # 2 fraud (high score), 3 legit (low score), known amounts.
    return pd.DataFrame(
        {
            "is_fraud": [1, 1, 0, 0, 0],
            "fraud_score": [0.90, 0.60, 0.40, 0.20, 0.10],
            "amount": [1000.0, 500.0, 200.0, 100.0, 50.0],
        }
    )


def test_confusion_counts():
    df = _toy()
    c = A.confusion_at_threshold(df["is_fraud"], df["fraud_score"], 0.55)
    # declined: 0.90, 0.60 -> both fraud -> tp=2, fp=0
    assert c == {"tp": 2, "fp": 0, "fn": 0, "tn": 3}


def test_confusion_with_false_positive():
    df = _toy()
    c = A.confusion_at_threshold(df["is_fraud"], df["fraud_score"], 0.35)
    # declined: 0.90,0.60 (fraud) + 0.40 (legit) -> tp=2, fp=1, fn=0, tn=2
    assert c == {"tp": 2, "fp": 1, "fn": 0, "tn": 2}


def test_metrics_rates():
    df = _toy()
    m = A.metrics_at_threshold(df, 0.55, review_floor=0.30)
    assert abs(m["fraud_capture_rate"] - 1.0) < 1e-9       # caught both fraud
    assert abs(m["false_positive_rate"] - 0.0) < 1e-9
    assert abs(m["approval_rate"] - 3 / 5) < 1e-9          # 3 approved of 5


def test_metrics_counts_sum_to_total():
    df = generate_transactions(n=5000, seed=1)
    m = A.metrics_at_threshold(df, 0.5)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["total"] == len(df)


def test_capture_rate_monotonic_decreasing_in_threshold():
    df = generate_transactions(n=8000, seed=7)
    hi = A.metrics_at_threshold(df, 0.8)["fraud_capture_rate"]
    lo = A.metrics_at_threshold(df, 0.3)["fraud_capture_rate"]
    assert lo >= hi  # lower cutoff catches at least as much fraud


def test_economics_fraud_prevented():
    df = _toy()
    e = A.economics_at_threshold(df, 0.55, A.EconomicAssumptions())
    # fraud declined = amounts 1000 + 500
    assert abs(e["fraud_prevented_usd"] - 1500.0) < 1e-6
    assert abs(e["fraud_loss_usd"] - 0.0) < 1e-6


def test_economics_false_decline_cost():
    df = _toy()
    a = A.EconomicAssumptions(false_decline_cost_rate=0.25, review_cost_per_txn=0.0)
    e = A.economics_at_threshold(df, 0.35, a)
    # legit declined at 0.35: the 0.40-score txn, amount 200 -> cost 50
    assert abs(e["false_decline_value_usd"] - 200.0) < 1e-6
    assert abs(e["false_decline_cost_usd"] - 50.0) < 1e-6


def test_roi_sign_and_fields():
    df = generate_transactions(n=10000, seed=3)
    roi = A.roi_vs_baseline(df, chosen_threshold=0.5, baseline_threshold=0.8)
    # net savings == baseline cost - chosen cost
    assert abs(
        roi["net_savings_usd"]
        - (roi["baseline_total_cost_usd"] - roi["chosen_total_cost_usd"])
    ) < 1e-6


def test_auc_in_unit_interval_and_better_than_chance():
    df = generate_transactions(n=10000, seed=5)
    auc = A.auc_roc(df)
    assert 0.5 < auc <= 1.0  # model separates fraud from good traffic


def test_curve_points_shape():
    df = generate_transactions(n=4000, seed=9)
    c = A.curve_points(df, n_thresholds=51)
    assert len(c) == 51
    assert set(["threshold", "tpr", "fpr", "precision", "recall"]).issubset(c.columns)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
