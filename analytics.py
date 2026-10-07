"""
analytics.py — Framework-free fraud-model analytics.

Pure compute: no Streamlit, no plotting. Everything here takes arrays
or a DataFrame plus parameters and returns plain dicts / DataFrames, so
it is directly unit-testable.

Decision model
--------------
A transaction is DECLINED when its fraud_score >= decline_threshold,
otherwise it is APPROVED. Among approved transactions, those scoring
below `review_floor` are AUTO-APPROVED (no human touch); the rest are
sent to MANUAL REVIEW (an operational cost).

    score >= decline_threshold        -> DECLINE
    review_floor <= score < threshold -> APPROVE (manual review)
    score < review_floor              -> APPROVE (auto)

Economic model (all assumptions are explicit and SYNTHETIC)
-----------------------------------------------------------
    * Fraud that is DECLINED  -> loss prevented   (benefit)
    * Fraud that is APPROVED  -> chargeback loss   (= full amount)
    * Good txn FALSELY DECLINED -> lost customer value
          (= amount * false_decline_cost_rate)
    * Each MANUAL REVIEW       -> fixed ops cost (review_cost_per_txn)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconomicAssumptions:
    """Dollar assumptions for the ROI model. All synthetic / editable."""

    false_decline_cost_rate: float = 0.25  # share of a declined good txn's
    #                                         value treated as lost (margin
    #                                         + downstream customer value)
    review_cost_per_txn: float = 4.00       # fully-loaded cost of one manual
    #                                         review

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


# --------------------------------------------------------------------------
# Confusion + rate metrics
# --------------------------------------------------------------------------
def confusion_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, int]:
    """Counts of TP / FP / FN / TN for 'decline when score >= threshold'.

    TP = fraud correctly declined
    FP = good customer wrongly declined
    FN = fraud wrongly approved
    TN = good customer correctly approved
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    declined = scores >= threshold
    fraud = labels == 1

    tp = int(np.sum(declined & fraud))
    fp = int(np.sum(declined & ~fraud))
    fn = int(np.sum(~declined & fraud))
    tn = int(np.sum(~declined & ~fraud))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def metrics_at_threshold(
    df: pd.DataFrame,
    threshold: float,
    review_floor: float = 0.30,
    score_col: str = "fraud_score",
    label_col: str = "is_fraud",
) -> Dict[str, float]:
    """Headline rate metrics at a decline threshold.

    Returns fraud capture rate (recall), false-positive rate, precision,
    overall approval rate, and auto-approve rate.
    """
    scores = df[score_col].to_numpy()
    labels = df[label_col].to_numpy().astype(int)
    c = confusion_at_threshold(labels, scores, threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    total = tp + fp + fn + tn

    approved = tn + fn
    auto_approved = int(np.sum(scores < review_floor))
    manual_review = int(np.sum((scores >= review_floor) & (scores < threshold)))

    return {
        "threshold": float(threshold),
        "fraud_capture_rate": _safe_div(tp, tp + fn),   # recall / TPR
        "false_positive_rate": _safe_div(fp, fp + tn),  # FPR among good
        "precision": _safe_div(tp, tp + fp),
        "approval_rate": _safe_div(approved, total),
        "auto_approve_rate": _safe_div(auto_approved, total),
        "manual_review_rate": _safe_div(manual_review, total),
        "decline_rate": _safe_div(tp + fp, total),
        **{k: int(v) for k, v in c.items()},
        "manual_review_count": manual_review,
        "total": int(total),
    }


# --------------------------------------------------------------------------
# Curves (ROC + precision/recall across thresholds)
# --------------------------------------------------------------------------
def curve_points(
    df: pd.DataFrame,
    n_thresholds: int = 101,
    score_col: str = "fraud_score",
    label_col: str = "is_fraud",
) -> pd.DataFrame:
    """Sweep thresholds and return rate metrics at each.

    Columns: threshold, tpr (fraud capture), fpr, precision, recall,
    approval_rate.
    """
    scores = df[score_col].to_numpy()
    labels = df[label_col].to_numpy().astype(int)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    rows = []
    for t in thresholds:
        c = confusion_at_threshold(labels, scores, t)
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        total = tp + fp + fn + tn
        rows.append(
            {
                "threshold": float(t),
                "tpr": _safe_div(tp, tp + fn),
                "fpr": _safe_div(fp, fp + tn),
                "precision": _safe_div(tp, tp + fp),
                "recall": _safe_div(tp, tp + fn),
                "approval_rate": _safe_div(tn + fn, total),
            }
        )
    return pd.DataFrame(rows)


def auc_roc(df: pd.DataFrame, score_col: str = "fraud_score",
            label_col: str = "is_fraud") -> float:
    """Area under the ROC curve via the rank (Mann-Whitney) statistic."""
    labels = df[label_col].to_numpy().astype(int)
    scores = df[score_col].to_numpy(dtype=float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    avg_ranks = sum_ranks / counts
    ranks = avg_ranks[inv]
    sum_pos = ranks[labels == 1].sum()
    n_pos, n_neg = pos.size, neg.size
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# Economics + ROI
# --------------------------------------------------------------------------
def economics_at_threshold(
    df: pd.DataFrame,
    threshold: float,
    assumptions: EconomicAssumptions | None = None,
    review_floor: float = 0.30,
    score_col: str = "fraud_score",
    label_col: str = "is_fraud",
    amount_col: str = "amount",
) -> Dict[str, float]:
    """Translate a threshold into dollars."""
    a = assumptions or EconomicAssumptions()
    scores = df[score_col].to_numpy()
    labels = df[label_col].to_numpy().astype(int)
    amount = df[amount_col].to_numpy(dtype=float)

    declined = scores >= threshold
    fraud = labels == 1

    tp_mask = declined & fraud          # fraud prevented
    fp_mask = declined & ~fraud         # good customers falsely declined
    fn_mask = ~declined & fraud         # fraud that slipped through

    fraud_prevented = float(amount[tp_mask].sum())
    fraud_loss = float(amount[fn_mask].sum())
    false_decline_value = float(amount[fp_mask].sum())
    false_decline_cost = false_decline_value * a.false_decline_cost_rate

    review_count = int(np.sum((scores >= review_floor) & (scores < threshold)))
    review_cost = review_count * a.review_cost_per_txn

    total_cost = fraud_loss + false_decline_cost + review_cost

    return {
        "threshold": float(threshold),
        "fraud_prevented_usd": fraud_prevented,
        "fraud_loss_usd": fraud_loss,
        "false_decline_value_usd": false_decline_value,
        "false_decline_cost_usd": false_decline_cost,
        "review_count": review_count,
        "review_cost_usd": review_cost,
        "total_cost_usd": total_cost,
    }


def roi_vs_baseline(
    df: pd.DataFrame,
    chosen_threshold: float,
    baseline_threshold: float,
    assumptions: EconomicAssumptions | None = None,
    review_floor: float = 0.30,
) -> Dict[str, float]:
    """Compare a chosen threshold to a baseline threshold, in dollars.

    `net_savings_usd` > 0 means the chosen threshold costs the client
    LESS than the baseline (lower total of fraud losses + false-decline
    cost + review cost).
    """
    a = assumptions or EconomicAssumptions()
    chosen = economics_at_threshold(df, chosen_threshold, a, review_floor)
    base = economics_at_threshold(df, baseline_threshold, a, review_floor)

    return {
        "chosen_threshold": float(chosen_threshold),
        "baseline_threshold": float(baseline_threshold),
        "chosen_total_cost_usd": chosen["total_cost_usd"],
        "baseline_total_cost_usd": base["total_cost_usd"],
        "net_savings_usd": base["total_cost_usd"] - chosen["total_cost_usd"],
        "fraud_prevented_delta_usd": (
            chosen["fraud_prevented_usd"] - base["fraud_prevented_usd"]
        ),
        "false_decline_cost_delta_usd": (
            chosen["false_decline_cost_usd"] - base["false_decline_cost_usd"]
        ),
        "review_cost_delta_usd": (
            chosen["review_cost_usd"] - base["review_cost_usd"]
        ),
        "fraud_loss_delta_usd": (
            chosen["fraud_loss_usd"] - base["fraud_loss_usd"]
        ),
    }
