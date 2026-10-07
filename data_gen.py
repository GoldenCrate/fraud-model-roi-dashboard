"""
data_gen.py — Synthetic scored-transaction generator.

=====================================================================
SYNTHETIC DATA ONLY. Nothing here is real. No real Socure model, no
real customer, and no real transaction is represented. Every number is
produced by a fixed-seed random generator so the dashboard is fully
reproducible and safe to share publicly.
=====================================================================

Each row is one transaction that has already been *scored* by a
hypothetical fraud model:

    txn_id        unique id
    amount        dollar value of the transaction
    is_fraud      ground-truth label (1 = fraud, 0 = legitimate)
    fraud_score   model's risk score in [0, 1] (higher = riskier)
    segment       a client-style business segment, for slicing

The score distributions deliberately OVERLAP between fraud and
legitimate transactions — that overlap is the whole point of a
threshold trade-off: you cannot separate the two groups perfectly, so
every cutoff caught-fraud-vs-false-positive is a business decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed seed -> identical data every run.
DEFAULT_SEED = 42
DEFAULT_N = 60_000
DEFAULT_FRAUD_RATE = 0.018  # ~1.8% base rate, realistic for many flows

SEGMENTS = ("Marketplace", "Digital Goods", "Lending", "Money Movement")


def generate_transactions(
    n: int = DEFAULT_N,
    fraud_rate: float = DEFAULT_FRAUD_RATE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate a reproducible, SYNTHETIC set of scored transactions.

    Returns a DataFrame with one row per transaction. The generator is
    deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)

    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    n_fraud = int(is_fraud.sum())
    n_legit = n - n_fraud

    # --- Fraud scores -------------------------------------------------
    # Legit transactions score low but with a tail (Beta skewed low).
    # Fraud transactions score high but with a tail into the low range
    # (Beta skewed high). The overlap creates the trade-off.
    scores = np.empty(n, dtype=float)
    scores[is_fraud == 0] = rng.beta(2.0, 9.0, size=n_legit)
    scores[is_fraud == 1] = rng.beta(6.0, 3.0, size=n_fraud)
    scores = np.clip(scores, 0.0, 1.0)

    # --- Dollar amounts ----------------------------------------------
    # Lognormal spend. Fraud skews toward larger tickets.
    amount = np.empty(n, dtype=float)
    amount[is_fraud == 0] = rng.lognormal(mean=4.1, sigma=0.9, size=n_legit)
    amount[is_fraud == 1] = rng.lognormal(mean=4.8, sigma=1.0, size=n_fraud)
    amount = np.round(np.clip(amount, 5.0, 25_000.0), 2)

    segment = rng.choice(SEGMENTS, size=n, p=[0.40, 0.30, 0.18, 0.12])

    df = pd.DataFrame(
        {
            "txn_id": np.arange(1, n + 1),
            "amount": amount,
            "is_fraud": is_fraud,
            "fraud_score": np.round(scores, 4),
            "segment": segment,
        }
    )
    return df


def main() -> None:
    """Write a copy to data/transactions.csv for inspection."""
    import os

    df = generate_transactions()
    os.makedirs("data", exist_ok=True)
    out = os.path.join("data", "transactions.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} SYNTHETIC transactions -> {out}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
    print(f"Mean amount: ${df['amount'].mean():,.2f}")


if __name__ == "__main__":
    main()
