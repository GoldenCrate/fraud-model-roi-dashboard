# Fraud-Model Client ROI Dashboard

A narrative Streamlit dashboard that turns a fraud model's risk scores into a
**client-facing business story**: how much fraud is stopped, how many good
customers are protected from false declines, and what the right decision
cutoff is worth in dollars versus the client's current rule.

Built as a portfolio piece for the kind of work a **Client Analysis** data
scientist does at an identity / fraud-prevention company — Python + SQL-style
analytics translated into "convincing data stories" for clients.

> ### Synthetic data disclaimer
> **Every transaction, fraud score, label, and dollar figure in this project is
> randomly generated with a fixed seed (`data_gen.py`).** There is no real
> model, no real customer, and no real company data of any kind. The numbers
> exist only to demonstrate the analysis and the storytelling.

---

## Why this project

A fraud model produces a risk score; the business value only appears once you
choose a **decision cutoff** and translate it into money. This dashboard makes
that translation explicit and interactive, so a client can decide with their
own economics and risk appetite instead of reading an abstract ROC curve.

## Views

1. **Executive summary** — headline KPI tiles: fraud capture rate,
   false-positive rate, approval / auto-approve rate, `$` fraud prevented, and
   `$` of good customers saved from false declines (vs. the client's current
   rule). Plus the score-distribution chart that shows *why* the cutoff is a
   judgment call.
2. **Model performance** — an ROC curve (with AUC) and precision/recall vs.
   threshold, both with the selected operating point marked, plus a confusion
   matrix at the chosen cutoff.
3. **Threshold explorer** — a decision-cutoff slider that shows fraud caught,
   good customers declined, fraud $ prevented, and false-decline cost trading
   off as you move the line — and the cutoff that minimizes total cost.
4. **Client ROI story** — a waterfall translating the chosen cutoff into net
   dollars saved versus a baseline rule, with a plain-language recommendation.

Every view ends with a **"So what / recommendation"** callout.

## Storytelling with Data

Charts follow Cole Nussbaumer Knaflic's *Storytelling with Data*: takeaway
sentence titles (not labels), decluttered axes and gridlines, direct
annotation instead of legends, and a restrained palette — gray for context,
one accent color to carry each view's point (red = fraud/loss, green =
savings, blue = the recommended action).

## Project structure

| File | Role |
|------|------|
| `app.py` | Multi-view Streamlit app (UI + charts only). |
| `analytics.py` | **Framework-free, unit-tested** compute: metrics at a threshold, ROC/PR curves, AUC, economics, ROI vs. baseline. |
| `data_gen.py` | Fixed-seed synthetic scored-transaction generator. |
| `test_analytics.py` | Unit tests for the analytics layer. |
| `requirements.txt` | Dependencies. |

## Run it

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate    macOS/Linux:  source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the dashboard
streamlit run app.py
```

The app generates its synthetic data in-memory on first load (cached), so no
data file is required. To inspect the data directly:

```bash
python data_gen.py          # writes data/transactions.csv
python -m pytest -q         # run the analytics unit tests
```

## Modeling assumptions

The dollar model is intentionally simple and fully adjustable in the sidebar:

- Fraud that is **declined** → loss prevented; fraud that is **approved** →
  chargeback loss equal to the full transaction amount.
- A **falsely declined** good customer costs a configurable share of the
  transaction's value (default 25%), standing in for lost margin and
  downstream customer value.
- Each **manual review** carries a fixed operational cost (default `$4`).

These are illustrative; the point is the *method* of turning a threshold into
a client ROI conversation.

---

**Author:** Leo Chan — [github.com/GoldenCrate](https://github.com/GoldenCrate)
