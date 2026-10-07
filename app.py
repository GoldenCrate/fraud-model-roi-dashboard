"""
app.py — Fraud-Model Client ROI dashboard (Streamlit).

A narrative dashboard that turns a fraud model's score distribution into
a client-facing business story: how many fraud dollars are stopped, how
many good customers are protected from false declines, and what the
right decision cutoff is worth versus the client's current rule.

SYNTHETIC DATA ONLY — see README and the banner below.

Charts follow Cole Nussbaumer Knaflic's "Storytelling with Data":
takeaway titles, decluttered axes, direct labels, and restrained color
(gray for context, one accent to carry the point).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A
from data_gen import generate_transactions

# --------------------------------------------------------------------------
# Restrained palette (Knaflic): gray carries context, one accent per view.
# --------------------------------------------------------------------------
GRAY = "#B4B4B4"
GRAY_DK = "#6E6E6E"
BLUE = "#2E6E9E"   # primary accent
RED = "#C0392B"    # fraud / loss
GREEN = "#2E8B57"  # savings / good outcome
INK = "#2B2B2B"

st.set_page_config(
    page_title="Fraud-Model Client ROI",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------------------------------
# Data + cached compute
# --------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return generate_transactions()


@st.cache_data
def load_curves(_df: pd.DataFrame) -> pd.DataFrame:
    return A.curve_points(_df, n_thresholds=101)


df = load_data()
curves = load_curves(df)
auc = A.auc_roc(df)

# --------------------------------------------------------------------------
# Sidebar: global controls that drive the whole narrative
# --------------------------------------------------------------------------
st.sidebar.title("Fraud-Model Client ROI")
st.sidebar.caption("Synthetic demo • Leo Chan")

view = st.sidebar.radio(
    "View",
    [
        "1 · Executive summary",
        "2 · Model performance",
        "3 · Threshold explorer",
        "4 · Client ROI story",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Decision cutoff")
threshold = st.sidebar.slider(
    "Decline when fraud score ≥",
    min_value=0.05, max_value=0.95, value=0.55, step=0.01,
    help="Transactions scoring at or above this are declined.",
)
baseline_threshold = st.sidebar.slider(
    "Client's current cutoff (baseline)",
    min_value=0.05, max_value=0.95, value=0.80, step=0.01,
    help="The rule the client runs today; the ROI view compares to this.",
)
review_floor = st.sidebar.slider(
    "Auto-approve below score",
    min_value=0.05, max_value=0.50, value=0.30, step=0.01,
    help="Approved transactions below this score skip manual review.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dollar assumptions")
fd_rate = st.sidebar.slider(
    "False-decline cost (share of lost txn value)",
    0.05, 0.60, 0.25, 0.05,
)
review_cost = st.sidebar.number_input(
    "Manual review cost ($/txn)", min_value=0.0, value=4.0, step=0.5,
)
assume = A.EconomicAssumptions(
    false_decline_cost_rate=fd_rate, review_cost_per_txn=review_cost
)

# Metrics shared across views
m = A.metrics_at_threshold(df, threshold, review_floor)
m_base = A.metrics_at_threshold(df, baseline_threshold, review_floor)
roi = A.roi_vs_baseline(df, threshold, baseline_threshold, assume, review_floor)
econ = A.economics_at_threshold(df, threshold, assume, review_floor)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x/1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}${x/1e3:.0f}K"
    return f"{sign}${x:,.0f}"


def so_what(text: str) -> None:
    st.markdown(
        f"""
        <div style="background:#F4F7FA;border-left:5px solid {BLUE};
        padding:14px 18px;border-radius:4px;margin-top:8px;">
        <b style="color:{BLUE};">So what &rarr;</b> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def banner() -> None:
    st.markdown(
        f"""
        <div style="background:#FFF6E6;border:1px solid #E8C97A;
        padding:8px 14px;border-radius:4px;font-size:0.85rem;color:{INK};">
        <b>Synthetic data.</b> All transactions, scores, and dollars are
        randomly generated with a fixed seed. No real model, customer, or
        Socure data is used.
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=INK)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=INK, size=13),
        margin=dict(l=60, r=30, t=70, b=50),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRAY)
    fig.update_yaxes(showgrid=True, gridcolor="#EDEDED", zeroline=False,
                     linecolor="white")
    return fig


# ==========================================================================
# VIEW 1 — EXECUTIVE SUMMARY (headline KPI tiles)
# ==========================================================================
if view.startswith("1"):
    st.title("Fraud-model performance, in the client's terms")
    banner()
    st.write("")
    st.caption(
        f"At the selected cutoff of **{threshold:.2f}** across "
        f"{m['total']:,} synthetic transactions "
        f"(model AUC = {auc:.3f})."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Fraud capture rate",
        f"{m['fraud_capture_rate']:.1%}",
        f"{(m['fraud_capture_rate']-m_base['fraud_capture_rate'])*100:+.1f} pts vs current",
    )
    c2.metric(
        "False-positive rate",
        f"{m['false_positive_rate']:.2%}",
        f"{(m['false_positive_rate']-m_base['false_positive_rate'])*100:+.2f} pts vs current",
        delta_color="inverse",
    )
    c3.metric(
        "Approval rate",
        f"{m['approval_rate']:.1%}",
        f"{(m['approval_rate']-m_base['approval_rate'])*100:+.1f} pts vs current",
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Auto-approve rate", f"{m['auto_approve_rate']:.1%}")
    c5.metric("$ fraud prevented", money(econ["fraud_prevented_usd"]))
    c6.metric(
        "$ good customers saved from false declines",
        money(-roi["false_decline_cost_delta_usd"]),
        help="Reduction in false-decline cost versus the client's current cutoff.",
    )

    st.markdown("---")
    left, right = st.columns([3, 2])
    with left:
        # Where the model separates fraud from good traffic.
        fig = go.Figure()
        fig.add_histogram(
            x=df.loc[df.is_fraud == 0, "fraud_score"], nbinsx=50,
            marker_color=GRAY, name="Legitimate", histnorm="probability density",
            opacity=0.85,
        )
        fig.add_histogram(
            x=df.loc[df.is_fraud == 1, "fraud_score"], nbinsx=50,
            marker_color=RED, name="Fraud", histnorm="probability density",
            opacity=0.7,
        )
        fig.add_vline(x=threshold, line_width=2, line_dash="dash",
                      line_color=INK)
        fig.add_annotation(x=threshold, y=1, yref="paper",
                           text=f"Decline ≥ {threshold:.2f}",
                           showarrow=False, yshift=10, font=dict(color=INK))
        fig.update_layout(barmode="overlay")
        fig = clean_layout(
            fig, "Fraud scores high, but the tails overlap — so the cutoff is a choice")
        fig.add_annotation(x=0.12, y=0.9, yref="paper", text="Good traffic",
                           showarrow=False, font=dict(color=GRAY_DK, size=13))
        fig.add_annotation(x=0.85, y=0.5, yref="paper", text="Fraud",
                           showarrow=False, font=dict(color=RED, size=13))
        st.plotly_chart(fig, width='stretch')
    with right:
        st.subheader("Read this dashboard as a client story")
        st.markdown(
            "- **View 2** shows the model can rank risk well (AUC "
            f"{auc:.3f}).\n"
            "- **View 3** lets you move the decline cutoff and watch fraud "
            "caught trade off against good customers declined.\n"
            "- **View 4** turns the chosen cutoff into dollars versus the "
            "client's current rule."
        )

    so_what(
        f"At a {threshold:.2f} cutoff the model catches "
        f"**{m['fraud_capture_rate']:.0%} of fraud** while declining only "
        f"**{m['false_positive_rate']:.2%}** of good customers. Moving off the "
        f"client's current {baseline_threshold:.2f} rule changes both fraud "
        f"losses and good-customer friction — quantified in View 4."
    )


# ==========================================================================
# VIEW 2 — MODEL PERFORMANCE
# ==========================================================================
elif view.startswith("2"):
    st.title("The model ranks risk well before any cutoff is picked")
    banner()
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        # ROC curve with the operating point marked.
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(color=GRAY, dash="dot"), name="Chance")
        fig.add_scatter(x=curves["fpr"], y=curves["tpr"], mode="lines",
                        line=dict(color=BLUE, width=3), name="Model")
        fig.add_scatter(x=[m["false_positive_rate"]],
                        y=[m["fraud_capture_rate"]], mode="markers",
                        marker=dict(color=RED, size=13),
                        name="Selected cutoff")
        fig.add_annotation(
            x=m["false_positive_rate"], y=m["fraud_capture_rate"],
            text=f"  cutoff {threshold:.2f}", showarrow=True, arrowcolor=INK,
            ax=40, ay=20, font=dict(color=INK),
        )
        fig = clean_layout(fig, f"ROC curve — AUC = {auc:.3f}")
        fig.update_xaxes(title="False-positive rate (good customers declined)",
                         range=[0, 0.3])
        fig.update_yaxes(title="Fraud capture rate", range=[0, 1.02])
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Precision / recall vs threshold with operating point.
        fig = go.Figure()
        fig.add_scatter(x=curves["threshold"], y=curves["recall"],
                        mode="lines", line=dict(color=BLUE, width=3),
                        name="Fraud capture (recall)")
        fig.add_scatter(x=curves["threshold"], y=curves["precision"],
                        mode="lines", line=dict(color=GRAY_DK, width=2.5),
                        name="Precision")
        fig.add_vline(x=threshold, line_dash="dash", line_color=INK)
        fig = clean_layout(
            fig, "Raising the cutoff trades fraud capture for precision")
        fig.update_xaxes(title="Decline threshold")
        fig.update_yaxes(title="Rate", range=[0, 1.02])
        fig.add_annotation(x=0.3, y=A.metrics_at_threshold(df, 0.3, review_floor)["fraud_capture_rate"],
                           text="Fraud capture", showarrow=False,
                           font=dict(color=BLUE), xshift=0, yshift=18)
        fig.add_annotation(x=0.7, y=0.95, text="Precision", showarrow=False,
                           font=dict(color=GRAY_DK))
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader(f"Confusion at the selected cutoff ({threshold:.2f})")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Approved", "Declined"], y=["Legit (actual)", "Fraud (actual)"],
            text=[[f"{cm[0,0]:,}<br>correct approve", f"{cm[0,1]:,}<br>false decline"],
                  [f"{cm[1,0]:,}<br>missed fraud", f"{cm[1,1]:,}<br>fraud caught"]],
            texttemplate="%{text}", textfont=dict(size=14, color=INK),
            colorscale=[[0, "white"], [1, "#CFE0EC"]], showscale=False,
        )
    )
    fig = clean_layout(fig, "Most volume is correctly approved; the costly cells are the off-diagonal")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width='stretch')

    so_what(
        f"An AUC of **{auc:.3f}** means the model reliably ranks fraud above "
        "good traffic, so there is real room to tune the cutoff. The question "
        "is not *whether* the model works but *where* to set the line — "
        "that is a dollars decision, handled next."
    )


# ==========================================================================
# VIEW 3 — THRESHOLD EXPLORER
# ==========================================================================
elif view.startswith("3"):
    st.title("Every cutoff is a trade-off: fraud caught vs. good customers lost")
    banner()
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud caught", f"{m['tp']:,}", f"of {m['tp']+m['fn']:,} fraud txns")
    c2.metric("Good customers declined", f"{m['fp']:,}",
              f"{m['false_positive_rate']:.2%} of good", delta_color="inverse")
    c3.metric("Fraud missed", f"{m['fn']:,}", delta_color="inverse")
    c4.metric("Sent to manual review", f"{m['manual_review_count']:,}")

    # Trade-off curve: as threshold rises, fraud caught falls and good
    # declines fall too. Show fraud $ prevented vs false-decline cost.
    sweep = []
    for t in np.linspace(0.1, 0.9, 33):
        e = A.economics_at_threshold(df, t, assume, review_floor)
        sweep.append({"t": t, "prevented": e["fraud_prevented_usd"],
                      "fd_cost": e["false_decline_cost_usd"],
                      "total_cost": e["total_cost_usd"]})
    sweep = pd.DataFrame(sweep)

    fig = go.Figure()
    fig.add_scatter(x=sweep["t"], y=sweep["prevented"], mode="lines",
                    line=dict(color=GREEN, width=3), name="Fraud $ prevented")
    fig.add_scatter(x=sweep["t"], y=sweep["fd_cost"], mode="lines",
                    line=dict(color=RED, width=3), name="False-decline cost")
    fig.add_vline(x=threshold, line_dash="dash", line_color=INK)
    fig = clean_layout(
        fig, "A lower cutoff stops more fraud $ but pushes up false-decline cost")
    fig.update_xaxes(title="Decline threshold")
    fig.update_yaxes(title="USD")
    fig.add_annotation(x=0.2, y=sweep.iloc[1]["prevented"], text="Fraud $ prevented",
                       showarrow=False, font=dict(color=GREEN), yshift=15)
    fig.add_annotation(x=0.35, y=sweep[sweep.t.between(0.3,0.4)]["fd_cost"].mean(),
                       text="False-decline cost", showarrow=False,
                       font=dict(color=RED), yshift=15)
    st.plotly_chart(fig, width='stretch')

    # Total-cost curve with the minimum highlighted.
    best_row = sweep.loc[sweep["total_cost"].idxmin()]
    fig2 = go.Figure()
    fig2.add_scatter(x=sweep["t"], y=sweep["total_cost"], mode="lines",
                     line=dict(color=GRAY_DK, width=3), name="Total cost")
    fig2.add_scatter(x=[best_row["t"]], y=[best_row["total_cost"]],
                     mode="markers", marker=dict(color=BLUE, size=14))
    fig2.add_vline(x=threshold, line_dash="dash", line_color=INK)
    fig2 = clean_layout(
        fig2, f"Total cost is minimized near a {best_row['t']:.2f} cutoff")
    fig2.update_xaxes(title="Decline threshold")
    fig2.update_yaxes(title="Total cost (fraud loss + false declines + review)")
    fig2.add_annotation(x=best_row["t"], y=best_row["total_cost"],
                        text=f"  lowest cost @ {best_row['t']:.2f}",
                        showarrow=True, arrowcolor=BLUE, ax=50, ay=-30,
                        font=dict(color=BLUE))
    st.plotly_chart(fig2, width='stretch')

    gap = "above" if threshold > best_row["t"] else "below"
    so_what(
        f"On these assumptions total cost bottoms out near a "
        f"**{best_row['t']:.2f}** cutoff. Your selected **{threshold:.2f}** sits "
        f"{gap} that point — move the sidebar slider toward "
        f"{best_row['t']:.2f} to cut the client's total fraud + friction cost."
    )


# ==========================================================================
# VIEW 4 — CLIENT ROI STORY
# ==========================================================================
else:
    st.title("What the recommended cutoff is worth to the client")
    banner()
    st.write("")
    st.caption(
        f"Recommended cutoff **{threshold:.2f}** vs. the client's current "
        f"**{baseline_threshold:.2f}**, over {m['total']:,} synthetic "
        "transactions (one scoring period)."
    )

    net = roi["net_savings_usd"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Net savings vs. current rule", money(net),
              delta="lower total cost" if net >= 0 else "higher total cost",
              delta_color="normal" if net >= 0 else "inverse")
    c2.metric("Extra fraud $ prevented", money(roi["fraud_prevented_delta_usd"]))
    c3.metric("Change in false-decline cost",
              money(roi["false_decline_cost_delta_usd"]),
              delta_color="inverse")

    # Waterfall: baseline cost -> recommended cost.
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Current-rule<br>total cost", "Fraud lossΔ",
           "False-declineΔ", "Review costΔ", "Recommended<br>total cost"],
        y=[roi["baseline_total_cost_usd"], roi["fraud_loss_delta_usd"],
           roi["false_decline_cost_delta_usd"], roi["review_cost_delta_usd"],
           roi["chosen_total_cost_usd"]],
        text=[money(roi["baseline_total_cost_usd"]),
              money(roi["fraud_loss_delta_usd"]),
              money(roi["false_decline_cost_delta_usd"]),
              money(roi["review_cost_delta_usd"]),
              money(roi["chosen_total_cost_usd"])],
        textposition="outside",
        connector=dict(line=dict(color=GRAY)),
        decreasing=dict(marker=dict(color=GREEN)),
        increasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=BLUE)),
    ))
    fig = clean_layout(
        fig, f"Moving to a {threshold:.2f} cutoff changes total cost by "
             f"{money(-net)} per period")
    fig.update_yaxes(title="Total cost (USD)")
    st.plotly_chart(fig, width='stretch')

    annual = net * 12
    st.markdown("### Recommendation")
    if net >= 0:
        st.success(
            f"**Adopt the {threshold:.2f} cutoff.** Versus the client's current "
            f"{baseline_threshold:.2f} rule it lowers total cost by "
            f"**{money(net)} per period** (~**{money(annual)}/yr** at monthly "
            f"scoring) — capturing {money(roi['fraud_prevented_delta_usd'])} "
            f"more fraud while shifting false-decline cost by "
            f"{money(roi['false_decline_cost_delta_usd'])}."
        )
    else:
        st.warning(
            f"The {threshold:.2f} cutoff costs **{money(-net)} more per period** "
            f"than the current {baseline_threshold:.2f} rule here. Use View 3 to "
            "find the cost-minimizing cutoff before recommending a change."
        )

    so_what(
        "Framing the cutoff as dollars — fraud prevented, false-decline "
        "cost, and review load — lets a client decide with their own risk "
        "appetite and economics, not just an abstract ROC curve. Every input in "
        "the sidebar is theirs to set."
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Leo Chan — github.com/GoldenCrate\n\nSynthetic portfolio demo.")
