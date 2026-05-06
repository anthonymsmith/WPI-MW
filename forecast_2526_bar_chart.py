"""
Forecast vs Actual — 25-26 season, event-by-event bar chart.

Each event is a row with two bars (Predicted, Actual) ordered chronologically.
Bars are stacked: paid tickets on the left, comps on the right. Upcoming
events show only the predicted bar.

Pulls data from Forecast_2526_FullSeason.xlsx (produced by
forecast_2526_full_season.py).

Output: forecast_2526_bar_chart.png, forecast_2526_bar_chart_wide.png

Run with `python forecast_2526_bar_chart.py --anon` to also produce
*_anon.png variants with event names + per-bar totals + x-axis tick numbers
stripped (for public/portfolio use).
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ANON = "--anon" in sys.argv

WORKING_DIR = "/Users/antho/Documents/WPI-MW"
os.chdir(WORKING_DIR)

# Navy family — predicted
PRED_PAID  = "#1A3A5C"
PRED_COMP  = "#7E9FBF"
# Orange family — actual
ACT_PAID   = "#E8922A"
ACT_COMP   = "#F4BE80"
# Neutrals
LGRAY  = "#E8EDF2"
DGRAY  = "#5A6A7A"
NAVY   = "#1A3A5C"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelcolor":   DGRAY,
    "xtick.color":       DGRAY,
    "ytick.color":       DGRAY,
    "text.color":        NAVY,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})


def load_forecast():
    df = pd.read_excel("Forecast_2526_FullSeason.xlsx")
    df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
    df = df.sort_values("EventDate").reset_index(drop=True)
    for c in ["Actual", "Actual_Paid", "Actual_Comp",
              "Pred_Adj", "Pred_Paid", "Pred_Comp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def shorten(name):
    name = str(name).replace(" 2025", "").replace(" 2026", "")
    return name if len(name) <= 48 else name[:45] + "…"


def _draw_stacked_pair(ax, y, bar_h, row, xmax, fontsize=8):
    """Draw predicted (paid|comp stacked) and — if completed — actual (paid|comp stacked)."""
    is_future = row["Status"] == "Upcoming"

    # Predicted (upper bar)
    p_paid = row["Pred_Paid"] or 0
    p_comp = row["Pred_Comp"] or 0
    ax.barh(y + bar_h / 2, p_paid, height=bar_h,
            color=PRED_PAID, edgecolor="white", linewidth=0.5, zorder=3)
    ax.barh(y + bar_h / 2, p_comp, left=p_paid, height=bar_h,
            color=PRED_COMP, edgecolor="white", linewidth=0.5, zorder=3)
    if not ANON:
        total_p = p_paid + p_comp
        ax.text(total_p + xmax * 0.006, y + bar_h / 2,
                f"{int(round(total_p)):,}",
                va="center", ha="left", fontsize=fontsize,
                color=PRED_PAID, fontweight="bold")

    # Actual (lower bar) — only if completed
    if is_future:
        return
    a_paid = row["Actual_Paid"] or 0
    a_comp = row["Actual_Comp"] or 0
    ax.barh(y - bar_h / 2, a_paid, height=bar_h,
            color=ACT_PAID, edgecolor="white", linewidth=0.5, zorder=3)
    ax.barh(y - bar_h / 2, a_comp, left=a_paid, height=bar_h,
            color=ACT_COMP, edgecolor="white", linewidth=0.5, zorder=3)
    if not ANON:
        total_a = a_paid + a_comp
        ax.text(total_a + xmax * 0.006, y - bar_h / 2,
                f"{int(round(total_a)):,}",
                va="center", ha="left", fontsize=fontsize,
                color=ACT_PAID, fontweight="bold")


def _summary_line(df):
    done = df[df["Status"] == "Completed"].copy()
    done["abs_pct"]    = (done["Pred_Adj"] - done["Actual"]).abs() / done["Actual"]
    done["signed_pct"] = (done["Pred_Adj"] - done["Actual"])       / done["Actual"]
    mape = done["abs_pct"].mean() * 100
    bias = done["signed_pct"].mean() * 100
    n_done = len(done)
    n_upc  = (df["Status"] == "Upcoming").sum()
    if ANON:
        return (f"Live-season backtest  ·  MAPE {mape:.0f}%  ·  "
                f"Bias {bias:+.0f}%  ·  anonymized")
    paid_share = done["Actual_Paid"].sum() / done["Actual"].sum() * 100
    return (f"{n_done} completed  ·  MAPE {mape:.0f}%  ·  "
            f"Bias {bias:+.0f}%  ·  paid share {paid_share:.0f}%  ·  "
            f"{n_upc} upcoming (prediction only)")


def _anon_label(idx, row):
    return f"Event {idx + 1:02d}"


def _legend_handles():
    return [
        Patch(facecolor=PRED_PAID, label="Predicted — paid"),
        Patch(facecolor=PRED_COMP, label="Predicted — comp"),
        Patch(facecolor=ACT_PAID,  label="Actual — paid"),
        Patch(facecolor=ACT_COMP,  label="Actual — comp"),
    ]


def plot_bar_chart(df):
    n = len(df)
    fig_h = max(7.5, 0.32 * n + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y = np.arange(n)[::-1]
    bar_h = 0.38

    totals = (df["Pred_Paid"].fillna(0) + df["Pred_Comp"].fillna(0)).tolist() + \
             (df["Actual_Paid"].fillna(0) + df["Actual_Comp"].fillna(0)).tolist()
    xmax = max(totals) * 1.14

    for yi, (_, row) in zip(y, df.iterrows()):
        _draw_stacked_pair(ax, yi, bar_h, row, xmax, fontsize=8)

    if ANON:
        labels = [_anon_label(i, r) for i, r in enumerate(df.itertuples())]
    else:
        labels = [f"{shorten(r.EventName)}   {r.EventDate:%b %d}"
                  for r in df.itertuples()]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Tickets" + (" (scale removed)" if ANON else ""),
                  fontsize=10, color=DGRAY)
    if ANON:
        ax.set_xticklabels([])
    ax.grid(axis="x", color=LGRAY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(LGRAY)
    ax.spines["bottom"].set_color(LGRAY)

    ax.legend(handles=_legend_handles(), loc="lower right",
              fontsize=9, frameon=True, framealpha=0.95, edgecolor=LGRAY)

    title = ("Season Forecast: Predicted vs Actual Attendance (paid + comp)"
             if ANON else
             "25–26 Season: Predicted vs Actual Attendance (paid + comp)")
    fig.suptitle(title, fontsize=14, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.972, _summary_line(df),
             ha="left", va="top", fontsize=10, color=DGRAY)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = ("forecast_2526_bar_chart_anon.png" if ANON
           else "forecast_2526_bar_chart.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_bar_chart_wide(df):
    n = len(df)
    half = (n + 1) // 2
    groups = [df.iloc[:half].reset_index(drop=True),
              df.iloc[half:].reset_index(drop=True)]

    totals = (df["Pred_Paid"].fillna(0) + df["Pred_Comp"].fillna(0)).tolist() + \
             (df["Actual_Paid"].fillna(0) + df["Actual_Comp"].fillna(0)).tolist()
    xmax = max(totals) * 1.18

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6),
                             gridspec_kw={"wspace": 0.42})

    for ax, grp in zip(axes, groups):
        m = len(grp)
        y = np.arange(m)[::-1]
        bar_h = 0.38
        for yi, (_, row) in zip(y, grp.iterrows()):
            _draw_stacked_pair(ax, yi, bar_h, row, xmax, fontsize=7.2)

        if ANON:
            labels = [_anon_label(i + (0 if grp is groups[0] else len(groups[0])),
                                  r) for i, r in enumerate(grp.itertuples())]
        else:
            labels = [f"{shorten(r.EventName)}   {r.EventDate:%b %d}"
                      for r in grp.itertuples()]
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", color=LGRAY, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color(LGRAY)
        ax.spines["bottom"].set_color(LGRAY)
        ax.tick_params(axis="x", labelsize=8)
        if ANON:
            ax.set_xticklabels([])

    title = ("Season Forecast: Predicted vs Actual Attendance (paid + comp)"
             if ANON else
             "25–26 Season: Predicted vs Actual Attendance (paid + comp)")
    fig.suptitle(title, fontsize=14, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.955, _summary_line(df),
             ha="left", va="top", fontsize=10, color=DGRAY)

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.015))

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out = ("forecast_2526_bar_chart_wide_anon.png" if ANON
           else "forecast_2526_bar_chart_wide.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_bar_chart_event_axis(df):
    """Single wide chart: events on x-axis, paired vertical bars (Predicted | Actual)."""
    n = len(df)
    fig_w = max(11.0, 0.42 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))

    x = np.arange(n)
    bar_w = 0.38

    totals = (df["Pred_Paid"].fillna(0) + df["Pred_Comp"].fillna(0)).tolist() + \
             (df["Actual_Paid"].fillna(0) + df["Actual_Comp"].fillna(0)).tolist()
    ymax = max(totals) * 1.18

    for xi, (_, row) in zip(x, df.iterrows()):
        is_future = row["Status"] == "Upcoming"
        p_paid = row["Pred_Paid"] or 0
        p_comp = row["Pred_Comp"] or 0
        ax.bar(xi - bar_w / 2, p_paid, width=bar_w,
               color=PRED_PAID, edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar(xi - bar_w / 2, p_comp, width=bar_w, bottom=p_paid,
               color=PRED_COMP, edgecolor="white", linewidth=0.5, zorder=3)
        if is_future:
            continue
        a_paid = row["Actual_Paid"] or 0
        a_comp = row["Actual_Comp"] or 0
        ax.bar(xi + bar_w / 2, a_paid, width=bar_w,
               color=ACT_PAID, edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar(xi + bar_w / 2, a_comp, width=bar_w, bottom=a_paid,
               color=ACT_COMP, edgecolor="white", linewidth=0.5, zorder=3)

    if ANON:
        labels = [_anon_label(i, r) for i, r in enumerate(df.itertuples())]
    else:
        labels = [f"{shorten(r.EventName)}\n{r.EventDate:%b %d}"
                  for r in df.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Tickets" + (" (scale removed)" if ANON else ""),
                  fontsize=10, color=DGRAY)
    if ANON:
        ax.set_yticklabels([])
    ax.grid(axis="y", color=LGRAY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(LGRAY)
    ax.spines["bottom"].set_color(LGRAY)

    ax.legend(handles=_legend_handles(), loc="upper right",
              fontsize=9, frameon=True, framealpha=0.95, edgecolor=LGRAY)

    title = ("Season Forecast: Predicted vs Actual Attendance (paid + comp)"
             if ANON else
             "25–26 Season: Predicted vs Actual Attendance (paid + comp)")
    fig.suptitle(title, fontsize=14, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.945, _summary_line(df),
             ha="left", va="top", fontsize=10, color=DGRAY)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = ("forecast_2526_bar_chart_eventaxis_anon.png" if ANON
           else "forecast_2526_bar_chart_eventaxis.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    df = load_forecast()
    print(f"Loaded: {len(df)} events "
          f"({(df['Status']=='Completed').sum()} completed, "
          f"{(df['Status']=='Upcoming').sum()} upcoming)")
    plot_bar_chart(df)
    plot_bar_chart_wide(df)
    plot_bar_chart_event_axis(df)


if __name__ == "__main__":
    main()
