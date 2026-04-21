"""
Forecast vs Actual — 25-26 season, event-by-event bar chart.

Built for a nontechnical audience: each event is a row with two bars
(Predicted and Actual) ordered chronologically. Events that have not yet
happened show only the prediction in a lighter shade.

Output: forecast_2526_bar_chart.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from forecast_2526_comparison import (
    load_data, get_training_df, build_hierarchy_models,
    predict_model_a, cap_at_capacity, build_pwyw_lift,
)
from forecast_artist_adjustment import apply_artist_adjustment

WORKING_DIR = "/Users/antho/Documents/WPI-MW"
os.chdir(WORKING_DIR)

NAVY   = "#1A3A5C"
ORANGE = "#E8922A"
TEAL   = "#2A9EA0"
LGRAY  = "#E8EDF2"
DGRAY  = "#5A6A7A"
PRED_FUT = "#B8C9D8"  # muted navy for future predictions

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


def forecast_upcoming():
    """Run the full model on 25-26 events that haven't happened yet."""
    em, merged = load_data()

    prior_seasons = sorted([s for s in merged["Season"].dropna().unique() if s < "25-26"])
    train = get_training_df(merged, prior_seasons)
    repeat_model, primary_sf, sf_ratio, primary, f1, f2, f3, f4, f5 = build_hierarchy_models(train)
    pwyw_lift, _ = build_pwyw_lift(
        merged, prior_seasons, repeat_model, primary_sf, sf_ratio,
        primary, f1, f2, f3, f4, f5)

    em["EventDate"] = pd.to_datetime(em["EventDate"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    upc = (
        em[(em["EventDate"] >= today)
           & (em["EventDate"] < "2026-09-01")
           & (em["EventType"] == "Live")]
        .drop_duplicates("EventName")
        .copy()
    )

    upc["Actual"] = np.nan
    upc["EventCapacity"] = pd.to_numeric(upc["EventCapacity"], errors="coerce")

    fc = predict_model_a(upc, repeat_model, primary_sf, sf_ratio,
                         primary, f1, f2, f3, f4, f5, pwyw_lift=pwyw_lift)
    fc["Pred_A"] = cap_at_capacity(fc["Pred_A"], fc["EventCapacity"])

    hist_actuals = (
        train.groupby(["EventId", "EventName", "EventClass", "EventVenue",
                       "EventGenre", "EventLoB", "EventSubGenre",
                       "EventRepeat", "SeatFormat"],
                      group_keys=False, dropna=False)
        .agg(Actual=("Quantity", "sum"))
        .reset_index()
    )
    cap = em.drop_duplicates("EventId")[["EventId", "EventCapacity"]].copy()
    cap["EventCapacity"] = pd.to_numeric(cap["EventCapacity"], errors="coerce")
    hist_actuals = hist_actuals.merge(cap, on="EventId", how="left")
    hist_fc = predict_model_a(hist_actuals, repeat_model, primary_sf, sf_ratio,
                              primary, f1, f2, f3, f4, f5)

    fc = apply_artist_adjustment(
        fc,
        merged_history=hist_fc,
        actuals_history=hist_fc["Actual"],
        bucket_preds_history=hist_fc["Pred_A"],
    )
    fc["EventDate"] = upc.set_index("EventName").loc[fc["EventName"], "EventDate"].values
    return fc[["EventName", "EventDate", "EventClass", "EventVenue",
               "EventCapacity", "Pred_A", "Pred_Adj"]].copy()


def build_combined():
    comp = pd.read_excel("Forecast_2526_Comparison.xlsx", sheet_name="2526_Comparison")
    em = pd.read_excel("EventManifest.xlsx", sheet_name="EventManifest")
    em["EventDate"] = pd.to_datetime(em["EventDate"], errors="coerce")
    dates = em.drop_duplicates("EventName")[["EventName", "EventDate"]]
    comp = comp.merge(dates, on="EventName", how="left")
    comp["Status"] = "Completed"

    upc = forecast_upcoming()
    upc["Actual"] = np.nan
    upc["Status"] = "Upcoming"

    cols = ["EventName", "EventDate", "EventClass", "EventVenue",
            "EventCapacity", "Actual", "Pred_Adj", "Status"]
    combined = pd.concat([comp[cols], upc[cols]], ignore_index=True)
    combined = combined.sort_values("EventDate").reset_index(drop=True)
    return combined


def shorten(name):
    name = str(name)
    # Strip year tails and trim long prefixes for readability
    name = name.replace(" 2025", "").replace(" 2026", "")
    if len(name) > 48:
        name = name[:45] + "…"
    return name


def plot_bar_chart(df):
    n = len(df)
    fig_h = max(7.5, 0.32 * n + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y = np.arange(n)[::-1]   # top-to-bottom chronological
    bar_h = 0.38

    actual_vals = df["Actual"].fillna(0).values
    pred_vals = df["Pred_Adj"].fillna(0).values
    is_future = df["Status"].eq("Upcoming").values

    # Predicted bars (upper of the pair)
    pred_colors = [PRED_FUT if f else NAVY for f in is_future]
    ax.barh(y + bar_h / 2, pred_vals, height=bar_h,
            color=pred_colors, edgecolor="white", linewidth=0.5,
            label="Predicted", zorder=3)

    # Actual bars (lower of the pair) — only for completed events
    actual_mask = ~is_future
    ax.barh(y[actual_mask] - bar_h / 2, actual_vals[actual_mask],
            height=bar_h, color=ORANGE, edgecolor="white", linewidth=0.5,
            label="Actual", zorder=3)

    # Value labels
    xmax = max(pred_vals.max(), actual_vals.max()) * 1.14
    for yi, v, future in zip(y, pred_vals, is_future):
        color = DGRAY if future else NAVY
        ax.text(v + xmax * 0.006, yi + bar_h / 2, f"{int(round(v)):,}",
                va="center", ha="left", fontsize=8, color=color,
                fontweight="bold" if not future else "normal")
    for yi, v, future in zip(y, actual_vals, is_future):
        if future:
            continue
        ax.text(v + xmax * 0.006, yi - bar_h / 2, f"{int(round(v)):,}",
                va="center", ha="left", fontsize=8, color=ORANGE,
                fontweight="bold")

    # Y labels = event names + date
    labels = [f"{shorten(r.EventName)}   {r.EventDate:%b %d}"
              for r in df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlim(0, xmax)
    ax.set_xlabel("Tickets", fontsize=10, color=DGRAY)
    ax.grid(axis="x", color=LGRAY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(LGRAY)
    ax.spines["bottom"].set_color(LGRAY)

    # Legend
    handles = [
        Patch(facecolor=ORANGE, label="Actual"),
        Patch(facecolor=NAVY,   label="Predicted (completed)"),
        Patch(facecolor=PRED_FUT, label="Predicted (upcoming)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              frameon=True, framealpha=0.95, edgecolor=LGRAY)

    # Summary box — completed events only
    done = df[df["Status"] == "Completed"].copy()
    done["abs_pct"] = (done["Pred_Adj"] - done["Actual"]).abs() / done["Actual"]
    done["signed_pct"] = (done["Pred_Adj"] - done["Actual"]) / done["Actual"]
    mape = done["abs_pct"].mean() * 100
    bias = done["signed_pct"].mean() * 100
    n_done = len(done)
    n_upc = is_future.sum()

    subtitle = (f"{n_done} completed events  ·  MAPE {mape:.0f}%  ·  "
                f"Bias {bias:+.0f}%  ·  {n_upc} upcoming (prediction only)")
    fig.suptitle("25–26 Season: Predicted vs Actual Attendance",
                 fontsize=14, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.972, subtitle, ha="left", va="top",
             fontsize=10, color=DGRAY)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = "forecast_2526_bar_chart.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    df = build_combined()
    print(f"Combined: {len(df)} events "
          f"({(df['Status']=='Completed').sum()} completed, "
          f"{(df['Status']=='Upcoming').sum()} upcoming)")
    print(df[["EventDate", "EventName", "Status", "Actual", "Pred_Adj"]].to_string(index=False))
    plot_bar_chart(df)


if __name__ == "__main__":
    main()
