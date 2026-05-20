"""
Forecast CI chart — 25-26 season.

Horizontal whisker plot: each event is a row showing the 90% prediction
interval (Lo—Hi) with the point prediction marked, and the actual
overlaid for completed events. Upcoming events show the band only.

Input:  forecasting/Forecast_2526_FullSeason.xlsx (sheet "FullSeason")
Output: forecasting/forecast_2526_ci_chart.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

WORKING_DIR = "/Users/antho/Documents/WPI-MW"
os.chdir(WORKING_DIR)

NAVY   = "#1A3A5C"
ORANGE = "#E8922A"
BAND   = "#B8C9D8"
LGRAY  = "#E8EDF2"
DGRAY  = "#5A6A7A"
GREEN  = "#2A9E6E"
RED    = "#C94B4B"

IN_XLSX  = "forecasting/Forecast_2526_FullSeason.xlsx"
OUT_PNG  = "forecasting/forecast_2526_ci_chart.png"

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


def shorten(name, n=42):
    return name if len(name) <= n else name[: n - 1] + "…"


def main():
    df = pd.read_excel(IN_XLSX, sheet_name="Detail")
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    df = df.sort_values("EventDate").reset_index(drop=True)

    n = len(df)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.32)))

    lo   = df["Pred_Adj_Lo"].values
    hi   = df["Pred_Adj_Hi"].values
    pred = df["Pred_Adj"].values
    act  = df["Actual"].values
    is_upc = df["Status"].eq("Upcoming").values

    xmax = max(np.nanmax(hi), np.nanmax(act)) * 1.12

    has_ci = np.isfinite(lo) & np.isfinite(hi) & (hi - lo > 0)

    # 90% CI band — only drawn for events with quantified uncertainty
    for yi, l, h, up, hc in zip(y, lo, hi, is_upc, has_ci):
        if hc:
            color = BAND if up else NAVY
            alpha = 0.55 if up else 0.28
            ax.plot([l, h], [yi, yi], color=color, linewidth=7.0,
                    alpha=alpha, zorder=2, solid_capstyle="round")

    # Point prediction
    pred_colors = [BAND if up else NAVY for up in is_upc]
    ax.scatter(pred, y, marker="D", s=28,
               color=pred_colors, edgecolor="white", linewidth=0.8,
               zorder=4)

    # Actuals:
    #   - green when inside a quantified CI
    #   - orange when outside a quantified CI
    #   - grey when no CI (bucket-only fallback)
    in_ci   = has_ci & np.isfinite(act) & (act >= lo) & (act <= hi)
    out_ci  = has_ci & np.isfinite(act) & ~in_ci
    no_ci   = ~has_ci & np.isfinite(act)

    ax.scatter(act[in_ci],  y[in_ci],  marker="o", s=62,
               color=GREEN,  edgecolor="white", linewidth=1.1, zorder=5)
    ax.scatter(act[out_ci], y[out_ci], marker="o", s=62,
               color=ORANGE, edgecolor="white", linewidth=1.1, zorder=5)
    ax.scatter(act[no_ci],  y[no_ci],  marker="o", s=52,
               color=DGRAY,  edgecolor="white", linewidth=1.1, zorder=5)

    labels = [f"{shorten(r.EventName)}   {r.EventDate:%b %d}"
              for r in df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Attendance", fontsize=9)
    ax.grid(axis="x", color=LGRAY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(LGRAY)
    ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(axis="x", labelsize=8)

    n_done     = int((~is_upc).sum())
    n_upc      = int(is_upc.sum())
    n_done_ci  = int((has_ci & ~is_upc).sum())
    n_in_ci    = int(in_ci.sum())
    coverage   = n_in_ci / n_done_ci * 100 if n_done_ci else 0.0

    fig.suptitle("25–26 Season: Prediction with 90% CI vs. Actual",
                 fontsize=13, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.99)
    subtitle = (f"{n_done} completed  ·  {n_in_ci}/{n_done_ci} inside CI "
                f"where CI quantified ({coverage:.0f}% coverage)  ·  "
                f"{n_done - n_done_ci} bucket-only (no CI)  ·  "
                f"{n_upc} upcoming")
    fig.text(0.02, 0.955, subtitle, ha="left", va="top",
             fontsize=9.5, color=DGRAY)

    legend_handles = [
        Line2D([0], [0], color=NAVY, linewidth=7, alpha=0.28,
               label="90% prediction interval"),
        Line2D([0], [0], marker="D", linestyle="",
               markerfacecolor=NAVY, markeredgecolor="white",
               markersize=7, label="Point prediction"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=GREEN, markeredgecolor="white",
               markersize=8, label="Actual — inside CI"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=ORANGE, markeredgecolor="white",
               markersize=8, label="Actual — outside CI"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=DGRAY, markeredgecolor="white",
               markersize=7, label="Actual — no CI (bucket-only)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              frameon=False, fontsize=8.5, ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"✓ {OUT_PNG}  ({n_done} completed, {n_upc} upcoming, "
          f"CI coverage {coverage:.0f}%)")


if __name__ == "__main__":
    main()
