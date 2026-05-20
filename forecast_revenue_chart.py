"""
Per-event revenue chart — 25-26 season.

Two bars per event (Forecast vs Actual revenue), ordered chronologically.
Uses the primary tiered uplift scenario from forecast_revenue_hindcast.py
(H/P +15%, Stdular +10%, Chorus/TCB/Mission/AiR 0%).

Pulls data from forecasting/Forecast_2526_Revenue_Hindcast.xlsx (Per-Event sheet)
produced by forecast_revenue_hindcast.py.

Output: forecasting/forecast_revenue_chart.png
Run with --anon for the anonymized variant.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ANON       = "--anon" in sys.argv
SCALE      = "--scale" in sys.argv      # keep dollar scale even in anon (for visuals)
HORIZONTAL = "--horizontal" in sys.argv # events on Y axis, side-by-side bars (larger fonts)
WORKING_DIR = "/Users/antho/Documents/WPI-MW"
os.chdir(WORKING_DIR)

PRED   = "#1A3A5C"   # navy
ACT    = "#E8922A"   # orange
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


def shorten(name, cap=24):
    name = str(name).replace(" 2025", "").replace(" 2026", "")
    name = name.replace("BACHtoberfest 2025:", "B'Fest:")
    name = name.replace("BACHtoberfest:", "B'Fest:")
    name = name.replace("Bach's Birthday Bash 2026:", "BBB:")
    name = name.replace("Bach's Birthday Bash:", "BBB:")
    name = name.replace("American Patchwork Quartet", "APQ")
    name = name.replace("American Spiritual Ensemble", "ASE")
    name = name.replace("Worcester Chamber Music Society", "WCMS")
    name = name.replace("Refugee Orchestra Project", "ROP")
    name = name.replace("Dance Theatre of Harlem", "DTH")
    name = name.replace("Women's Ensemble", "Wom Chor")
    name = name.replace("Chorus:", "Chor:")
    name = name.replace("Ladysmith Black Mambazo", "Ladysmith")
    name = name.replace("Emi Ferguson & Ruckus", "Ferguson/Ruckus")
    name = name.replace("Orchestre National de France, Daniil Trifonov", "ONF/Trifonov")
    name = name.replace("TCB: Christmas Oratorio with Winchendon Players", "TCB: X-mas Orat")
    name = name.replace("Catherine Russell & Sean Mason", "Russell/Mason")
    name = name.replace(", CONCORA, Baroklyn, Simone Dinnerstein", " CONCORA+Dnst")
    name = name.replace("BACHtoberfest 2025: Simone Dinnerstein Recital", "B'Fest: Dinnerstein")
    name = name.replace(": Simone Dinnerstein Recital", ": Dinnerstein")
    name = name.replace("Alexandre Kantorow: Piano Recital", "Kantorow")
    name = name.replace("Jordi Savall & Hesperion XXI", "Savall")
    name = name.replace("Handel Messiah", "Messiah")
    name = name.replace("Kyung-Wha Chung", "Chung")
    name = name.replace("Nelson Goerner", "Goerner")
    name = name.replace("Hermitage Trio", "Hermitage")
    name = name.replace("Aaron Diehl Trio", "Diehl Trio")
    name = name.replace("The Sebastians", "Sebastians")
    name = name.replace("Cantatathon", "Cantatathon")
    return name if len(name) <= cap else name[:cap-1] + "…"


def load():
    df = pd.read_excel(
        "forecasting/Forecast_2526_Revenue_Hindcast.xlsx", sheet_name="Per-Event")
    df = df[df["Event"].notna() & (df["Event"] != "Total")].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    for c in ["Pred Revenue", "Actual Revenue", "Pred Paid", "Actual Paid"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_chart(df, out_path, anon=False, scale=False):
    """Events on the X axis, paired vertical bars (Forecast | Actual).
    Matches the layout of forecast_2526_bar_chart_eventaxis.png.

    anon=True  → use Event #NN labels (org-anonymized).
    scale=True (with anon) → keep dollar scale on Y axis + in subtitle.
    """
    n = len(df)
    fig_w = max(11.0, 0.42 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))

    x = np.arange(n)
    bar_w = 0.38
    pred = df["Pred Revenue"].values
    act  = df["Actual Revenue"].values

    ymax = max(np.nanmax(pred), np.nanmax(act)) * 1.18

    ax.bar(x - bar_w / 2, pred, width=bar_w, color=PRED,
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + bar_w / 2, act,  width=bar_w, color=ACT,
           edgecolor="white", linewidth=0.5, zorder=3)

    # X labels with dates (event name + date stacked, rotated)
    if anon:
        labels = [f"Event #{i+1:02d}\n{r.Date:%b %-d}"
                  for i, r in enumerate(df.itertuples())]
    else:
        labels = [f"{shorten(r.Event)}\n{r.Date:%b %-d}"
                  for r in df.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=8)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, ymax)

    hide_scale = anon and not scale
    ylabel_suffix = " (scale removed)" if hide_scale else ""
    ax.set_ylabel(f"Revenue (paid tickets){ylabel_suffix}",
                  fontsize=10, color=DGRAY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: rf"\${int(y/1000):,}K"))
    if hide_scale:
        ax.set_yticklabels([])
    ax.grid(axis="y", color=LGRAY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(LGRAY)
    ax.spines["bottom"].set_color(LGRAY)
    ax.tick_params(axis="y", labelsize=8)

    handles = [
        Patch(facecolor=PRED, label="Forecast Revenue"),
        Patch(facecolor=ACT,  label="Actual Revenue"),
    ]
    ax.legend(handles=handles, loc="upper left",
              fontsize=9, frameon=True, framealpha=0.95, edgecolor=LGRAY)

    # Title + summary
    pred_total = int(df["Pred Revenue"].sum())
    act_total  = int(df["Actual Revenue"].sum())
    gap        = pred_total - act_total
    gap_pct    = gap / act_total * 100

    title_org = "" if anon else "Music Worcester — "
    fig.suptitle(f"{title_org}25–26 Season Revenue: Forecast vs. Actual",
                 fontsize=14, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.99)
    if anon and not scale:
        subtitle = (f"Forecast within {abs(gap_pct):.1f}% of actual at the season level  ·  "
                    f"Tier-uplifted price prior (Marquee +40%, H/P +15%, Std +10%)  ·  "
                    f"n={n} events  ·  scale removed")
    else:
        # Named version OR anon+scale: show dollar totals
        subtitle = (rf"Forecast \${pred_total:,} vs. actual \${act_total:,}  "
                    f"({gap:+,}, {gap_pct:+.1f}%)  ·  "
                    f"Tier-uplifted price prior (Marquee +40%, H/P +15%, Std +10%)  ·  "
                    f"n={n} events")
    fig.text(0.02, 0.945, subtitle, ha="left", va="top",
             fontsize=10, color=DGRAY)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def build_chart_horizontal(df, out_path, anon=False, scale=False):
    """Two-column horizontal layout: events split across left/right panes,
    horizontal bars (Forecast above, Actual below), larger fonts. Matches
    forecast_2526_bar_chart_wide.png layout. Used in the finance brief."""
    n = len(df)
    half = (n + 1) // 2
    groups = [df.iloc[:half].reset_index(drop=True),
              df.iloc[half:].reset_index(drop=True)]

    pred_all = df["Pred Revenue"].values
    act_all  = df["Actual Revenue"].values
    xmax = max(np.nanmax(pred_all), np.nanmax(act_all)) * 1.15

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6),
                             gridspec_kw={"wspace": 0.42})

    bar_h = 0.36
    hide_scale = anon and not scale

    for ax_i, (ax, grp) in enumerate(zip(axes, groups)):
        m = len(grp)
        y = np.arange(m)[::-1]
        y_pred = y + bar_h / 2
        y_act  = y - bar_h / 2

        pred = grp["Pred Revenue"].values
        act  = grp["Actual Revenue"].values

        ax.barh(y_pred, pred, height=bar_h, color=PRED,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax.barh(y_act,  act,  height=bar_h, color=ACT,
                edgecolor="white", linewidth=0.5, zorder=3)

        if anon:
            offset = 0 if ax_i == 0 else half
            labels = [f"Event #{i+offset+1:02d}   {r.Date:%b %-d}"
                      for i, r in enumerate(grp.itertuples())]
        else:
            labels = [f"{shorten(r.Event)}   {r.Date:%b %-d}"
                      for r in grp.itertuples()]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: rf"\${int(x/1000):,}K"))
        if hide_scale:
            ax.set_xticklabels([])
        ax.grid(axis="x", color=LGRAY, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color(LGRAY)
        ax.spines["bottom"].set_color(LGRAY)
        ax.tick_params(axis="x", labelsize=9.5)

    pred_total = int(df["Pred Revenue"].sum())
    act_total  = int(df["Actual Revenue"].sum())
    gap        = pred_total - act_total
    gap_pct    = gap / act_total * 100

    title_org = "" if anon else "Music Worcester — "
    fig.suptitle(f"{title_org}25–26 Season Revenue: Forecast vs. Actual",
                 fontsize=15, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.995)
    if anon and not scale:
        subtitle = (f"Forecast within {abs(gap_pct):.1f}% of actual at the season level  ·  "
                    f"Tier-uplifted price prior (Marquee +40%, H/P +15%, Std +10%)  ·  "
                    f"n={n} events  ·  scale removed")
    else:
        subtitle = (rf"Forecast \${pred_total:,} vs. actual \${act_total:,}  "
                    f"({gap:+,}, {gap_pct:+.1f}%)  ·  "
                    f"Tier-uplifted price prior (Marquee +40%, H/P +15%, Std +10%)  ·  "
                    f"n={n} events")
    fig.text(0.02, 0.960, subtitle, ha="left", va="top",
             fontsize=10.5, color=DGRAY)

    handles = [
        Patch(facecolor=PRED, label="Forecast Revenue"),
        Patch(facecolor=ACT,  label="Actual Revenue"),
    ]
    axes[1].legend(handles=handles, loc="upper right",
                   fontsize=9.5, frameon=True, framealpha=0.95,
                   edgecolor=LGRAY, ncol=1)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main():
    df = load()
    if HORIZONTAL:
        suffix = "_horizontal"
        builder = build_chart_horizontal
    else:
        suffix = ""
        builder = build_chart

    if ANON and SCALE:
        out_path = f"forecasting/forecast_revenue_chart_anon_scaled{suffix}.png"
    elif ANON:
        out_path = f"forecasting/forecast_revenue_chart_anon{suffix}.png"
    else:
        out_path = f"forecasting/forecast_revenue_chart{suffix}.png"
    builder(df, out_path, anon=ANON, scale=SCALE)


if __name__ == "__main__":
    main()
