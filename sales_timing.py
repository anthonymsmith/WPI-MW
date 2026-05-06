"""
Two complementary views of concert sales timing, stratified by event
comp intensity (tertiles). Non-Headliner events only.

Chart 1 — Cumulative fill (% of capacity over time):
  Stacked area per group: paid fill + comp fill vs. days-before-event.
  Answers "how full is the house at day t, and by what mix?"

Chart 2 — Volume mix + price trajectory:
  Stacked bars per 5-day bucket (paid + comp) with an overlaid mean paid-
  unit-price line. One small multiple per comp-intensity group.
  Answers "when do comps go out, and what's happening to paid price?"

Clock: DaysBeforeEvent = EventDate - CreatedDate.

Outputs:
  sales_capacity_fill.png
  sales_volume_price.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

WORKING_DIR = "/Users/antho/Documents/WPI-MW"
os.chdir(WORKING_DIR)

START_DATE       = "2022-07-01"
MAX_DAYS_BEFORE  = 200
BUCKET_DAYS      = 5
EXCLUDE_VENUES   = ["The Hanover Theatre"]
EXCLUDE_CLASSES  = ["Headliner"]

NAVY   = "#1A3A5C"    # paid
ORANGE = "#E8922A"    # comp
TEAL   = "#2A9EA0"    # price line
LGRAY  = "#E8EDF2"
DGRAY  = "#5A6A7A"

GROUP_COLORS = {
    "Low comp":  "#3E6E93",
    "Mid comp":  "#5FA9AB",
    "High comp": "#D97B2A",
}

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


def load():
    df = pd.read_csv("anon_DataMerge.csv",
                     parse_dates=["CreatedDate", "EventDate"])
    df = df[df["EventDate"] >= START_DATE]
    df = df[~df["EventVenue"].isin(EXCLUDE_VENUES)]
    df["DaysBeforeEvent"] = (df["EventDate"] - df["CreatedDate"]).dt.days
    df = df[(df["DaysBeforeEvent"] >= 0)
            & (df["DaysBeforeEvent"] <= MAX_DAYS_BEFORE)]
    df = df.dropna(subset=["EventClass"])
    df["EventClass"] = df["EventClass"].replace("Local Favorite", "Headliner")
    df = df[(df["EventStatus"] == "Complete")
            & (df["EventType"] == "Live")
            & (df["TicketStatus"] == "Active")
            & (df["Quantity"] > 0)]
    df = df[~df["EventClass"].isin(EXCLUDE_CLASSES)]
    df = df.dropna(subset=["EventCapacity"])
    return df


def tertile_groups(df):
    shares = (df.assign(comp_qty=np.where(df["IsComp"], df["Quantity"], 0))
                .groupby("EventName")
                .agg(comp_qty=("comp_qty", "sum"),
                     total_qty=("Quantity", "sum"),
                     capacity=("EventCapacity", "first")))
    shares["comp_share"] = shares["comp_qty"] / shares["total_qty"]
    t1, t2 = shares["comp_share"].quantile([1/3, 2/3]).values

    def bucket(x):
        if x < t1:  return "Low comp"
        if x < t2:  return "Mid comp"
        return "High comp"

    shares["group"] = shares["comp_share"].apply(bucket)
    return shares, (t1, t2)


def per_event_cumfill(df, shares):
    """Returns per-event cumulative fill (paid + comp) at each DaysBeforeEvent."""
    frames = []
    for ev, sub in df.groupby("EventName"):
        cap = shares.at[ev, "capacity"]
        if cap <= 0 or pd.isna(cap):
            continue
        sub = sub.sort_values("DaysBeforeEvent", ascending=False).copy()
        sub["PaidQ"] = np.where(sub["IsComp"], 0, sub["Quantity"])
        sub["CompQ"] = np.where(sub["IsComp"], sub["Quantity"], 0)
        # Reverse-cumulative: at day t, tickets issued with DaysBefore >= t
        agg = (sub.groupby("DaysBeforeEvent")
                  .agg(Paid=("PaidQ", "sum"), Comp=("CompQ", "sum"))
                  .sort_index(ascending=False))
        agg["CumPaid"] = agg["Paid"].cumsum()
        agg["CumComp"] = agg["Comp"].cumsum()
        agg["CumTotal"] = agg["CumPaid"] + agg["CumComp"]
        agg["FillPaid"] = agg["CumPaid"] / cap
        agg["FillComp"] = agg["CumComp"] / cap
        agg["FillTotal"] = agg["CumTotal"] / cap
        agg["EventName"] = ev
        agg["group"] = shares.at[ev, "group"]
        frames.append(agg.reset_index())
    return pd.concat(frames, ignore_index=True)


def plot_fill(shares_cumfill, thresholds):
    """Three-panel mean cumulative fill, stacked paid + comp, capacity reference."""
    t1, t2 = thresholds
    groups = ["Low comp", "Mid comp", "High comp"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             sharey=True, gridspec_kw={"wspace": 0.12})

    # Resample each event onto daily grid for averaging
    day_grid = np.arange(0, MAX_DAYS_BEFORE + 1)

    for ax, group in zip(axes, groups):
        sub = shares_cumfill[shares_cumfill["group"] == group]
        events = sub["EventName"].unique()
        if len(events) == 0:
            continue

        # For each event, forward-fill cumulative fill onto daily grid
        paid_grid, comp_grid = [], []
        for ev in events:
            ev_df = sub[sub["EventName"] == ev].sort_values("DaysBeforeEvent",
                                                             ascending=False)
            # Map DaysBefore → fill, then forward-fill over day_grid (descending)
            fp = pd.Series(ev_df["FillPaid"].values,
                           index=ev_df["DaysBeforeEvent"].values)
            fc = pd.Series(ev_df["FillComp"].values,
                           index=ev_df["DaysBeforeEvent"].values)
            fp = fp.reindex(day_grid[::-1]).ffill().fillna(0.0)
            fc = fc.reindex(day_grid[::-1]).ffill().fillna(0.0)
            paid_grid.append(fp.values)
            comp_grid.append(fc.values)

        paid_mean = np.mean(paid_grid, axis=0)
        comp_mean = np.mean(comp_grid, axis=0)
        x = day_grid[::-1]

        ax.fill_between(x, 0, paid_mean, color=NAVY, alpha=0.88, label="Paid")
        ax.fill_between(x, paid_mean, paid_mean + comp_mean,
                        color=ORANGE, alpha=0.85, label="Comp")
        ax.axhline(1.0, color=DGRAY, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(5, 1.01, "capacity", fontsize=8, color=DGRAY, ha="right")

        final_total = paid_mean[-1] + comp_mean[-1]
        final_paid  = paid_mean[-1]
        ax.text(2, final_total + 0.03,
                f"{final_total:.0%} full  ·  {final_paid/final_total:.0%} paid"
                if final_total > 0 else "",
                fontsize=9, color=NAVY, fontweight="bold", ha="left")

        ax.set_xlim(MAX_DAYS_BEFORE, 0)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{group}  (n={len(events)})",
                     fontsize=11, color=GROUP_COLORS[group],
                     fontweight="bold", loc="left")
        ax.set_xlabel("Days before event", fontsize=10, color=DGRAY)
        ax.grid(True, linestyle="--", linewidth=0.6, color=LGRAY)
        ax.spines["left"].set_color(LGRAY)
        ax.spines["bottom"].set_color(LGRAY)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    axes[0].set_ylabel("Share of capacity filled", fontsize=10, color=DGRAY)

    handles = [Patch(color=NAVY, label="Paid"), Patch(color=ORANGE, label="Comp")]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=9,
                    frameon=True, framealpha=0.95, edgecolor=LGRAY)

    fig.suptitle("Average house fill over time — paid vs comp",
                 fontsize=14, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.955,
             f"Non-Headliner events, {START_DATE[:7]}+  ·  "
             f"comp-intensity tertiles: Low <{t1:.0%}  "
             f"·  Mid {t1:.0%}–{t2:.0%}  ·  High ≥{t2:.0%}",
             ha="left", va="top", fontsize=10, color=DGRAY)

    fig.tight_layout(rect=[0, 0, 1, 0.935])
    out = "sales_capacity_fill.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_volume_price(df, shares, thresholds, facet="group"):
    """Stacked bars of paid+comp volume per 5-day bucket, with paid-price line.

    facet: "group" for comp-intensity tertiles, "season" for per-season rows.
    """
    t1, t2 = thresholds
    df = df.copy()
    df["group"] = df["EventName"].map(shares["group"])
    df["UnitPrice"] = df["TicketTotal"] / df["Quantity"]

    edges = np.arange(0, MAX_DAYS_BEFORE + BUCKET_DAYS, BUCKET_DAYS)
    df["Bucket"] = pd.cut(df["DaysBeforeEvent"], bins=edges, right=False,
                           labels=edges[:-1])

    if facet == "group":
        facet_col = "group"
        facet_values = ["Low comp", "Mid comp", "High comp"]
        facet_colors = GROUP_COLORS
        out_name = "sales_volume_price.png"
        subtitle_extra = (f"tertiles: Low <{t1:.0%}  "
                          f"·  Mid {t1:.0%}–{t2:.0%}  ·  High ≥{t2:.0%}")
    else:  # season
        facet_col = "Season"
        facet_values = sorted(df["Season"].dropna().unique())
        facet_colors = {s: NAVY for s in facet_values}
        out_name = "sales_volume_price_by_season.png"
        subtitle_extra = "all non-Headliner events, pooled across comp intensity"

    n_facets = len(facet_values)
    fig, axes = plt.subplots(n_facets, 1,
                              figsize=(13, 2.4 * n_facets + 1.0),
                              sharex=True, gridspec_kw={"hspace": 0.35})
    if n_facets == 1:
        axes = [axes]

    for ax, fv in zip(axes, facet_values):
        sub = df[df[facet_col] == fv]
        n_events = sub["EventName"].nunique()

        vol = (sub.assign(PaidQ=np.where(sub["IsComp"], 0, sub["Quantity"]),
                          CompQ=np.where(sub["IsComp"], sub["Quantity"], 0))
                  .groupby("Bucket", observed=False)
                  [["PaidQ", "CompQ"]]
                  .sum()
                  .reindex(edges[:-1], fill_value=0))

        # Per-event normalization so bars don't reflect group size
        vol_per_event = vol / max(n_events, 1)
        x = vol.index.astype(int).values
        ax.bar(x, vol_per_event["PaidQ"], width=BUCKET_DAYS - 0.6,
               color=NAVY, label="Paid", align="edge", zorder=3)
        ax.bar(x, vol_per_event["CompQ"], width=BUCKET_DAYS - 0.6,
               bottom=vol_per_event["PaidQ"],
               color=ORANGE, label="Comp", align="edge", zorder=3)

        # Paid unit price (weighted by paid quantity) per bucket
        paid = sub[~sub["IsComp"]].copy()
        paid_bucket = (paid.assign(px=paid["UnitPrice"] * paid["Quantity"])
                            .groupby("Bucket", observed=False)
                            .agg(px=("px", "sum"), q=("Quantity", "sum"))
                            .reindex(edges[:-1]))
        price = (paid_bucket["px"] / paid_bucket["q"]).values

        ax2 = ax.twinx()
        ax2.plot(x + (BUCKET_DAYS - 0.6) / 2, price,
                 color=TEAL, linewidth=2.2, marker="o", markersize=3.5,
                 label="Avg paid $ / ticket", zorder=4)
        ax2.set_ylabel("Avg paid $", fontsize=9, color=TEAL)
        ax2.tick_params(axis="y", labelcolor=TEAL, labelsize=8)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(LGRAY)
        ax2.grid(False)
        y_max = max(np.nanmax(price) * 1.15, 10) if np.any(~np.isnan(price)) else 60
        ax2.set_ylim(0, y_max)

        ax.set_xlim(MAX_DAYS_BEFORE, 0)
        ax.set_title(f"{fv}  (n={n_events} events)",
                     fontsize=11, color=facet_colors.get(fv, NAVY),
                     fontweight="bold", loc="left")
        ax.set_ylabel("Tickets / event", fontsize=9, color=DGRAY)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, color=LGRAY)
        ax.set_axisbelow(True)
        ax.spines["left"].set_color(LGRAY)
        ax.spines["bottom"].set_color(LGRAY)

    axes[-1].set_xlabel("Days before event", fontsize=10, color=DGRAY)

    handles = [
        Patch(color=NAVY, label="Paid tickets / event"),
        Patch(color=ORANGE, label="Comp tickets / event"),
        plt.Line2D([0], [0], color=TEAL, linewidth=2, marker="o",
                   markersize=4, label="Avg paid $ / ticket"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=9,
                   frameon=True, framealpha=0.95, edgecolor=LGRAY)

    title_suffix = "by comp intensity" if facet == "group" else "by season"
    fig.suptitle(f"Ticket issuance timing & paid price — {title_suffix}",
                 fontsize=14, fontweight="bold", color=NAVY,
                 x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.97,
             f"Non-Headliner events, {START_DATE[:7]}+  ·  "
             f"5-day buckets  ·  {subtitle_extra}",
             ha="left", va="top", fontsize=10, color=DGRAY)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_name}")


def main():
    df = load()
    print(f"Rows after filters: {len(df):,}  "
          f"(events: {df['EventName'].nunique()})")
    shares, thresholds = tertile_groups(df)
    print(f"Comp-intensity tertiles: Low <{thresholds[0]:.1%}  "
          f"·  Mid {thresholds[0]:.1%}–{thresholds[1]:.1%}  "
          f"·  High ≥{thresholds[1]:.1%}")
    print(shares.groupby("group").size().rename("events").to_string())

    cumfill = per_event_cumfill(df, shares)
    plot_fill(cumfill, thresholds)
    plot_volume_price(df, shares, thresholds, facet="group")
    plot_volume_price(df, shares, thresholds, facet="season")


if __name__ == "__main__":
    main()
