"""
Festival Side training set growth — 25-26 → 26-27 forecast.

Shows why FestivalRole couldn't help for 25-26 but should for 26-27:
adding 25-26's actuals to training grows the Side bucket from n=3 to n=10
overall, and the (Standard, Mechanics Hall) bucket from n=2 to n=6 with
the prior dropping 464 → 357 — the direction that fixes the +14% mid-cap
over-prediction.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("/Users/antho/Documents/WPI-MW")

NAVY   = "#1A3A5C"
ORANGE = "#E8922A"
INK    = "#1A2330"
INK_LT = "#5A6A7A"
LGRAY  = "#E8EDF2"


def collect():
    em = pd.read_excel("EventManifest.xlsx", sheet_name="EventManifest")
    df = pd.read_csv("anon_DataMerge.csv")
    df.columns = df.columns.str.strip()
    em.columns = em.columns.str.strip()
    role = em.drop_duplicates("EventId")[
        ["EventId", "EventName", "Season", "EventClass", "EventVenue", "FestivalRole"]
    ]
    m = df.merge(role, on="EventId", how="left", suffixes=("", "_m"))
    for c in ["EventClass", "EventVenue"]:
        if c + "_m" in m.columns:
            m[c] = m[c].combine_first(m[c + "_m"])
    for c in ["EventType", "EventStatus", "TicketStatus"]:
        m[c] = m[c].astype(str).str.strip().str.title()
    m["Quantity"] = pd.to_numeric(m["Quantity"], errors="coerce").fillna(0)
    liv = m[(m["EventType"] == "Live")
            & (m["EventStatus"] == "Complete")
            & (m["TicketStatus"] == "Active")
            & (m["Quantity"] > 0)]
    ev = (liv.groupby(["EventId", "EventName", "Season", "EventClass",
                        "EventVenue", "FestivalRole"], dropna=False)
          .agg(Q=("Quantity", "sum")).reset_index())

    overlay = {
        "Dance Theatre of Harlem 2026": 1487,
        "Chorus: Frederick Douglass": 895,
        "TCB: Bach Organ & Arias": 193,
        "Women's Ensemble & Cantilena": 134,
    }
    for name, q in overlay.items():
        msk = (ev["EventName"] == name) & (ev["Season"] == "25-26")
        if msk.any():
            ev.loc[msk, "Q"] = q
    return ev


def summarize(ev, cutoff_season):
    s = ev[(ev["FestivalRole"] == "Side") & (ev["Season"] < cutoff_season)]
    all_n, all_mean = len(s), s["Q"].mean()
    bucket = s[(s["EventClass"] == "Standard")
               & (s["EventVenue"] == "Mechanics Hall")]
    bk_n, bk_mean = len(bucket), bucket["Q"].mean()
    return all_n, all_mean, bk_n, bk_mean


def main():
    ev = collect()
    s2526 = summarize(ev, "25-26")
    s2627 = summarize(ev, "26-27")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "Festival side-event training data: 25-26 vs 26-27 forecast",
        fontsize=12, fontweight="bold", color=INK, y=0.99,
    )

    labels = ["25-26 forecast\n(through 24-25)", "26-27 forecast\n(through 25-26)"]
    colors = [INK_LT, ORANGE]

    ax = axes[0]
    counts = [s2526[0], s2627[0]]
    means = [s2526[1], s2627[1]]
    bars = ax.bar(labels, counts, color=colors, edgecolor=NAVY, linewidth=1.2)
    for bar, n, mu in zip(bars, counts, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25,
                f"n = {n}", ha="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"mean\n{mu:.0f}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_title("All Side events in training set",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_ylabel("Training events", color=INK_LT)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK_LT)

    ax = axes[1]
    counts = [s2526[2], s2627[2]]
    means = [s2526[3], s2627[3]]
    bars = ax.bar(labels, means, color=colors, edgecolor=NAVY, linewidth=1.2)
    for bar, n, mu in zip(bars, counts, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 12,
                f"prior = {mu:.0f}", ha="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"n = {n}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_title("(Standard, Mechanics Hall) Side bucket — the over-predicting one",
                 fontsize=10.5, color=INK, pad=8)
    ax.set_ylabel("Bucket weighted-mean prior", color=INK_LT)
    ax.set_ylim(0, max(means) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK_LT)

    fig.text(
        0.5, 0.01,
        "Side-event prior drops 464 → 357 once 25-26 actuals join training — "
        "23% reduction in the direction that fixes the +14% mid-cap over-prediction.",
        ha="center", fontsize=9, color=INK_LT, style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = "forecasting/festival_side_training_growth.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"✓ {out}")
    print(f"  25-26 forecast: all Side n={s2526[0]} mean={s2526[1]:.0f}; "
          f"(Std,MH) n={s2526[2]} prior={s2526[3]:.0f}")
    print(f"  26-27 forecast: all Side n={s2627[0]} mean={s2627[1]:.0f}; "
          f"(Std,MH) n={s2627[2]} prior={s2627[3]:.0f}")


if __name__ == "__main__":
    main()
