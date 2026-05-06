"""
Paid/comp split layer for the attendance forecast.

Computes historical comp rate (comp_qty / total_qty) at each level of the
forecast hierarchy from training-season data, then splits Pred_Adj into
Pred_Paid and Pred_Comp via a per-event waterfall lookup:

    EventRepeat  →  Primary (Class, Venue, LoB, SubGenre)
                 →  F1 (Class, Venue, SubGenre)
                 →  F2 (Class, Venue, Genre)
                 →  F3 (Class, Venue)
                 →  F4 (SubGenre)
                 →  _overall

A level is trusted once it has at least MIN_N prior training observations.
Per-event rates stay between [0, MAX_RATE] to guard against pathological
buckets (e.g. a single fully-comped showing).
"""
import pandas as pd
import numpy as np

MIN_N     = 3
MAX_RATE  = 0.75   # cap individual rates to 75% comp

# Match the forecast's season weights so recent comp policy dominates.
# (Imported lazily to avoid circular import.)
def _season_weights():
    from forecast_2526_comparison import WEIGHTS
    return WEIGHTS


def _rate_for(df, key_cols):
    """Return {tuple(keys): (comp_qty_wtd, total_qty_wtd, rate)}.
    n-threshold uses the unweighted row count so thin buckets still pool up.
    """
    sub = df.dropna(subset=key_cols)
    if sub.empty:
        return {}
    g = sub.groupby(key_cols, dropna=False).agg(
        comp=("CompWtd", "sum"),
        total=("TotalWtd", "sum"),
        n=("Quantity", "size"),
    )
    g = g[g["total"] > 0]
    g["rate"] = (g["comp"] / g["total"]).clip(0.0, MAX_RATE)
    out = {}
    for idx, row in g.iterrows():
        key = idx if isinstance(idx, tuple) else (idx,)
        out[key] = (float(row["comp"]), float(row["total"]),
                    float(row["rate"]), int(row["n"]))
    return out


def build_comp_rates(merged, training_seasons):
    """Return a nested dict of comp-rate lookups keyed by level."""
    df = merged[
        (merged["Season"].isin(training_seasons))
        & (merged["EventType"] == "Live")
        & (merged["EventStatus"] == "Complete")
        & (merged["TicketStatus"] == "Active")
        & (merged["Quantity"] > 0)
    ].copy()

    if "IsComp" in df.columns:
        is_comp = df["IsComp"].astype(bool)
    else:
        is_comp = df["TicketTotal"] == 0
    df["CompQty"] = np.where(is_comp, df["Quantity"], 0)

    weights = _season_weights()
    df["W"] = df["Season"].map(weights).fillna(1.0)
    df["TotalWtd"] = df["Quantity"] * df["W"]
    df["CompWtd"]  = df["CompQty"]  * df["W"]

    levels = {
        "EventRepeat": ["EventRepeat"],
        "Primary":     ["EventClass", "EventVenue", "EventLoB", "EventSubGenre"],
        "F1":          ["EventClass", "EventVenue", "EventSubGenre"],
        "F2":          ["EventClass", "EventVenue", "EventGenre"],
        "F3":          ["EventClass", "EventVenue"],
        "F4":          ["EventSubGenre"],
    }
    rates = {name: _rate_for(df, cols) for name, cols in levels.items()}

    total = df["TotalWtd"].sum()
    comp  = df["CompWtd"].sum()
    rates["_overall"] = {
        "comp":  float(comp),
        "total": float(total),
        "rate":  float(comp / total) if total else 0.0,
    }
    rates["_levels"] = levels
    rates["_min_n"]  = MIN_N
    return rates


def _lookup_rate(row, rates):
    """Waterfall lookup for a single event row."""
    min_n = rates["_min_n"]
    levels = rates["_levels"]

    # Entry tuple: (comp_wtd, total_wtd, rate, n_rows)
    # EventRepeat first (captures event-specific comp policy like Messiah)
    er = row.get("EventRepeat")
    if pd.notna(er):
        entry = rates["EventRepeat"].get((er,))
        if entry and entry[3] >= min_n:
            return entry[2], f"EventRepeat ({er})"

    # Primary → F1 → F2 → F3 → F4
    for level in ["Primary", "F1", "F2", "F3", "F4"]:
        cols = levels[level]
        key = tuple(row.get(c) for c in cols)
        if any(pd.isna(v) for v in key):
            continue
        entry = rates[level].get(key)
        if entry and entry[3] >= min_n:
            return entry[2], level

    return rates["_overall"]["rate"], "_overall"


def apply_comp_split(fc, comp_rates):
    """Adds CompRate, CompRate_Source, Pred_Paid, Pred_Comp columns.

    Also splits Actual into Actual_Paid / Actual_Comp when those upstream
    columns are available; otherwise leaves them blank.
    """
    result = fc.copy()
    rates, sources = [], []
    for _, row in result.iterrows():
        r, s = _lookup_rate(row, comp_rates)
        rates.append(r)
        sources.append(s)
    result["CompRate"]        = rates
    result["CompRate_Source"] = sources

    result["Pred_Comp"] = (result["Pred_Adj"] * result["CompRate"]).round(0)
    result["Pred_Paid"] = (result["Pred_Adj"] - result["Pred_Comp"]).round(0)

    for lo, hi in [("Pred_Adj_Lo", "Pred_Adj_Hi")]:
        if lo in result.columns and hi in result.columns:
            result["Pred_Paid_Lo"] = (result[lo] * (1 - result["CompRate"])).round(0)
            result["Pred_Paid_Hi"] = (result[hi] * (1 - result["CompRate"])).round(0)
            result["Pred_Comp_Lo"] = (result[lo] * result["CompRate"]).round(0)
            result["Pred_Comp_Hi"] = (result[hi] * result["CompRate"]).round(0)

    return result
