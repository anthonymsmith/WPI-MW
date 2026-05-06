"""
Temporal-holdout residual bootstrap per FallbackLevel.

For each target season in HOLDOUT_SEASONS, train on everything before it and
predict that season's events. Collect log-residuals log(actual / Pred_A)
grouped by FallbackLevel. Bootstrap the 5th and 95th percentiles of each
stratum. Pool strata with fewer than MIN_N observations toward the overall
distribution.

Output: fallback_ci_calibration.json
"""
import os
import json
import numpy as np
import pandas as pd

from forecast_2526_comparison import (
    load_data, get_training_df, build_hierarchy_models,
    build_pwyw_lift, predict_model_a, cap_at_capacity,
    INCLUDE_COMPS,
)

os.chdir("/Users/antho/Documents/WPI-MW")

OUT_JSON       = "fallback_ci_calibration.json"
HOLDOUT_SEASONS = ["22-23", "23-24", "24-25"]
BOOTSTRAP_N    = 1000
ALPHA          = 0.10     # 90% CI → 5th / 95th percentiles
MIN_N          = 5        # strata with fewer observations pool to _overall
RNG_SEED       = 42


def _holdout_predictions():
    em, merged = load_data()
    all_seasons = sorted(s for s in merged["Season"].dropna().unique())
    frames = []
    for target in HOLDOUT_SEASONS:
        if target not in all_seasons:
            continue
        prior = [s for s in all_seasons if s < target]
        train = get_training_df(merged, prior)
        (repeat_model, primary_sf, sf_ratio,
         primary, f1, f2, f3, f3a, f3b, f4, f5) = build_hierarchy_models(train)
        pwyw_lift, _ = build_pwyw_lift(
            merged, prior, repeat_model, primary_sf, sf_ratio,
            primary, f1, f2, f3, f3a, f3b, f4, f5)

        gb = ["EventId", "EventName", "EventClass", "EventVenue",
              "EventGenre", "EventLoB", "EventSubGenre", "EventRepeat"]
        for c in ("SeatFormat", "Pricing", "VenueType"):
            if c in merged.columns:
                gb.append(c)
        actuals = (
            merged[(merged["Season"] == target)
                   & (merged["EventType"] == "Live")
                   & (merged["EventStatus"] == "Complete")
                   & (merged["TicketStatus"] == "Active")
                   & (merged["Quantity"] > 0 if INCLUDE_COMPS else merged["TicketTotal"] > 0)]
            .groupby(gb, group_keys=False, dropna=False)
            .agg(Actual=("Quantity", "sum"))
            .reset_index()
        )
        cap = em.drop_duplicates("EventId")[["EventId", "EventCapacity"]].copy()
        cap["EventCapacity"] = pd.to_numeric(cap["EventCapacity"], errors="coerce")
        actuals = actuals.merge(cap, on="EventId", how="left")

        fc = predict_model_a(actuals, repeat_model, primary_sf, sf_ratio,
                             primary, f1, f2, f3, f3a, f3b, f4, f5,
                             pwyw_lift=pwyw_lift)
        fc["Pred_A"] = cap_at_capacity(fc["Pred_A"], fc["EventCapacity"])
        fc["Target"] = target
        frames.append(fc[["Target", "EventName", "FallbackLevel",
                          "Actual", "Pred_A"]])
    return pd.concat(frames, ignore_index=True)


def _bootstrap_ci(arr, n_boot=BOOTSTRAP_N, alpha=ALPHA, seed=RNG_SEED):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    los, his = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        los[i] = np.quantile(sample, alpha / 2)
        his[i] = np.quantile(sample, 1 - alpha / 2)
    return float(np.mean(los)), float(np.mean(his))


def _strip_pwyw(level):
    return str(level).split(" × PWYW lift")[0]


def main():
    df = _holdout_predictions()
    df = df[(df["Pred_A"] > 0) & df["Actual"].notna() & (df["Actual"] > 0)].copy()
    df["Level"] = df["FallbackLevel"].apply(_strip_pwyw)
    df["LogR"]  = np.log(df["Actual"] / df["Pred_A"])

    calib = {}
    for level, g in df.groupby("Level"):
        lo, hi = _bootstrap_ci(g["LogR"].values)
        calib[level] = {
            "n":        int(len(g)),
            "q_lo":     lo,
            "q_hi":     hi,
            "median":   float(np.median(g["LogR"])),
            "mean":     float(g["LogR"].mean()),
        }
    lo, hi = _bootstrap_ci(df["LogR"].values)
    calib["_overall"] = {
        "n":        int(len(df)),
        "q_lo":     lo,
        "q_hi":     hi,
        "median":   float(np.median(df["LogR"])),
        "mean":     float(df["LogR"].mean()),
        "min_n":    MIN_N,
        "alpha":    ALPHA,
        "seasons":  HOLDOUT_SEASONS,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"✓ {OUT_JSON}  (holdouts: {', '.join(HOLDOUT_SEASONS)})")
    print(f"  {'Level':<55s}  {'n':>4s}  "
          f"{'Lo mult':>7s}  {'Hi mult':>7s}  {'Width':>6s}  Pooled?")
    for level in sorted(calib):
        e = calib[level]
        pooled = "" if e["n"] >= MIN_N or level == "_overall" else "→ _overall"
        if e["q_lo"] is None:
            print(f"  {level:<55s}  {e['n']:>4d}  (no data)")
            continue
        lo_mult = np.exp(e["q_lo"])
        hi_mult = np.exp(e["q_hi"])
        width   = hi_mult - lo_mult
        print(f"  {level:<55s}  {e['n']:>4d}  "
              f"{lo_mult:>7.2f}  {hi_mult:>7.2f}  {width:>6.2f}  {pooled}")


if __name__ == "__main__":
    main()
