"""
Diagnostic: evaluate 25-26 forecast on events with EventDate <= 2025-12-11.
These are the only events with reliable actuals given the 2024-12-11 SF export
cutoff. Uses the patched artist-adjustment layer (tight priors + centering).
"""

import pandas as pd
import numpy as np

from forecast_2526_comparison import (
    load_data, get_training_df, build_hierarchy_models, build_name_model,
    predict_model_a, predict_model_b, cap_at_capacity, print_metrics,
    FORECAST_SEASON, INCLUDE_COMPS,
)
from forecast_artist_adjustment import apply_artist_adjustment

CUTOFF = pd.Timestamp("2025-12-11")


def main():
    em, merged = load_data()
    merged["EventDate"] = pd.to_datetime(merged["EventDate"], errors="coerce")

    all_prior = sorted([s for s in merged["Season"].dropna().unique() if s < FORECAST_SEASON])
    filtered_train = get_training_df(merged, all_prior)

    (repeat_model, primary_sf, sf_ratio, primary, f1, f2, f3, f3a, f3b, f4, f5
     ) = build_hierarchy_models(filtered_train)
    name_model = build_name_model(filtered_train)

    qty_mask = merged["Quantity"] > 0 if INCLUDE_COMPS else merged["TicketTotal"] > 0
    actuals_gb = ["EventId", "EventName", "EventClass", "EventVenue",
                  "EventGenre", "EventLoB", "EventSubGenre", "EventRepeat"]
    if "VenueType" in merged.columns:
        actuals_gb.append("VenueType")
    actuals = (
        merged[
            (merged["Season"] == FORECAST_SEASON)
            & (merged["EventType"] == "Live")
            & (merged["EventStatus"] == "Complete")
            & (merged["TicketStatus"] == "Active")
            & (merged["EventDate"] <= CUTOFF)
            & qty_mask
        ]
        .groupby(actuals_gb, group_keys=False, dropna=False)
        .agg(Actual=("Quantity", "sum"), EventDate=("EventDate", "max"))
        .reset_index()
    )

    cap = em.drop_duplicates("EventId")[["EventId", "EventCapacity"]].copy()
    cap["EventCapacity"] = pd.to_numeric(cap["EventCapacity"], errors="coerce")
    actuals = actuals.merge(cap, on="EventId", how="left")

    fc = predict_model_a(actuals, repeat_model, primary_sf, sf_ratio,
                         primary, f1, f2, f3, f3a, f3b, f4, f5)
    fc = predict_model_b(fc, name_model)
    fc["Pred_A"] = cap_at_capacity(fc["Pred_A"], fc["EventCapacity"])
    fc["Pred_B"] = cap_at_capacity(fc["Pred_B"], fc["EventCapacity"])

    hist_gb = ["EventId", "EventName", "EventClass", "EventVenue",
               "EventGenre", "EventLoB", "EventSubGenre", "EventRepeat"]
    if "VenueType" in filtered_train.columns:
        hist_gb.append("VenueType")
    hist_actuals = (
        filtered_train
        .groupby(hist_gb, group_keys=False, dropna=False)
        .agg(Actual=("Quantity", "sum"))
        .reset_index()
        .merge(cap, on="EventId", how="left")
    )
    hist_fc = predict_model_a(hist_actuals, repeat_model, primary_sf, sf_ratio,
                              primary, f1, f2, f3, f3a, f3b, f4, f5)

    fc = apply_artist_adjustment(
        fc,
        merged_history=hist_fc,
        actuals_history=hist_fc["Actual"],
        bucket_preds_history=hist_fc["Pred_A"],
    )

    # --- posterior inspection ---------------------------------------------
    from forecast_artist_adjustment import _build_model
    model, _, _, centers = _build_model(hist_fc, hist_fc["Actual"], hist_fc["Pred_A"])
    print(f"\nPosterior (n_obs={model.n_obs}):")
    for k, v in model.posterior_summary().items():
        print(f"  {k:<35} mean={v[0]:+.4f}  std={v[1]:.4f}")
    print(f"Feature centers: { {k: round(v, 3) for k, v in centers.items()} }")

    # --- metrics ----------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"25-26 through {CUTOFF.date()}  (n={len(fc)} completed events)")
    print("=" * 75)
    print_metrics("Model A: Hierarchy (base)",              fc, "Pred_A")
    print_metrics("Model A + Artist Bayesian adj (patched)", fc, "Pred_Adj")
    print_metrics("Model B: Name history + fallback",        fc, "Pred_B")

    # --- per-event detail -------------------------------------------------
    print("\nEVENT DETAIL")
    hdr = f"{'EventDate':<12}{'EventName':<42} {'Actual':>7} {'Pred_A':>7} {'Pred_Adj':>9} {'ΔA%':>6} {'ΔAdj%':>7}  {'Factor':>7}  Source"
    print(hdr)
    print("-" * len(hdr))
    for _, row in fc.sort_values("EventDate").iterrows():
        err_a   = (row["Pred_A"]   - row["Actual"]) / row["Actual"] * 100
        err_adj = (row["Pred_Adj"] - row["Actual"]) / row["Actual"] * 100
        factor  = row.get("Adj_LogFactor", 0.0)
        factor_s = f"{factor:+.2f}" if "ArtistSignal" in str(row.get("Adj_Source")) else "  n/a"
        print(f"{row['EventDate'].date()!s:<12}"
              f"{str(row['EventName'])[:41]:<42} "
              f"{row['Actual']:>7.0f} {row['Pred_A']:>7.0f} {row['Pred_Adj']:>9.0f} "
              f"{err_a:>+5.0f}% {err_adj:>+6.0f}%  {factor_s:>7}  {row.get('Adj_Source', '')}")


if __name__ == "__main__":
    main()
