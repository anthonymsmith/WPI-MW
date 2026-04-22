"""
Full 25-26 season forecast — one clean tab.

Includes every 25-26 live event from the manifest (completed + upcoming),
with the final adjusted prediction and 90% CI. Meant for planning, budgeting,
and sharing — no diagnostic columns.

Output: Forecast_2526_FullSeason.xlsx, sheet "FullSeason"
Columns: EventDate, EventName, EventVenue, EventClass, EventCapacity,
         Status, Actual, Pred_Adj, Pred_Adj_Lo, Pred_Adj_Hi
"""
import os
import pandas as pd
import numpy as np

os.chdir("/Users/antho/Documents/WPI-MW")

from forecast_2526_comparison import (
    load_data, get_training_df, build_hierarchy_models,
    predict_model_a, cap_at_capacity, build_pwyw_lift,
)
from forecast_artist_adjustment import apply_artist_adjustment

FORECAST_SEASON = "25-26"
OUT_XLSX = "Forecast_2526_FullSeason.xlsx"


def main():
    em, merged = load_data()
    prior = sorted([s for s in merged["Season"].dropna().unique()
                    if s < FORECAST_SEASON])
    train = get_training_df(merged, prior)
    (repeat_model, primary_sf, sf_ratio, primary, f1, f2, f3, f3a, f3b, f4, f5
     ) = build_hierarchy_models(train)
    pwyw_lift, _ = build_pwyw_lift(
        merged, prior, repeat_model, primary_sf, sf_ratio,
        primary, f1, f2, f3, f3a, f3b, f4, f5)

    # All 25-26 live events from manifest
    em["EventDate"] = pd.to_datetime(em["EventDate"], errors="coerce")
    season = (
        em[(em["Season"] == FORECAST_SEASON)
           & (em["EventType"] == "Live")]
        .drop_duplicates("EventName")
        .copy()
    )
    season["EventCapacity"] = pd.to_numeric(season["EventCapacity"], errors="coerce")
    season["Actual"] = np.nan

    # Completed actuals — merge in where available
    comp_actuals = (
        merged[(merged["Season"] == FORECAST_SEASON)
               & (merged["EventType"] == "Live")
               & (merged["EventStatus"] == "Complete")
               & (merged["TicketStatus"] == "Active")
               & (merged["Quantity"] > 0)]
        .groupby("EventName")["Quantity"].sum()
        .rename("ActualObs")
        .reset_index()
    )
    season = season.merge(comp_actuals, on="EventName", how="left")
    season["Actual"] = season["ActualObs"]
    season = season.drop(columns=["ActualObs"])
    season["Status"] = np.where(season["Actual"].notna(), "Completed", "Upcoming")

    # Predict
    fc = predict_model_a(season, repeat_model, primary_sf, sf_ratio,
                         primary, f1, f2, f3, f3a, f3b, f4, f5, pwyw_lift=pwyw_lift)
    fc["Pred_A"] = cap_at_capacity(fc["Pred_A"], fc["EventCapacity"])

    # Artist adjustment needs training-season labelled history
    hist_gb = ["EventId", "EventName", "EventClass", "EventVenue",
               "EventGenre", "EventLoB", "EventSubGenre", "EventRepeat"]
    for c in ("SeatFormat", "VenueType"):
        if c in train.columns:
            hist_gb.append(c)
    hist_actuals = (
        train.groupby(hist_gb, group_keys=False, dropna=False)
        .agg(Actual=("Quantity", "sum"))
        .reset_index()
    )
    cap = em.drop_duplicates("EventId")[["EventId", "EventCapacity"]].copy()
    cap["EventCapacity"] = pd.to_numeric(cap["EventCapacity"], errors="coerce")
    hist_actuals = hist_actuals.merge(cap, on="EventId", how="left")
    hist_fc = predict_model_a(hist_actuals, repeat_model, primary_sf, sf_ratio,
                               primary, f1, f2, f3, f3a, f3b, f4, f5)
    fc = apply_artist_adjustment(
        fc,
        merged_history=hist_fc,
        actuals_history=hist_fc["Actual"],
        bucket_preds_history=hist_fc["Pred_A"],
    )

    # Cap CI bounds at capacity too
    fc["Pred_Adj"]    = cap_at_capacity(fc["Pred_Adj"],    fc["EventCapacity"])
    fc["Pred_Adj_Lo"] = cap_at_capacity(fc["Pred_Adj_Lo"], fc["EventCapacity"])
    fc["Pred_Adj_Hi"] = cap_at_capacity(fc["Pred_Adj_Hi"], fc["EventCapacity"])

    fc["Status"] = np.where(fc["Actual"].notna(), "Completed", "Upcoming")

    out = (
        fc[["EventDate", "EventName", "EventVenue", "EventClass",
            "EventCapacity", "Status", "Actual",
            "Pred_Adj", "Pred_Adj_Lo", "Pred_Adj_Hi"]]
        .sort_values("EventDate")
        .reset_index(drop=True)
    )
    for c in ["Pred_Adj", "Pred_Adj_Lo", "Pred_Adj_Hi", "Actual", "EventCapacity"]:
        out[c] = out[c].round(0)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="FullSeason", index=False)

    n_comp = (out["Status"] == "Completed").sum()
    n_upc  = (out["Status"] == "Upcoming").sum()
    total_actual   = int(out.loc[out["Actual"].notna(), "Actual"].sum())
    total_fcst     = int(out["Pred_Adj"].sum())
    total_fcst_upc = int(out.loc[out["Status"] == "Upcoming", "Pred_Adj"].sum())
    print(f"✓ {OUT_XLSX}")
    print(f"  Events: {len(out)}  ({n_comp} completed, {n_upc} upcoming)")
    print(f"  Completed attendance to date: {total_actual:,}")
    print(f"  Forecast (upcoming only):     {total_fcst_upc:,}")
    print(f"  Forecast (full season):       {total_fcst:,}")


if __name__ == "__main__":
    main()
