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
    INCLUDE_COMPS,
)
from forecast_artist_adjustment import apply_artist_adjustment
from forecast_bucket_ci import apply_bucket_ci
from forecast_comp_split import build_comp_rates, apply_comp_split

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

    # Completed actuals — split paid vs comp so the output mirrors Pred_Paid / Pred_Comp
    completed = merged[
        (merged["Season"] == FORECAST_SEASON)
        & (merged["EventType"] == "Live")
        & (merged["EventStatus"] == "Complete")
        & (merged["TicketStatus"] == "Active")
        & (merged["Quantity"] > 0)
    ].copy()
    is_comp = (completed["IsComp"].astype(bool) if "IsComp" in completed.columns
               else completed["TicketTotal"] == 0)
    completed["CompQty"] = np.where(is_comp, completed["Quantity"], 0)
    completed["PaidQty"] = np.where(is_comp, 0, completed["Quantity"])
    actuals = (
        completed.groupby("EventName")
        .agg(Actual=("Quantity", "sum"),
             Actual_Paid=("PaidQty", "sum"),
             Actual_Comp=("CompQty", "sum"))
        .reset_index()
    )
    season = season.merge(actuals, on="EventName", how="left")
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

    # Fill CIs for events without artist signal using bucket-level residuals
    fc = apply_bucket_ci(fc)

    # Cap CI bounds at capacity too
    fc["Pred_Adj"]    = cap_at_capacity(fc["Pred_Adj"],    fc["EventCapacity"])
    fc["Pred_Adj_Lo"] = cap_at_capacity(fc["Pred_Adj_Lo"], fc["EventCapacity"])
    fc["Pred_Adj_Hi"] = cap_at_capacity(fc["Pred_Adj_Hi"], fc["EventCapacity"])

    # Paid/comp split — historical comp rate from training seasons, routed
    # by event bucket (EventRepeat → Primary → F1 → F2 → F3 → F4 → overall)
    comp_rates = build_comp_rates(merged, prior)
    fc = apply_comp_split(fc, comp_rates)

    fc["Status"] = np.where(fc["Actual"].notna(), "Completed", "Upcoming")

    out = (
        fc[["EventDate", "EventName", "EventVenue", "EventClass",
            "EventCapacity", "Status",
            "Actual", "Actual_Paid", "Actual_Comp",
            "Pred_Adj", "Pred_Adj_Lo", "Pred_Adj_Hi",
            "Pred_Paid", "Pred_Paid_Lo", "Pred_Paid_Hi",
            "Pred_Comp", "Pred_Comp_Lo", "Pred_Comp_Hi",
            "CompRate", "CompRate_Source"]]
        .sort_values("EventDate")
        .reset_index(drop=True)
    )
    for c in ["Pred_Adj", "Pred_Adj_Lo", "Pred_Adj_Hi",
              "Pred_Paid", "Pred_Paid_Lo", "Pred_Paid_Hi",
              "Pred_Comp", "Pred_Comp_Lo", "Pred_Comp_Hi",
              "Actual", "Actual_Paid", "Actual_Comp", "EventCapacity"]:
        out[c] = out[c].round(0)
    out["CompRate"] = out["CompRate"].round(3)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="FullSeason", index=False)

    n_comp = (out["Status"] == "Completed").sum()
    n_upc  = (out["Status"] == "Upcoming").sum()
    done   = out["Actual"].notna()
    upc    = out["Status"] == "Upcoming"
    print(f"✓ {OUT_XLSX}")
    print(f"  Events: {len(out)}  ({n_comp} completed, {n_upc} upcoming)")
    print(f"  Completed to date — total: {int(out.loc[done, 'Actual'].sum()):,}  "
          f"paid: {int(out.loc[done, 'Actual_Paid'].sum()):,}  "
          f"comp: {int(out.loc[done, 'Actual_Comp'].sum()):,}")
    print(f"  Upcoming forecast  — total: {int(out.loc[upc, 'Pred_Adj'].sum()):,}  "
          f"paid: {int(out.loc[upc, 'Pred_Paid'].sum()):,}  "
          f"comp: {int(out.loc[upc, 'Pred_Comp'].sum()):,}")
    print(f"  Full season        — total: {int(out['Pred_Adj'].sum()):,}  "
          f"paid: {int(out['Pred_Paid'].sum()):,}  "
          f"comp: {int(out['Pred_Comp'].sum()):,}")


if __name__ == "__main__":
    main()
