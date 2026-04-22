"""
What-if: re-predict Ladysmith Black Mambazo under different EventClass tags.
Shows the raw bucket prediction (pre-capacity cap) and the final capped value.
"""
import os
import pandas as pd
import numpy as np

os.chdir("/Users/antho/Documents/WPI-MW")

from forecast_2526_comparison import (
    load_data, get_training_df, build_hierarchy_models,
    predict_model_a, cap_at_capacity, build_pwyw_lift,
)

EVENT = "Ladysmith Black Mambazo"
CLASSES = ["Standard", "Prestige", "Headliner", "Mission"]


def main():
    em, merged = load_data()
    prior = sorted([s for s in merged["Season"].dropna().unique() if s < "25-26"])
    train = get_training_df(merged, prior)
    (repeat_model, primary_sf, sf_ratio,
     primary, f1, f2, f3, f3a, f3b, f4, f5) = build_hierarchy_models(train)
    pwyw_lift, _ = build_pwyw_lift(
        merged, prior, repeat_model, primary_sf, sf_ratio,
        primary, f1, f2, f3, f3a, f3b, f4, f5)

    row = em[em["EventName"] == EVENT].drop_duplicates("EventName").copy()
    if row.empty:
        raise SystemExit(f"Could not find {EVENT}")
    row["EventCapacity"] = pd.to_numeric(row["EventCapacity"], errors="coerce")

    # Actual from completed data
    comp = pd.read_excel("Forecast_2526_Comparison.xlsx",
                         sheet_name="2526_Comparison")
    actual = int(comp.loc[comp["EventName"] == EVENT, "Actual"].iloc[0])

    print(f"Event: {EVENT}")
    print(f"Venue: {row.iloc[0]['EventVenue']}  ·  Capacity: "
          f"{int(row.iloc[0]['EventCapacity'])}  ·  "
          f"SubGenre: {row.iloc[0]['EventSubGenre']}  ·  "
          f"Current tag: {row.iloc[0]['EventClass']}")
    print(f"Actual attendance: {actual}\n")

    out = []
    for cls in CLASSES:
        probe = row.copy()
        probe["EventClass"] = cls
        probe["Actual"] = np.nan
        fc = predict_model_a(probe,
                              repeat_model, primary_sf, sf_ratio,
                              primary, f1, f2, f3, f3a, f3b, f4, f5,
                              pwyw_lift=pwyw_lift)
        raw = float(fc["Pred_A"].iloc[0])
        capped = float(cap_at_capacity(fc["Pred_A"], fc["EventCapacity"]).iloc[0])
        level = str(fc["FallbackLevel"].iloc[0])
        err_pct = (capped - actual) / actual * 100
        out.append((cls, raw, capped, err_pct, level))

    print(f"{'Class':10s}  {'Raw bucket':>10s}  {'Capped (900)':>12s}  "
          f"{'Err vs actual':>13s}  Fallback level")
    for cls, raw, capped, err, level in out:
        print(f"{cls:10s}  {raw:10.0f}  {capped:12.0f}  {err:+12.1f}%  {level}")


if __name__ == "__main__":
    main()
