"""
Honest hindcast evaluation with the patched (5-level hierarchy + artist
Bayesian adjustment) model. For each eval season, train on all OTHER seasons
and predict the target season — no data leakage.

Output: Forecast_Honest_Eval_Patched.xlsx
  All_Seasons sheet carries both Pred_A (base) and Pred_Adj (patched) for
  each event, plus PredictedAttendance alias = Pred_Adj for viz compatibility.
"""

import pandas as pd
import numpy as np
import warnings

from forecast_2526_comparison import (
    load_data, get_training_df, build_hierarchy_models,
    predict_model_a, cap_at_capacity,
)
from forecast_artist_adjustment import apply_artist_adjustment

warnings.filterwarnings("ignore", category=DeprecationWarning)

EVAL_SEASONS = ["22-23", "23-24", "24-25"]


def eval_one_season(merged, em, target_season):
    other_seasons = sorted([
        s for s in merged["Season"].dropna().unique()
        if s != target_season
    ])
    train = get_training_df(merged, other_seasons)
    primary, f1, f2, f3, f4, f5 = build_hierarchy_models(train)

    # Target-season actuals
    actuals = (
        merged[
            (merged["Season"] == target_season)
            & (merged["EventType"] == "Live")
            & (merged["EventStatus"] == "Complete")
            & (merged["TicketStatus"] == "Active")
            & (merged["Quantity"] > 0)
        ]
        .groupby(
            ["EventId", "EventName", "EventClass", "EventVenue",
             "EventGenre", "EventLoB", "EventSubGenre"],
            group_keys=False,
        )
        .agg(Actual=("Quantity", "sum"))
        .reset_index()
    )

    cap = em.drop_duplicates("EventId")[["EventId", "EventCapacity"]].copy()
    cap["EventCapacity"] = pd.to_numeric(cap["EventCapacity"], errors="coerce")
    actuals = actuals.merge(cap, on="EventId", how="left")

    fc = predict_model_a(actuals, primary, f1, f2, f3, f4, f5)
    fc["Pred_A"] = cap_at_capacity(fc["Pred_A"], fc["EventCapacity"])

    # Artist-adjustment training history (same holdout — exclude target season)
    hist_actuals = (
        train
        .groupby(
            ["EventId", "EventName", "EventClass", "EventVenue",
             "EventGenre", "EventLoB", "EventSubGenre"],
            group_keys=False,
        )
        .agg(Actual=("Quantity", "sum"))
        .reset_index()
        .merge(cap, on="EventId", how="left")
    )
    hist_fc = predict_model_a(hist_actuals, primary, f1, f2, f3, f4, f5)

    fc = apply_artist_adjustment(
        fc,
        merged_history=hist_fc,
        actuals_history=hist_fc["Actual"],
        bucket_preds_history=hist_fc["Pred_A"],
    )

    fc["Season"] = target_season
    return fc


def main():
    em, merged = load_data()

    all_rows = []
    print("HONEST HINDCAST (patched model — 5-level hierarchy + artist adj)")
    print("=" * 70)

    for season in EVAL_SEASONS:
        fc = eval_one_season(merged, em, season)
        valid = fc[fc["Actual"] > 0].copy()
        valid["AbsPctA"]   = (valid["Pred_A"]   - valid["Actual"]).abs() / valid["Actual"]
        valid["AbsPctAdj"] = (valid["Pred_Adj"] - valid["Actual"]).abs() / valid["Actual"]
        valid["SignedPctA"]   = (valid["Pred_A"]   - valid["Actual"]) / valid["Actual"]
        valid["SignedPctAdj"] = (valid["Pred_Adj"] - valid["Actual"]) / valid["Actual"]

        mape_a    = valid["AbsPctA"].mean() * 100
        mape_adj  = valid["AbsPctAdj"].mean() * 100
        bias_a    = valid["SignedPctA"].mean() * 100
        bias_adj  = valid["SignedPctAdj"].mean() * 100

        print(f"\n--- {season}  n={len(valid)} ---")
        print(f"  base   MAPE={mape_a:.1f}%  Bias={bias_a:+.1f}%")
        print(f"  patched MAPE={mape_adj:.1f}%  Bias={bias_adj:+.1f}%")
        all_rows.append(valid)

    combined = pd.concat(all_rows, ignore_index=True)

    # Viz-compatible columns
    out = combined.rename(columns={
        "Actual":   "ActualAttendance",
        "Pred_A":   "PredictedAttendance_A",
        "Pred_Adj": "PredictedAttendance",   # patched is primary
    }).copy()
    out["PercentError"]   = out["AbsPctAdj"]
    out["SignedPctError"] = out["SignedPctAdj"]
    out["AbsoluteError"]  = (out["PredictedAttendance"] - out["ActualAttendance"]).abs()
    out["SignedError"]    = out["PredictedAttendance"] - out["ActualAttendance"]

    keep_cols = [
        "Season", "EventId", "EventName", "EventClass", "EventGenre",
        "EventSubGenre", "EventLoB", "EventVenue", "EventCapacity",
        "PredictedAttendance", "PredictedAttendance_A", "ActualAttendance",
        "AbsoluteError", "SignedError", "PercentError", "SignedPctError",
        "FallbackLevel", "Adj_Source", "Adj_LogFactor",
    ]
    out = out[[c for c in keep_cols if c in out.columns]]

    out_path = "Forecast_Honest_Eval_Patched.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out.sort_values(["Season", "EventName"]).to_excel(
            writer, sheet_name="All_Seasons", index=False)

        summary = (
            combined.groupby("Season")
            .agg(n=("Actual", "size"),
                 MAPE_A=("AbsPctA", lambda x: x.mean() * 100),
                 MAPE_Adj=("AbsPctAdj", lambda x: x.mean() * 100),
                 Bias_A=("SignedPctA", lambda x: x.mean() * 100),
                 Bias_Adj=("SignedPctAdj", lambda x: x.mean() * 100))
            .reset_index()
        )
        summary.to_excel(writer, sheet_name="Season_Summary", index=False)

    print(f"\n✅ Written to {out_path}")


if __name__ == "__main__":
    main()
