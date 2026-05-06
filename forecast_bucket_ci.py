"""
Bucket-level CI fill-in using holdout residual bootstrap.

For events without artist-signal CI (Pred_Adj_Lo == Pred_Adj_Hi), attach a
90% prediction interval from the FallbackLevel-calibrated log-residual
quantiles in fallback_ci_calibration.json.

Pools strata with n < min_n toward _overall.
"""
import json
import numpy as np
import pandas as pd


def _strip_pwyw(level):
    return str(level).split(" × PWYW lift")[0]


def apply_bucket_ci(fc, calib_path="fallback_ci_calibration.json"):
    with open(calib_path) as f:
        calib = json.load(f)

    overall = calib["_overall"]
    min_n   = overall.get("min_n", 5)

    result = fc.copy()

    # Degenerate CI = artist adjustment did not fire
    degenerate = (
        np.isclose(result["Pred_Adj_Lo"], result["Pred_Adj_Hi"])
        & result["Pred_Adj"].notna()
        & (result["Pred_Adj"] > 0)
    )

    if "Bucket_CI_Source" not in result.columns:
        result["Bucket_CI_Source"] = ""

    for idx in result.index[degenerate]:
        base  = result.at[idx, "Pred_Adj"]
        level = _strip_pwyw(result.at[idx, "FallbackLevel"])

        entry = calib.get(level)
        if entry is None or entry.get("n", 0) < min_n or entry.get("q_lo") is None:
            entry = overall
            source = f"Bucket _overall (level={level})"
        else:
            source = f"Bucket {level}"

        lo = base * np.exp(entry["q_lo"])
        hi = base * np.exp(entry["q_hi"])

        cap = result.at[idx, "EventCapacity"] if "EventCapacity" in result.columns else None
        if pd.notna(cap) and cap > 0:
            hi = min(hi, cap)
        lo = max(lo, 0.0)

        result.at[idx, "Pred_Adj_Lo"]      = round(lo)
        result.at[idx, "Pred_Adj_Hi"]      = round(hi)
        result.at[idx, "Bucket_CI_Source"] = source

    return result
