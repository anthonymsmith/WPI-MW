"""
Calibration improvements to the attendance forecast:
  1. Per-class calibration factor (multiplicative bias correction)
  2. Shrinkage toward class median
  3. Trend measurement (reported, not applied)

All evaluated with proper temporal holdout.
"""

import pandas as pd
import numpy as np
import os
import warnings
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore", category=DeprecationWarning)

WORKING_DIR = '/Users/antho/Documents/WPI-MW'
INCLUDE_COMPS = True
EVAL_SEASONS = ['22-23', '23-24', '24-25']

ALL_SEASONS_ORDERED = ['14-15', '15-16', '16-17', '17-18', '18-19',
                       '19-20', '20-21', '21-22', '22-23', '23-24', '24-25']

BEST_WEIGHTS = {'19-20': 1.0, '20-21': 0.3, '21-22': 0.7,
                '22-23': 2.0, '23-24': 3.0, '24-25': 3.0}


def seasons_before(target):
    idx = ALL_SEASONS_ORDERED.index(target) if target in ALL_SEASONS_ORDERED else -1
    return ALL_SEASONS_ORDERED[:idx] if idx >= 0 else [s for s in ALL_SEASONS_ORDERED if s < target]


def assign_season_weight(season, weights=BEST_WEIGHTS):
    if pd.isnull(season): return 1.0
    if season < "19-20": return 1.0
    return weights.get(season, 1.0)


def load_data():
    os.chdir(WORKING_DIR)
    history_df = pd.read_csv("anon_DataMerge.csv")
    event_manifest = pd.read_excel("EventManifest.xlsx", sheet_name="EventManifest")

    history_df.columns = history_df.columns.str.strip()
    history_df['EventDate'] = pd.to_datetime(history_df.get('EventDate'), errors='coerce')
    history_df = history_df.drop(columns=[
        'EventName', 'Season', 'EventType', 'EventStatus',
        'EventClass', 'EventVenue', 'EventGenre'
    ], errors='ignore')

    manifest_deduped = event_manifest.drop_duplicates(subset='EventId')
    merged_history = history_df.merge(
        manifest_deduped[['EventId', 'EventName', 'Season', 'EventType', 'EventStatus',
                           'EventClass', 'EventVenue', 'EventGenre', 'EventCapacity']],
        on='EventId', how='left'
    )

    for c in ['EventType', 'EventStatus', 'TicketStatus']:
        if c in merged_history.columns:
            merged_history[c] = merged_history[c].astype(str).str.strip().str.title()

    merged_history['Quantity'] = pd.to_numeric(merged_history.get('Quantity'), errors='coerce').fillna(0)
    merged_history['TicketTotal'] = pd.to_numeric(merged_history.get('TicketTotal'), errors='coerce').fillna(0)
    merged_history['EventCapacity'] = pd.to_numeric(merged_history.get('EventCapacity'), errors='coerce')
    return event_manifest, merged_history


def build_filtered_df(merged_history, training_seasons):
    mh = merged_history[merged_history['Season'].isin(training_seasons)].copy()
    mask = mh['Quantity'] > 0 if INCLUDE_COMPS else mh['TicketTotal'] > 0
    filtered = mh[
        (mh['EventType'] == 'Live') &
        (mh['EventStatus'] == 'Complete') &
        (mh['TicketStatus'] == 'Active') &
        mask
    ].copy()
    filtered['Weight'] = filtered['Season'].apply(assign_season_weight)
    filtered['WeightedTickets'] = filtered['Quantity'] * filtered['Weight']
    return filtered


def compute_event_attendance(filtered_df):
    return (
        filtered_df
        .groupby(['EventId', 'EventName', 'EventClass', 'EventVenue', 'EventGenre', 'Season'],
                 group_keys=False)
        .agg(
            EventCapacity=('EventCapacity', 'max'),
            TotalTickets=('Quantity', 'sum'),
            Weight=('Weight', 'first')
        )
        .reset_index()
    )


def build_models(event_attendance):
    def wgt_avg(g):
        wt = (g['TotalTickets'] * g['Weight']).sum()
        tw = g['Weight'].sum()
        return wt / tw if tw > 0 else np.nan

    wg = (
        event_attendance
        .groupby(['EventClass', 'EventVenue', 'EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WeightedAvgAttendance': wgt_avg(g)}))
        .reset_index()
    )
    f1 = (
        event_attendance
        .groupby(['EventClass', 'EventVenue'], group_keys=False)
        .apply(lambda g: pd.Series({'WeightedAvgAttendance_f1': wgt_avg(g)}))
        .reset_index()
    )
    f2 = (
        event_attendance
        .groupby(['EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WeightedAvgAttendance_f2': wgt_avg(g)}))
        .reset_index()
    )
    return wg, f1, f2


def apply_model(events_df, wg, f1, f2):
    """Generate raw predictions for a set of events."""
    fc = events_df.copy()
    for col in ['EventClass', 'EventVenue', 'EventGenre']:
        fc[col] = fc[col].astype(str).str.strip()
    fc['EventCapacity'] = pd.to_numeric(fc.get('EventCapacity'), errors='coerce')

    fc = fc.merge(wg[['EventClass', 'EventVenue', 'EventGenre', 'WeightedAvgAttendance']],
                  on=['EventClass', 'EventVenue', 'EventGenre'], how='left')
    fc = fc.merge(f1[['EventClass', 'EventVenue', 'WeightedAvgAttendance_f1']],
                  on=['EventClass', 'EventVenue'], how='left')
    fc = fc.merge(f2[['EventGenre', 'WeightedAvgAttendance_f2']],
                  on='EventGenre', how='left')

    fc['RawPred'] = (
        fc['WeightedAvgAttendance']
        .combine_first(fc['WeightedAvgAttendance_f1'])
        .combine_first(fc['WeightedAvgAttendance_f2'])
    )
    fc['RawPred'] = fc.apply(
        lambda r: min(r['RawPred'], r['EventCapacity'])
        if pd.notnull(r['RawPred']) and pd.notnull(r['EventCapacity'])
        else r['RawPred'],
        axis=1
    )
    return fc


def get_actuals(merged_history, season):
    mask = merged_history['Quantity'] > 0 if INCLUDE_COMPS else merged_history['TicketTotal'] > 0
    return (
        merged_history[
            (merged_history['Season'] == season) &
            (merged_history['EventType'] == 'Live') &
            (merged_history['EventStatus'] == 'Complete') &
            (merged_history['TicketStatus'] == 'Active') &
            mask
        ]
        .groupby(['EventId', 'EventName', 'EventClass', 'EventVenue', 'EventGenre'],
                 group_keys=False)
        .agg(Actual=('Quantity', 'sum'))
        .reset_index()
    )


def compute_calibration_factors(event_manifest, merged_history, training_seasons):
    """
    Compute per-EventClass calibration factors from training data using
    leave-one-season-out within the training set.
    Returns dict: {EventClass -> calibration_factor}
    where factor < 1 means the model over-forecasts that class.
    """
    ratios = []  # (EventClass, actual/predicted)

    for s in training_seasons:
        prior = [x for x in training_seasons if x < s]
        if not prior:
            continue

        filtered_prior = build_filtered_df(merged_history, prior)
        if filtered_prior.empty:
            continue
        ea = compute_event_attendance(filtered_prior)
        wg, f1, f2 = build_models(ea)

        events_s = event_manifest[
            (event_manifest['Season'] == s) &
            (event_manifest['EventType'].str.lower().str.strip() == 'live')
        ].drop_duplicates(subset='EventId').copy()

        if events_s.empty:
            continue

        preds = apply_model(events_s, wg, f1, f2)
        actuals = get_actuals(merged_history, s)
        comp = preds.merge(actuals[['EventId', 'Actual']], on='EventId', how='inner')
        comp = comp[(comp['Actual'] > 0) & comp['RawPred'].notna() & (comp['RawPred'] > 0)]

        for _, row in comp.iterrows():
            ratios.append({'EventClass': row['EventClass'],
                           'ratio': row['Actual'] / row['RawPred']})

    if not ratios:
        return {}

    ratios_df = pd.DataFrame(ratios)
    # Use median ratio per class (more robust to outliers than mean)
    factors = ratios_df.groupby('EventClass')['ratio'].median().to_dict()
    return factors


def compute_class_medians(merged_history, training_seasons):
    """Median actual attendance per EventClass from training data."""
    actuals_all = []
    for s in training_seasons:
        a = get_actuals(merged_history, s)
        actuals_all.append(a)
    if not actuals_all:
        return {}
    all_a = pd.concat(actuals_all, ignore_index=True)
    return all_a.groupby('EventClass')['Actual'].median().to_dict()


def apply_calibration(pred, event_class, cal_factors):
    factor = cal_factors.get(event_class, 1.0)
    return pred * factor if pd.notnull(pred) else pred


def apply_shrinkage(pred, event_class, class_medians, alpha):
    """Blend prediction toward class median: (1-alpha)*pred + alpha*median."""
    median = class_medians.get(event_class)
    if pd.notnull(pred) and median is not None:
        return (1 - alpha) * pred + alpha * median
    return pred


def score_predictions(comp):
    valid = comp[comp['Actual'] > 0].dropna(subset=['Pred'])
    if valid.empty:
        return np.nan, np.nan, np.nan
    mape = ((valid['Actual'] - valid['Pred']).abs() / valid['Actual']).mean() * 100
    wape = (valid['Actual'] - valid['Pred']).abs().sum() / valid['Actual'].sum() * 100
    bias = ((valid['Pred'] - valid['Actual']) / valid['Actual']).mean() * 100
    return mape, wape, bias


def run_hindcast(event_manifest, merged_history, alpha=0.0, use_calibration=True):
    """Full temporal holdout with optional calibration and shrinkage."""
    all_comps = []
    for season in EVAL_SEASONS:
        train = seasons_before(season)
        if not train:
            continue

        filtered = build_filtered_df(merged_history, train)
        if filtered.empty:
            continue
        ea = compute_event_attendance(filtered)
        wg, f1, f2 = build_models(ea)

        events = event_manifest[
            (event_manifest['Season'] == season) &
            (event_manifest['EventType'].str.lower().str.strip() == 'live')
        ].drop_duplicates(subset='EventId').copy()

        preds = apply_model(events, wg, f1, f2)
        actuals = get_actuals(merged_history, season)
        comp = preds.merge(actuals[['EventId', 'Actual']], on='EventId', how='inner')
        comp['Season'] = season

        # Calibration
        if use_calibration:
            cal_factors = compute_calibration_factors(event_manifest, merged_history, train)
            comp['CalFactor'] = comp['EventClass'].map(cal_factors).fillna(1.0)
            comp['Pred'] = comp['RawPred'] * comp['CalFactor']
        else:
            comp['CalFactor'] = 1.0
            comp['Pred'] = comp['RawPred']

        # Shrinkage
        if alpha > 0:
            class_medians = compute_class_medians(merged_history, train)
            comp['ClassMedian'] = comp['EventClass'].map(class_medians)
            comp['Pred'] = comp.apply(
                lambda r: apply_shrinkage(r['Pred'], r['EventClass'], class_medians, alpha),
                axis=1
            )

        # Cap at capacity
        comp['Pred'] = comp.apply(
            lambda r: min(r['Pred'], r['EventCapacity'])
            if pd.notnull(r['Pred']) and pd.notnull(r['EventCapacity'])
            else r['Pred'],
            axis=1
        )

        all_comps.append(comp)

    return pd.concat(all_comps, ignore_index=True) if all_comps else pd.DataFrame()


def measure_attendance_trend(merged_history):
    """
    Measure linear trend in attendance per EventClass over time.
    Returns a DataFrame with slope (tickets/season) and % change per season.
    """
    mask = merged_history['Quantity'] > 0 if INCLUDE_COMPS else merged_history['TicketTotal'] > 0
    actuals = merged_history[
        (merged_history['EventType'] == 'Live') &
        (merged_history['EventStatus'] == 'Complete') &
        (merged_history['TicketStatus'] == 'Active') &
        mask
    ].copy()

    # Per-event totals
    event_totals = (
        actuals
        .groupby(['EventId', 'EventClass', 'Season'], group_keys=False)
        .agg(Attendance=('Quantity', 'sum'))
        .reset_index()
    )

    # Season index
    season_list = sorted(event_totals['Season'].dropna().unique())
    season_idx = {s: i for i, s in enumerate(season_list)}
    event_totals['SeasonIdx'] = event_totals['Season'].map(season_idx)

    trend_rows = []
    for ec in sorted(event_totals['EventClass'].dropna().unique()):
        sub = event_totals[event_totals['EventClass'] == ec].dropna(subset=['SeasonIdx'])
        if len(sub) < 4:
            continue
        # Season-level mean attendance (per event)
        season_means = sub.groupby('SeasonIdx')['Attendance'].mean().reset_index()
        if len(season_means) < 3:
            continue
        x = season_means['SeasonIdx'].values.astype(float)
        y = season_means['Attendance'].values.astype(float)
        # Linear fit
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        baseline = y.mean()
        pct_per_season = (slope / baseline * 100) if baseline > 0 else np.nan
        # R²
        y_hat = np.polyval(coeffs, x)
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        trend_rows.append({
            'EventClass': ec,
            'N_Events': len(sub),
            'N_Seasons': len(season_means),
            'AvgAttendance': round(baseline, 1),
            'Slope_TicketsPerSeason': round(slope, 1),
            'PctChangePerSeason': round(pct_per_season, 1),
            'R2': round(r2, 3)
        })

    return pd.DataFrame(trend_rows)


def main():
    event_manifest, merged_history = load_data()

    # ── Step 0: Baseline ──────────────────────────────────────────────────
    print("=" * 60)
    print("BASELINE (no calibration, no shrinkage)")
    base_comp = run_hindcast(event_manifest, merged_history, alpha=0.0, use_calibration=False)
    mape, wape, bias = score_predictions(base_comp)
    print(f"  MAPE={mape:.1f}%  WAPE={wape:.1f}%  Bias={bias:+.1f}%")

    # ── Step 1: Calibration only ──────────────────────────────────────────
    print("\nCALIBRATION only (alpha=0)")
    cal_comp = run_hindcast(event_manifest, merged_history, alpha=0.0, use_calibration=True)
    mape_c, wape_c, bias_c = score_predictions(cal_comp)
    print(f"  MAPE={mape_c:.1f}%  WAPE={wape_c:.1f}%  Bias={bias_c:+.1f}%")

    # Show calibration factors from full training data (for inspection)
    print("\n  Calibration factors (median actual/predicted per class, all training data):")
    all_train = [s for s in ALL_SEASONS_ORDERED if s < '24-25']
    cal_factors_full = compute_calibration_factors(event_manifest, merged_history, all_train)
    for ec, f in sorted(cal_factors_full.items()):
        direction = "under" if f > 1.0 else "over"
        print(f"    {ec:<20} factor={f:.3f}  (model {direction}-forecasts by {abs(1-f)*100:.1f}%)")

    # ── Step 2: Calibration + shrinkage sweep ─────────────────────────────
    print("\n" + "=" * 60)
    print("CALIBRATION + SHRINKAGE SWEEP")
    print(f"  {'Alpha':>6}  {'MAPE':>7}  {'WAPE':>7}  {'Bias':>8}")
    print(f"  {'-----':>6}  {'----':>7}  {'----':>7}  {'----':>8}")

    alpha_results = []
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        comp = run_hindcast(event_manifest, merged_history, alpha=alpha, use_calibration=True)
        m, w, b = score_predictions(comp)
        alpha_results.append({'alpha': alpha, 'MAPE': m, 'WAPE': w, 'Bias': b})
        print(f"  {alpha:>6.2f}  {m:>6.1f}%  {w:>6.1f}%  {b:>+7.1f}%")

    best = min(alpha_results, key=lambda x: x['WAPE'])
    print(f"\n  Best alpha (min WAPE): {best['alpha']}  →  WAPE={best['WAPE']:.1f}%  Bias={best['Bias']:+.1f}%")

    # ── Step 3: Trend measurement ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ATTENDANCE TREND (measured, not applied)")
    trend_df = measure_attendance_trend(merged_history)
    print(trend_df.to_string(index=False))

    # ── Step 4: Per-season detail with best config ────────────────────────
    print("\n" + "=" * 60)
    print(f"PER-SEASON DETAIL  (calibration + shrinkage alpha={best['alpha']})")
    best_comp = run_hindcast(event_manifest, merged_history,
                             alpha=best['alpha'], use_calibration=True)
    for season in EVAL_SEASONS:
        sc = best_comp[best_comp['Season'] == season]
        m, w, b = score_predictions(sc)
        print(f"  {season}: MAPE={m:.1f}%  WAPE={w:.1f}%  Bias={b:+.1f}%  (n={len(sc[sc['Actual']>0])})")

    # ── Step 5: Bias by class with best config ────────────────────────────
    print(f"\nBIAS BY CLASS (calibrated + shrinkage alpha={best['alpha']}):")
    valid = best_comp[best_comp['Actual'] > 0].dropna(subset=['Pred']).copy()
    valid['AbsErr'] = (valid['Actual'] - valid['Pred']).abs()
    valid['SignedPct'] = (valid['Pred'] - valid['Actual']) / valid['Actual']
    for ec, g in valid.groupby('EventClass'):
        wape_ec = g['AbsErr'].sum() / g['Actual'].sum() * 100
        bias_ec = g['SignedPct'].mean() * 100
        print(f"  {ec:<20} WAPE={wape_ec:.1f}%  Bias={bias_ec:+.1f}%  (n={len(g)})")

    # ── Write results ─────────────────────────────────────────────────────
    output_file = "Forecast_Calibration_Results.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        best_comp[['Season', 'EventId', 'EventName', 'EventClass', 'EventVenue', 'EventGenre',
                   'EventCapacity', 'RawPred', 'CalFactor', 'Pred', 'Actual']].sort_values(
            ['Season', 'EventName']).to_excel(writer, sheet_name="BestConfig_Detail", index=False)

        pd.DataFrame(alpha_results).to_excel(writer, sheet_name="Shrinkage_Sweep", index=False)

        cal_factors_df = pd.DataFrame(
            [{'EventClass': k, 'CalibrationFactor': v,
              'ImpliedBias_Pct': round((1 - v) * 100, 1)}
             for k, v in sorted(cal_factors_full.items())]
        )
        cal_factors_df.to_excel(writer, sheet_name="CalibrationFactors", index=False)

        trend_df.to_excel(writer, sheet_name="AttendanceTrend", index=False)

    print(f"\n✅ Results written to {output_file}")


if __name__ == "__main__":
    main()
