"""
Season weight optimization via proper temporal holdout cross-validation.

For each hindcast season, training uses ONLY data from prior seasons
(mimicking what would have been available at forecast time).

Optimizes weights to minimize WAPE across available hindcast seasons.
"""

import pandas as pd
import numpy as np
import os
import warnings
from itertools import product
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=DeprecationWarning)

WORKING_DIR = '/Users/antho/Documents/WPI-MW'
INCLUDE_COMPS = True

# Seasons in chronological order for temporal holdout
ALL_SEASONS_ORDERED = ['14-15', '15-16', '16-17', '17-18', '18-19',
                       '19-20', '20-21', '21-22', '22-23', '23-24', '24-25']

# Hindcast targets — must have at least a few prior seasons for meaningful training
EVAL_SEASONS = ['22-23', '23-24', '24-25']

# Current weights (baseline)
BASELINE_WEIGHTS = {
    '19-20': 1.0, '20-21': 0.3, '21-22': 0.7,
    '22-23': 2.0, '23-24': 3.0, '24-25': 3.0,
}


def assign_season_weight(season, weights):
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


def seasons_before(target):
    """Return seasons strictly before target (temporal holdout)."""
    idx = ALL_SEASONS_ORDERED.index(target) if target in ALL_SEASONS_ORDERED else -1
    if idx < 0:
        return [s for s in ALL_SEASONS_ORDERED if s < target]
    return ALL_SEASONS_ORDERED[:idx]


def build_filtered_df(merged_history, training_seasons, weights):
    mh = merged_history[merged_history['Season'].isin(training_seasons)].copy()

    if INCLUDE_COMPS:
        mask = mh['Quantity'] > 0
    else:
        mask = mh['TicketTotal'] > 0

    filtered = mh[
        (mh['EventType'] == 'Live') &
        (mh['EventStatus'] == 'Complete') &
        (mh['TicketStatus'] == 'Active') &
        mask
    ].copy()

    filtered['Weight'] = filtered['Season'].apply(lambda s: assign_season_weight(s, weights))
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
    def wgt_avg(g, ticket_col='TotalTickets', wgt_col='Weight'):
        wt = (g[ticket_col] * g[wgt_col]).sum()
        tw = g[wgt_col].sum()
        return wt / tw if tw > 0 else np.nan

    wg = (
        event_attendance
        .groupby(['EventClass', 'EventVenue', 'EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WeightedAvgAttendance': wgt_avg(g),
            'N': len(g)
        }))
        .reset_index()
    )
    f1 = (
        event_attendance
        .groupby(['EventClass', 'EventVenue'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WeightedAvgAttendance_f1': wgt_avg(g)
        }))
        .reset_index()
    )
    f2 = (
        event_attendance
        .groupby(['EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WeightedAvgAttendance_f2': wgt_avg(g)
        }))
        .reset_index()
    )
    return wg, f1, f2


def forecast_season(event_manifest, wg, f1, f2, season):
    fc = event_manifest[
        (event_manifest['Season'] == season) &
        (event_manifest['EventType'].str.lower().str.strip() == 'live')
    ].drop_duplicates(subset='EventId').copy()

    for col in ['EventClass', 'EventVenue', 'EventGenre']:
        fc[col] = fc[col].astype(str).str.strip()
    fc['EventCapacity'] = pd.to_numeric(fc.get('EventCapacity'), errors='coerce')

    fc = fc.merge(wg[['EventClass', 'EventVenue', 'EventGenre', 'WeightedAvgAttendance']],
                  on=['EventClass', 'EventVenue', 'EventGenre'], how='left')
    fc = fc.merge(f1[['EventClass', 'EventVenue', 'WeightedAvgAttendance_f1']],
                  on=['EventClass', 'EventVenue'], how='left')
    fc = fc.merge(f2[['EventGenre', 'WeightedAvgAttendance_f2']],
                  on='EventGenre', how='left')

    fc['Pred'] = (
        fc['WeightedAvgAttendance']
        .combine_first(fc['WeightedAvgAttendance_f1'])
        .combine_first(fc['WeightedAvgAttendance_f2'])
    )
    fc['Pred'] = fc.apply(
        lambda r: min(r['Pred'], r['EventCapacity'])
        if pd.notnull(r['Pred']) and pd.notnull(r['EventCapacity']) else r['Pred'],
        axis=1
    )
    return fc[['EventId', 'EventClass', 'EventGenre', 'EventVenue', 'Pred']]


def get_actuals(merged_history, season):
    mh = merged_history.copy()
    mask = mh['Quantity'] > 0 if INCLUDE_COMPS else mh['TicketTotal'] > 0
    return (
        mh[
            (mh['Season'] == season) &
            (mh['EventType'] == 'Live') &
            (mh['EventStatus'] == 'Complete') &
            (mh['TicketStatus'] == 'Active') &
            mask
        ]
        .groupby(['EventId', 'EventClass', 'EventGenre', 'EventVenue'], group_keys=False)
        .agg(Actual=('Quantity', 'sum'))
        .reset_index()
    )


def evaluate_weights(event_manifest, merged_history, weights, seasons=EVAL_SEASONS):
    """Run temporal holdout for given weights; return combined WAPE."""
    all_abs_err, all_actual = [], []
    for season in seasons:
        train_seasons = seasons_before(season)
        if not train_seasons:
            continue
        filtered = build_filtered_df(merged_history, train_seasons, weights)
        if filtered.empty:
            continue
        ea = compute_event_attendance(filtered)
        wg, f1, f2 = build_models(ea)
        preds = forecast_season(event_manifest, wg, f1, f2, season)
        actuals = get_actuals(merged_history, season)
        comp = preds.merge(actuals[['EventId', 'Actual']], on='EventId', how='inner')
        comp = comp[comp['Actual'] > 0].dropna(subset=['Pred'])
        all_abs_err.append((comp['Actual'] - comp['Pred']).abs())
        all_actual.append(comp['Actual'])

    if not all_abs_err:
        return 999.0
    total_err = pd.concat(all_abs_err).sum()
    total_act = pd.concat(all_actual).sum()
    return total_err / total_act * 100


def main():
    event_manifest, merged_history = load_data()

    # ── Baseline (proper temporal holdout) ──────────────────────────────────
    baseline_wape = evaluate_weights(event_manifest, merged_history, BASELINE_WEIGHTS)
    print("=" * 60)
    print("BASELINE weights (temporal holdout)")
    print(f"  Weights: {BASELINE_WEIGHTS}")
    print(f"  Combined WAPE: {baseline_wape:.1f}%")

    # ── Per-season breakdown with baseline ──────────────────────────────────
    print("\nPer-season (temporal holdout):")
    for season in EVAL_SEASONS:
        train_seasons = seasons_before(season)
        if not train_seasons:
            print(f"  {season}: insufficient training data")
            continue
        filtered = build_filtered_df(merged_history, train_seasons, BASELINE_WEIGHTS)
        if filtered.empty:
            print(f"  {season}: no training data")
            continue
        ea = compute_event_attendance(filtered)
        wg, f1, f2 = build_models(ea)
        preds = forecast_season(event_manifest, wg, f1, f2, season)
        actuals = get_actuals(merged_history, season)
        comp = preds.merge(actuals[['EventId', 'Actual']], on='EventId', how='inner')
        comp = comp[comp['Actual'] > 0].dropna(subset=['Pred'])
        mape = ((comp['Actual'] - comp['Pred']).abs() / comp['Actual']).mean() * 100
        wape = (comp['Actual'] - comp['Pred']).abs().sum() / comp['Actual'].sum() * 100
        bias = ((comp['Pred'] - comp['Actual']) / comp['Actual']).mean() * 100
        n_train = merged_history[merged_history['Season'].isin(train_seasons)]['Season'].nunique()
        print(f"  {season}: MAPE={mape:.1f}%  WAPE={wape:.1f}%  Bias={bias:+.1f}%"
              f"  (trained on {n_train} seasons: {', '.join(train_seasons[-3:])}{'...' if n_train > 3 else ''})")

    # ── Grid search over key weights ────────────────────────────────────────
    # Tune 4 weights: 21-22 (recovery), 22-23 (bounce-back), 23-24, 24-25
    # Pre-19-20 and 19-20 stay at 1.0; 20-21 stays at 0.3 (clear anomaly)
    print("\n" + "=" * 60)
    print("GRID SEARCH (w_2122, w_2223, w_2324, w_2425)")
    print("Optimizing combined WAPE across 22-23, 23-24, 24-25 ...")

    grid_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    best_wape = baseline_wape
    best_weights = BASELINE_WEIGHTS.copy()
    results = []

    for w2122, w2223, w2324, w2425 in product(grid_options, repeat=4):
        w = {'19-20': 1.0, '20-21': 0.3, '21-22': w2122,
             '22-23': w2223, '23-24': w2324, '24-25': w2425}
        wape = evaluate_weights(event_manifest, merged_history, w)
        results.append((wape, w.copy()))
        if wape < best_wape:
            best_wape = wape
            best_weights = w.copy()

    results.sort(key=lambda x: x[0])
    print(f"\nTop 5 weight sets:")
    for wape, w in results[:5]:
        print(f"  WAPE={wape:.1f}%  21-22={w['21-22']}  22-23={w['22-23']}"
              f"  23-24={w['23-24']}  24-25={w['24-25']}")

    print(f"\nBest weights:   {best_weights}")
    print(f"Best WAPE:      {best_wape:.1f}%  (was {baseline_wape:.1f}%)")
    print(f"Improvement:    {baseline_wape - best_wape:.1f}pp")

    # ── Per-season with optimized weights ───────────────────────────────────
    print("\nPer-season (optimized weights):")
    for season in EVAL_SEASONS:
        train_seasons = seasons_before(season)
        if not train_seasons:
            continue
        filtered = build_filtered_df(merged_history, train_seasons, best_weights)
        if filtered.empty:
            continue
        ea = compute_event_attendance(filtered)
        wg, f1, f2 = build_models(ea)
        preds = forecast_season(event_manifest, wg, f1, f2, season)
        actuals = get_actuals(merged_history, season)
        comp = preds.merge(actuals[['EventId', 'Actual']], on='EventId', how='inner')
        comp = comp[comp['Actual'] > 0].dropna(subset=['Pred'])
        mape = ((comp['Actual'] - comp['Pred']).abs() / comp['Actual']).mean() * 100
        wape = (comp['Actual'] - comp['Pred']).abs().sum() / comp['Actual'].sum() * 100
        bias = ((comp['Pred'] - comp['Actual']) / comp['Actual']).mean() * 100
        print(f"  {season}: MAPE={mape:.1f}%  WAPE={wape:.1f}%  Bias={bias:+.1f}%")

    # ── Write results ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(
        [{'WAPE': w, '21-22': ws['21-22'], '22-23': ws['22-23'],
          '23-24': ws['23-24'], '24-25': ws['24-25']}
         for w, ws in results[:50]]
    ).sort_values('WAPE')

    output_file = "Forecast_Weight_Optimization.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Top50_GridSearch", index=False)

    print(f"\n✅ Top-50 grid search results written to {output_file}")


if __name__ == "__main__":
    main()
