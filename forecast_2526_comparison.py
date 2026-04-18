"""
Compare two forecasting approaches for 25-26 season to date:
  Model A: Current weighted average by (EventClass, EventVenue, EventGenre)
  Model B: Event name matching — use event's own history first, fall back to Model A

Training: all seasons through 24-25 (temporal holdout for 25-26).
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

WORKING_DIR = '/Users/antho/Documents/WPI-MW'
INCLUDE_COMPS = True
FORECAST_SEASON = '25-26'

WEIGHTS = {'19-20': 1.0, '20-21': 0.3, '21-22': 0.7,
           '22-23': 2.0, '23-24': 3.0, '24-25': 3.0}


def assign_season_weight(season):
    if pd.isnull(season): return 1.0
    if season < "19-20": return 1.0
    return WEIGHTS.get(season, 1.0)


def load_data():
    os.chdir(WORKING_DIR)
    df = pd.read_csv("anon_DataMerge.csv")
    em = pd.read_excel("EventManifest.xlsx", sheet_name="EventManifest")
    df.columns = df.columns.str.strip()
    em.columns = em.columns.str.strip()

    for c in ['EventType', 'EventStatus', 'TicketStatus']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()

    df['Quantity'] = pd.to_numeric(df.get('Quantity'), errors='coerce').fillna(0)
    df['TicketTotal'] = pd.to_numeric(df.get('TicketTotal'), errors='coerce').fillna(0)

    manifest_meta = em.drop_duplicates(subset='EventId')[
        ['EventId', 'EventName', 'EventStatus', 'EventType', 'EventClass',
         'EventVenue', 'EventGenre', 'EventCapacity']
    ]
    for c in ['EventStatus', 'EventType']:
        manifest_meta[c] = manifest_meta[c].astype(str).str.strip().str.title()

    merged = df.merge(manifest_meta, on='EventId', how='left', suffixes=('', '_m'))
    for c in ['EventStatus', 'EventType', 'EventClass', 'EventVenue', 'EventGenre']:
        col_m = c + '_m'
        if col_m in merged.columns:
            merged[c] = merged[c].combine_first(merged[col_m])
            merged.drop(columns=[col_m], inplace=True, errors='ignore')
    for c in ['EventType', 'EventStatus', 'TicketStatus']:
        if c in merged.columns:
            merged[c] = merged[c].astype(str).str.strip().str.title()

    merged['EventCapacity'] = pd.to_numeric(merged.get('EventCapacity'), errors='coerce')
    return em, merged


def get_training_df(merged, training_seasons):
    mask = merged['Quantity'] > 0 if INCLUDE_COMPS else merged['TicketTotal'] > 0
    filtered = merged[
        (merged['Season'].isin(training_seasons)) &
        (merged['EventType'] == 'Live') &
        (merged['EventStatus'] == 'Complete') &
        (merged['TicketStatus'] == 'Active') &
        mask
    ].copy()
    filtered['Weight'] = filtered['Season'].apply(assign_season_weight)
    filtered['WeightedTickets'] = filtered['Quantity'] * filtered['Weight']
    return filtered


def build_class_models(filtered):
    """Model A: weighted avg by (EventClass, EventVenue, EventGenre) with fallbacks."""
    ea = (
        filtered
        .groupby(['EventId', 'EventName', 'EventClass', 'EventVenue', 'EventGenre', 'Season'],
                 group_keys=False)
        .agg(TotalTickets=('Quantity', 'sum'), Weight=('Weight', 'first'),
             EventCapacity=('EventCapacity', 'max'))
        .reset_index()
    )

    def wg_avg(g):
        wt = (g['TotalTickets'] * g['Weight']).sum()
        tw = g['Weight'].sum()
        return wt / tw if tw > 0 else np.nan

    primary = (
        ea.groupby(['EventClass', 'EventVenue', 'EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WeightedAvg': wg_avg(g)})).reset_index()
    )
    f1 = (
        ea.groupby(['EventClass', 'EventVenue'], group_keys=False)
        .apply(lambda g: pd.Series({'WeightedAvg_f1': wg_avg(g)})).reset_index()
    )
    f2 = (
        ea.groupby(['EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WeightedAvg_f2': wg_avg(g)})).reset_index()
    )
    return primary, f1, f2


def build_name_model(filtered):
    """Model B component: weighted avg attendance per EventName."""
    ea = (
        filtered
        .groupby(['EventId', 'EventName', 'Season'], group_keys=False)
        .agg(TotalTickets=('Quantity', 'sum'), Weight=('Weight', 'first'))
        .reset_index()
    )
    name_model = (
        ea.groupby('EventName', group_keys=False)
        .apply(lambda g: pd.Series({
            'NameWeightedAvg': (g['TotalTickets'] * g['Weight']).sum() / g['Weight'].sum(),
            'NameOccurrences': len(g),
            'NameSeasons': g['Season'].nunique()
        }))
        .reset_index()
    )
    return name_model


def predict_model_a(events_df, primary, f1, f2):
    fc = events_df.copy()
    for col in ['EventClass', 'EventVenue', 'EventGenre']:
        fc[col] = fc[col].astype(str).str.strip()

    fc = fc.merge(primary, on=['EventClass', 'EventVenue', 'EventGenre'], how='left')
    fc = fc.merge(f1, on=['EventClass', 'EventVenue'], how='left')
    fc = fc.merge(f2, on='EventGenre', how='left')

    fc['Pred_A'] = (
        fc['WeightedAvg']
        .combine_first(fc['WeightedAvg_f1'])
        .combine_first(fc['WeightedAvg_f2'])
    )
    fc['FallbackLevel'] = 'Primary'
    fc.loc[fc['WeightedAvg'].isna() & fc['WeightedAvg_f1'].notna(), 'FallbackLevel'] = 'F1 (Class+Venue)'
    fc.loc[fc['WeightedAvg'].isna() & fc['WeightedAvg_f1'].isna(), 'FallbackLevel'] = 'F2 (Genre)'
    return fc


def predict_model_b(events_df_with_a, name_model):
    """
    Model B: use event name history if available; otherwise use Model A prediction.
    Events with ≥2 prior seasons of name history get the name-based prediction.
    """
    fc = events_df_with_a.copy()
    fc = fc.merge(name_model, on='EventName', how='left')

    # Use name model if event has appeared in ≥2 prior seasons
    has_name_history = fc['NameSeasons'] >= 2
    fc['Pred_B'] = np.where(has_name_history, fc['NameWeightedAvg'], fc['Pred_A'])
    fc['Model_B_Source'] = np.where(has_name_history,
                                    fc['NameSeasons'].apply(lambda n: f'Name history ({int(n) if pd.notna(n) else 0} seasons)'),
                                    'Fallback to Model A')
    return fc


def cap_at_capacity(pred_series, capacity_series):
    return pred_series.combine(capacity_series, lambda p, c:
        min(p, c) if pd.notna(p) and pd.notna(c) else p)


def print_metrics(label, comp, pred_col):
    valid = comp[(comp['Actual'] > 0) & comp[pred_col].notna()].copy()
    valid['AbsErr'] = (valid['Actual'] - valid[pred_col]).abs()
    valid['PctErr'] = valid['AbsErr'] / valid['Actual']
    valid['SignedPct'] = (valid[pred_col] - valid['Actual']) / valid['Actual']
    mape = valid['PctErr'].mean() * 100
    wape = valid['AbsErr'].sum() / valid['Actual'].sum() * 100
    bias = valid['SignedPct'].mean() * 100
    print(f"  {label:<35} MAPE={mape:.1f}%  WAPE={wape:.1f}%  Bias={bias:+.1f}%  (n={len(valid)})")


def main():
    em, merged = load_data()

    # Training: everything before 25-26
    all_prior = sorted([s for s in merged['Season'].dropna().unique() if s < FORECAST_SEASON])
    filtered_train = get_training_df(merged, all_prior)

    # Build models
    primary, f1, f2 = build_class_models(filtered_train)
    name_model = build_name_model(filtered_train)

    # Get 25-26 completed events
    actuals_2526 = (
        merged[
            (merged['Season'] == FORECAST_SEASON) &
            (merged['EventType'] == 'Live') &
            (merged['EventStatus'] == 'Complete') &
            (merged['TicketStatus'] == 'Active') &
            (merged['Quantity'] > 0)
        ]
        .groupby(['EventId', 'EventName', 'EventClass', 'EventVenue', 'EventGenre'], group_keys=False)
        .agg(Actual=('Quantity', 'sum'))
        .reset_index()
    )

    # Get capacity from manifest
    cap = em.drop_duplicates('EventId')[['EventId', 'EventCapacity']].copy()
    cap['EventCapacity'] = pd.to_numeric(cap['EventCapacity'], errors='coerce')
    actuals_2526 = actuals_2526.merge(cap, on='EventId', how='left')

    # Generate predictions
    fc = predict_model_a(actuals_2526, primary, f1, f2)
    fc = predict_model_b(fc, name_model)
    fc['Pred_A'] = cap_at_capacity(fc['Pred_A'], fc['EventCapacity'])
    fc['Pred_B'] = cap_at_capacity(fc['Pred_B'], fc['EventCapacity'])

    # ── Summary metrics ──────────────────────────────────────────────────
    print("=" * 70)
    print(f"25-26 FORECAST vs ACTUALS (n={len(fc[fc['Actual']>0])} completed events)")
    print("=" * 70)
    print_metrics("Model A: Class/Venue/Genre weighted avg", fc, 'Pred_A')
    print_metrics("Model B: Event name matching + fallback", fc, 'Pred_B')

    # ── Event-level detail ───────────────────────────────────────────────
    print("\nEVENT-LEVEL DETAIL")
    print(f"{'EventName':<50} {'Class':<15} {'Actual':>7} {'Pred_A':>7} {'Pred_B':>7} {'Err_A':>7} {'Err_B':>7}  Source")
    print("-" * 130)
    for _, row in fc.sort_values('EventName').iterrows():
        err_a = f"{(row['Pred_A'] - row['Actual']) / row['Actual'] * 100:+.0f}%" if pd.notna(row['Pred_A']) else "N/A"
        err_b = f"{(row['Pred_B'] - row['Actual']) / row['Actual'] * 100:+.0f}%" if pd.notna(row['Pred_B']) else "N/A"
        print(f"{str(row['EventName']):<50} {str(row['EventClass']):<15} "
              f"{row['Actual']:>7.0f} {row['Pred_A']:>7.0f} {row['Pred_B']:>7.0f} "
              f"{err_a:>7} {err_b:>7}  {row['Model_B_Source']}")

    # ── Name model coverage ──────────────────────────────────────────────
    print("\nNAME MODEL COVERAGE")
    has_history = fc['NameSeasons'] >= 2
    print(f"  Events with ≥2 seasons of name history: {has_history.sum()} / {len(fc)}")
    print(f"  Events using fallback (new events):      {(~has_history).sum()} / {len(fc)}")

    # Show what name history looks like for matched events
    matched = fc[has_history][['EventName', 'EventClass', 'NameSeasons', 'NameWeightedAvg', 'Actual', 'Pred_A']].copy()
    if not matched.empty:
        print("\n  Matched events — name history summary:")
        print(f"  {'EventName':<50} {'Seasons':>7} {'NameAvg':>8} {'Model_A':>8} {'Actual':>8}")
        for _, r in matched.iterrows():
            print(f"  {str(r['EventName']):<50} {r['NameSeasons']:>7.0f} {r['NameWeightedAvg']:>8.0f} {r['Pred_A']:>8.0f} {r['Actual']:>8.0f}")

    # ── Trend note ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ATTENDANCE TREND CONTEXT (from prior analysis)")
    print("  Mission:   -8.4%/season  R²=0.82  (strong decline)")
    print("  Standard:  -3.5%/season  R²=0.41  (moderate decline)")
    print("  Headliner: +0.7%/season  R²=0.01  (flat)")

    # ── Write output ─────────────────────────────────────────────────────
    out = fc[['EventName', 'EventClass', 'EventVenue', 'EventGenre', 'EventCapacity',
              'Actual', 'Pred_A', 'FallbackLevel', 'Pred_B', 'Model_B_Source',
              'NameSeasons', 'NameWeightedAvg']].copy()
    out['Error_A_Pct'] = ((out['Pred_A'] - out['Actual']) / out['Actual'] * 100).round(1)
    out['Error_B_Pct'] = ((out['Pred_B'] - out['Actual']) / out['Actual'] * 100).round(1)

    with pd.ExcelWriter("Forecast_2526_Comparison.xlsx", engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="2526_Comparison", index=False)
        name_model.sort_values('NameSeasons', ascending=False).to_excel(
            writer, sheet_name="Name_History", index=False)

    print(f"\n✅ Written to Forecast_2526_Comparison.xlsx")


if __name__ == "__main__":
    main()
