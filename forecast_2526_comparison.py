"""
Compare forecasting approaches for 25-26 season to date:
  Model A: Current hierarchy — (Class,Venue,LoB,SubGenre) → (Class,Venue,SubGenre)
           → (Class,Venue,Genre) → (Class,Venue) → (SubGenre)
  Model B: Event name matching — use event's own history first, fall back to Model A

Training: all seasons through 24-25 (temporal holdout for 25-26).
"""

import pandas as pd
import numpy as np
import os
import warnings
from forecast_artist_adjustment import apply_artist_adjustment, print_adjustment_summary

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

    # Merge from manifest: EventLoB only (EventSubGenre stays from CSV to avoid collision)
    manifest_meta = em.drop_duplicates(subset='EventId')[
        ['EventId', 'EventName', 'EventStatus', 'EventType', 'EventClass',
         'EventVenue', 'EventGenre', 'EventLoB', 'EventCapacity']
    ].copy()
    for c in ['EventStatus', 'EventType']:
        manifest_meta[c] = manifest_meta[c].astype(str).str.strip().str.title()

    merged = df.merge(manifest_meta, on='EventId', how='left', suffixes=('', '_m'))
    for c in ['EventStatus', 'EventType', 'EventClass', 'EventVenue', 'EventGenre', 'EventLoB']:
        col_m = c + '_m'
        if col_m in merged.columns:
            merged[c] = merged[c].combine_first(merged[col_m])
            merged.drop(columns=[col_m], inplace=True, errors='ignore')
    for c in ['EventType', 'EventStatus', 'TicketStatus']:
        if c in merged.columns:
            merged[c] = merged[c].astype(str).str.strip().str.title()

    merged['EventCapacity'] = pd.to_numeric(merged.get('EventCapacity'), errors='coerce')
    merged['EventLoB'] = merged['EventLoB'].fillna('Concert')
    if 'EventSubGenre' not in merged.columns:
        merged['EventSubGenre'] = np.nan

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
    return filtered


def build_hierarchy_models(filtered):
    """5-level hierarchy: (Class,Venue,LoB,SubGenre) → (Class,Venue,SubGenre)
       → (Class,Venue,Genre) → (Class,Venue) → (SubGenre)"""
    ea = (
        filtered
        .groupby(['EventId', 'EventName', 'EventClass', 'EventVenue',
                  'EventGenre', 'EventLoB', 'EventSubGenre', 'Season'],
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
        ea.groupby(['EventClass', 'EventVenue', 'EventLoB', 'EventSubGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WA_p': wg_avg(g)})).reset_index()
    )
    f1 = (
        ea.groupby(['EventClass', 'EventVenue', 'EventSubGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WA_f1': wg_avg(g)})).reset_index()
    )
    f2 = (
        ea.groupby(['EventClass', 'EventVenue', 'EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WA_f2': wg_avg(g)})).reset_index()
    )
    f3 = (
        ea.groupby(['EventClass', 'EventVenue'], group_keys=False)
        .apply(lambda g: pd.Series({'WA_f3': wg_avg(g)})).reset_index()
    )
    f4 = (
        ea.groupby(['EventSubGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WA_f4': wg_avg(g)})).reset_index()
    )
    # F5 — ultimate fallback: EventClass mean. Guarantees no NaN for
    # one-off subgenre × new-venue combinations (e.g. Fusion @ JMAC).
    f5 = (
        ea.groupby(['EventClass'], group_keys=False)
        .apply(lambda g: pd.Series({'WA_f5': wg_avg(g)})).reset_index()
    )
    return primary, f1, f2, f3, f4, f5


def build_name_model(filtered):
    """Weighted avg attendance per EventName."""
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


def predict_model_a(events_df, primary, f1, f2, f3, f4, f5=None):
    fc = events_df.copy()
    for col in ['EventClass', 'EventVenue', 'EventGenre', 'EventLoB', 'EventSubGenre']:
        fc[col] = fc[col].astype(str).str.strip()

    fc = fc.merge(primary, on=['EventClass', 'EventVenue', 'EventLoB', 'EventSubGenre'], how='left')
    fc = fc.merge(f1, on=['EventClass', 'EventVenue', 'EventSubGenre'], how='left')
    fc = fc.merge(f2, on=['EventClass', 'EventVenue', 'EventGenre'], how='left')
    fc = fc.merge(f3, on=['EventClass', 'EventVenue'], how='left')
    fc = fc.merge(f4, on='EventSubGenre', how='left')
    if f5 is not None:
        fc = fc.merge(f5, on='EventClass', how='left')

    preds = (
        fc['WA_p']
        .combine_first(fc['WA_f1'])
        .combine_first(fc['WA_f2'])
        .combine_first(fc['WA_f3'])
        .combine_first(fc['WA_f4'])
    )
    if f5 is not None:
        preds = preds.combine_first(fc['WA_f5'])
    fc['Pred_A'] = preds

    conditions = [
        fc['WA_p'].notna(),
        fc['WA_p'].isna() & fc['WA_f1'].notna(),
        fc['WA_p'].isna() & fc['WA_f1'].isna() & fc['WA_f2'].notna(),
        fc['WA_p'].isna() & fc['WA_f1'].isna() & fc['WA_f2'].isna() & fc['WA_f3'].notna(),
        fc['WA_p'].isna() & fc['WA_f1'].isna() & fc['WA_f2'].isna() & fc['WA_f3'].isna() & fc['WA_f4'].notna(),
    ]
    choices = ['Primary (Class+Venue+LoB+SubGenre)',
               'F1 (Class+Venue+SubGenre)',
               'F2 (Class+Venue+Genre)',
               'F3 (Class+Venue)',
               'F4 (SubGenre)']
    fc['FallbackLevel'] = np.select(conditions, choices, default='F5 (EventClass)')
    return fc


def predict_model_b(events_df_with_a, name_model):
    """Model B: use event name history (≥2 seasons); otherwise fall back to Model A."""
    fc = events_df_with_a.copy()
    fc = fc.merge(name_model, on='EventName', how='left')

    has_name_history = fc['NameSeasons'] >= 2
    fc['Pred_B'] = np.where(has_name_history, fc['NameWeightedAvg'], fc['Pred_A'])
    fc['Model_B_Source'] = np.where(
        has_name_history,
        fc['NameSeasons'].apply(lambda n: f'Name ({int(n) if pd.notna(n) else 0} seasons)'),
        'Fallback → Model A'
    )
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
    print(f"  {label:<40} MAPE={mape:.1f}%  WAPE={wape:.1f}%  Bias={bias:+.1f}%  (n={len(valid)})")


def main():
    em, merged = load_data()

    # Training: everything before 25-26
    all_prior = sorted([s for s in merged['Season'].dropna().unique() if s < FORECAST_SEASON])
    filtered_train = get_training_df(merged, all_prior)

    # Build models
    primary, f1, f2, f3, f4, f5 = build_hierarchy_models(filtered_train)
    name_model = build_name_model(filtered_train)

    # Get 25-26 completed events with actuals
    actuals_2526 = (
        merged[
            (merged['Season'] == FORECAST_SEASON) &
            (merged['EventType'] == 'Live') &
            (merged['EventStatus'] == 'Complete') &
            (merged['TicketStatus'] == 'Active') &
            (merged['Quantity'] > 0)
        ]
        .groupby(['EventId', 'EventName', 'EventClass', 'EventVenue',
                  'EventGenre', 'EventLoB', 'EventSubGenre'], group_keys=False)
        .agg(Actual=('Quantity', 'sum'))
        .reset_index()
    )

    # Capacity from manifest
    cap = em.drop_duplicates('EventId')[['EventId', 'EventCapacity']].copy()
    cap['EventCapacity'] = pd.to_numeric(cap['EventCapacity'], errors='coerce')
    actuals_2526 = actuals_2526.merge(cap, on='EventId', how='left')

    # Generate predictions
    fc = predict_model_a(actuals_2526, primary, f1, f2, f3, f4, f5)
    fc = predict_model_b(fc, name_model)
    fc['Pred_A'] = cap_at_capacity(fc['Pred_A'], fc['EventCapacity'])
    fc['Pred_B'] = cap_at_capacity(fc['Pred_B'], fc['EventCapacity'])

    # Build labelled history for artist adjustment training
    # Event-level actuals from training seasons
    hist_actuals = (
        filtered_train
        .groupby(['EventId', 'EventName', 'EventClass', 'EventVenue',
                  'EventGenre', 'EventLoB', 'EventSubGenre'], group_keys=False)
        .agg(Actual=('Quantity', 'sum'))
        .reset_index()
    )
    hist_actuals = hist_actuals.merge(cap, on='EventId', how='left')
    hist_fc = predict_model_a(hist_actuals, primary, f1, f2, f3, f4, f5)

    fc = apply_artist_adjustment(
        fc,
        merged_history=hist_fc,
        actuals_history=hist_fc['Actual'],
        bucket_preds_history=hist_fc['Pred_A'],
    )
    print_adjustment_summary(fc)

    has_history = fc['NameSeasons'] >= 2
    n_total = len(fc)
    n_history = has_history.sum()

    # ── Summary metrics ──────────────────────────────────────────────────
    print("=" * 75)
    print(f"25-26 FORECAST vs ACTUALS  (n={n_total} completed events)")
    print("=" * 75)
    print_metrics("Model A: Hierarchy (all events)", fc, 'Pred_A')
    print_metrics("Model B: Name history + fallback (all events)", fc, 'Pred_B')
    print_metrics("Model A + Artist Bayesian adj (all events)", fc, 'Pred_Adj')
    print()
    print_metrics(f"Model A: Hierarchy  (prior-season events only, n={n_history})",
                  fc[has_history], 'Pred_A')
    print_metrics(f"Model B: Name history  (prior-season events only, n={n_history})",
                  fc[has_history], 'Pred_B')

    # ── Event-level detail ───────────────────────────────────────────────
    print("\nEVENT-LEVEL DETAIL")
    hdr = f"{'EventName':<45} {'Class':<10} {'SubGenre':<18} {'Actual':>7} {'Pred_A':>7} {'Pred_Adj':>8} {'Err_A':>7} {'Err_Adj':>8}  AdjSource"
    print(hdr)
    print("-" * len(hdr))
    for _, row in fc.sort_values(['EventClass', 'EventName']).iterrows():
        err_a   = f"{(row['Pred_A']   - row['Actual']) / row['Actual'] * 100:+.0f}%" if pd.notna(row['Pred_A']) else "N/A"
        err_adj = f"{(row['Pred_Adj'] - row['Actual']) / row['Actual'] * 100:+.0f}%" if pd.notna(row.get('Pred_Adj')) else "N/A"
        marker = "★" if row['NameSeasons'] >= 2 else " "
        print(f"{marker}{str(row['EventName']):<44} {str(row['EventClass']):<10} "
              f"{str(row['EventSubGenre']):<18} "
              f"{row['Actual']:>7.0f} {row['Pred_A']:>7.0f} {row.get('Pred_Adj', float('nan')):>8.0f} "
              f"{err_a:>7} {err_adj:>8}  {row.get('Adj_Source', '')}")
    print("  ★ = event has ≥2 seasons of prior name history")

    # ── Name model coverage ──────────────────────────────────────────────
    print(f"\nNAME MODEL COVERAGE")
    print(f"  Events with ≥2 seasons of name history: {n_history} / {n_total}")
    print(f"  New events (fallback to Model A):        {n_total - n_history} / {n_total}")

    # ── Write output ─────────────────────────────────────────────────────
    out = fc[['EventName', 'EventClass', 'EventVenue', 'EventGenre', 'EventLoB', 'EventSubGenre',
              'EventCapacity', 'Actual', 'Pred_A', 'FallbackLevel',
              'Pred_Adj', 'Pred_Adj_Lo', 'Pred_Adj_Hi', 'Adj_LogFactor', 'Adj_Source',
              'Pred_B', 'Model_B_Source', 'NameSeasons', 'NameWeightedAvg']].copy()
    out['Error_A_Pct']   = ((out['Pred_A']   - out['Actual']) / out['Actual'] * 100).round(1)
    out['Error_Adj_Pct'] = ((out['Pred_Adj'] - out['Actual']) / out['Actual'] * 100).round(1)
    out['Error_B_Pct']   = ((out['Pred_B']   - out['Actual']) / out['Actual'] * 100).round(1)

    with pd.ExcelWriter("Forecast_2526_Comparison.xlsx", engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="2526_Comparison", index=False)
        name_model.sort_values('NameSeasons', ascending=False).to_excel(
            writer, sheet_name="Name_History", index=False)

    print(f"\n✅ Written to Forecast_2526_Comparison.xlsx")


if __name__ == "__main__":
    main()
