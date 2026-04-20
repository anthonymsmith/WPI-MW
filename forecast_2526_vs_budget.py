"""
25-26 Season: Actuals vs Budget vs Attendance Forecast

Sources:
  - '25-26 Sales as of 4.9.26.xlsx'  — revenue actuals, budget goals, ticket counts
  - anon_DataMerge.csv + EventManifest.xlsx — attendance model training data
  - forecast_artist_adjustment.py     — Bayesian artist adjustment layer

Output: Forecast_2526_vs_Budget.xlsx with two sheets:
  EventDetail  — one row per event, revenue + attendance comparison
  Summary      — season totals and key metrics
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

WORKING_DIR = '/Users/antho/Documents/WPI-MW'
SALES_FILE  = '25-26 Sales as of 4.9.26.xlsx'
FORECAST_SEASON = '25-26'

WEIGHTS = {'19-20': 1.0, '20-21': 0.3, '21-22': 0.7,
           '22-23': 2.0, '23-24': 3.0, '24-25': 3.0}


def assign_season_weight(season):
    if pd.isnull(season): return 1.0
    if season < '19-20': return 1.0
    return WEIGHTS.get(season, 1.0)


# ── Load sales actuals ─────────────────────────────────────────────────────────

def load_sales():
    df = pd.read_excel(SALES_FILE)
    df.columns = ['PerfDate', 'EventName', 'ActualRevenue', 'AdjustedRevenue',
                  'BudgetGoal', 'PctToBudget', 'ActualTix', 'AvgYield', 'Notes']
    df = df.iloc[1:].reset_index(drop=True)
    df = df[df['EventName'].notna() & (df['EventName'] != 'EVENT')].copy()
    df['PerfDate']        = pd.to_datetime(df['PerfDate'], errors='coerce')
    df['ActualRevenue']   = pd.to_numeric(df['ActualRevenue'],   errors='coerce')
    df['AdjustedRevenue'] = pd.to_numeric(df['AdjustedRevenue'], errors='coerce')
    df['BudgetGoal']      = pd.to_numeric(df['BudgetGoal'],      errors='coerce')
    df['ActualTix']       = pd.to_numeric(df['ActualTix'],       errors='coerce')
    df['AvgYield']        = pd.to_numeric(df['AvgYield'],        errors='coerce')
    return df


# ── Load forecast model data ───────────────────────────────────────────────────

def load_model_data():
    from forecast_2526_comparison import (
        load_data, get_training_df, build_hierarchy_models,
        build_name_model, predict_model_a, predict_model_b, cap_at_capacity
    )
    em, merged = load_data()
    all_prior = sorted([s for s in merged['Season'].dropna().unique()
                        if s < FORECAST_SEASON])
    filtered_train = get_training_df(merged, all_prior)
    repeat_model, primary, f1, f2, f3, f4, f5 = build_hierarchy_models(filtered_train)
    name_model = build_name_model(filtered_train)

    cap = em.drop_duplicates('EventId')[['EventId', 'EventCapacity']].copy()
    cap['EventCapacity'] = pd.to_numeric(cap['EventCapacity'], errors='coerce')

    # 25-26 events from manifest (all, not just completed)
    events_2526 = (
        em[em['Season'] == FORECAST_SEASON]
        .drop_duplicates('EventId')
        [['EventId', 'EventName', 'EventClass', 'EventVenue',
          'EventGenre', 'EventLoB', 'EventSubGenre']]
        .copy()
    )
    events_2526 = events_2526.merge(cap, on='EventId', how='left')

    fc = predict_model_a(events_2526, repeat_model, primary, f1, f2, f3, f4, f5)
    fc = predict_model_b(fc, name_model)
    fc['Pred_A'] = cap_at_capacity(fc['Pred_A'], fc['EventCapacity'])
    fc['Pred_B'] = cap_at_capacity(fc['Pred_B'], fc['EventCapacity'])

    # Artist adjustment (trained on all prior seasons)
    hist_actuals = (
        filtered_train
        .groupby(['EventId', 'EventName', 'EventClass', 'EventVenue',
                  'EventGenre', 'EventLoB', 'EventSubGenre'], group_keys=False)
        .agg(Actual=('Quantity', 'sum'))
        .reset_index()
    )
    hist_actuals = hist_actuals.merge(cap, on='EventId', how='left')
    from forecast_2526_comparison import predict_model_a as pma
    hist_fc = pma(hist_actuals, primary, f1, f2, f3, f4)

    from forecast_artist_adjustment import apply_artist_adjustment
    fc = apply_artist_adjustment(
        fc,
        merged_history=hist_fc,
        actuals_history=hist_fc['Actual'],
        bucket_preds_history=hist_fc['Pred_A'],
    )
    return fc[['EventName', 'EventClass', 'EventVenue', 'EventGenre',
               'EventLoB', 'EventSubGenre', 'EventCapacity',
               'Pred_A', 'Pred_Adj', 'Pred_Adj_Lo', 'Pred_Adj_Hi',
               'Adj_LogFactor', 'Adj_Source', 'FallbackLevel',
               'NameSeasons', 'Model_B_Source']]


# ── Fuzzy name match between sales file and manifest ──────────────────────────

def _norm(s):
    """Normalize name for matching: lowercase, strip, collapse separators."""
    return (str(s).lower().strip()
            .replace(' & ', ' ')
            .replace(',', '')
            .replace('  ', ' '))


# Events that are virtual/streaming — exclude from attendance accuracy
VIRTUAL_KEYWORDS = ['livestream', 'on demand', 'live stream']


def match_events(sales, forecast):
    """Left-join sales→forecast on EventName with fuzzy fallback."""
    sales = sales.copy()
    forecast = forecast.copy()
    sales['_key']    = sales['EventName'].apply(_norm)
    forecast['_key'] = forecast['EventName'].apply(_norm)

    # Exact match first
    merged = sales.merge(
        forecast.rename(columns={'EventName': 'Manifest_EventName'}),
        on='_key', how='left'
    )

    # For unmatched rows try substring match (skip virtual events)
    unmatched = merged['Manifest_EventName'].isna()
    for idx in merged[unmatched].index:
        sale_key = merged.at[idx, '_key']
        # Don't fuzzy-match livestream/on-demand events to real events
        if any(kw in sale_key for kw in VIRTUAL_KEYWORDS):
            continue
        for _, frow in forecast.iterrows():
            fk = frow['_key']
            if any(kw in fk for kw in VIRTUAL_KEYWORDS):
                continue
            if sale_key in fk or fk in sale_key:
                for col in forecast.columns:
                    if col != '_key':
                        merged.at[idx, col if col != 'EventName'
                                  else 'Manifest_EventName'] = frow[col]
                break

    merged.drop(columns=['_key'], inplace=True)

    # Flag virtual events so they're excluded from attendance accuracy
    merged['IsVirtual'] = merged['EventName'].str.lower().apply(
        lambda x: any(kw in x for kw in VIRTUAL_KEYWORDS)
    )
    return merged


# ── Build comparison ───────────────────────────────────────────────────────────

def build_comparison(sales, forecast):
    df = match_events(sales, forecast)

    # Clear attendance forecasts for virtual/streaming events
    is_virtual = df['IsVirtual'].fillna(False)
    for col in ['Pred_A', 'Pred_Adj', 'Pred_Adj_Lo', 'Pred_Adj_Hi', 'Adj_LogFactor', 'Adj_Source']:
        if col in df.columns:
            df.loc[is_virtual, col] = np.nan

    df['RevVsBudget_$']   = df['AdjustedRevenue'] - df['BudgetGoal']
    df['RevVsBudget_Pct'] = (df['RevVsBudget_$'] / df['BudgetGoal'] * 100).round(1)

    df['TixVsForecast_$']   = df['ActualTix'] - df['Pred_Adj']
    df['TixVsForecast_Pct'] = (df['TixVsForecast_$'] / df['Pred_Adj'] * 100).round(1)
    df['TixVsPredA_$']      = df['ActualTix'] - df['Pred_A']
    df['TixVsPredA_Pct']    = (df['TixVsPredA_$'] / df['Pred_A'] * 100).round(1)

    return df


# ── Print summary ──────────────────────────────────────────────────────────────

def print_summary(df):
    has_budget = df['BudgetGoal'].notna() & (df['BudgetGoal'] > 0)
    has_tix    = df['ActualTix'].notna() & (df['Pred_Adj'].notna())

    total_actual  = df['AdjustedRevenue'].sum()
    total_budget  = df.loc[has_budget, 'BudgetGoal'].sum()
    total_tix     = df['ActualTix'].sum()
    total_pred_a  = df.loc[df['Pred_A'].notna(), 'Pred_A'].sum()
    total_pred_adj = df.loc[df['Pred_Adj'].notna(), 'Pred_Adj'].sum()

    print('=' * 80)
    print(f'25-26 SEASON: ACTUALS vs BUDGET  ({len(df)} events)')
    print('=' * 80)
    print(f'  Total Revenue (adj):  ${total_actual:>10,.0f}')
    print(f'  Total Budget Goal:    ${total_budget:>10,.0f}   ({total_actual/total_budget*100:.1f}% to budget)')
    print(f'  Total Tickets:         {total_tix:>10,.0f}')
    print(f'  Forecast (Model A):    {total_pred_a:>10,.0f}')
    print(f'  Forecast (Adj):        {total_pred_adj:>10,.0f}')
    print()

    # Revenue detail
    print(f"{'EventName':<50} {'AdjRev':>9} {'Budget':>9} {'Var$':>8} {'Var%':>6}  {'Tix':>5} {'PredAdj':>7} {'TixVar%':>8}")
    print('-' * 110)
    for _, row in df.sort_values('PerfDate').iterrows():
        budg  = f"${row['BudgetGoal']:>8,.0f}" if pd.notna(row['BudgetGoal']) and row['BudgetGoal'] > 0 else '       n/a'
        var_d = f"${row['RevVsBudget_$']:>+7,.0f}" if pd.notna(row.get('RevVsBudget_$')) and pd.notna(row['BudgetGoal']) and row['BudgetGoal'] > 0 else '      n/a'
        var_p = f"{row['RevVsBudget_Pct']:>+5.1f}%" if pd.notna(row.get('RevVsBudget_Pct')) and pd.notna(row['BudgetGoal']) and row['BudgetGoal'] > 0 else '   n/a'
        pred  = f"{row['Pred_Adj']:>7.0f}" if pd.notna(row.get('Pred_Adj')) else '    n/a'
        tvp   = f"{row['TixVsForecast_Pct']:>+7.1f}%" if pd.notna(row.get('TixVsForecast_Pct')) else '     n/a'
        name  = str(row['EventName'])[:49]
        print(f"{name:<50} ${row['AdjustedRevenue']:>8,.0f} {budg} {var_d} {var_p}  {row['ActualTix']:>5.0f} {pred} {tvp}")

    print()

    # MAPE/WAPE for attendance forecast (non-virtual events where we have both)
    comp = df[has_tix & df['Pred_A'].notna() & ~df['IsVirtual']].copy()
    if not comp.empty:
        comp['AbsErr_A']   = (comp['ActualTix'] - comp['Pred_A']).abs()
        comp['AbsErr_Adj'] = (comp['ActualTix'] - comp['Pred_Adj']).abs()
        mape_a   = (comp['AbsErr_A']   / comp['ActualTix']).mean() * 100
        mape_adj = (comp['AbsErr_Adj'] / comp['ActualTix']).mean() * 100
        wape_a   = comp['AbsErr_A'].sum()   / comp['ActualTix'].sum() * 100
        wape_adj = comp['AbsErr_Adj'].sum() / comp['ActualTix'].sum() * 100
        bias_a   = ((comp['Pred_A']   - comp['ActualTix']) / comp['ActualTix']).mean() * 100
        bias_adj = ((comp['Pred_Adj'] - comp['ActualTix']) / comp['ActualTix']).mean() * 100
        n = len(comp)
        print(f"ATTENDANCE FORECAST ACCURACY  (n={n} events with model coverage)")
        print(f"  Model A:            MAPE={mape_a:.1f}%  WAPE={wape_a:.1f}%  Bias={bias_a:+.1f}%")
        print(f"  + Artist Bayes adj: MAPE={mape_adj:.1f}%  WAPE={wape_adj:.1f}%  Bias={bias_adj:+.1f}%")


# ── Write Excel ────────────────────────────────────────────────────────────────

def write_excel(df):
    detail_cols = [
        'PerfDate', 'EventName', 'EventClass', 'EventSubGenre',
        'ActualTix', 'Pred_Adj', 'Pred_A', 'TixVsForecast_Pct', 'TixVsPredA_Pct',
        'Pred_Adj_Lo', 'Pred_Adj_Hi', 'Adj_LogFactor', 'Adj_Source',
        'AdjustedRevenue', 'ActualRevenue', 'BudgetGoal',
        'RevVsBudget_$', 'RevVsBudget_Pct', 'AvgYield', 'Notes',
        'FallbackLevel', 'NameSeasons',
    ]
    detail_cols = [c for c in detail_cols if c in df.columns]
    detail = df[detail_cols].sort_values('PerfDate').copy()
    detail['PerfDate'] = detail['PerfDate'].dt.strftime('%Y-%m-%d')

    has_budget = df['BudgetGoal'].notna() & (df['BudgetGoal'] > 0)
    summary = pd.DataFrame([{
        'Metric': 'Total Adjusted Revenue',
        'Value': df['AdjustedRevenue'].sum(),
    }, {
        'Metric': 'Total Budget Goal',
        'Value': df.loc[has_budget, 'BudgetGoal'].sum(),
    }, {
        'Metric': 'Revenue vs Budget ($)',
        'Value': df['AdjustedRevenue'].sum() - df.loc[has_budget, 'BudgetGoal'].sum(),
    }, {
        'Metric': 'Revenue vs Budget (%)',
        'Value': round(df['AdjustedRevenue'].sum() /
                       df.loc[has_budget, 'BudgetGoal'].sum() * 100, 1),
    }, {
        'Metric': 'Total Tickets',
        'Value': df['ActualTix'].sum(),
    }, {
        'Metric': 'Forecast Attendance (Adj)',
        'Value': df['Pred_Adj'].sum(),
    }, {
        'Metric': 'Events in Sales File',
        'Value': len(df),
    }, {
        'Metric': 'Events with Forecast Coverage',
        'Value': df['Pred_A'].notna().sum(),
    }])

    out_file = 'Forecast_2526_vs_Budget.xlsx'
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        detail.to_excel(writer, sheet_name='EventDetail', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
    print(f'\n✅ Written to {out_file}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.chdir(WORKING_DIR)
    sales    = load_sales()
    forecast = load_model_data()
    df       = build_comparison(sales, forecast)
    print_summary(df)
    write_excel(df)


if __name__ == '__main__':
    main()
