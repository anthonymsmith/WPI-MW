"""
Honest hindcast evaluation — no data leakage.
Models are rebuilt excluding the target season before evaluating accuracy.
Reports MAPE/WAPE per season and bias by EventClass/EventGenre/EventVenue.
"""

import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

WORKING_DIR = '/Users/antho/Documents/WPI-MW'
INCLUDE_COMPS = True  # match Forecast-v2 setting

EVAL_SEASONS = ['22-23', '23-24', '24-25']


def assign_season_weight(season):
    if pd.isnull(season): return 1.0
    if season < "19-20": return 1.0
    if season == "19-20": return 1.0
    if season == "20-21": return 0.3
    if season == "21-22": return 0.7
    if season == "22-23": return 2.0
    if season == "23-24": return 3.0
    if season == "24-25": return 3.0
    return 1.0


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


def build_filtered_df(merged_history, exclude_season=None):
    """Build filtered training data, optionally excluding a season."""
    mh = merged_history.copy()

    if INCLUDE_COMPS:
        qty_mask = mh['Quantity'] > 0
        paid_mask = True
    else:
        qty_mask = True
        paid_mask = mh['TicketTotal'] > 0

    filtered = mh[
        (mh['EventType'] == 'Live') &
        (mh['EventStatus'] == 'Complete') &
        (mh['TicketStatus'] == 'Active') &
        qty_mask & (paid_mask if not isinstance(paid_mask, bool) else True)
    ].copy()

    if exclude_season:
        filtered = filtered[filtered['Season'] != exclude_season]

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
            WeightedTickets=('WeightedTickets', 'sum'),
            Weight=('Weight', 'first')
        )
        .reset_index()
    )


def build_forecast_models(event_attendance):
    weighted_grouped = (
        event_attendance
        .groupby(['EventClass', 'EventVenue', 'EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WeightedAttendance': (g['TotalTickets'] * g['Weight']).sum(),
            'TotalWeight': g['Weight'].sum()
        }))
        .reset_index()
    )
    weighted_grouped['WeightedAvgAttendance'] = (
        weighted_grouped['WeightedAttendance'] / weighted_grouped['TotalWeight']
    )

    fallback_1 = (
        event_attendance
        .groupby(['EventClass', 'EventVenue'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WeightedAttendance_f1': (g['TotalTickets'] * g['Weight']).sum(),
            'TotalWeight_f1': g['Weight'].sum()
        }))
        .reset_index()
    )
    fallback_1['WeightedAvgAttendance_f1'] = (
        fallback_1['WeightedAttendance_f1'] / fallback_1['TotalWeight_f1']
    )

    fallback_2 = (
        event_attendance
        .groupby(['EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WeightedAttendance_f2': (g['TotalTickets'] * g['Weight']).sum(),
            'TotalWeight_f2': g['Weight'].sum()
        }))
        .reset_index()
    )
    fallback_2['WeightedAvgAttendance_f2'] = (
        fallback_2['WeightedAttendance_f2'] / fallback_2['TotalWeight_f2']
    )

    return weighted_grouped, fallback_1, fallback_2


def forecast_season(event_manifest, weighted_grouped, fallback_1, fallback_2, season):
    """Generate per-EventId forecasts for a season."""
    forecast_df = event_manifest[
        (event_manifest['Season'] == season) &
        (event_manifest['EventType'].str.lower().str.strip() == 'live')
    ].drop_duplicates(subset='EventId').copy()

    for col in ['EventClass', 'EventVenue', 'EventGenre']:
        forecast_df[col] = forecast_df[col].astype(str).str.strip()
    forecast_df['EventCapacity'] = pd.to_numeric(forecast_df.get('EventCapacity'), errors='coerce')

    forecast_df = forecast_df.merge(
        weighted_grouped[['EventClass', 'EventVenue', 'EventGenre', 'WeightedAvgAttendance']],
        on=['EventClass', 'EventVenue', 'EventGenre'], how='left'
    ).merge(
        fallback_1[['EventClass', 'EventVenue', 'WeightedAvgAttendance_f1']],
        on=['EventClass', 'EventVenue'], how='left'
    ).merge(
        fallback_2[['EventGenre', 'WeightedAvgAttendance_f2']],
        on='EventGenre', how='left'
    )

    forecast_df['FallbackLevel'] = 'Primary'
    no_primary = forecast_df['WeightedAvgAttendance'].isna()
    no_f1 = forecast_df['WeightedAvgAttendance_f1'].isna()
    forecast_df.loc[no_primary & ~no_f1, 'FallbackLevel'] = 'Fallback1'
    forecast_df.loc[no_primary & no_f1, 'FallbackLevel'] = 'Fallback2'

    forecast_df['WeightedAvgAttendance'] = (
        forecast_df['WeightedAvgAttendance']
        .combine_first(forecast_df['WeightedAvgAttendance_f1'])
        .combine_first(forecast_df['WeightedAvgAttendance_f2'])
    )

    forecast_df['PredictedAttendance'] = forecast_df.apply(
        lambda row: min(row['WeightedAvgAttendance'], row['EventCapacity'])
        if pd.notnull(row['WeightedAvgAttendance']) and pd.notnull(row['EventCapacity'])
        else row['WeightedAvgAttendance'],
        axis=1
    )

    return forecast_df[['EventId', 'EventName', 'EventVenue', 'EventGenre', 'EventClass',
                         'EventCapacity', 'PredictedAttendance', 'FallbackLevel']]


def get_actuals(merged_history, season):
    """Actual attendance per EventId for a season."""
    mh = merged_history.copy()

    if INCLUDE_COMPS:
        mask = mh['Quantity'] > 0
    else:
        mask = mh['TicketTotal'] > 0

    return (
        mh[
            (mh['Season'] == season) &
            (mh['EventType'] == 'Live') &
            (mh['EventStatus'] == 'Complete') &
            (mh['TicketStatus'] == 'Active') &
            mask
        ]
        .groupby(['EventId', 'EventName', 'EventVenue', 'EventGenre', 'EventClass'],
                 group_keys=False)
        .agg(ActualAttendance=('Quantity', 'sum'))
        .reset_index()
    )


def compute_metrics(comparison):
    valid = comparison[comparison['ActualAttendance'] > 0].copy()
    valid['AbsoluteError'] = (valid['ActualAttendance'] - valid['PredictedAttendance']).abs()
    valid['SignedError'] = valid['PredictedAttendance'] - valid['ActualAttendance']
    valid['PercentError'] = valid['AbsoluteError'] / valid['ActualAttendance']
    valid['SignedPctError'] = valid['SignedError'] / valid['ActualAttendance']

    mape = valid['PercentError'].mean() * 100
    wape = valid['AbsoluteError'].sum() / valid['ActualAttendance'].sum() * 100
    bias = valid['SignedPctError'].mean() * 100  # positive = over-forecasting
    return mape, wape, bias, valid


def bias_breakdown(valid, groupby_col):
    """Signed mean percent error by a dimension."""
    return (
        valid.groupby(groupby_col)
        .apply(lambda g: pd.Series({
            'Events': len(g),
            'TotalActual': g['ActualAttendance'].sum(),
            'MeanSignedPctError': g['SignedPctError'].mean() * 100,
            'WAPE': g['AbsoluteError'].sum() / g['ActualAttendance'].sum() * 100
        }))
        .reset_index()
        .sort_values('MeanSignedPctError')
    )


def main():
    event_manifest, merged_history = load_data()

    all_comparisons = []
    season_metrics = []

    print("=" * 60)
    print("HONEST HINDCAST — models exclude target season")
    print(f"INCLUDE_COMPS = {INCLUDE_COMPS}")
    print("=" * 60)

    for season in EVAL_SEASONS:
        # Build models WITHOUT the target season
        filtered_train = build_filtered_df(merged_history, exclude_season=season)
        event_attendance = compute_event_attendance(filtered_train)
        wg, f1, f2 = build_forecast_models(event_attendance)

        # Forecast target season
        preds = forecast_season(event_manifest, wg, f1, f2, season)
        actuals = get_actuals(merged_history, season)

        comparison = preds.merge(actuals[['EventId', 'ActualAttendance']], on='EventId', how='outer')
        comparison['Season'] = season

        mape, wape, bias, valid = compute_metrics(comparison)
        season_metrics.append({'Season': season, 'Events': len(valid),
                                'MAPE': mape, 'WAPE': wape, 'Bias': bias})
        all_comparisons.append(valid)

        print(f"\n--- {season} ---")
        print(f"  Events evaluated: {len(valid)}")
        print(f"  MAPE:  {mape:.1f}%")
        print(f"  WAPE:  {wape:.1f}%")
        print(f"  Bias:  {bias:+.1f}%  ({'over' if bias > 0 else 'under'}-forecasting)")
        print(f"  Fallback usage: {valid['FallbackLevel'].value_counts().to_dict()}")

    # Combined across all seasons
    combined = pd.concat(all_comparisons, ignore_index=True)
    overall_mape = combined['PercentError'].mean() * 100
    overall_wape = combined['AbsoluteError'].sum() / combined['ActualAttendance'].sum() * 100
    overall_bias = combined['SignedPctError'].mean() * 100

    print(f"\n{'=' * 60}")
    print(f"OVERALL ({', '.join(EVAL_SEASONS)})")
    print(f"  MAPE:  {overall_mape:.1f}%")
    print(f"  WAPE:  {overall_wape:.1f}%")
    print(f"  Bias:  {overall_bias:+.1f}%")

    # Bias breakdowns
    print(f"\n{'=' * 60}")
    print("BIAS BY EVENT CLASS (MeanSignedPctError: + = over, - = under)")
    print(bias_breakdown(combined, 'EventClass').to_string(index=False))

    print(f"\nBIAS BY GENRE")
    print(bias_breakdown(combined, 'EventGenre').to_string(index=False))

    print(f"\nBIAS BY VENUE")
    print(bias_breakdown(combined, 'EventVenue').to_string(index=False))

    # Write Excel output
    output_file = "Forecast_Honest_Eval.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        combined[['Season', 'EventId', 'EventName', 'EventClass', 'EventGenre', 'EventVenue',
                   'EventCapacity', 'PredictedAttendance', 'ActualAttendance',
                   'AbsoluteError', 'SignedError', 'PercentError', 'SignedPctError',
                   'FallbackLevel']].sort_values(['Season', 'EventName']).to_excel(
            writer, sheet_name="All_Seasons", index=False)

        pd.DataFrame(season_metrics).to_excel(writer, sheet_name="Season_Summary", index=False)

        bias_breakdown(combined, 'EventClass').to_excel(writer, sheet_name="Bias_EventClass", index=False)
        bias_breakdown(combined, 'EventGenre').to_excel(writer, sheet_name="Bias_Genre", index=False)
        bias_breakdown(combined, 'EventVenue').to_excel(writer, sheet_name="Bias_Venue", index=False)

    print(f"\n✅ Results written to {output_file}")


if __name__ == "__main__":
    main()
