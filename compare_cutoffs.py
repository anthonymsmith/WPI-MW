#!/usr/bin/env python3
"""
compare_cutoffs.py

Runs the MWSalesSumm pipeline at multiple data-age cutoffs and produces a
comparison report showing how key patron metrics and segments shift as older
data is removed.

Cutoffs tested: 5, 7, 9, 11, 13 years (15 = current full history, used as baseline)

Usage:
    python compare_cutoffs.py

Output:
    cutoff_comparison.xlsx  — one sheet per cutoff + summary sheet
"""

import os
import shutil
import logging
import sys
import warnings
from time import perf_counter

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
default_data_dir = '/Users/antho/Documents/WPI-MW'
data_dir = input(f'Enter data directory [{default_data_dir}]: ').strip() or default_data_dir
os.chdir(data_dir)

import MW_functions as mw
import Model_functions as mod

MANIFEST_FILE = 'EventManifest.xlsx'
PNL_FILE      = 'Budget/EventPnL.xlsx'
SALES_FILE    = 'SalesforceLatest.csv'
CHORUS_FILE   = 'Worcester Chorus current members.xlsx'
REGIONS_FILE  = 'regions_computed.csv'
BOARD_FILE    = 'BoardCorporators.csv'
PNL_OUT       = 'Budget/EventPnLProcessed.csv'

OUTPUT_EXCEL  = 'cutoff_comparison.xlsx'

# Cutoffs to test — 15 is the baseline (current full history)
CUTOFFS = [5, 7, 9, 11, 13, 15]

# Key metrics to compare
METRIC_COLS = [
    'Segment', 'RFMScore', 'GrowthScore', 'Regularity',
    'AYM', 'Lifespan', 'Frequency', 'FullPriceRate',
]

NEW_THRESHOLD       = 250
REENGAGED_THRESHOLD = 2.5
RFM_THRESHOLD       = 0

# ---------------------------------------------------------------------------
# Logging — minimal output (just progress)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
logger = logging.getLogger('compare_cutoffs')
logger.setLevel(logging.WARNING)

# Suppress MW/Model logger noise during batch runs
for name in ('MW_functions', 'Model_functions', '__main__', 'root'):
    logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(years):
    """Run the full pipeline for a given yearsOfData cutoff. Returns patron DataFrame."""
    print(f'  Running {years}-year cutoff...', end=' ', flush=True)
    t = perf_counter()

    # get_patron_details reads patrons_file to carry forward cached lat/long.
    # Pre-seed the temp file from the real Patrons.csv so that read doesn't fail.
    tmp_patrons = f'_tmp_patrons_{years}yr.csv'
    if os.path.exists('Patrons.csv') and not os.path.exists(tmp_patrons):
        shutil.copy('Patrons.csv', tmp_patrons)

    event_df        = mw.load_event_manifest(MANIFEST_FILE, logger)
    event_df        = mw.add_PnL_data(event_df, PNL_FILE, PNL_OUT, logger)
    sales_df        = mw.load_sales_file(SALES_FILE, years, logger)
    sales_df        = mw.sales_initial_prep(sales_df, logger)
    sales_df        = mw.venue_and_attribute_processing(sales_df, CHORUS_FILE, BOARD_FILE, logger)
    merged_df       = mw.combine_sales_and_events(sales_df, event_df, logger)
    merged_df       = mw.event_counts(merged_df, logger, event_column='EventGenre')
    merged_df       = mw.event_counts(merged_df, logger, event_column='EventClass')
    merged_df       = mw.event_counts(merged_df, logger, event_column='EventVenue')
    merged_df       = mw.final_processing_and_output(
                          merged_df, f'_tmp_datamerge_{years}yr.csv', logger,
                          processDonations=True)

    rates_df        = mw.calculate_retention_and_churn(merged_df, logger)

    patrons_df      = mw.get_patron_details(
                          merged_df, rates_df,
                          RFM_THRESHOLD,
                          False,              # GetLatLong=False — skip geocoding for speed
                          REGIONS_FILE,
                          tmp_patrons,
                          False,              # anonymized=False
                          NEW_THRESHOLD,
                          REENGAGED_THRESHOLD,
                          logger)

    print(f'{perf_counter() - t:.0f}s  ({len(patrons_df):,} patrons)')
    return patrons_df


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
SEGMENT_ORDER = [
    'Best', 'Upsell', 'Regular', 'Re-engaged',
    'Come Again', 'New', 'Lapsed', 'Fully Lapsed',
]

def segment_rank(s):
    try:
        return SEGMENT_ORDER.index(s)
    except ValueError:
        return len(SEGMENT_ORDER)


def compare_to_baseline(baseline: pd.DataFrame, other: pd.DataFrame, years: int) -> pd.DataFrame:
    """
    Join other cutoff to baseline on AccountId.
    Return per-patron diff rows for patrons present in both.
    """
    b = baseline[['AccountId', 'AccountName'] + METRIC_COLS].copy()
    o = other[['AccountId'] + METRIC_COLS].copy()

    merged = b.merge(o, on='AccountId', suffixes=('_base', f'_{years}yr'))

    rows = []
    for _, row in merged.iterrows():
        r = {'AccountId': row['AccountId'], 'AccountName': row['AccountName']}
        seg_base = row['Segment_base']
        seg_cut  = row[f'Segment_{years}yr']
        r['Segment_base']    = seg_base
        r[f'Segment_{years}yr'] = seg_cut
        r['SegmentChanged']  = seg_base != seg_cut
        r['SegmentDrift']    = segment_rank(seg_cut) - segment_rank(seg_base)
        for col in METRIC_COLS:
            if col == 'Segment':
                continue
            base_val = row[f'{col}_base']
            cut_val  = row[f'{col}_{years}yr']
            r[f'{col}_base']     = base_val
            r[f'{col}_{years}yr'] = cut_val
            if pd.notna(base_val) and pd.notna(cut_val) and base_val != 0:
                r[f'{col}_pct_chg'] = (cut_val - base_val) / abs(base_val) * 100
            else:
                r[f'{col}_pct_chg'] = np.nan
        rows.append(r)

    return pd.DataFrame(rows)


def summary_stats(diffs: pd.DataFrame, years: int, n_base: int, n_cut: int) -> dict:
    """Roll up key statistics from a diff DataFrame."""
    present = len(diffs)
    dropped = n_base - present    # patrons in baseline not in this cutoff (lost all records)

    seg_changed     = diffs['SegmentChanged'].sum()
    downgraded      = (diffs['SegmentDrift'] > 0).sum()   # drift > 0 = worse segment
    upgraded        = (diffs['SegmentDrift'] < 0).sum()   # drift < 0 = better segment

    stats = {
        'Cutoff (years)':          years,
        'Patrons in baseline':     n_base,
        'Patrons at cutoff':       n_cut,
        'Patrons lost (no data)':  dropped,
        'Segment changed':         seg_changed,
        'Segment changed %':       round(seg_changed / present * 100, 1),
        'Downgraded':              downgraded,
        'Upgraded':                upgraded,
    }

    for col in METRIC_COLS:
        if col == 'Segment':
            continue
        chg = diffs[f'{col}_pct_chg'].dropna()
        stats[f'{col} median % chg'] = round(chg.median(), 1) if len(chg) else np.nan
        stats[f'{col} >10% chg']     = int((chg.abs() > 10).sum())
        stats[f'{col} >25% chg']     = int((chg.abs() > 25).sum())

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f'\nRunning pipeline at {len(CUTOFFS)} cutoffs: {CUTOFFS} years')
    print('Geocoding disabled for speed. This will take a few minutes per cutoff.\n')

    results = {}
    for years in CUTOFFS:
        results[years] = run_pipeline(years)

    baseline = results[15]
    n_base   = len(baseline)

    print('\nBuilding comparison...')

    all_summaries = []
    all_diffs     = {}

    for years in [c for c in CUTOFFS if c != 15]:
        diffs = compare_to_baseline(baseline, results[years], years)
        all_diffs[years] = diffs
        stats = summary_stats(diffs, years, n_base, len(results[years]))
        all_summaries.append(stats)

    summary_df = pd.DataFrame(all_summaries)

    # ---------------------------------------------------------------------------
    # Write Excel output
    # ---------------------------------------------------------------------------
    print(f'\nWriting {OUTPUT_EXCEL}...')
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='xlsxwriter') as writer:
        book = writer.book

        # Summary sheet
        summary_df.T.to_excel(writer, sheet_name='Summary')
        ws = writer.sheets['Summary']
        ws.set_column(0, 0, 32)
        ws.set_column(1, len(CUTOFFS), 14)
        ws.set_tab_color('#1F497D')

        # Segment cross-tab per cutoff
        for years in [c for c in CUTOFFS if c != 15]:
            diffs = all_diffs[years]
            tab_name = f'{years}yr vs 15yr'

            # Segment crosstab
            ct = pd.crosstab(
                diffs['Segment_base'],
                diffs[f'Segment_{years}yr'],
                rownames=['Baseline (15yr)'],
                colnames=[f'Cutoff ({years}yr)'],
            )
            # Reindex to consistent segment order
            present_segs = [s for s in SEGMENT_ORDER if s in ct.index or s in ct.columns]
            ct = ct.reindex(index=[s for s in present_segs if s in ct.index],
                            columns=[s for s in present_segs if s in ct.columns],
                            fill_value=0)

            ct.to_excel(writer, sheet_name=tab_name, startrow=0)

            # Metric shift percentiles below the crosstab
            row_offset = len(ct) + 3
            shift_rows = []
            for col in [c for c in METRIC_COLS if c != 'Segment']:
                chg = diffs[f'{col}_pct_chg'].dropna()
                if len(chg):
                    shift_rows.append({
                        'Metric':      col,
                        'Median % chg': round(chg.median(), 2),
                        'P10 % chg':   round(chg.quantile(0.10), 2),
                        'P90 % chg':   round(chg.quantile(0.90), 2),
                        '# > 10% chg': int((chg.abs() > 10).sum()),
                        '# > 25% chg': int((chg.abs() > 25).sum()),
                    })
            pd.DataFrame(shift_rows).to_excel(
                writer, sheet_name=tab_name,
                startrow=row_offset, index=False
            )

            ws = writer.sheets[tab_name]
            ws.set_column(0, 0, 22)
            ws.set_column(1, 10, 14)

            # Highlight diagonal (unchanged segment) in crosstab
            green_fmt  = book.add_format({'bg_color': '#C6EFCE'})
            yellow_fmt = book.add_format({'bg_color': '#FFEB9C'})
            # (xlsxwriter doesn't do conditional format on specific cells easily,
            #  so just format the header)
            hdr_fmt = book.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
            ws.set_row(0, 20, hdr_fmt)

        # Patron-level detail for most aggressive cutoff (5yr)
        if 5 in all_diffs:
            detail = all_diffs[5][
                all_diffs[5]['SegmentChanged'] == True
            ].sort_values('SegmentDrift', ascending=False)[[
                'AccountName', 'Segment_base', 'Segment_5yr', 'SegmentDrift',
                'RFMScore_base', 'RFMScore_5yr', 'RFMScore_pct_chg',
                'GrowthScore_base', 'GrowthScore_5yr', 'GrowthScore_pct_chg',
                'Regularity_base', 'Regularity_5yr', 'Regularity_pct_chg',
                'AYM_base', 'AYM_5yr', 'AYM_pct_chg',
                'Lifespan_base', 'Lifespan_5yr', 'Lifespan_pct_chg',
            ]]
            detail.to_excel(writer, sheet_name='5yr Segment Changes', index=False)
            ws = writer.sheets['5yr Segment Changes']
            ws.set_column(0, 0, 30)
            ws.set_column(1, 20, 14)

    print(f'\nDone. Results written to {OUTPUT_EXCEL}')
    print('\n=== Quick Summary ===')
    print(summary_df[['Cutoff (years)', 'Patrons lost (no data)',
                       'Segment changed', 'Segment changed %']].to_string(index=False))

    # Cleanup temp files
    import glob
    for f in glob.glob('_tmp_patrons_*.csv') + glob.glob('_tmp_datamerge_*.csv'):
        try:
            os.remove(f)
        except OSError:
            pass


if __name__ == '__main__':
    main()
