"""
Copyright (c) Nolichucky Associates 2024. All Rights Reserved.

This software is the confidential and proprietary information of Nolichucky Associates.
You shall not disclose such Confidential Information and shall use it only in accordance
 with the terms of the license agreement you entered into with Nolichucky Associates.

Unauthorized copying of this file, via any medium, is strictly prohibited.
Proprietary and confidential.

Project: Music Worcester Patron and Event Analytics

Author: Anthony Smith
Date: 2025

Merges patron analytics (Patrons.csv) with donor history (DonationsLatest.xlsx),
then classifies patrons into five mutually exclusive donor prospect tranches.

Inputs:
    Patrons.csv          - output of MWSalesSumm.ipynb
    DonationsLatest.xlsx - Salesforce donation export

Output:
    Patron_Classification.xlsx - one sheet per tranche, plus unmatched donor review sheet
"""

import difflib
import logging
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
default_data_dir = '/Users/antho/Documents/WPI-MW'
data_dir = input(f'Enter data directory [{default_data_dir}]: ').strip() or default_data_dir
os.chdir(data_dir)

patrons_file  = 'Patrons.csv'
donor_file    = 'DonationsLatest.xlsx'
output_file   = 'Patron_Classification.xlsx'
anon_file     = 'Patron_Classification_Anon.xlsx'
review_file   = 'DonorNameMatchReview.xlsx'

# Tranche thresholds — adjust here without touching logic below
MAJOR_DONOR_AYM        = 1500  # Average Yearly Monetary threshold for Major Donors
MAJOR_DONOR_LIFETIME   = 5000  # Lifetime donation threshold for Major Donors
ACTIVE_DONATION_MONTHS = 18    # Months since last donation to be considered "active"
PRIME_RECENCY_MONTHS   = 12    # Ticket recency threshold for Prime Non-Donor Prospects
GROWTH_RECENCY_MONTHS  = 18    # Ticket recency threshold for Growth Prospects
MIN_GROWTH_SCORE       = 0.0   # Minimum GrowthScore for Growth Prospects
MIN_REGULARITY         = 0.3   # Minimum Regularity for Growth Prospects

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler(os.path.join(data_dir, 'donor_classifier.log'), mode='w')
    f_handler.setFormatter(logging.Formatter('%(asctime)s - %(lineno)s - %(levelname)s - %(message)s'))
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
logger.info('Loading patron data...')
patrons_df = pd.read_csv(patrons_file, low_memory=False)
logger.info('%s patrons loaded from %s', f'{len(patrons_df):,}', patrons_file)

logger.info('Loading donor data...')
donor_df = pd.read_excel(donor_file, parse_dates=['Close Date'])
donor_df['Last Donation Date'] = pd.to_datetime(donor_df['Last Donation Date'], errors='coerce')
donor_df['Close Date']         = pd.to_datetime(donor_df['Close Date'],         errors='coerce')
# Fall back to Close Date when Last Donation Date is missing
donor_df['Last Donation Date'] = donor_df['Last Donation Date'].fillna(donor_df['Close Date'])
logger.info('%s donation records loaded', f'{len(donor_df):,}')

# ---------------------------------------------------------------------------
# Aggregate donations — one row per donor account
# ---------------------------------------------------------------------------
logger.info('Aggregating donations by account...')
donor_summary = (
    donor_df
    .groupby('Account Name', as_index=False)
    .agg(
        DonorAccountId    = ('Account ID',         'first'),
        LifetimeDonations = ('Amount',             'sum'),
        AverageDonation   = ('Amount',             'mean'),
        DonationCount     = ('Amount',             'count'),
        FirstDonationDate = ('Close Date',         'min'),
        LastDonationDate  = ('Last Donation Date', 'max'),
    )
)
logger.info('%s unique donor accounts', f'{len(donor_summary):,}')

# ---------------------------------------------------------------------------
# Identify donor accounts with no matching patron record
# ---------------------------------------------------------------------------
matched_names    = set(patrons_df['AccountName'])
unmatched_donors = donor_summary[
    ~donor_summary['Account Name'].isin(matched_names)
].copy()
logger.info('%s donor accounts have no matching patron record (name mismatch or no ticket history)',
            f'{len(unmatched_donors):,}')

# ---------------------------------------------------------------------------
# Fuzzy-match unmatched donors against patron names
# Helps identify Account Name mismatches (whitespace, nicknames, title variants)
# ---------------------------------------------------------------------------
_patron_names = sorted(matched_names)  # sorted list for get_close_matches

def _best_match(name, candidates, cutoff=0.60):
    """Return (best_match, score) for name against candidates, or ('', 0.0)."""
    hits = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    if not hits:
        return '', 0.0
    score = difflib.SequenceMatcher(None, name.lower(), hits[0].lower()).ratio()
    return hits[0], round(score, 2)

_results = unmatched_donors['Account Name'].apply(
    lambda n: pd.Series(_best_match(n, _patron_names), index=['SuggestedMatch', 'MatchScore'])
)
unmatched_donors = pd.concat([unmatched_donors, _results], axis=1)
logger.info('Fuzzy matching complete for unmatched donor accounts')

# ---------------------------------------------------------------------------
# Build name-match review file
# Joins suggested matches back to patron IDs so staff can correct Salesforce
# ---------------------------------------------------------------------------
_patron_ids = patrons_df[['AccountName', 'AccountId', 'ContactId']].drop_duplicates('AccountName')
review_df = (
    unmatched_donors
    .merge(_patron_ids, left_on='SuggestedMatch', right_on='AccountName', how='left')
    .drop(columns='AccountName')
    .rename(columns={
        'Account Name':       'Donor Name (Salesforce)',
        'DonorAccountId':     'Donor Account ID (SF)',
        'SuggestedMatch':     'Suggested Patron Match',
        'MatchScore':         'Match Score',
        'AccountId':          'Patron Account ID',
        'ContactId':          'Patron Contact ID',
        'LifetimeDonations':  'Lifetime Donations',
        'AverageDonation':    'Average Donation',
        'DonationCount':      'Donation Count',
        'FirstDonationDate':  'First Donation Date',
        'LastDonationDate':   'Last Donation Date',
    })
)
# Reorder: identity columns first, then match columns, then donation summary
_review_cols = [
    'Donor Name (Salesforce)', 'Donor Account ID (SF)',
    'Suggested Patron Match', 'Match Score',
    'Patron Account ID', 'Patron Contact ID',
    'Lifetime Donations', 'Donation Count',
    'First Donation Date', 'Last Donation Date',
]
review_df = review_df[[c for c in _review_cols if c in review_df.columns]]
review_df = review_df.sort_values('Match Score', ascending=False)

# ---------------------------------------------------------------------------
# Join to patrons (single left join)
# ---------------------------------------------------------------------------
logger.info('Merging patron and donor data...')
df = patrons_df.merge(
    donor_summary,
    left_on  = 'AccountName',
    right_on = 'Account Name',
    how      = 'left',
).drop(columns='Account Name')

df[['LifetimeDonations', 'AverageDonation', 'DonationCount']] = (
    df[['LifetimeDonations', 'AverageDonation', 'DonationCount']].fillna(0)
)
df['FirstDonationDate'] = pd.to_datetime(df['FirstDonationDate'], errors='coerce')
df['LastDonationDate']  = pd.to_datetime(df['LastDonationDate'],  errors='coerce')

matched   = df['DonationCount'].gt(0).sum()
unmatched = len(df) - matched
logger.info('%s patrons matched to donation records; %s with no donation history',
            f'{matched:,}', f'{unmatched:,}')

# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------
today = pd.Timestamp('today')
df['MonthsSinceLastDonation'] = (
    (today - df['LastDonationDate']).dt.days / 30
).fillna(9999)
df['IsDonor'] = df['DonationCount'] > 0

numeric_cols = ['AYM', 'Frequency', 'Lifespan', 'Regularity',
                'GrowthScore', 'Recency (Months)', 'LifetimeDonations', 'RFMScore']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)  # guard against inf values

# ---------------------------------------------------------------------------
# Tranche classification (mutually exclusive — each patron appears at most once)
# ---------------------------------------------------------------------------
logger.info('Classifying tranches...')

# Tranche 1: Major Donors — high-value current donors, upgrade ask
# No segment filter: preserves donors who may show as attendance-Lapsed post-COVID
major_donors = df[
    df['IsDonor'] &
    (df['MonthsSinceLastDonation'] <= ACTIVE_DONATION_MONTHS) &
    ((df['AYM'] >= MAJOR_DONOR_AYM) | (df['LifetimeDonations'] >= MAJOR_DONOR_LIFETIME))
].copy()
classified_ids = set(major_donors['AccountId'])
logger.info('  Major Donors:                %s', f'{len(major_donors):,}')

# Tranche 2: Growth Prospects — active donors with strong attendance, upgrade ask
# Best/High/Upsell segments with recent attendance: both donating AND attending well.
# Sorted by GrowthScore descending (note: scores suppressed by festival clustering
# and will improve once GrowthScore is recalibrated; ordering is relative for now).
growth_prospects = df[
    ~df['AccountId'].isin(classified_ids) &
    df['IsDonor'] &
    (df['MonthsSinceLastDonation'] <= ACTIVE_DONATION_MONTHS) &
    df['Segment'].isin(['Best', 'High', 'Upsell']) &
    (df['Recency (Months)'] <= PRIME_RECENCY_MONTHS)
].sort_values('GrowthScore', ascending=False).copy()
classified_ids |= set(growth_prospects['AccountId'])
logger.info('  Growth Prospects:            %s', f'{len(growth_prospects):,}')

# Tranche 3: Active Donors — Renew — all remaining donors who gave recently
# No segment filter: preserves donors who may show as attendance-Lapsed post-COVID
active_donors = df[
    ~df['AccountId'].isin(classified_ids) &
    df['IsDonor'] &
    (df['MonthsSinceLastDonation'] <= ACTIVE_DONATION_MONTHS)
].copy()
classified_ids |= set(active_donors['AccountId'])
logger.info('  Active Donors — Renew:       %s', f'{len(active_donors):,}')

# Tranche 4: Dormant Donors — Reactivate
# Gave before but not recently; still attending (not Lapsed or One&Done)
# Re-engaged and Come Again included: they returned to events, worth a reactivation ask
dormant_donors = df[
    ~df['AccountId'].isin(classified_ids) &
    df['IsDonor'] &
    (df['MonthsSinceLastDonation'] > ACTIVE_DONATION_MONTHS) &
    ~df['Segment'].isin(['Lapsed', 'One&Done'])
].copy()
classified_ids |= set(dormant_donors['AccountId'])
logger.info('  Dormant Donors — Reactivate: %s', f'{len(dormant_donors):,}')

# Tranche 5: Prime Non-Donor Prospects — first gift ask, high confidence
# Best/High/Upsell segments, recently active; sorted by GrowthScore
# New, Re-engaged, Come Again excluded — cultivate attendance first
prime_prospects = df[
    ~df['AccountId'].isin(classified_ids) &
    ~df['IsDonor'] &
    df['Segment'].isin(['Best', 'High', 'Upsell']) &
    (df['Recency (Months)'] <= PRIME_RECENCY_MONTHS)
].sort_values('GrowthScore', ascending=False).copy()
classified_ids |= set(prime_prospects['AccountId'])
logger.info('  Prime Non-Donor Prospects:   %s', f'{len(prime_prospects):,}')

# Lapsed Donors — donors whose attendance segment is Lapsed or One&Done
# Not solicited for donations, but captured for review
lapsed_donors = df[
    df['IsDonor'] &
    ~df['AccountId'].isin(classified_ids)
].copy()
logger.info('  Lapsed Donors (review only): %s', f'{len(lapsed_donors):,}')

total = (len(major_donors) + len(growth_prospects) + len(active_donors) +
         len(dormant_donors) + len(prime_prospects))
logger.info('Total actionable: %s of %s patrons', f'{total:,}', f'{len(df):,}')

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
PII_COLS = {'Account Name', 'First Name', 'Last Name', 'Email Address'}

COL_MAP = {
    'AccountName':             'Account Name',
    'AccountId':               'Account ID',
    'FirstName':               'First Name',
    'LastName':                'Last Name',
    'Email':                   'Email Address',
    'Segment':                 'Customer Segment',
    'RFMScore':                'RFM Score',
    'Recency (Months)':        'Recency (Months)',
    'Frequency':               'Frequency',
    'Lifespan':                'Customer Lifespan',
    'AYM':                     'Avg Yearly Monetary',
    'GrowthScore':             'Growth Score',
    'Regularity':              'Regularity',
    'IsDonor':                 'Is Donor',
    'LifetimeDonations':       'Lifetime Donations',
    'AverageDonation':         'Average Donation',
    'DonationCount':           'Donation Count',
    'FirstDonationDate':       'First Donation Date',
    'LastDonationDate':        'Last Donation Date',
    'MonthsSinceLastDonation': 'Months Since Last Donation',
    'PreferredEventGenre':     'Favorite Genre',
    'RegionAssignment':        'Geo Region',
}

CURRENCY_COLS = {'Lifetime Donations', 'Average Donation', 'Avg Yearly Monetary'}
DATE_COLS     = {'First Donation Date', 'Last Donation Date'}


def _prepare(frame):
    """Select and rename columns, keeping only those present in the frame."""
    available = {k: v for k, v in COL_MAP.items() if k in frame.columns}
    return frame[list(available.keys())].rename(columns=available)


def _write_sheet(writer, data, sheet_name, currency_cols=None, date_cols=None):
    """Write a DataFrame to an Excel sheet with auto column widths, filters, and formats."""
    if currency_cols is None:
        currency_cols = CURRENCY_COLS
    if date_cols is None:
        date_cols = DATE_COLS
    out = data if isinstance(data, pd.DataFrame) else _prepare(data)
    out.to_excel(writer, sheet_name=sheet_name, index=False)
    ws   = writer.sheets[sheet_name]
    book = writer.book
    last_col = len(out.columns) - 1
    ws.freeze_panes(1, 0)                          # freeze header row
    ws.autofilter(0, 0, len(out), last_col)        # enable filter dropdowns
    for i, col in enumerate(out.columns):
        col_width = max(out[col].astype(str).map(len).max(), len(col)) + 2
        fmt = None
        if col in date_cols:
            fmt = book.add_format({'num_format': 'mm/dd/yyyy'})
        elif col in currency_cols:
            fmt = book.add_format({'num_format': '$#,##0'})
        elif out[col].dtype.kind in 'fi':
            fmt = book.add_format({'num_format': '0.0'})
        ws.set_column(i, i, col_width, fmt)


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
_DATE_FMT = {'datetime_format': 'mm/dd/yyyy', 'date_format': 'mm/dd/yyyy'}

unmatched_out = unmatched_donors.rename(columns={
    'LifetimeDonations': 'Lifetime Donations',
    'AverageDonation':   'Average Donation',
    'DonationCount':     'Donation Count',
    'FirstDonationDate': 'First Donation Date',
    'LastDonationDate':  'Last Donation Date',
})

def _write_tranches(writer, drop_pii=False):
    """Write all tranche sheets to an ExcelWriter. If drop_pii, omit PII columns."""
    def prep(frame):
        out = _prepare(frame)
        if drop_pii:
            out = out.drop(columns=[c for c in PII_COLS if c in out.columns])
        return out

    _write_sheet(writer, prep(major_donors),    'Major Donors')
    _write_sheet(writer, prep(growth_prospects),'Growth Prospects')
    _write_sheet(writer, prep(active_donors),   'Active Donors - Renew')
    _write_sheet(writer, prep(dormant_donors),  'Dormant Donors - Reactivate')
    _write_sheet(writer, prep(prime_prospects), 'Prime Non-Donor Prospects')
    _write_sheet(writer, prep(lapsed_donors),   'Lapsed Donors - Review')
    if not drop_pii:
        _write_sheet(writer, unmatched_out, 'Donors - No Attendance Match',
                     currency_cols={'Lifetime Donations', 'Average Donation'},
                     date_cols={'First Donation Date', 'Last Donation Date'})

logger.info('Writing %s...', output_file)
with pd.ExcelWriter(output_file, engine='xlsxwriter', **_DATE_FMT) as writer:
    _write_tranches(writer, drop_pii=False)
logger.info('Done. Output written to: %s', output_file)

logger.info('Writing %s...', anon_file)
with pd.ExcelWriter(anon_file, engine='xlsxwriter', **_DATE_FMT) as writer:
    _write_tranches(writer, drop_pii=True)
logger.info('Done. Anonymized output written to: %s', anon_file)

# ---------------------------------------------------------------------------
# Write donor name match review (shareable, standalone file)
# ---------------------------------------------------------------------------
logger.info('Writing %s...', review_file)
with pd.ExcelWriter(review_file, engine='xlsxwriter', **_DATE_FMT) as writer:
    _write_sheet(
        writer, review_df, 'Donor Name Match Review',
        currency_cols={'Lifetime Donations', 'Average Donation'},
        date_cols={'First Donation Date', 'Last Donation Date'},
    )
logger.info('Name match review written to: %s', review_file)
