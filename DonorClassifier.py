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
    Patron_Classification.xlsx - one sheet per tranche
"""

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
        LifetimeDonations = ('Amount',              'sum'),
        AverageDonation   = ('Amount',              'mean'),
        DonationCount     = ('Amount',              'count'),
        FirstDonationDate = ('Close Date',          'min'),
        LastDonationDate  = ('Last Donation Date',  'max'),
    )
)
logger.info('%s unique donor accounts', f'{len(donor_summary):,}')

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

# ---------------------------------------------------------------------------
# Tranche classification (mutually exclusive — each patron appears once)
# ---------------------------------------------------------------------------
logger.info('Classifying tranches...')

# Tranche 1: Top Prospects — best customers who are already active donors
top_prospects = df[
    df['Segment'].isin(['Best', 'Upsell']) &
    df['IsDonor'] &
    (df['AYM'] >= 1000) &
    (df['Frequency'] >= 20) &
    (df['Recency (Months)'] <= 12) &
    (df['Lifespan'] >= 5)
].copy()
classified_ids = set(top_prospects['AccountId'])
logger.info('  Top Prospects:          %s', f'{len(top_prospects):,}')

# Tranche 2: Continued Contributors — loyal donors worth retaining
continued_contributors = df[
    ~df['AccountId'].isin(classified_ids) &
    df['Segment'].isin(['Best', 'Upsell', 'Slipping']) &
    df['IsDonor'] &
    (df['AYM'] >= 100) &
    (df['Frequency'] >= 10) &
    (df['Recency (Months)'] <= 24) &
    (df['Regularity'] >= 0.4)
].copy()
classified_ids |= set(continued_contributors['AccountId'])
logger.info('  Continued Contributors: %s', f'{len(continued_contributors):,}')

# Tranche 3: Growth Opportunities — high-value attendees who have never donated
#            or whose last donation was more than 2 years ago
growth_opportunities = df[
    ~df['AccountId'].isin(classified_ids) &
    df['Segment'].isin(['Best', 'Upsell']) &
    (df['Frequency'] >= 10) &
    (df['Recency (Months)'] <= 18) &
    (~df['IsDonor'] | (df['MonthsSinceLastDonation'] > 24)) &
    (df['LifetimeDonations'] < 5000)
].copy()
classified_ids |= set(growth_opportunities['AccountId'])
logger.info('  Growth Opportunities:   %s', f'{len(growth_opportunities):,}')

# Tranche 4: Reactivation Targets — lapsed donors with strong ticket history
reactivation_targets = df[
    ~df['AccountId'].isin(classified_ids) &
    df['Segment'].isin(['Slipping', 'Lapsed']) &
    df['IsDonor'] &
    (df['AYM'] >= 500) &
    (df['Frequency'] >= 15) &
    (df['Recency (Months)'] > 18)
].copy()
classified_ids |= set(reactivation_targets['AccountId'])
logger.info('  Reactivation Targets:   %s', f'{len(reactivation_targets):,}')

# Tranche 5: New Watch List — newer patrons with early donation signals
new_watch_list = df[
    ~df['AccountId'].isin(classified_ids) &
    (df['Lifespan'] < 2.0) &
    (df['LifetimeDonations'] > 100) &
    (df['AYM'] >= 200)
].copy()
logger.info('  New Watch List:         %s', f'{len(new_watch_list):,}')

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
# Columns to include in each sheet, with readable labels
COL_MAP = {
    'AccountName':              'Account Name',
    'AccountId':                'Account ID',
    'FirstName':                'First Name',
    'LastName':                 'Last Name',
    'Email':                    'Email Address',
    'Segment':                  'Customer Segment',
    'RFMScore':                 'RFM Score',
    'Recency (Months)':         'Recency (Months)',
    'Frequency':                'Frequency',
    'Lifespan':                 'Customer Lifespan',
    'AYM':                      'Avg Yearly Monetary',
    'GrowthScore':              'Growth Score',
    'Regularity':               'Regularity',
    'IsDonor':                  'Is Donor',
    'LifetimeDonations':        'Lifetime Donations',
    'AverageDonation':          'Average Donation',
    'DonationCount':            'Donation Count',
    'FirstDonationDate':        'First Donation Date',
    'LastDonationDate':         'Last Donation Date',
    'MonthsSinceLastDonation':  'Months Since Last Donation',
    'PreferredEventGenre':      'Favorite Genre',
    'RegionAssignment':         'Geo Region',
}

CURRENCY_COLS = {'Lifetime Donations', 'Average Donation', 'Avg Yearly Monetary'}
DATE_COLS     = {'First Donation Date', 'Last Donation Date'}


def _prepare(frame):
    """Select and rename columns, keeping only those present in the frame."""
    available = {k: v for k, v in COL_MAP.items() if k in frame.columns}
    return frame[list(available.keys())].rename(columns=available)


def _write_sheet(writer, data, sheet_name):
    """Write a DataFrame to an Excel sheet with auto column widths and formats."""
    out = _prepare(data)
    out.to_excel(writer, sheet_name=sheet_name, index=False)
    ws   = writer.sheets[sheet_name]
    book = writer.book
    for i, col in enumerate(out.columns):
        col_width = max(out[col].astype(str).map(len).max(), len(col)) + 2
        fmt = None
        if col in DATE_COLS:
            fmt = book.add_format({'num_format': 'mm/dd/yyyy'})
        elif col in CURRENCY_COLS:
            fmt = book.add_format({'num_format': '$#,##0'})
        elif out[col].dtype.kind in 'fi':
            fmt = book.add_format({'num_format': '0.0'})
        ws.set_column(i, i, col_width, fmt)


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
logger.info('Writing %s...', output_file)
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    _write_sheet(writer, top_prospects,          'Top Prospects')
    _write_sheet(writer, continued_contributors, 'Continued Contributors')
    _write_sheet(writer, growth_opportunities,   'Growth Opportunities')
    _write_sheet(writer, reactivation_targets,   'Reactivation Targets')
    _write_sheet(writer, new_watch_list,         'New Watch List')

logger.info('Done. Output written to: %s', output_file)
