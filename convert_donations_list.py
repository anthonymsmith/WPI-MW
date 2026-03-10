#!/usr/bin/env python3
"""
convert_donations_list.py

Converts DonationsList.xlsx (PatronManager grouped export) to a flat
tabular format compatible with DonorClassifier.py (matching DonationsLatest.xlsx).

Input:  DonationsList.xlsx  — grouped by Account Name, sub-grouped by FY
Output: DonationsList_converted.xlsx

Close Date note:
  The source file only records fiscal year (e.g. "FY 2025"), not an actual date.
  FY YYYY runs July 1 of YYYY-1 through June 30 of YYYY.
  We use January 1, YYYY as a mid-FY proxy (falls in Fall season).
  Giving season and recency calculations in DonorClassifier will be approximate.
"""

import pandas as pd

INPUT  = 'DonationsList.xlsx'
OUTPUT = 'DonationsList_converted.xlsx'

# Row 13 is the header row (0-indexed), data starts at row 14.
HEADER_ROW  = 13
DATA_END    = 3245   # exclusive — row 3245 is the grand total

# FY → proxy date mapping (mid-FY = Jan 1 of end year)
FY_DATE_MAP = {
    'FY 2022': pd.Timestamp('2022-01-01'),
    'FY 2023': pd.Timestamp('2023-01-01'),
    'FY 2024': pd.Timestamp('2024-01-01'),
    'FY 2025': pd.Timestamp('2025-01-01'),
    'FY 2026': pd.Timestamp('2026-01-01'),
}

PLACEHOLDER_ACCOUNT_ID = '000000000000000'


def convert():
    raw = pd.read_excel(INPUT, header=None)

    # Extract column names from header row
    header = raw.iloc[HEADER_ROW].tolist()
    # header: [nan, 'Account Name  ↑', 'Close Date  ↑', nan, 'Account ID',
    #          'Billing Street', 'Billing City', 'Billing State/Province',
    #          'Billing Zip/Postal Code', 'Amount', 'Donation Record Type']

    # Slice data rows only
    data = raw.iloc[HEADER_ROW + 1 : DATA_END].copy()
    data.columns = range(data.shape[1])

    # Rename columns by position
    data = data.rename(columns={
        1:  'Account Name',
        2:  'Close Date',
        4:  'Account ID',
        5:  'Billing Street',
        6:  'Billing City',
        7:  'Billing State/Province',
        8:  'Billing Zip/Postal Code',
        9:  'Amount',
        10: 'Donation Record Type',
    }).drop(columns=[0, 3])

    # Forward-fill Account Name and Close Date (grouped export leaves blanks)
    data['Account Name'] = data['Account Name'].ffill()
    data['Close Date']   = data['Close Date'].ffill()

    # Drop placeholder / unattributed donations
    data = data[data['Account ID'] != PLACEHOLDER_ACCOUNT_ID]

    # Map FY labels to proxy dates
    unmapped = data[~data['Close Date'].isin(FY_DATE_MAP)]['Close Date'].unique()
    if len(unmapped):
        print(f'WARNING: unmapped Close Date values (will be NaT): {unmapped}')
    data['Close Date'] = data['Close Date'].map(FY_DATE_MAP)

    # Ensure numeric
    data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')
    data = data.dropna(subset=['Amount', 'Account ID'])

    # Add columns expected by DonorClassifier (derived from per-account aggregates)
    # These are populated per-account from the transaction rows themselves.
    acct = (
        data.groupby('Account ID')
        .agg(
            _LastDonationDate=('Close Date', 'max'),
            _LastDonationAmount=('Amount', lambda s: s.loc[s.index[s == s.max()][0]]),
            _LifetimeAmount=('Amount', 'sum'),
            _LifetimeCount=('Amount', 'count'),
        )
        .reset_index()
    )
    # Simpler last donation amount: amount on the most recent Close Date row
    acct['_LastDonationDate'] = data.groupby('Account ID')['Close Date'].max().values
    last_rows = (
        data.sort_values('Close Date')
            .groupby('Account ID')
            .last()
            .reset_index()[['Account ID', 'Amount', 'Close Date']]
            .rename(columns={'Amount': 'Last Donation Amount',
                             'Close Date': 'Last Donation Date'})
    )
    lifetime = (
        data.groupby('Account ID')
            .agg(
                **{'Lifetime Donation History (Amount)': ('Amount', 'sum'),
                   'Lifetime Donation History (Number)': ('Amount', 'count')}
            )
            .reset_index()
    )

    data = (
        data
        .merge(last_rows, on='Account ID', how='left')
        .merge(lifetime,  on='Account ID', how='left')
    )

    # Build output matching DonationsLatest.xlsx column order
    out = data[[
        'Account Name',
        'Account ID',
        'Amount',
        'Close Date',
        'Last Donation Amount',
        'Last Donation Date',
        'Lifetime Donation History (Amount)',
        'Lifetime Donation History (Number)',
        'Billing Street',
        'Billing City',
        'Billing State/Province',
        'Billing Zip/Postal Code',
        'Donation Record Type',
    ]].copy()

    # Add placeholder columns present in DonationsLatest but not available here
    for col in ['Donation Name', 'Lifetime Order Count',
                'Lifetime Single Ticket $', 'Lifetime Subscription $']:
        out[col] = pd.NA

    # Final column order matching DonationsLatest.xlsx
    out = out[[
        'Donation Name',
        'Amount',
        'Close Date',
        'Account Name',
        'Account ID',
        'Last Donation Amount',
        'Last Donation Date',
        'Lifetime Donation History (Amount)',
        'Lifetime Donation History (Number)',
        'Billing Street',
        'Billing City',
        'Billing State/Province',
        'Billing Zip/Postal Code',
        'Lifetime Order Count',
        'Lifetime Single Ticket $',
        'Lifetime Subscription $',
        'Donation Record Type',
    ]]

    out.to_excel(OUTPUT, index=False)
    print(f'Written {len(out):,} rows → {OUTPUT}')
    print(f'\nRecord type breakdown:')
    print(out['Donation Record Type'].value_counts().to_string())
    print(f'\nDate range: {out["Close Date"].min().date()} – {out["Close Date"].max().date()}')
    print(f'Total amount: ${out["Amount"].sum():,.2f}')
    print(f'Unique accounts: {out["Account ID"].nunique():,}')


if __name__ == '__main__':
    convert()
