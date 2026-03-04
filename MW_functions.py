"""
Copyright (c) Nolichucky Associates 2024. All Rights Reserved.

This software is the confidential and proprietary information of Nolichucky Associates.
You shall not disclose such Confidential Information and shall use it only in accordance
 with the terms of the license agreement you entered into with Nolichucky Associates.

Unauthorized copying of this file, via any medium, is strictly prohibited.
Proprietary and confidential.

Project: Music Worcester Patron and Event Analytics

Author: Anthony Smith
Date: September, 2024
"""

import pandas as pd
import numpy as np
import re
import Model_functions as mod
from datetime import datetime, timedelta
from time import perf_counter
import hashlib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import requests
from requests.exceptions import RequestException

# Processing functions
def _elapsed(start):
    """Return elapsed seconds since `start` as a '%.2f' string."""
    return f'{timedelta(seconds=perf_counter() - start).total_seconds():.2f}'


def load_event_manifest(manifest_file, logger):
    """Load event manifest from Excel; remove test events, standardize dates and defaults, sort by date."""
    start = perf_counter()

    # Load event manifest file and fix column names
    event_df = pd.read_excel(manifest_file, sheet_name='EventManifest')
    logger.debug(f'Raw Event Manifest {event_df.head()}')

    # Fix dates
    event_df['EventDate'] = pd.to_datetime(event_df['EventDate'].copy()).dt.date

    # Remove test events
    event_df = event_df[event_df['EventName'].str.contains('test|TEST|Test', case=False) == False]
    logger.debug(f'Event shape {event_df.shape}')

    # Fill missing values
    columns_to_fill = ['EventGenre', 'EventClass', 'EventStatus', 'EventSubGenre', 'EventVenue']
    event_df[columns_to_fill] = event_df[columns_to_fill].fillna('None')
    event_df['EventCapacity'] = event_df['EventCapacity'].fillna(1)

    # Sort dataframe
    event_df = event_df.sort_values(by=['EventDate', 'EventName', 'EventInstance'], ascending=False)

    logger.info(f'Events loaded. Execution Time: {_elapsed(start)}')

    return event_df
def add_PnL_data(event_df, Pnl_file, PnLProcessed_file, logger):
    """Merge event data with QuickBooks P&L export, compute revenue/expense ratios, and write processed P&L to CSV."""
    start = perf_counter()

    PnL_df = pd.read_excel(Pnl_file,sheet_name='PnLSourceData')
    logger.debug(f'raw PnL {PnL_df.columns}')

    # Fix QuickBooks names to be more readable
    PnL_df = PnL_df.rename(columns={
        'Total 40000 · Direct Public Support': 'EventPublicSupport',
        'Total 41000 · Endowment Income': 'EventEndowment',
        '42010 · Corporate Sponsorship': 'CorporateSponsor',
        'Total 42005 · Concert Sponsorship': 'EventSponsor',
        'Total 42015 · Concert Tickets': 'EventTickets',
        'Total 50100 · Advertising': 'EventAdvertising',
        '50200 · Artistic Fees': 'EventArtistFees',
        '50500 · Hall/Tech Fees': 'EventVenueFees',
        '50550 · Hospitality Costs': 'EventHospitality',
        'G/A': 'EventG/A'})
    logger.debug(f'PnL shape {PnL_df.shape}')
    logger.debug(f'PnL {PnL_df.columns}')

    # Merge dataframes
    logger.debug(f'PnL pre merge: {PnL_df.shape}')
    logger.debug(f'PnL columns {PnL_df.columns}')
    logger.debug(f'event_df columns{event_df.columns}')
    res_df = event_df.merge(PnL_df, how='left', on='EventId').sort_values('EventDate', ascending=False)
    logger.debug(f'PnL post merge: {PnL_df.shape}')

    # Clean up resulting dataframe, keeping the Event Manifest EventName
    res_df.drop(['Type', 'EventInstance', 'InstanceId', 'EventName_PnL', 'EventDate_PnL'], axis=1,
                inplace=True)
    # We only care about EventNames, not multiple EventInstances, so keep only the first EventName.
    res_df = res_df.groupby('EventId').first().reset_index()
    logger.debug(f'Results columns: {res_df.columns}')

    # Convert any numeric strings column values to floats
    numeric_columns = ['EventCapacity', 'EventPublicSupport', 'CorporateSponsor', 'EventSponsor', 'EventAdvertising', 'EventArtistFees', 'EventVenueFees',
        'EventHospitality', 'EventG/A', 'EventTickets', 'EventTotalSponsorship', 'EventRevenue', 'EventExpense', 'EventSalesPlan']
    for col in numeric_columns:
        res_df[col] = pd.to_numeric(res_df[col], errors='coerce').fillna(0.0)

    # calculate ratios, being careful to avoid division errors
    res_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    res_df['EventRev/Expense'] = safe_divide(res_df['EventRevenue'], res_df['EventExpense'])
    res_df['EventPctSalesPlan'] = safe_divide(res_df['EventTickets'], res_df['EventSalesPlan'])

    #replace with 0 if salesplan is 0. Belt-and-suspenders...
    #res_df['EventPctSalesPlan'].replace([np.inf, -np.inf,np.nan], 0, inplace=True)
    res_df['EventPctSalesPlan'] = (res_df['EventPctSalesPlan']
                                   .replace([np.inf, -np.inf, np.nan], 0)
    )

    keep_cols = ['EventName',
                 'EventDate',
                 'EventVenue',
                 'EventPublicSupport',
                 'EventSponsor',
                 'EventTickets',
                 'EventAdvertising',
                 'EventArtistFees',
                 'EventVenueFees',
                 'EventHospitality',
                 'EventG/A',
                 'EventTotalSponsorship',
                 'EventRevenue',
                 'EventExpense',
                 'EventProfit',
                 'EventSalesPlan',
                 'EventRev/Expense',
                 'EventPctSalesPlan',
                 'EventCapacity',
                 'EventId',
                 'Season',
                 'EventStatus',
                 'EventType',
                 'EventClass',
                 'EventGenre',
                 'EventSubGenre']

    res_df = res_df[keep_cols]

    # Save the processed data
    res_df.to_csv(PnLProcessed_file, index=False)

    logger.info(f'PnL data written to {PnLProcessed_file}. Execution Time: {_elapsed(start)}')

    return event_df
def load_sales_file(sales_file, yearsOfData, logger):
    """Load Salesforce sales CSV, rename columns to readable names, and prune records older than yearsOfData years."""
    start = perf_counter()

    # load sales file and fix column names
    sales_df = pd.read_csv(sales_file, encoding= 'latin1', low_memory=False)
    logger.debug(f'Raw sales file: {sales_df.shape}')
    logger.debug(f'Raw sales file:/n{sales_df.columns}')

    # Rename Salesforce column names to be more readable.
    sales_df = sales_df.rename(columns={
        'Ticket Order': 'OrderNumber',
        'Ticket Order Item': 'OrderItem',
        'Ticket Order: First Name': 'FirstName',
        'Ticket Order: Last Name': 'LastName',
        'Ticket Allocation': 'Allocation',
        'Ticket Price Level: Name': 'TicketType',
        # 'Quantity' not renamed
        'Effective Item Price': 'ItemPrice',
        'Ticket Order: Order Status': 'OrderStatus',
        'Ticket Order: Street Address': 'Address',
        'Ticket Order: City': 'City',
        'Ticket Order: State': 'State',
        'Ticket Order: Postal Code': 'ZIP',
        'Ticket Order: Amount Paid': 'AmountPaid',
        'Ticket Order: Delivery Method': 'Method',
        'Ticket Order: Order Origin': 'Origin',
        'Ticket Order: Order Total': 'Total',
        'Ticket Order: Payment Method': 'PaymentMethod',
        'Contact: Contact ID': 'ContactId',
        'Contact: Account Name: Account Name': 'AccountName',
        'Ticket Order: Order Source': 'OrderSource',
        'Ticket Price Level: Price': 'PriceLevel',
        'Ticket Order: Created Date': 'OrderCreatedDate',
        'Event Instance: Date': 'EventDate_sales',
        'Ticket Order: Donation Amount': 'DonationAmount',
        'Ticket Order: Donation': 'DonationName',
        'Ticket Order: Payment Transaction ID': 'PaymentTxnId',
        'Venue Name': 'VenueName_sales',
        # 'Ticketable Event: 18 Digit Event ID' not renamed and deleted
        'Ticketable Event: Ticketable Event ID': 'EventId_sales',
        # 'Ticketable Event: 18 Digit Event ID' redundant and deleted
        # 'Ticketable Event: Ticketable Event ID' redundant and deleted
        'Ticket Price Level: Ticket Allocation: Event Instance: Event Instance ID': 'InstanceId_sales',
        'Status': 'TicketStatus',
        # 'Status' is redundant and deleted
        'Discount Code: Name': 'DiscountCode',
        'Discount Total': 'DiscountTotal',
        'Pre-Discount Total': 'PreDiscountTotal',
        'Unit Discount Amount': 'UnitDiscount',
        'Unit Discount Type': 'UnitDiscountType',
        'Ticket Order: Email': 'Email',
        'Contact: Email': 'ContactEmail',
        # 'Contact: Email' is redundant and deleted
        'Created Date': 'CreatedDate', # This is really the order modified date.
        'Entry Date': 'EntryDate', # is often missing, so we'll delete it.
        'Ticketable Event: Name': 'EventName_sales',
        'Event Instance': 'EventInstance_sales',
        # Months, Quarters, Years not renamed or used.
    })

    # Convert CreatedDate to datetime if it's not already
    sales_df['CreatedDate'] = pd.to_datetime(sales_df['CreatedDate'].copy(), errors='coerce')

    # Calculate earliest date to keep and prune earlier rows
    beginning_date = datetime.now() - timedelta(days=365*yearsOfData)
    sales_df = sales_df[sales_df['CreatedDate'] > beginning_date]

    logger.info(f'Sales loaded. Execution Time: {_elapsed(start)}')

    logger.debug(sales_df.shape)
    return sales_df
def sales_initial_prep(df, logger):
    """Remove deleted tickets, fill missing values, generate SHA-256 AccountIds, and drop test events."""
    start = perf_counter()
    df = df.copy()

    df['EventDate_sales'] = pd.to_datetime(df['EventDate_sales'], format='mixed').dt.date

    # Remove deleted tickets (placeholders for subscriptions or cancelled orders)
    df = df[df['TicketStatus'] != 'Deleted']

    # AccountName steps done at transaction-level because we need a name to generate AccountId.
    df['AccountName'] = df['AccountName'].fillna('Walk Up Sales')

    # Assign AccountId (ContactId is at contact level, not account level)
    df['AccountId'] = df['AccountName'].apply(generate_identifier)

    # Numeric columns — coerce and fill
    numeric_fillna = {
        'ItemPrice': 0, 'Quantity': 0, 'DonationAmount': 0,
        'DiscountTotal': 0, 'UnitDiscount': 0, 'PreDiscountTotal': 0,
        'PriceLevel': 0, 'Total': 0,
    }
    for col, fill_value in numeric_fillna.items():
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_value)

    df.fillna({'DonationName': 'None', 'DiscountCode': 'None', 'UnitDiscountType': 'None'}, inplace=True)

    # Clean up Event Names and Instances
    df['EventName_sales'] = df['EventName_sales'].astype(str)
    df['EventInstance_sales'] = df['EventInstance_sales'].astype(str)

    # Remove test events and instances
    df = df.loc[~df['EventName_sales'].str.contains(' test', case=False)]
    df = df.loc[~df['EventInstance_sales'].str.contains(' test', case=False)]

    logger.info(f'Initial sales prep complete. Execution Time: {_elapsed(start)}')
    return df
def venue_and_attribute_processing(sales_df, chorus_list_file, board_file, logger):
    """
    Standardize venue names and add patron attribute flags.

        Flags: ChorusMember (chorus list or Dues discount), DuesTxn, Student,
        and Subscriber (current season computed dynamically). Board/corporator
        status applied from board_file.
    """
    start = perf_counter()
    # Clean up venue names
    sales_df['VenueName_sales'] = sales_df['VenueName_sales'].fillna('None')
    sales_df.loc[sales_df['VenueName_sales'].str.contains("Mechanics"), 'VenueName_sales'] = 'Mechanics Hall'

    # Chorus members: from the chorus member list or from use of DUES discount
    chorus_members = set(pd.read_excel(chorus_list_file, usecols=['Account Name']).dropna()['Account Name'])
    sales_df['ChorusMember'] = sales_df['AccountName'].isin(chorus_members)

    # Board/corporator status: merge on FirstName + LastName
    board_df = pd.read_csv(board_file).dropna()
    board_df[['FirstName', 'LastName']] = board_df['AccountName'].str.split(pat=' ', n=1, expand=True)
    sales_df = sales_df.merge(board_df[['FirstName', 'LastName', 'PatronStatus']],
                               on=['FirstName', 'LastName'], how='left')
    sales_df['PatronStatus'] = sales_df['PatronStatus'].fillna('patron')

    sales_df['DuesTxn'] = sales_df['DiscountCode'].str.contains("Chorus Dues", na=False)
    sales_df['ChorusMember'] = sales_df['ChorusMember'] | sales_df['DuesTxn']
    sales_df['Student'] = sales_df['TicketType'].str.contains("Student", na=False)

    sales_df['EventName_sales'] = sales_df['EventName_sales'].astype(str)

    # Calculate the current season dynamically (season starts in July)
    _today = datetime.today()
    _season_start = _today.year if _today.month > 6 else _today.year - 1
    _season_end = _season_start + 1
    _season_long = f'{_season_start}-{_season_end}'                          # e.g. '2025-2026'
    _season_short = f'{str(_season_start)[-2:]}-{str(_season_end)[-2:]}'    # e.g. '25-26'

    # Define the ordered categorical column for correct aggregation
    sales_df['Subscriber'] = pd.Categorical(
        sales_df['EventName_sales'].apply(
            lambda x: (
                'current' if _season_long in x.lower() or _season_short in x.lower()
                else 'previous' if 'subscri' in x.lower()
                else 'never'
            )
        ),
        categories=['never', 'previous', 'current'],  # Ensures correct order for max()
        ordered=True
    )

    logger.debug('Subscriber initial totals: %s', sales_df["Subscriber"].value_counts())

    logger.debug(f'Venue and Attribute columns: {sales_df.columns}')

    logger.info(f'Venue and attribute processing complete. Execution Time: {_elapsed(start)}')

    return sales_df
def event_counts(df, logger, event_column):
    """
    Generalized function to count occurrences of a specified event attribute (e.g., EventGenre, EventClass, EventVenue)
    for each AccountId.

    Parameters:
        df (DataFrame): The input DataFrame containing event data.
        logger (Logger): Logger instance for logging messages.
        event_column (str): The column name to count occurrences of (e.g., 'EventGenre', 'EventClass', 'EventVenue').

    Returns:
        DataFrame: Original DataFrame merged with event counts for the specified column.
    """
    start = perf_counter()

    # Ensure we are not modifying the original DataFrame
    df = df.copy()

    # Fill NaN values in the specified event column
    df[event_column] = df[event_column].fillna('None')

    # Filter relevant event types
    df_filtered = df[df['EventType'].isin(['Live', 'Virtual', 'Subscriptions'])]

    logger.debug('Subscriber totals before %s merge: %s', event_column, df["Subscriber"].value_counts())

    # Log all unique values of the event column
    unique_values = df_filtered[event_column].unique()
    logger.debug(f'Unique values in {event_column}: {list(unique_values)}')

    # Drop duplicates to count only unique events per AccountId for the given event column
    unique_events_df = df_filtered.drop_duplicates(subset=['AccountId', event_column, 'EventId'])

    # Pivot table to count the number of events per category for each AccountId
    event_counts_df = unique_events_df.pivot_table(index='AccountId',
                                                   columns=event_column,
                                                   values='EventId',
                                                   aggfunc='count',
                                                   fill_value=0).reset_index()

    # Merge only the necessary event count columns back to the original DataFrame
    merged_df = df.merge(event_counts_df, on='AccountId', how='left')

    # Logging details
    logger.debug('Subscriber totals after %s merge: %s', event_column, merged_df["Subscriber"].value_counts())
    logger.debug(f'Final DataFrame shape after {event_column} merge: {merged_df.shape}')
    logger.debug(f'Final columns after {event_column} merge: {merged_df.columns}')

    # Execution time calculation
    logger.info(f'{event_column} counts complete. Execution Time: {_elapsed(start)}')

    return merged_df
def state_and_city_processing(sales_df, logger):
    """Standardize State and City fields: extract ZIPs misplaced in State, fix typos and abbreviations via regex."""
    start = perf_counter()

    logger.debug(f'State & City input columns: {sales_df.shape}')
    logger.debug(f'State & City input shape: {sales_df.shape}')

    # set aside State and City and clean up
    sales_df['State'] = sales_df['State'].astype(str)
    sales_df['State-orig'] = sales_df['State']
    logger.debug(sales_df.State.dtypes)

    # If State and ZIP are in the same field, which happens a fair amount, parse them out.
    # Regular expression to match 'State' and 'ZIP' patterns
    pattern = r'(?P<State>[A-Z]{2})(?P<ZIP>\d{5}(?:-\d{4})?)'

    # Extract 'State' and 'ZIP' into new columns
    extracted_df = sales_df['State'].str.extract(pattern)

    # Update 'State' and 'ZIP' columns in the original dataframe where a match is found
    sales_df.loc[extracted_df['State'].notna(), 'State'] = extracted_df['State']
    sales_df.loc[extracted_df['ZIP'].notna(), 'ZIP'] = extracted_df['ZIP']

    # Fix other common State typos, mostly focusing on MA
    state_replacements = {'Ma': 'MA',
                          '01': 'MA',
                          'Mechanics': 'MA',
                          '2472': 'MA'}
    sales_df['State'] = sales_df['State'].replace(state_replacements)

    #Fix cases where the ZIP is in the state field.
    # Regular expression for ZIP code pattern
    zip_pattern = r'\d{5}(?:-\d{4})?'

    # Identify rows with ZIP code in 'State' column and move them to the ZIP columns
    zip_in_state_mask = sales_df['State'].str.match(zip_pattern) & sales_df['ZIP'].isna()
    sales_df.loc[zip_in_state_mask, 'ZIP'] = sales_df.loc[zip_in_state_mask, 'State']
    # for these, we assume the right state is MA. Not ideal, but a safe guess.
    sales_df.loc[zip_in_state_mask, 'State'] = 'MA'

    #replace any "Mass" string with MA
    sales_df.loc[sales_df['State'].str.contains('Mass', case=False, na=False), 'State'] = 'MA'

    #Fix Burncoat school ZIP. It's so common that it deserves special treatment.
    mask = (sales_df['Address'] == '179 Burncoat St') & (sales_df['City'] == 'Worcester')
    sales_df.loc[mask, 'ZIP'] = '01606'

    # Standardize city names: convert to title case, remove ' MA',
    # replace punctuation with spaces, and clean up extra spaces
    sales_df['City'] = (
        sales_df['City']
        .str.title()
        .str.replace(r" MA", "", case=False)
        .str.replace(r"[,.]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )
    logger.debug(f'Begin cleaning up city names')

    # Define a dictionary for state-specific city name corrections
    city_corrections_by_state = {
        "MA": {
            r'Arlin.*': 'Arlington',
            r'Attl.*': 'Attleboro',
            r'Ashkand': 'Ashland',
            r'Aburn': 'Auburn',
            r'.*barre.*': 'Barre',
            r'.*lmont': 'Belmont',
            r'Bylston|Boylson|Biylston': 'Boylston',
            r'Baldwinville': 'Baldwinville',
            r'Charlt.*': 'Charlton',
            r'Chelms.*': 'Chelmsford',
            r'ouglas': 'Douglas',
            r'Fithburg': 'Fitchburg',
            r'Fisk.*': 'Fiskdale',
            r'Framingham|Framngham': 'Framingham',
            r'Gardiner': 'Gardner',
            r'Hold.*': 'Holden',
            r'Hopkin.*': 'Hopkinton',
            r'aster|ancaster': 'Lancaster',
            r'Lanesboro': 'Lanesborough',
            r'East Longmeadow': 'E Longmeadow',
            r'Flirence': 'Florence',
            r'Foxboro': 'Foxborough',
            r'Hub.*': 'Hubbardston',
            r'Jeffer.*': 'Jefferson',
            r'JP': 'Jamaica Plain',
            r'Leomin.*': 'Leominster',
            r'Lunenberg': 'Lunenburg',
            r'Middleboro': 'Middleborough',
            r'Marl.*': 'Marlborough',
            r'Natix': 'Natick',
            r'Newton U.*': 'Newton Upper Falls',
            r'Newton H.*': 'Newton Highlands',
            r'Northboro.*|01532': 'Northborough',
            r'Northbr.*': 'Northbridge',
            r'Princet.*': 'Princeton',
            r'1543|Rutalnd': 'Rutland',
            r'Sherb.*': 'Sherborn',
            r'Southboro.*': 'Southborough',
            r'Shre.*': 'Shrewsbury',
            r'Stow.*': 'Stow',
            r'Sturb.*': 'Sturbridge',
            r'Surtton|Suttom': 'Sutton',
            r'Uuton': 'Upton',
            r'^West.*bo.*': 'Westborough',
            r'Westfield': 'Westfield',
            r'.*insville': 'Whitinsville',
            r'^Wor.*': 'Worcester'
        },
        "CT": {
            r'Haddam.*': 'E Haddam',
            r'Grosvenordale.*': 'N Grosvenordale',
            r'Pomfret.*': 'Pomfret',
            r'Claymont': 'Claymont',
            r'Storrs': 'Storrs'
        },
        "NY": {
            r'Brokln|Brookj|Broook|Brooklyn.*': 'Brooklyn',
            r'Ny.*': 'New York City'
        }
    }

    # Function to correct city names
    def correct_city(city, corrections):
        if pd.isna(city):  # Skip if the city is NaN
            return city

        # Iterate over each pattern and apply the first match
        for pattern, replacement in corrections.items():
            if re.search(pattern, city):  # Check if the pattern matches
                return re.sub(pattern, replacement, city)  # Apply the match
        return city  # Return original city if no match is found

    # Apply corrections for each state
    for state, corrections in city_corrections_by_state.items():
        mask = sales_df['State'] == state
        sales_df.loc[mask, 'City'] = sales_df.loc[mask, 'City'].apply(lambda city: correct_city(city, corrections))


    # Align NSEW and Centre in City names
    direction_mappings = {
        'North': 'N',
        'South': 'S',
        'East': 'E',
        'West': 'W',
        'No': 'N',
        'So': 'S',
        'Wedt': 'W',
    }

    sales_df['City'] = (
        sales_df['City']
        .str.replace(r"(?i)\bCentre\b", "Center", regex=True)
        .str.replace(r"\b(" + "|".join(direction_mappings.keys()) + r")\b",
                     lambda x: direction_mappings[x.group(0)], regex=True)
    )

    city_state_mappings = {
        'Amherst': 'MA',
        'Auburn': 'MA',
        'Boylston': 'MA',
        'Charlton': 'MA',
        'Douglas': 'MA',
        'Grafton': 'MA',
        'Holden': 'MA',
        'Jamaica Plain': 'MA',
        'Malden': 'MA',
        'Newton': 'MA',
        'Northborough': 'MA',
        'Shrewsbury': 'MA',
        'Southbridge': 'MA',
        'Sterling': 'MA',
        'Westborough': 'MA',
        'Worcester': 'MA',
    }

    sales_df['State'] = sales_df['State'].fillna(sales_df['City'].map(city_state_mappings))

    logger.debug('States containing "MA"')
    logger.debug(np.unique(sales_df.loc[sales_df['State'].str.contains("MA")]['State']))
    logger.debug('MA cities with null state')
    logger.debug(sales_df.shape)

    logger.info(f'State and city processing complete. Execution Time: {_elapsed(start)}')

    return sales_df
def address_and_ZIP_processing(sales_df, logger):
    """Convert addresses to title case, abbreviate street types, and pad/correct ZIP codes based on city mappings."""
    start = perf_counter()
    logger.debug(sales_df.shape)

    # Clean up addresses
    sales_df['Address'] = sales_df['Address'].str.title()

    # Perform abbreviation replacements
    replacements = {
        r'\bRoad\b': 'Rd',
        r'\bAvenue\b': 'Ave',
        r'\bStreet\b': 'St',
        r'\bDrive\b': 'Dr'
    }
    sales_df['Address'] = sales_df['Address'].replace(replacements, regex=True)

    # Perform other replacements
    sales_df['Address'] = sales_df['Address'].str.replace(r"(Po|Pob) ", lambda x: x.group(1).upper() + ' ', regex=True)

    # Now, deal with periods and other punctuation
    sales_df['Address'] = sales_df['Address'].str.replace(r"[.]", " ", regex=True) # replace single period with a space
    sales_df['Address'] = sales_df['Address'].replace(r"[,]", "", regex=True) # remove commas

    # Strip ZIP+4
    sales_df['ZIP'] = sales_df['ZIP'].astype(str).str[:5].str.pad(width=5, side='left', fillchar='0')

    # Clean up bad ZIPs by city

    # Function to clean up ZIP codes for a given city
    def clean_zip_codes(df, city, zip_mappings):
        mask = (df['City'] == city) & (df['ZIP'].str.contains('|'.join(zip_mappings.keys())))
        df.loc[mask, 'ZIP'] = df.loc[mask, 'ZIP'].map(zip_mappings).fillna(df['ZIP'])

    # Consolidated ZIP code mappings
    zip_mappings = {
        '0162': '01602', r'\(': '01609', r'\)': '01602', '10608': '01608',
        '00880': '01609', "0'720": '01720', "02091": '02891',"01250": '01520',
        '91545': '01545', '01555': '01545', r"\(016": '01601', '00756': '01545',
        '00162': '01602', '01533': '01532', '00158': '01581','01594': '01504',
        '014540': '01545'
    }

    def clean_zip_codes(df, zip_column):
        for wrong_zip, correct_zip in zip_mappings.items():
            df[zip_column] = df[zip_column].str.replace(wrong_zip, correct_zip, regex=True)
        return df

    # Apply the clean_zip_codes function to the sales_df DataFrame
    sales_df = clean_zip_codes(sales_df, 'ZIP')

    logger.info(f'address & ZIP processing complete. Execution Time: {_elapsed(start)}')

    logger.debug(sales_df.shape)

    return sales_df
def combine_sales_and_events(sales_df, event_df, logger):
    """Merge sales and event data on EventId/InstanceId (left join), log unmatched records, sort by event date."""
    start = perf_counter()

    logger.debug('Sales + Events:')

    logger.debug(f'sales + events pre merge: {sales_df.shape}')
    se_df = sales_df.merge(event_df.drop_duplicates(),
                     left_on=['EventId_sales', 'InstanceId_sales'],
                     right_on=['EventId', 'InstanceId'],
                     how='left')
    logger.debug(f'Sales columns {sales_df.columns}')
    logger.debug(f'Manifest columns {event_df.columns}')
    logger.debug(f'sales + events post merge columns {se_df.columns}')

    logger.debug(f'sales + events post merge: {se_df.shape}')

    logger.debug(se_df)

    # find any EventInstances in sales that are not in events
    # Filter to keep only rows where InstanceId is null
    filtered_df = se_df[se_df['InstanceId'].isnull()]

    # Select the InstanceId_sales column and write to a file.
    result_df = filtered_df[['EventName_sales','EventId_sales','EventInstance_sales','InstanceId_sales','EventDate_sales']].drop_duplicates()
    logger.debug(f'Missing Events:{result_df}')
    result_df.to_csv('missing_events.csv', index=False)

    # sort reverse chronologically.
    se_df = se_df.sort_values(by='EventDate_sales', ascending=False)

    logger.debug(f'Raw Results:')
    logger.debug(se_df.shape)

    logger.debug('Subscriber totals after event merge: %s', se_df["Subscriber"].value_counts())

    logger.info(f'Sales and events merging. Execution Time: {_elapsed(start)}')

    return se_df
def final_processing_and_output(df, output_file, logger, processDonations):
    """
    Aggregate sales by order/event/ticket type, optionally adjust subscription
        donation amounts, and write full and anonymized output CSVs.
    """
    start = perf_counter()

    # Clean up Event Types:
    df['EventType'] = df['EventType'].str.title()
    logger.debug(f'Columns before final aggregation {df.columns}')
    logger.debug('Subscriber totals before final aggregation: %s', df["Subscriber"].value_counts())

    # aggregate individual records to the lowest unique set.
    # Assumes each order (Order Number, CreatedDate) can contain = multiple Events and quantities
    # Note: including OrderStatus and TicketStatus may be excessive. TODO is to prune that down.
    # groupby by columns. Can't include any columns to keep.
    groupby_cols = [
        'OrderNumber',
        'EventName',
        'EventInstance',
        'Allocation',
        'TicketType',
        'OrderStatus',
        'TicketStatus']

    # aggregation dict
    agg_dict = {
        'EventCapacity': 'max',
        'Quantity': 'sum',
        'ItemPrice': 'max',
        'AmountPaid': 'sum',
        'PriceLevel': 'max',
        'DonationAmount': 'max',
        'Total': 'sum',
        'DiscountTotal': 'sum',
        'PreDiscountTotal': 'sum',
        'UnitDiscount': 'sum',
        'Subscriber': 'max',
        'ChorusMember': 'max',
        'DuesTxn': 'max',
        'Student': 'max'}

    # and use latest record for any columns using the 'last' aggregation
    df = df.sort_values(by=['CreatedDate'])
    logger.info(f'starting sales aggregation')


    for col in df.columns:
        # Check if the column is not in the groupby_cols and agg_dict
        if col not in groupby_cols and col not in agg_dict:
            # If not, add it to the agg_dict using 'max'
            agg_dict[col] = 'last'

    # aggregate to the min unique set of records for our purposes.
    df = df.groupby(groupby_cols).agg(agg_dict).reset_index()

    logger.info(f'Sales aggregation complete')
    logger.debug('Subscriber totals after final aggregation: %s', df["Subscriber"].value_counts())

    logger.debug(df.shape)

    # Now calculate totals
    df['TicketTotal'] = df['ItemPrice'] * df['Quantity']

    logger.debug(df[df['EventId_sales'].isna()].groupby('EventId_sales').count())
    logger.debug(df[df['EventName_sales'].isna()])
    logger.debug(df.shape)

    # This deals with that Salesforce overcounts donations made as part of a subscription.
    # Create a boolean mask for 'Subscription' in the EventName
    #subscription_mask = df['EventName'].str.contains('Subscription')
    #subscription_mask = df['EventName_sales'].str.contains('subscription', case=False)
    subscription_mask = df['EventName_sales'].fillna('').str.contains('subscription', case=False)

    # Get the OrderNumbers with 'Subscription' in the EventName
    subscription_orders = df.loc[subscription_mask, 'OrderNumber'].unique()

    # Create a boolean mask for OrderNumber in subscription_orders
    #order_mask = df['OrderNumber'].isin(subscription_orders)
    order_mask = df['OrderNumber'].fillna('').isin(subscription_orders)

    # Get the indices of records with OrderNumber in subscription_orders and without 'Subscription' in EventName
    indices_to_zero = df.loc[order_mask & ~subscription_mask].index

    # Set DonationAmount to 0 for those records
    df.loc[indices_to_zero, 'DonationAmount'] = 0
    logger.debug(f'Donation handling complete.')
    logger.debug(f'raw columns: {df.columns}')

    # Need to Calculate discount amounts based on ItemPrice and PriceLevel. Can't trust Salesforce numbers.
    #df['NetTxn'] = df['TicketTotal'] + df['DonationAmount']

    # Strip to final set up columns for transaction file output.
    # We're dropping patron attribute information from the transaction file, but keeping them for patron details.
    output_cols = [
                   'AccountName','AccountId','ContactId',
                   #'FirstName', 'LastName', 'Address', 'City', 'State', 'ZIP', 'ContactEmail', 'Email',
                    'EventName','EventInstance','EventId', 'InstanceId', 'EventDate', 'EventVenue', 'EventCapacity', 'Season',
                   'Method', 'Origin', 'CreatedDate',
                   'OrderNumber', 'TicketStatus', 'OrderStatus', 'Allocation', 'TicketType','OrderSource',
                   'EventStatus', 'EventType', 'EventClass', 'EventGenre', 'EventSubGenre',
                   'Quantity', 'ItemPrice', 'TicketTotal', 'PriceLevel', 'AmountPaid', 'DonationName', 'DonationAmount',#'Total',
                   'DiscountCode', 'DiscountTotal', 'PreDiscountTotal', 'UnitDiscount', 'UnitDiscountType',
                   'ChorusMember','DuesTxn','Student','Subscriber','PatronStatus',
                   'Choral','Bach','Classical', 'Contemporary', 'Dance',
                    'Standard', 'Headliner', 'Mission', 'Local Favorite',
                    'Mechanics Hall', 'JMAC', 'Trinity Lutheran', 'The Hanover Theatre', 'Prior Center', 'Tuckerman Hall', 'First Unitarian','Curtis Hall', 'First Baptist',
                    #'Washburn', 'None', 'Curtis Hall', 'First Baptist', 'St Johns', 'Razzo Hall', 'Indian Ranch',  'Wesley United', 'Wamsworks', 'Brooks Hall', 'Shapiro Hall', 'Harvard Unitarian', 'St Paul'
    ]

    # write results to output file for only output columns.
    output_df = df[output_cols]
    logger.debug(f'Sales Output Columns:{output_df.columns}')

    logger.debug('Subscriber totals after final processing: %s', df["Subscriber"].value_counts())

    output_df.to_csv(output_file, index=False)
    logger.debug(f'full results written.')

    PII_columns = ['AccountName','DonationName','PatronStatus']
    anon_df = output_df.drop(PII_columns, axis=1)
    anon_df.to_csv('anon_' + output_file, index=False)
    logger.debug(f'PII safe Output written. {anon_df.columns}')

    logger.info(f'Final sales results written to file: {output_file}. Execution Time: {_elapsed(start)}')

    logger.debug(f'final processing return df columns:{df.columns}')

    return df # the full data frame is needed for Patron details

def add_regions(df, regions_file, logger):
    """Assign geographic region labels by merging on ZIP code from the regions reference file."""
    start = perf_counter()

    # Load the regions file with appropriate data types
    raw_regions_df = pd.read_csv(regions_file, dtype={'PHYSICAL ZIP': str, 'ZIP': str})

    # Normalize ZIP codes: take the first 5 digits and pad with leading zeros if necessary
    raw_regions_df['ZIP'] = raw_regions_df['ZIP'].str[:5].str.zfill(5)

    # Deduplicate based on ZIP and RegionAssignment to handle inconsistent ZIP-to-region mappings
    raw_regions_df = raw_regions_df[['ZIP', 'RegionAssignment']].drop_duplicates()

    # Handle potential duplicate ZIP codes by keeping the latest or most consistent assignment
    regions_df = (
        raw_regions_df
        .sort_values(by=['ZIP', 'RegionAssignment'])  # Sort if needed (customize as necessary)
        .drop_duplicates(subset='ZIP', keep='first')  # Drop duplicates, keeping the first assignment
    )

    logger.debug(regions_df)
    logger.debug(regions_df.shape)
    logger.debug(regions_df.drop_duplicates().shape)

    logger.debug(f'Regions pre merge: {df.shape}')

    # Merge the regions DataFrame with the original DataFrame on ZIP codes
    dr_df = df.merge(regions_df, on='ZIP', how='left')

    logger.debug(f'Regions post merge: {dr_df.shape}')
    logger.debug(f'Results with regions {dr_df.columns}')

    logger.info(f'Region Processing complete. Execution Time: {_elapsed(start)}')

    return dr_df


def load_anonymized_dataset(anon_data_file, logger):
    """Load anonymized sales/patron dataset from CSV."""
    start = perf_counter()

    # Load event manifest file and fix column names
    event_df = pd.read_csv(anon_data_file, low_memory=False)

    logger.info(f'Anon Dataset loaded. Execution Time: {_elapsed(start)}')

    return event_df

def add_key_events(df, logger):
    """For each AccountId extract first, second, penultimate, and latest event names and dates."""
    from datetime import timedelta
    from timeit import default_timer as timer
    import numpy as np
    import pandas as pd

    start = perf_counter()

    # Log the initial state of the EventDate column

    # Ensure EventDate is a datetime64 type
    if not pd.api.types.is_datetime64_any_dtype(df['EventDate']):
        df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')

    # Convert EventDate to date only (remove time component)
    df['EventDate'] = df['EventDate'].dt.date

    # Only consider actual events, not subscriptions.
    #todo: gotta fix this!!!!!!
    df = df[df['EventStatus'].isin(['Complete', 'Future'])]

    # Sort by AccountId and EventDate
    df = df.sort_values(by=['AccountId', 'EventDate'])

    # Ensure unique events per AccountId and EventDate
    df = df.drop_duplicates(subset=['AccountId', 'EventDate', 'EventName'])


    # Helper functions for second and penultimate events
    def get_second_event(series):
        return series.iloc[1] if len(series) > 1 else pd.NA

    def get_penultimate_event(series):
        return series.iloc[-2] if len(series) > 1 else pd.NA

    # Group by AccountId and extract key events
    logger.info("Grouping by AccountId and extracting key events...")
    first_event_df = df.groupby('AccountId')['EventName'].first().rename('FirstEvent')
    first_event_date = df.groupby('AccountId')['EventDate'].first().rename('FirstEventDate')
    second_event_df = df.groupby('AccountId')['EventName'].apply(get_second_event).rename('SecondEvent')
    second_event_date = df.groupby('AccountId')['EventDate'].apply(get_second_event).rename('SecondEventDate')
    penultimate_event_df = df.groupby('AccountId')['EventName'].apply(get_penultimate_event).rename('PenultimateEvent')
    penultimate_event_date = df.groupby('AccountId')['EventDate'].apply(get_penultimate_event).rename('PenultimateEventDate')
    latest_event_df = df.groupby('AccountId')['EventName'].last().rename('LatestEvent')
    latest_event_date = df.groupby('AccountId')['EventDate'].last().rename('LatestEventDate')

    # Combine the results into a single DataFrame
    key_events = pd.concat(
        [first_event_df, first_event_date,
         second_event_df, second_event_date,
         penultimate_event_df, penultimate_event_date,
         latest_event_df, latest_event_date],
        axis=1
    ).reset_index()

    # Log final dataframe sample

    # Log execution time
    logger.info(f'Key events added: Execution Time: {_elapsed(start)}')

    return key_events



def add_bulk_buyers(df, logger):
    """Flag BulkBuyer (>=12 tickets/event) and FrequentBulkBuyer (bulk purchases at >=4 events) accounts."""
    start = perf_counter()

    bulk_threshold = 12
    event_count_threshold = 4

    # Gather event quantities for all accounts
    grouped = df.groupby(['AccountId', 'EventName'])['Quantity'].sum().reset_index()

    # Identify accounts that bought more than the bulk threshold tickets for any event
    bulk_purchases = grouped[grouped['Quantity'] >= bulk_threshold]

    # Add a new column for BulkBuyers
    df['BulkBuyer'] = df['AccountId'].isin(bulk_purchases['AccountId'])

    # Count the number of bulk purchases for each account
    bulk_purchase_counts = bulk_purchases['AccountId'].value_counts()

    # Identify accounts with bulk purchases for more than the event count threshold
    frequent_bulk_buyers = bulk_purchase_counts[bulk_purchase_counts > event_count_threshold].index

    # Add a new column for FrequentBulkBuyers
    df['FrequentBulkBuyer'] = df['AccountId'].isin(frequent_bulk_buyers)

    logger.debug(f'Frequent Bulk buyers list: {frequent_bulk_buyers}')

    logger.info(f'Bulk Buyers added: Execution Time: {_elapsed(start)}')

    return df
def get_patron_details(df,
                       rates_df,
                       RFMScoreThreshold,
                       GetLatLong,
                       regions_file,
                       patrons_file,
                       anonymized,
                       new_threshold,
                       reengaged_threshold,
                       logger):
    """
    Orchestrate the full patron detail pipeline: filter transactions, calculate
        event preference scores (genre/class/venue), compute RFM/regularity metrics,
        assign segments, geocode addresses, and write full and anonymized patron CSVs.
    """

    try:
        df['EventDate'] = pd.to_datetime(df['EventDate'].copy(), errors='coerce')

        # Keep completed/future events and subscriptions; drop cancelled, test, etc.
        df = df[df['EventStatus'].isin(['Complete', 'Future', 'Subscription'])]

        initial_columns = ['AccountName', 'AccountId', 'ContactId', 'EventStatus', 'EventClass', 'EventVenue',
                           'FirstName', 'LastName', 'Address', 'City', 'State', 'ZIP', 'Email',
                           'Quantity', 'ItemPrice', 'CreatedDate', 'EventDate', 'EventName',
                           'PatronStatus', 'Subscriber', 'ChorusMember', 'DuesTxn', 'Season', 'Student',
                           'EventGenre', 'Choral', 'Bach', 'Classical', 'Contemporary', 'Dance',
                           'Headliner', 'Local Favorite', 'Standard', 'Mission',
                           'Mechanics Hall', 'Tuckerman Hall', 'JMAC', 'The Hanover Theatre']
        df = df[initial_columns]
        df = df.sort_values(['AccountId', 'CreatedDate'])
        df = df.rename(columns={'Season': 'LatestSeason'})

        key_events = add_key_events(df, logger)
        df = df.merge(key_events, on='AccountId', how='left')

        df = add_bulk_buyers(df, logger)

        # Genre/class/venue preference scores — aggregate full history before patron rollup
        genre_scores = mod.calculate_event_scores(df, logger, event_column='EventGenre')
        class_scores = mod.calculate_event_scores(df, logger, event_column='EventClass', use_tfidf=True)
        venue_scores = mod.calculate_event_scores(df, logger, event_column='EventVenue')
        df = df.merge(genre_scores, on='AccountId', how='left')
        df = df.merge(class_scores, on='AccountId', how='left')
        df = df.merge(venue_scores, on='AccountId', how='left')
        del genre_scores, class_scores, venue_scores

        logger.info('Calculating Patron metrics...')
        metrics_df = mod.calculate_patron_metrics(df.copy(), logger)

        # CLV score (not ready for primetime — needs to account for dormant patrons)
        metrics_df = calculate_CLV_score(metrics_df, logger)

        metrics_df['Segment'] = metrics_df.apply(mod.assign_segment, args=(new_threshold, reengaged_threshold), axis=1)

        # Prune to one row per patron (most recent transaction address)
        df = df.sort_values(by=['CreatedDate'])
        last_entry_df = df.drop_duplicates('AccountId', keep='last')

        keep_columns = ['AccountName', 'AccountId', 'ContactId', 'FirstName', 'LastName', 'Email', 'Address', 'City', 'State', 'ZIP',
                        'PatronStatus',
                        'ClassicalScore', 'ChoralScore', 'ContemporaryScore', 'DanceScore', 'BachScore',
                        'HeadlinerScore', 'StandardScore', 'Local FavoriteScore', 'MissionScore',
                        'Mechanics HallScore', 'The Hanover TheatreScore', 'Tuckerman HallScore', 'JMACScore',
                        'PreferredEventGenre', 'EventGenreTopScore', 'EventGenreStrength', 'EventGenreEntropy',
                        'PreferredEventVenue', 'EventVenueTopScore', 'EventVenueStrength', 'EventVenueEntropy',
                        'PreferredEventClass', 'EventClassTopScore', 'EventClassStrength', 'EventClassEntropy',
                        'FirstEvent', 'FirstEventDate', 'SecondEvent', 'SecondEventDate',
                        'PenultimateEvent', 'PenultimateEventDate', 'LatestEvent', 'LatestEventDate', 'LatestSeason']
        last_entry_df = last_entry_df[keep_columns]

        df = last_entry_df.merge(metrics_df, on='AccountId', how='left').sort_values(
            by=['MonetaryScore', 'FrequencyScore', 'RecencyScore'], ascending=False)

        # Convert days to months
        for col in ['DaysFromFirstEvent', 'DaysToReturn', 'DaysFromPenultimateEvent']:
            df[col] = df[col] / 30.4
            df.rename(columns={col: col.replace('Days', 'Months')}, inplace=True)
        df['Recency'] = df['Recency'] / 30.4
        df.rename(columns={'Recency': 'Recency (Months)'}, inplace=True)

        # Write anonymized output (always)
        anon_df = df.drop(columns=['AccountName', 'FirstName', 'LastName', 'Email', 'Address', 'City', 'State'])
        anon_df.to_csv('anon_' + patrons_file, index=False)
        logger.info(f'Anon Patron results written to file: anon_{patrons_file}')

        if not anonymized:
            df = state_and_city_processing(df, logger)
            df = address_and_ZIP_processing(df, logger)
            df = add_regions(df, regions_file, logger)

            # Carry forward existing Lat/Long from previous run to avoid re-geocoding
            orig_df = pd.read_csv(patrons_file, low_memory=False)
            orig_df = orig_df.sort_values(['AccountName', 'Recency (Months)']).groupby('AccountName').first().reset_index()
            df = df.merge(orig_df[['AccountName', 'Latitude', 'Longitude', 'ZIP+4']], on='AccountName', how='left').drop_duplicates()

            missing = df[(df['Latitude'].isna() | df['Longitude'].isna()) & (df['RFMScore'] > RFMScoreThreshold)]
            count_missing_before = missing['AccountName'].nunique()

            if GetLatLong:
                logger.info('Getting any new Lat/Long data...')
                df = df.apply(update_geocode_info, args=(RFMScoreThreshold, logger), axis=1)
            else:
                logger.info('Bypassing Lat/Long...')

            missing_after = df[(df['Latitude'].isna() | df['Longitude'].isna()) &
                               (df['RFMScore'] > RFMScoreThreshold)][['AccountName', 'Address', 'City', 'State', 'ZIP', 'RFMScore', 'Recency (Months)']]
            count_missing_after = missing_after['AccountName'].nunique()
            missing_after.to_csv('bad_addresses.csv', index=False)
            logger.info(f'{count_missing_after} contacts missing Lat/Long (bad addresses); {count_missing_before - count_missing_after} new geocoded.')

            output_cols = ['AccountName', 'ContactId', 'Segment', 'RFMScore', 'Lifespan', 'LatestSeason', 'RegionAssignment',
                           'Recency (Months)', 'Frequency', 'AYM', 'GrowthScore', 'Regularity', 'Monetary',
                           'RecencyScore', 'FrequencyScore', 'MonetaryScore', 'CLV_Score',
                           'PreferredEventGenre', 'EventGenreTopScore', 'EventGenreStrength', 'EventGenreEntropy',
                           'PreferredEventVenue', 'EventVenueTopScore', 'EventVenueStrength', 'EventVenueEntropy',
                           'PreferredEventClass', 'EventClassTopScore', 'EventClassStrength', 'EventClassEntropy',
                           'ClassicalScore', 'ChoralScore', 'ContemporaryScore', 'DanceScore', 'BachScore',
                           'HeadlinerScore', 'StandardScore', 'Local FavoriteScore', 'MissionScore',
                           'Mechanics HallScore', 'The Hanover TheatreScore', 'Tuckerman HallScore', 'JMACScore',
                           'PatronStatus', 'Subscriber', 'ChorusMember', 'DuesTxn', 'FrequentBulkBuyer', 'Student',
                           'MonthsFromFirstEvent', 'MonthsToReturn', 'RecentEventYearsGap',
                           'FirstEvent', 'FirstEventDate', 'SecondEvent', 'SecondEventDate',
                           'PenultimateEvent', 'PenultimateEventDate', 'LatestEvent', 'LatestEventDate',
                           'FirstName', 'LastName', 'Email', 'Address', 'City', 'State', 'ZIP',
                           'Latitude', 'Longitude', 'ZIP+4', 'AccountId']

            df[output_cols].to_csv(patrons_file, index=False)
            logger.info(f'Full Patron results written to file: {patrons_file}')

            summary_rename_map = {
                'AccountName': 'Account Name', 'ContactId': 'Contact ID',
                'Segment': 'Customer Segment', 'RFMScore': 'RFM Score',
                'Lifespan': 'Customer Lifespan', 'LatestSeason': 'Most Recent Season',
                'RegionAssignment': 'Geo Region', 'Recency (Months)': 'Recency (Months)',
                'Frequency': 'Frequency', 'AYM': 'Average Yearly Monetary',
                'GrowthScore': 'Growth Score', 'Regularity': 'Regularity',
                'Monetary': 'Total Monetary', 'RecencyScore': 'Recency Score',
                'FrequencyScore': 'Frequency Score', 'MonetaryScore': 'Monetary Score',
                'PreferredEventGenre': 'Favorite Genre', 'EventGenreStrength': 'Genre Strength',
                'PreferredEventVenue': 'Favorite Venue', 'EventVenueStrength': 'Venue Strength',
                'PreferredEventClass': 'Favorite Class', 'EventClassStrength': 'Class Strength',
                'PatronStatus': 'Patron Status', 'Subscriber': 'Is Subscriber',
                'ChorusMember': 'In Chorus', 'DuesTxn': 'Dues Txn',
                'FrequentBulkBuyer': 'Frequent Bulk Buyer', 'Student': 'Is Student',
                'MonthsFromFirstEvent': 'Months Since First Event', 'MonthsToReturn': 'Months to Return',
                'RecentEventYearsGap': 'Recent Event Gap (Years)',
                'FirstEvent': 'First Event', 'FirstEventDate': 'First Event Date',
                'SecondEvent': 'Second Event', 'SecondEventDate': 'Second Event Date',
                'PenultimateEvent': 'Second-to-Last Event', 'PenultimateEventDate': 'Second-to-Last Event Date',
                'LatestEvent': 'Most Recent Event', 'LatestEventDate': 'Most Recent Event Date',
                'FirstName': 'First Name', 'LastName': 'Last Name', 'Email': 'Email Address',
                'Address': 'Address', 'City': 'City', 'State': 'State', 'ZIP': 'ZIP Code',
                'Latitude': 'Latitude', 'Longitude': 'Longitude', 'AccountId': 'Account ID',
            }
            summary_df = df[list(summary_rename_map.keys())].rename(columns=summary_rename_map)
            summary_output_file = 'summary_' + patrons_file.replace('.csv', '.xlsx')
            summary_df.to_excel(summary_output_file, index=False)

            return df

    except PermissionError:
        print('The output file is already open.')
    except FileNotFoundError:
        print('The output file was not found. Please check the file path.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

# General functions
def safe_divide(x, y):
    """Element-wise division that replaces inf and NaN results with 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(x, y)
        result[~np.isfinite(result)] = 0  # Set NaN, inf, -inf to 0
    return result

def generate_identifier(account_name):
    """Return a stable SHA-256 hex digest of account_name for use as an anonymized AccountId."""
    # Use SHA-256 hash function
    return hashlib.sha256(account_name.encode()).hexdigest()
def plot_RFM(df, logger):
    """3-D scatter plot of Recency, Frequency, and Monetary values."""
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    xs = df['Recency']
    ys = df['Frequency']
    zs = df['Monetary']

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
def plot_3D_scatter(xs,x_label, ys, y_label,zs, z_label, logger):
    """Generic 3-D scatter plot with configurable axis labels."""
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
def plot_2D_scatter(x,x_label, y, y_label, logger):
    """2-D scatter plot with configurable axis labels and a descriptive title."""
    plt.figure(figsize=(20, 20))
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Distribution of {x_label} and {y_label}')
    plt.show()
def get_geocode_info(address):
    """Call Google Geocoding API for address; return location dict or None on failure."""

    google_api_key = 'AIzaSyAC4jkZD-p7bkor1InDTyw2Q2ULXK23yLw'
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': address, 'key': google_api_key}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

        res_json = response.json()

        if 'results' in res_json and len(res_json['results']) > 0:
            return res_json['results'][0]['geometry']['location']
        else:
            return None

    except RequestException as e:
        print(f"Error during requests to {base_url}: {str(e)}")
        return None
    except ValueError as e:
        print(f"Value error: {str(e)}")
        return None
    except KeyError as e:
        print(f"Key error: {str(e)}")
        return None
def update_geocode_info(df, RFMScoreThreshold, logger):
    """Update lat/long for records above RFMScoreThreshold that are missing geo coordinates."""
    # Check if the latitude and longitude exist and if conditions are met
    if df['RFMScore'] > RFMScoreThreshold and pd.isnull(df['Latitude']) and pd.isnull(df['Longitude']):
        # Additional checks for valid Address, City, State, and ZIP
        if pd.notnull(df['Address']) and pd.notnull(df['City']) and pd.notnull(df['State']) and pd.notnull(df['ZIP']):
            # Call get_geocode_info() and update latitude and longitude
            geocode_info = get_geocode_info(f"{df['Address']}, {df['City']}, {df['State']}, {df['ZIP']}")
            if geocode_info is not None:
                #start = perf_counter()
                df['Latitude'] = geocode_info['lat']
                df['Longitude'] = geocode_info['lng']

    return df
def get_zip_plus4(street, city, state, zipcode):
    """Call USPS API to retrieve the ZIP+4 extension for a given address."""
    base_url = 'http://production.shippingapis.com/ShippingAPI.dll'
    params = {
        'API': 'ZipCodeLookup',
        'XML': f'<ZipCodeLookupRequest USERID="usps_userid">'
               f'<Address><Address1></Address1><Address2>{street}</Address2>'
               f'<City>{city}</City><State>{state}</State>'
               f'<Zip5>{zipcode}</Zip5><Zip4></Zip4></Address>'
               f'</ZipCodeLookupRequest>'
    }
    response = requests.get(base_url, params=params)
    root = ET.fromstring(response.text)
    address_element = root.find('Address')
    if address_element is not None:
        zip4_element = address_element.find('Zip4')
        if zip4_element is not None:
            return zip4_element.text
    return None
def update_zip_plus4_info(df, RFMScoreThreshold):
    """Add ZIP+4 extension for records above RFMScoreThreshold that are missing it."""
    # Check if ZIP+4 exists and if conditions are met
    if df['RFMScore'] > RFMScoreThreshold and pd.isnull(df['ZIP+4']):
        # Call get_zip_plus4
        zip_plus4 = get_zip_plus4(df['Address'], df['City'], df['State'], df['ZIP'])
        # If get_zip_plus4 returns a value, assign it to df['ZIP+4']
        if zip_plus4 is not None:
            df['ZIP+4'] = df['ZIP'] + '-' + zip_plus4
    return df

import pandas as pd

import pandas as pd

def calculate_retention_and_churn(df, logger=None):
    """
    Calculate Year-over-Year (YoY) and 3-Year cumulative retention and churn rates for patrons.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with at least the following columns:
        - 'AccountId': Unique identifier for each patron
        - 'EventDate': Date of the patron's activity or attendance

    logger : logging.Logger, optional
        Logger instance for status logging.

    Returns:
    --------
    pd.DataFrame
        A DataFrame indexed by FiscalYear, containing:
        - Total Patrons Previous Year
        - New Patrons Previous Year
        - Retained New Patrons
        - Retained All Patrons
        - Churned Existing Patrons
        - New Patron Retention & Churn Rates
        - 1-Year (YoY) Retention & Churn Rates
        - 3-Year Retention & Churn Rates (excluding recent joiners)
    """

    # Convert to datetime and assign fiscal year
    df['EventDate'] = pd.to_datetime(df['EventDate'])
    df['FiscalYear'] = df['EventDate'].dt.year + (df['EventDate'].dt.month >= 7)
    df['FirstFiscalYear'] = df.groupby('AccountId')['FiscalYear'].transform('min')

    retention_data = []
    fiscal_years = sorted(df['FiscalYear'].unique())

    for i in range(1, len(fiscal_years)):
        current_year = fiscal_years[i]
        prev_year = fiscal_years[i - 1]

        #if logger:
        #    logger.info(f"Calculating retention for fiscal year {current_year}")

        # Slice patrons
        patrons_prev = df[df['FiscalYear'] == prev_year]['AccountId'].unique()
        new_prev = df[df['FirstFiscalYear'] == prev_year]['AccountId'].unique()

        # New Patron Retention
        retained_new = df[(df['AccountId'].isin(new_prev)) &
                          (df['FiscalYear'] == current_year)]['AccountId'].nunique()
        new_retention_rate = retained_new / len(new_prev) if len(new_prev) > 0 else None
        new_churn_rate = 1 - new_retention_rate if new_retention_rate is not None else None

        # Overall YoY Retention
        retained_all = df[(df['AccountId'].isin(patrons_prev)) &
                          (df['FiscalYear'] == current_year)]['AccountId'].nunique()
        overall_retention_rate = retained_all / len(patrons_prev) if len(patrons_prev) > 0 else None
        overall_churn_rate = 1 - overall_retention_rate if overall_retention_rate is not None else None

        # Existing Patron Churn (those not new last year)
        existing_prev = list(set(patrons_prev) - set(new_prev))
        retained_existing = df[(df['AccountId'].isin(existing_prev)) &
                               (df['FiscalYear'] == current_year)]['AccountId'].nunique()
        churned_existing = len(existing_prev) - retained_existing
        churn_rate_existing = churned_existing / len(existing_prev) if len(existing_prev) > 0 else None

        # 3-Year Retention
        if i >= 3:
            # Exclude recent joiners: only include patrons with FirstFiscalYear <= current_year - 2
            eligible_3y_patrons = df[df['FirstFiscalYear'] <= current_year - 2]['AccountId'].unique()
            recent_years = [current_year - 2, current_year - 1, current_year]

            patrons_3y = set(df[(df['AccountId'].isin(eligible_3y_patrons)) &
                                (df['FiscalYear'].isin(recent_years))]['AccountId'])

            retained_3y = set(df[(df['AccountId'].isin(patrons_3y)) &
                                 (df['FiscalYear'] == current_year)]['AccountId'])

            retained_count = len(retained_3y)
            total_3y = len(patrons_3y)
            retention_3y = retained_count / total_3y if total_3y > 0 else None
            churn_3y = 1 - retention_3y if retention_3y is not None else None
        else:
            total_3y = None
            retained_count = None
            retention_3y = None
            churn_3y = None

        # Append results
        retention_data.append({
            'FiscalYear': current_year,
            'Total Patrons Previous Year': len(patrons_prev),
            'New Patrons Previous Year': len(new_prev),
            'Retained New Patrons': retained_new,
            'New Patron Retention Rate': new_retention_rate,
            'New Patron Churn Rate': new_churn_rate,
            'Retained All Patrons': retained_all,
            '1-Year Retention Rate': overall_retention_rate,
            '1-Year Churn Rate': overall_churn_rate,
            'Churned Existing Patrons': churned_existing,
            'Existing Patron Churn Rate': churn_rate_existing,
            'Three Year Patrons': total_3y,
            'Three Year Retained Patrons': retained_count,
            '3-Year Retention Rate': retention_3y,
            '3-Year Churn Rate': churn_3y
        })

    return pd.DataFrame(retention_data).set_index('FiscalYear')



# Here is the corrected version of the provided function:

def calculate_CLV_score(df, logger):
    """
    This function calculates CLV scores and ranks based on Recency, Frequency, and Monetary scores.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing Recency, Frequency, and Monetary columns.
    churn_rates_df (pandas.DataFrame): (Placeholder) The DataFrame containing churn rates (not used in this implementation).
    logger (logging.Logger): Logger object for logging information.

    Returns:
    pandas.DataFrame: The DataFrame with calculated CLV scores and ranks.
    """
    # Define the weights for Recency, Frequency, and Monetary
    weights = {'Recency': 0.2, 'Frequency': 0.3, 'Monetary': 0.5}

    # Calculate CLV Score based on the weighted average
    df['CLV_Score'] = (weights['Recency'] * df['RecencyScore'] +
                       weights['Frequency'] * df['FrequencyScore'] +
                       weights['Monetary'] * df['MonetaryScore'])

    """
    # Assign CLV rank based on the score thresholds
    df['CLV_Rank'] = np.select(
        [
            df['CLV_Score'] >= 4.5,
            df['CLV_Score'] >= 3.5,
            df['CLV_Score'] >= 2.5,
            df['CLV_Score'] >= 1.5,
            df['CLV_Score'] < 1.5
        ],
        [5,4,3,2,1]
    )
    """
    logger.info(f"CLV score calculation complete")
    return df

# This corrected version includes fixes such as properly closing the 'df['RecencyScore']' bracket and removing the duplicate 'return df' statement.
