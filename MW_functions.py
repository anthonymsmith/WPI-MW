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
import Model_functions as mod
from datetime import datetime, timedelta
from timeit import default_timer as timer
import hashlib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import requests
from requests.exceptions import RequestException

# Processing functions
"""
Function: load_event_manifest

Description:
    This function loads and processes an event manifest file from an Excel sheet. 
    It performs initial cleaning, such as removing test events, fixing date formats, filling missing values, 
    and sorting the events by date, name, and instance. The function prepares the event data for further analysis 
    and logs the process along with execution time.

Parameters:
    manifest_file (str): The file path to the event manifest Excel file.
    logger (logging.Logger): A logger object for logging debug information and execution time.

Process:
    1. **Load the Event Manifest**: Loads the event manifest data from the specified Excel file.
    2. **Remove Test Events**: Filters out any rows where 'EventName' contains variations of the word "test" (case-insensitive).
    3. **Fix Dates**: Converts the 'EventDate' column to a simple date format.
    4. **Fill Missing Values**: Fills missing values in the columns 'EventGenre', 'EventClass', 'EventStatus', 'EventSubGenre', and 'EventVenue' with the placeholder 'None', and fills missing 'EventCapacity' values with 1.
    5. **Sort the Data**: Sorts the DataFrame by 'EventDate', 'EventName', and 'EventInstance' in descending order.
    6. **Logging and Execution Time**: Logs the shape of the DataFrame and records the execution time.

Returns:
    pd.DataFrame: The cleaned and processed DataFrame containing event data.
"""
def load_event_manifest(manifest_file, logger):
    start = timer()

    # Load event manifest file and fix column names
    event_df = pd.read_excel(manifest_file)

    # Fix dates
    event_df['EventDate'] = pd.to_datetime(event_df['EventDate'].copy()).dt.date

    # Remove test events
    event_df = event_df[event_df['EventName'].str.contains('test|TEST|Test', case=False) == False]
    logger.debug(f'Event shape {event_df.shape}')

    # Fill missing values
    columns_to_fill = ['EventGenre', 'EventClass', 'EventStatus', 'EventSubGenre', 'EventVenue']
    event_df[columns_to_fill] = event_df[columns_to_fill].fillna('None')
    event_df['EventCapacity'].fillna(1, inplace=True)

    # Sort dataframe
    event_df = event_df.sort_values(by=['EventDate', 'EventName', 'EventInstance'], ascending=False)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Events loaded. Execution Time: {formatted_timing}')

    return event_df
"""
Function: add_PnL_data

Description:
    This function merges event data with profit and loss (PnL) data from a QuickBooks export Excel file. 
    It processes financial information such as public support, sponsorship, ticket sales, and expenses for events, 
    calculating key financial ratios. The result is a cleaned and enriched event dataset that includes financial metrics. 
    The processed data is saved to a specified CSV file.

Parameters:
    event_df (pd.DataFrame): A DataFrame containing event data with columns like 'EventId' and 'EventDate'.
    Pnl_file (str): The file path to the Excel file containing PnL data.
    PnLProcessed_file (str): The file path where the processed PnL data should be saved as a CSV.
    logger (logging.Logger): A logger object for logging debug information and execution time.

Process:
    1. **Load and Rename PnL Columns**: Reads the PnL data from the Excel file and renames the columns to more readable names for easier processing.
    2. **Merge PnL and Event Data**: Merges the PnL data with the event data on the 'EventId' column.
    3. **Clean and Organize Data**:
        - Drops unnecessary columns from the merged DataFrame.
        - Groups by 'EventId' to keep only the first event name and prevent duplicates.
    4. **Convert Numeric Data**: Converts columns related to financials to numeric types and fills missing or invalid values with 0.
    5. **Calculate Financial Ratios**:
        - Calculates the revenue-to-expense ratio and the percentage of ticket sales plan achieved.
        - Ensures division errors (e.g., division by zero) are handled by replacing invalid values with 0.
    6. **Save Processed Data**: Saves the processed data, including PnL and event metrics, to the specified CSV file.
    7. **Logging and Execution Time**: Logs the process and records the execution time.

Returns:
    pd.DataFrame: The original event DataFrame (unmodified), though the processed PnL data is saved to the specified CSV file.
"""
def add_PnL_data(event_df, Pnl_file, PnLProcessed_file, logger):
    start = timer()

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
    res_df['EventPctSalesPlan'].replace([np.inf, -np.inf,np.nan], 0, inplace=True)

    cols = ['EventName', 'EventDate', 'EventVenue', 'EventPublicSupport', 'EventSponsor', 'EventTickets',
            'EventAdvertising', 'EventArtistFees', 'EventVenueFees', 'EventHospitality', 'EventG/A','EventTotalSponsorship',
            'EventRevenue', 'EventExpense', 'EventProfit', 'EventSalesPlan', 'EventRev/Expense', 'EventPctSalesPlan', 'EventCapacity', 'EventId',
            'Season',
            'EventStatus', 'EventType', 'EventClass', 'EventGenre', 'EventSubGenre']

    res_df = res_df[cols]

    # Save the processed data
    res_df.to_csv(PnLProcessed_file, index=False)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'PnL data written to {PnLProcessed_file}. Execution Time: {formatted_timing}')

    return event_df
"""
Function: load_sales_file

Description:
    This function loads and processes a raw sales data file. It renames columns from their original Salesforce naming conventions to more readable names, filters the data to retain only recent sales records (based on a specified number of years), and prepares the data for further analysis. The function also handles date conversion and logs the shape of the dataset before and after pruning older records.

Parameters:
    sales_file (str): The file path to the sales data CSV file.
    yearsOfData (int): The number of years of data to retain. Rows older than this are pruned.
    logger (logging.Logger): A logger object for logging debug information and execution time.

Process:
    1. **Load the Sales Data**: Reads the sales data from a CSV file using `pd.read_csv`, with `latin1` encoding and low memory optimization.
    2. **Rename Columns**: Renames Salesforce-specific column names to more readable names for easier manipulation.
    3. **Convert Dates**: Converts the 'CreatedDate' column to a proper datetime format.
    4. **Prune Older Records**: Filters the DataFrame to retain only sales records within the last `yearsOfData` years.
    5. **Logging**: Logs the initial and pruned size of the DataFrame, as well as the execution time.

Returns:
    pd.DataFrame: The processed DataFrame containing the sales data with renamed columns and only recent records.
"""
def load_sales_file(sales_file, yearsOfData, logger):
    start = timer()

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
        'Contact ID': 'ContactId',
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
        'Ticket Order: Email': 'OrderEmail',
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
    logger.info(f'Starting sales size: {sales_df.shape}')
    beginning_date = datetime.now() - timedelta(days=365*yearsOfData)
    sales_df = sales_df[sales_df['CreatedDate'] > beginning_date]
    logger.info(f'Pruned sales size: {sales_df.shape}')

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Sales loaded. Execution Time: {formatted_timing}')

    logger.debug(sales_df.shape)
    return sales_df
"""
Function: sales_initial_prep

Description:
    This function performs initial preprocessing of the sales data, including cleaning up columns, 
    filling missing values, and removing unnecessary or test records. It ensures that numeric and non-numeric columns 
    have appropriate default values, formats dates, and prepares the DataFrame for further analysis.

Parameters:
    df (pd.DataFrame): A DataFrame containing sales data, with columns like 'TicketStatus', 'ItemPrice', 'EventDate_sales', 'AccountName', and more.
    Account_file (str): The path to a CSV file containing account information, which can be used to merge Account IDs (this part is currently commented out).
    logger (logging.Logger): A logger object for logging debug information and execution time.

Process:
    1. **Remove Deleted Tickets**: Filters out records where 'TicketStatus' is marked as 'Deleted', which represent placeholders or canceled orders.
    2. **Drop Redundant Columns**: Removes unnecessary columns from the sales data.
    3. **Handle Missing Numeric Values**:
        - Converts specific columns to numeric and fills missing values with default values (e.g., 0 for price and quantity).
    4. **Handle Missing Non-Numeric Values**:
        - Fills missing values for non-numeric columns (e.g., 'DonationName', 'DiscountCode') with placeholder values.
    5. **Generate Account Names**:
        - If 'AccountName' is missing, fills it with a generated value based on 'FirstName' and 'LastName'.
    6. **Date Conversion**: Converts the 'EventDate_sales' column to a simple date format.
    7. **Remove Test Events**: Filters out test events and instances based on their names.
    8. **Logging and Execution Time**: Logs the initial shape of the DataFrame, changes made during processing, and the final shape, as well as the total execution time.

Returns:
    pd.DataFrame: The cleaned and preprocessed sales DataFrame, ready for further analysis.
"""
def sales_initial_prep(df, Account_file, logger):
    start = timer()
    logger.debug(df.shape)
    logger.debug(f'Sales pre-prep columns: {df.columns}')

    # Convert date to a simple date format
    df['EventDate_sales'] = pd.to_datetime(df['EventDate_sales'].copy()).dt.date

    df = df.copy()

    #Remove deleted tickets, as these were placeholders for subscriptions or canceled orders.
    df = df[df['TicketStatus'] != 'Deleted']

    # TODO: These AccountName steps are done at transaction-level
    #  only because we need an AccountName to generate the ContactId.
    #  We should get ContactId from saleforce and all of this could be
    #  then moved to Patron Detail Processing.
    # set any null AccountNames to walk up
    df['AccountName'] = df['AccountName'].fillna('Walk Up Sales')

    # for debug only and expensive at transaction-level.
    #null_accountnames = df[df['AccountName'].isnull()]
    #logger.debug(f'List of records missing account names: {null_accountnames}')

    # add ContactId on the assumption that this function generates a unique, repeatable ID.
    # TODO When the salesforce ContactId problem gets solved, can delete this.
    df['ContactId'] = df['AccountName'].apply(generate_identifier)

    # Remove redundant columns from Salesforce data file.
    redundant_columns = [
        'Ticketable Event: 18 Digit Event ID',
        'Ticketable Event: 18 Digit Event ID.1',
        'Ticketable Event: Ticketable Event ID.1',
        'Status.1',
        'Contact: Email.1'
    ]
    df.drop(redundant_columns, axis=1, inplace=True)

    # Define a dictionary for numeric columns and their fillna values
    numeric_fillna_dict = {
        'ItemPrice': 0,
        'Quantity': 0,
        'DonationAmount': 0,
        'DiscountTotal': 0,
        'UnitDiscount': 0,
        'PreDiscountTotal': 0,
        'PriceLevel': 0,
        'Total': 0
        }
    # Convert columns to numeric and fill NaN values
    for col, fill_value in numeric_fillna_dict.items():
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_value)

    # For non-numeric columns that may be Null. May need to add more, as needed.
    non_numeric_fillna_dict = {
        'DonationName': 'None',
        'DiscountCode': 'None',
        'UnitDiscountType': 'None'
    }
    # Fill NaN values for non-numeric columns
    df.fillna(non_numeric_fillna_dict, inplace=True)

    # Fill in null AccountNames with First Name + Last Name
    logger.debug(f"Number of missing AccountNames: {df['AccountName'].isna().sum()}")
    df.loc[df['AccountName'].isna(), 'AccountName'] = df['FirstName'] + ' ' + df['LastName'] + ' ' + '(generated)'

    #merge in AccountIds - this method assumes an AccountId file has been created. Salesforce doesn't do that, though.
    #acc_df = pd.read_csv(Account_file).rename(columns={'Account Name': 'AccountName','Account ID': 'AccountId'})
    #df = df.merge(acc_df,on='AccountName',how='left')

    # Clean up Event Names and Instances
    df['EventName_sales'] = df['EventName_sales'].astype(str)
    df['EventInstance_sales'] = df['EventInstance_sales'].astype(str)

    # Remove test events and instances
    test_event_mask = ~(df['EventName_sales'].str.contains(' test', case=False))
    df = df.loc[test_event_mask]
    test_instance_mask = ~(df['EventInstance_sales'].str.contains(' test', case=False))
    df = df.loc[test_instance_mask]
    logger.debug('Removed Test sales')
    logger.debug(df.head)
    logger.debug(df.shape)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Initial sales prep complete. Execution Time: {formatted_timing}')

    logger.debug(f'Sales post initial-prep columns: {df.columns}')

    return df
"""
Function: venue_and_attribute_processing

Description:
    This function processes venue and customer attributes in the sales DataFrame. 
    It cleans up venue names, adds chorus membership information, and flags transactions with specific attributes 
    such as chorus dues, student discounts, and subscriptions. The function also integrates an external chorus member 
    list to identify chorus members and updates relevant customer attributes.

Parameters:
    sales_df (pd.DataFrame): A DataFrame containing sales data with columns like 'VenueName_sales', 'AccountName', 'DiscountCode', and 'TicketType'.
    chorus_list_file (str): The file path to the CSV file containing the list of chorus members, which includes 'AccountName'.
    logger (logging.Logger): A logger object for logging debug information and execution time.

Process:
    1. **Venue Cleaning**:
        - Fills missing venue names with 'None'.
        - Standardizes venue names by replacing all variations of "Mechanics" with "Mechanics Hall".

    2. **Chorus Member Identification**:
        - Loads the chorus member list and creates a set of unique 'AccountName' values from it.
        - Adds a 'ChorusMember' column to the sales DataFrame by checking if the 'AccountName' exists in the chorus member list or has used a "Chorus Dues" discount code in any transaction.

    3. **Flagging Specific Attributes**:
        - Adds a 'DuesTxn' column to flag any transactions where the discount code contains "Chorus Dues".
        - Updates the 'ChorusMember' column to include accounts that have used a "Chorus Dues" coupon.
        - Adds a 'Student' column to flag transactions with a student discount based on 'TicketType'.
        - Adds a 'Subscriber' column to flag transactions involving subscriptions, based on the event name containing "Subscription".

    4. **Logging and Execution Time**:
        - Logs the initial shape and column information, as well as changes made to the DataFrame during processing.
        - Records the execution time for the entire process.

Returns:
    pd.DataFrame: The processed DataFrame with updated venue names and additional columns for customer attributes ('ChorusMember', 'DuesTxn', 'Student', 'Subscriber').
"""
def venue_and_attribute_processing(sales_df, chorus_list_file, logger):
    start = timer()
    logger.debug(f'Sales columns: {sales_df.columns}')

    # Clean up venue names
    sales_df['VenueName_sales'].fillna('None', inplace=True)
    sales_df.loc[sales_df['VenueName_sales'].str.contains("Mechanics"), 'VenueName_sales'] = 'Mechanics Hall'

    logger.debug(np.unique(sales_df.loc[sales_df['VenueName_sales'].str.contains("Mechanics")]['VenueName_sales']))
    logger.debug(sales_df.dtypes)

    # Load the chorus member list and drop missing AccountNames in one step
    chorus_df = pd.read_csv(chorus_list_file, usecols=['AccountName']).dropna()

    # Create a set of unique AccountNames
    chorus_members = set(chorus_df['AccountName'])

    # Add the 'ChorusMember' column to sales_df
    sales_df['ChorusMember'] = sales_df['AccountName'].isin(chorus_members)

    logger.debug(f'ChorusMember column added: {sales_df["ChorusMember"].head()}')
    logger.debug(f'sales_df shape: {sales_df.shape}')

    # Now add a field for any Accounts who bought a ticket with a DUES coupon
    sales_df['DuesTxn'] = sales_df['DiscountCode'].str.contains("Chorus Dues", na=False)

    # Chorus member is true if in the chorus member list or has ever used DUES for tickets.
    sales_df['ChorusMember'] = sales_df['ChorusMember'] | sales_df['DuesTxn']

    # Add Student field based on use of 'Student' discount code.
    sales_df['Student'] = sales_df['TicketType'].str.contains("Student", na=False)

    # Add Subscriber field based on subscription practice.
    sales_df['Subscriber'] = sales_df['EventName_sales'].str.contains("Subscription", na=False)

    logger.debug(sales_df.head)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Venue and attribute processing complete. Execution Time: {formatted_timing}')

    return sales_df
"""
Function: genre_counts

Description:
    This function processes event data to create genre-specific event counts for each account. 
    It filters events to include only 'Live' and 'Virtual' types, then calculates the number of unique events 
    attended by each account in each genre. The resulting genre counts are merged back into the original DataFrame.

Parameters:
    df (pd.DataFrame): A DataFrame containing event and account data, including columns such as 'ContactId', 'EventGenre', 'EventId', and 'EventType'.
    logger (logging.Logger): A logger object for logging debug information and execution time.

Process:
    1. **Data Filtering**:
        - Fills missing values in the 'EventGenre' column with a placeholder ('None').
        - Filters the DataFrame to include only 'Live' and 'Virtual' events, excluding subscriptions, test events, and other irrelevant event types.

    2. **Unique Event Counting**:
        - Removes duplicate events for each combination of 'ContactId' and 'EventGenre'.
        - Uses a pivot table to count the number of unique events in each genre for each account ('ContactId').

    3. **Merging Genre Data**:
        - Merges the original DataFrame with the genre-specific event counts.

    4. **Logging**:
        - Logs the shape of the DataFrame before and after merging, and records the execution time of the processing steps.

Returns:
    pd.DataFrame: The processed DataFrame with additional columns for each genre, representing the count of events attended by each account.
"""
def genre_counts(df, logger):
    start = timer()

    # Fill NaN values in 'EventGenre' with a placeholder
    df['EventGenre'].fillna('None', inplace=True)

    # only consider live or virtual events. Exclude suscriptions, test, etc. from genre counts.
    df1 = df[df['EventType'].isin(['Live', 'Virtual'])]

    # First remove duplicate events for each ContactId and EventGenre
    unique_events_df = df1.drop_duplicates(subset=['ContactId', 'EventGenre', 'EventId'])

    # Calculate the number of events of each genre for each ContactId, using a pivot_table.
    genre_df = unique_events_df.pivot_table(index='ContactId',
                                            columns='EventGenre',
                                            values='EventId',  # Assuming 'EventID' is the unique identifier for an event
                                            aggfunc='count',
                                            fill_value=0)

    genre_df.reset_index(inplace=True)

    # Merge the original DataFrame with the pivoted genre DataFrame
    logger.debug(f'Genre pre merge: {df.shape}')
    merged_df = df.merge(genre_df, on='ContactId', how='left')
    logger.debug(f'Genre post merge: {df.shape}')
    logger.debug(f'Sales columns: {df.columns}')
    logger.debug(f'genre columns: {genre_df.columns}')

    logger.debug(f'Sales with genre columns: {merged_df.columns}')
    logger.debug(df.head)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'genre counts complete. Execution Time: {formatted_timing}')

    return merged_df
"""
Function: state_and_city_processing

Description:
    This function standardizes and cleans the 'State' and 'City' fields in the sales DataFrame. It corrects common issues, 
    such as ZIP codes incorrectly placed in the 'State' field, fixes common state and city name errors, 
    and applies standardized formatting. Additionally, it ensures consistency in directional abbreviations (e.g., North to N) 
    and corrects specific known errors in city and state names.

Parameters:
    sales_df (pd.DataFrame): A DataFrame containing sales data with columns like 'State', 'City', and 'ZIP'.
    logger (logging.Logger): A logger object for logging debug information and execution time.

Process:
    1. **State Cleaning**:
        - Extracts and separates ZIP codes accidentally placed in the 'State' field.
        - Corrects common state typos, focusing mainly on 'MA' (Massachusetts).
        - Replaces any "Mass" with "MA" for consistency.
    
    2. **City Cleaning**:
        - Standardizes city names (e.g., title case, removes " MA" suffix, cleans punctuation).
        - Applies state-specific city corrections using regex patterns to fix common misspellings and abbreviations.
        - Replaces directional names (e.g., 'North' to 'N') and changes 'Centre' to 'Center'.

    3. **ZIP Code Alignment**:
        - Ensures that ZIP codes found in the 'State' field are corrected and moved to the correct column.

    4. **City-State Mapping**:
        - For certain cities known to belong to specific states (e.g., 'Worcester' belongs to 'MA'), it fills in missing state data based on city name.

    5. **Execution Time Logging**:
        - Logs the initial and final shape of the DataFrame and records the execution time of the processing steps.

Returns:
    pd.DataFrame: The processed DataFrame with cleaned and standardized 'State' and 'City' fields.
"""
def state_and_city_processing(sales_df, logger):
    start = timer()

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

    # clean up Cities
    # Define a dictionary for state-specific city name corrections
    city_corrections_by_state = {
        "MA": {
            r'Arlin': 'Arlington',
            r'Attl': 'Attleboro',
            r'Ashkand': 'Ashland',
            r'Aburn': 'Auburn',
            r'barre': 'Barre',
            r'lmont': 'Belmont',
            r'Bylston|Boylson|Biylston': 'Boylston',
            r'Baldwinville': 'Baldwinville',
            r'Charlt': 'Charlton',
            r'Chelms|Chelmsfp': 'Chelmsford',
            r'ouglas': 'Douglas',
            r'Fithburg': 'Fitchburg',
            r'Fisk': 'Fiskdale',
            r'Framingham|Framngham': 'Framingham',
            r'Gardiner': 'Gardner',
            r'Hold': 'Holden',
            r'Hopkin': 'Hopkinton',
            r'aster|ancaster': 'Lancaster',
            r'Lanesboro': 'Lanesborough',
            r'East Longmeadow': 'E Longmeadow',
            r'Flirence': 'Florence',
            r'Foxboro': 'Foxborough',
            r'Hub': 'Hubbardston',
            r'Jeffer': 'Jefferson',
            r'JP': 'Jamaica Plain',
            r'Leominister': 'Leominster',
            r'Lunenberg': 'Lunenburg',
            r'Middleboro': 'Middleborough',
            r'Marlboro|Marboro|Malb': 'Marlborough',
            r'Natix': 'Natick',
            r'Newton U': 'Newton Upper Falls',
            r'Newton H': 'Newton Highlands',
            r'Northboro|Northbrough|Nothbrough|Norhto|North Borough|01532': 'Northborough',
            r'Northbr': 'Northbridge',
            r'Princet': 'Princeton',
            r'1543|Rutalnd': 'Rutland',
            r'Sherb': 'Sherborn',
            r'Southboro|Southbrough': 'Southborough',
            r'Shrew|sbury': 'Shrewsbury',
            r'Stowe': 'Stow',
            r'Sturb': 'Sturbridge',
            r'Surtton|Suttom': 'Sutton',
            r'Uuton': 'Upton',
            r'Westboro|Weestborough|Wesborough|Westtbo|Westboorough|Westobor': 'Westborough',
            r'Westfield': 'Westfield',
            r'insville': 'Whitinsville',
            r'^Wor.*': 'Worcester'
        },
        "CT": {
            r'Haddam': 'E Haddam',
            r'Grosvenordale': 'N Grosvenordale',
            r'Pomfret': 'Pomfret',
            r'Claymont': 'Claymont',
            r'Storrs': 'Storrs'
        },
        "NY": {
            r'Brokln|Brookj|Broook|Brooklyn': 'Brooklyn',
            r'Ny': 'New York City'
        }
    }

    # Iterate over the dictionary and apply corrections for each state
    for state, corrections in city_corrections_by_state.items():
        mask = sales_df['State'] == state
        for pattern, replacement in corrections.items():
            sales_df.loc[mask, 'City'] = sales_df.loc[mask, 'City'].replace(pattern, replacement, regex=True)


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

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'State and city processing complete. Execution Time: {formatted_timing}')

    return sales_df
"""
Function: address_and_ZIP_processing

Description:
    This function standardizes and cleans the address and ZIP code fields in the sales DataFrame. 
    It formats addresses with title casing, applies abbreviation replacements for street types (e.g., 'Road' to 'Rd'), 
    and removes unwanted punctuation. Additionally, it corrects and cleans ZIP codes, ensuring proper formatting 
    (e.g., handling ZIP+4 codes and fixing incorrect or partial ZIP codes based on predefined mappings).

Parameters:
    sales_df (pd.DataFrame): A DataFrame containing sales data, with columns such as 'Address', 'City', and 'ZIP'.
    logger (logging.Logger): A logger object for logging the processing details and execution time.

Process:
    1. **Address Cleaning**:
        - Converts addresses to title case.
        - Applies replacements for common street abbreviations (e.g., 'Road' to 'Rd').
        - Removes unwanted punctuation (e.g., periods and commas) from addresses.
        - Normalizes PO Box abbreviations to uppercase.
    
    2. **ZIP Code Processing**:
        - Extracts the first five digits of ZIP codes and ensures they are zero-padded to five digits.
        - Cleans incorrect or invalid ZIP codes based on predefined mappings for specific cities.
    
    3. **Execution Time Logging**:
        - Logs the initial and final shape of the DataFrame and records the execution time of the processing steps.

Returns:
    pd.DataFrame: The processed DataFrame with cleaned and standardized 'Address' and 'ZIP' fields.
"""
def address_and_ZIP_processing(sales_df, logger):
    start = timer()
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
        '0162': '01602', '\(': '01609', '\)': '01602', '10608': '01608',
        '00880': '01609', "0'720": '01720', "02091": '02891',"01250": '01520',
        '91545': '01545', '01555': '01545', "\(016": '01601', '00756': '01545',
        '00162': '01602', '01533': '01532', '00158': '01581'
    }

    def clean_zip_codes(df, zip_column):
        for wrong_zip, correct_zip in zip_mappings.items():
            df[zip_column] = df[zip_column].str.replace(wrong_zip, correct_zip, regex=True)
        return df

    # Apply the clean_zip_codes function to the sales_df DataFrame
    sales_df = clean_zip_codes(sales_df, 'ZIP')

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'address & ZIP processing complete. Execution Time: {formatted_timing}')

    logger.debug(sales_df.shape)

    return sales_df
"""
Function: combine_sales_and_events

Description:
    This function merges sales data with event data based on matching 'EventId' and 'InstanceId' fields from both data sources. 
    It also identifies and logs any sales records that do not have corresponding event instances. 
    The result is a combined dataset of sales and event information, sorted in reverse chronological order by event date.

Parameters:
    sales_df (pd.DataFrame): A DataFrame containing sales data, with columns like 'EventId_sales', 'InstanceId_sales', and 'EventDate_sales'.
    event_df (pd.DataFrame): A DataFrame containing event data, with columns like 'EventId', 'InstanceId', and event details.
    logger (logging.Logger): A logger object for logging execution details and debug information.

Process:
    1. Logs the initial shape of both sales and event data.
    2. Merges the two datasets based on 'EventId' and 'InstanceId' using a left join, keeping all sales records.
    3. Logs the columns and shape of the merged dataset.
    4. Identifies any sales records that do not have matching event instances in the event data and logs these missing records.
    5. Saves the missing event instances to a CSV file ('missing_events.csv').
    6. Sorts the merged dataset in reverse chronological order by 'EventDate_sales'.
    7. Logs the final shape of the merged dataset and the total execution time.

Returns:
    pd.DataFrame: The merged DataFrame containing combined sales and event data, sorted by 'EventDate_sales'.
"""
def combine_sales_and_events(sales_df, event_df, logger):
    start = timer()

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

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Sales and events merging. Execution Time: {formatted_timing}')

    return se_df
"""
Function: final_processing_and_output

Description:
    This function performs the final processing and aggregation of sales and event data, calculating relevant metrics 
    like ticket totals and donation adjustments, and outputs the results to a CSV file. The function aggregates records 
    at the most granular level and optionally processes donations by handling subscription-related adjustments.

Parameters:
    df (pd.DataFrame): The DataFrame containing the processed sales and event data.
    output_file (str): The file path where the final processed data will be saved as a CSV file.
    logger (logging.Logger): A logger object for logging debug information and execution time.
    processDonations (bool): A flag indicating whether to adjust donation amounts based on subscriptions.

Process:
    1. **Event Type Cleanup**: Standardizes event type capitalization by converting strings to title case.
    2. **Data Aggregation**: 
        - Groups data by specific columns (like 'OrderNumber', 'EventName', 'EventInstance') and applies aggregation 
            functions (e.g., sum for quantities, max for prices).
        - Dynamically adds additional columns for aggregation by keeping the latest record where necessary.
    3. **Total Calculations**: Calculates ticket totals by multiplying item price by quantity.
    4. **Optional Donation Processing**: 
        - Adjusts donation amounts for orders containing subscriptions by setting donations to 0 for non-subscription items
            in the same order.
    5. **Final Cleanup**: Strips down the DataFrame to a final set of output columns and writes the results to a CSV file.

Returns:
    pd.DataFrame: The processed DataFrame containing aggregated and cleaned sales and event data.

Exceptions:
    - Handles any errors by logging debug information during processing.
"""
def final_processing_and_output(df, output_file, logger, processDonations):
    start = timer()

    # Clean up Event Types:
    df['EventType'] = df['EventType'].str.title()
    logger.debug(f'Columns before final aggregation {df.columns}')

    # aggregate individual records to the lowest unique set.
    # Assumes each order (Order Number, CreatedDate) can contain = multiple Events and quantities
    # Note: including OrderStatus and TicketStatus may be excessive. TODO is to prune that down.
    # groupby by columns. Can't include any columns to keep.
    groupby_cols = [
        'OrderNumber',
        'EventName',
        'EventInstance',
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
        'UnitDiscount': 'sum'}

    # and use latest record for any columns using the 'last' aggregation
    df = df.sort_values(by=['CreatedDate'])
    logger.debug(f'Sales aggregation created')

    for col in df.columns:
        # Check if the column is not in the groupby_cols and agg_dict
        if col not in groupby_cols and col not in agg_dict:
            # If not, add it to the agg_dict using 'first' chronologically
            agg_dict[col] = 'first'

    # aggregate to the min unique set of records for our purposes.
    df = df.groupby(groupby_cols).agg(agg_dict).reset_index()
    logger.debug(f'Sales aggregation complete')
    logger.debug(df.shape)

    # Now calculate totals
    df['TicketTotal'] = df['ItemPrice'] * df['Quantity']

    # logger.debug(df[dm_df['EventId_sales'].isna()].groupby('EventId_sales').count())
    # logger.debug(df[dm_df['EventName_sales'].isna()])
    logger.debug(df.shape)

    # Create a boolean mask for 'Subscription' in the EventName
    subscription_mask = df['EventName'].str.contains('Subscription')

    # Get the OrderNumbers with 'Subscription' in the EventName
    subscription_orders = df.loc[subscription_mask, 'OrderNumber'].unique()

    # Create a boolean mask for OrderNumber in subscription_orders
    order_mask = df['OrderNumber'].isin(subscription_orders)

    # Get the indices of records with OrderNumber in subscription_orders and without 'Subscription' in EventName
    indices_to_zero = df.loc[order_mask & ~subscription_mask].index

    # Set DonationAmount to 0 for those records
    df.loc[indices_to_zero, 'DonationAmount'] = 0
    logger.debug(f'Donation handling complete.')

    # TODO Calculate discount amounts based on ItemPrice and PriceLevel. Can't trust Salesforce numbers.

    #df['NetTxn'] = df['TicketTotal'] + df['DonationAmount']

    #Strip to final set up columns for transaction file output.
    # We're dropping patron attribute information from the transaction file, but keeping them for patron details.
    output_cols = [
                   'AccountName','ContactId',
                   #'FirstName', 'LastName', 'Address', 'City', 'State', 'ZIP', 'ContactEmail', 'OrderEmail',
                    'EventName','EventInstance','EventId', 'InstanceId', 'EventDate', 'EventVenue', 'EventCapacity', 'Season',
                   'PaymentMethod', 'Method', 'Origin', 'CreatedDate', 'EntryDate',
                   'OrderNumber', 'TicketStatus', 'OrderStatus', 'Allocation', 'TicketType','OrderSource',
                   'EventStatus', 'EventType', 'EventClass', 'EventGenre', 'EventSubGenre',
                   'Quantity', 'ItemPrice', 'TicketTotal','PriceLevel', 'AmountPaid', 'DonationName', 'DonationAmount','Total',
                   'DiscountCode', 'DiscountTotal', 'PreDiscountTotal', 'UnitDiscount', 'UnitDiscountType',
                   'ChorusMember','DuesTxn','Student','Subscriber',
                   'Choral','Brass','Classical', 'Contemporary', 'Dance'
                   ]

    # write results to output file for only output columns.
    output_df = df[output_cols]
    logger.debug(f'Sales Output Columns:{output_df.columns}')

    output_df.to_csv(output_file, index=False)
    logger.debug(f'full results written.')

    PII_columns = ['AccountName','DonationName']
    anon_df = output_df.drop(PII_columns, axis=1)
    anon_df.to_csv('anon_' + output_file, index=False)
    logger.debug(f'PII safe Output written. {anon_df.columns}')

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Final sales results written to file: {output_file}. Execution Time: {formatted_timing}')

    return df # the full data frame is needed for Patron details
"""
def region_processing(df, filename, logger):
    start = timer()

    # add in ZIP-Region Assignments
    raw_regions_df = pd.read_csv(filename, dtype={'PHYSICAL ZIP': str,'ZIP': str})

    # Extract the first 5 digits of the ZIP code and add leading zeros if necessary
    raw_regions_df['ZIP'] = raw_regions_df['ZIP'].str[:5].str.zfill(5)
    raw_regions_df = raw_regions_df[['ZIP','RegionAssignment']].drop_duplicates()
    #regions_df = raw_regions_df.groupby('ZIP',as_index=False).first()
    regions_df = raw_regions_df
    # strip ZIP+4
    # regions_df['ZIP'] = regions_df['ZIP'].astype(str)
    # regions_df['ZIP'] = regions_df['ZIP'].str[:5].astype(str)
    # regions_df['ZIP'] = regions_df['ZIP'].apply(lambda x: '{0:0>5}'.format(x))
    logger.debug(regions_df)
    logger.debug(regions_df.shape)
    logger.debug(regions_df.drop_duplicates().shape)

    # dm_df = dm_df.merge(regions_df, on='ZIP', how='left')
    logger.debug(f'regions pre merge: {df.shape}')

    dm_df = df.merge(regions_df, on='ZIP', how='left')

    logger.debug(f'regions post merge: {dm_df.shape}')
    logger.debug(f'Results With regions {dm_df.columns}')

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())

    return dm_df, formatted_timing
"""

def add_regions(df, regions_file, logger):
    start = timer()

    # add in ZIP-Region Assignments
    raw_regions_df = pd.read_csv(regions_file, dtype={'PHYSICAL ZIP': str,'ZIP': str})

    # Extract the first 5 digits of the ZIP code and add leading zeros if necessary
    raw_regions_df['ZIP'] = raw_regions_df['ZIP'].str[:5].str.zfill(5)
    raw_regions_df = raw_regions_df[['ZIP','RegionAssignment']].drop_duplicates()

    #regions_df = raw_regions_df.groupby('ZIP',as_index=False).first()
    regions_df = raw_regions_df
    # strip ZIP+4
    # regions_df['ZIP'] = regions_df['ZIP'].astype(str)
    # regions_df['ZIP'] = regions_df['ZIP'].str[:5].astype(str)
    # regions_df['ZIP'] = regions_df['ZIP'].apply(lambda x: '{0:0>5}'.format(x))

    logger.debug(regions_df)
    logger.debug(regions_df.shape)
    logger.debug(regions_df.drop_duplicates().shape)

    logger.debug(f'regions pre merge: {df.shape}')

    dr_df = df.merge(regions_df, on='ZIP', how='left')

    logger.debug(f'regions post merge: {dr_df.shape}')
    logger.debug(f'Results With regions {dr_df.columns}')

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Region Processing complete. Execution Time: {formatted_timing}')

    return dr_df


def load_anonymized_dataset(anon_data_file, logger):
    start = timer()

    # Load event manifest file and fix column names
    event_df = pd.read_csv(anon_data_file, low_memory=False)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Anon Dataset loaded. Execution Time: {formatted_timing}')

    return event_df
6
def add_first_latest_events(df, logger):
    start = timer()
    df = df.sort_values(by=['ContactId','EventDate'])

    # Get the first and latest event for each 'ContactId'
    first_event_df = df.groupby('ContactId')['EventName'].first().rename('FirstEvent')
    first_event_date = df.groupby('ContactId')['EventDate'].first().rename('FirstEventDate')
    latest_event_df = df.groupby('ContactId')['EventName'].last().rename('LatestEvent')
    latest_event_date = df.groupby('ContactId')['EventDate'].last().rename('LatestEventDate')


    # For the penultimate event, we work with a grouped object and then apply a custom function to get the penultimate values
    def get_penultimate(series):
        if len(series) > 1:
            return series.iloc[-2]
        return series.iloc[0]

    penultimate_event_df = df.groupby('ContactId')['EventName'].apply(get_penultimate).rename('PenultimateEvent')
    penultimate_event_date = df.groupby('ContactId')['EventDate'].apply(get_penultimate).rename('PenultimateEventDate')

    # Concatenate the first and latest event dataframes along the column axis
    first_latest_events = pd.concat([first_event_df, first_event_date,
                                     latest_event_df,latest_event_date,
                                     penultimate_event_df,penultimate_event_date],
                                    axis=1).reset_index()
    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'first/last events added: Execution Time: {formatted_timing}')

    return first_latest_events
"""
Function: add_bulk_buyers

Description:
    This function identifies and flags "bulk buyers" and "frequent bulk buyers" based on the number of tickets purchased for events. It adds two new columns, 'BulkBuyer' and 'FrequentBulkBuyer', to the DataFrame, indicating accounts that have made large purchases and accounts that frequently make large purchases across multiple events.

Parameters:
    df (pd.DataFrame): A DataFrame containing event data, including columns like 'ContactId', 'EventName', and 'Quantity'.
    logger (logging.Logger): A logger object for logging debug information and execution time.

Process:
    1. Defines thresholds:
        - `bulk_threshold`: The minimum number of tickets purchased in a single event to qualify as a bulk purchase.
        - `event_count_threshold`: The minimum number of events with bulk purchases required to qualify as a frequent bulk buyer.
    2. Groups the data by 'ContactId' and 'EventName', summing the quantities of tickets purchased.
    3. Identifies accounts that purchased more than the bulk threshold of tickets for any event.
    4. Adds a 'BulkBuyer' column to flag accounts that made bulk purchases.
    5. Counts the number of events where each account made bulk purchases.
    6. Adds a 'FrequentBulkBuyer' column to flag accounts that made bulk purchases in more than the event count threshold of events.
    7. Logs the list of frequent bulk buyers and the execution time.

Returns:
    pd.DataFrame: The updated DataFrame with two new columns:
        - 'BulkBuyer': A boolean flag indicating if the account made any bulk purchases.
        - 'FrequentBulkBuyer': A boolean flag indicating if the account made bulk purchases in more than the specified number of events.
"""
def add_bulk_buyers(df, logger):
    start = timer()

    bulk_threshold = 12
    event_count_threshold = 3

    # Gather event quantities for all accounts
    grouped = df.groupby(['ContactId', 'EventName'])['Quantity'].sum().reset_index()

    # Identify accounts that bought more than the bulk threshold tickets for any event
    bulk_purchases = grouped[grouped['Quantity'] >= bulk_threshold]

    # Add a new column for BulkBuyers
    df['BulkBuyer'] = df['ContactId'].isin(bulk_purchases['ContactId'])

    # Count the number of bulk purchases for each account
    bulk_purchase_counts = bulk_purchases['ContactId'].value_counts()

    # Identify accounts with bulk purchases for more than the event count threshold
    frequent_bulk_buyers = bulk_purchase_counts[bulk_purchase_counts > event_count_threshold].index

    # Add a new column for FrequentBulkBuyers
    df['FrequentBulkBuyer'] = df['ContactId'].isin(frequent_bulk_buyers)

    logger.debug(f'Frequent Bulk buyers list: {frequent_bulk_buyers}')

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Bulk Buyers added: Execution Time: {formatted_timing}')

    return df
"""
Function: get_patron_details

Description:
    This function processes event and patron data to calculate RFM scores, identify key events, 
    assess bulk buying behavior, and update geographical details like latitude and longitude 
    based on address information. It also handles data cleaning, such as fixing city/state inconsistencies 
    and merging with pre-existing patron data. The processed data is then saved, and patrons without 
    latitude/longitude details are identified.

Parameters:
    df (pd.DataFrame): A DataFrame containing event and transaction data, with columns like 'EventDate', 
        'ContactId', 'EventName', 'Quantity', and 'ItemPrice'.
    RFMScoreThreshold (int): The threshold above which a patron's RFM score is considered for geolocation updates.
    GetLatLong (bool): A flag indicating whether to update latitude, longitude, and ZIP+4 for missing values.
    regions_file (str): Path to the file containing region mapping data.
    patron_details_file (str): Path to the file containing existing patron details (latitude, longitude, ZIP+4, etc.).
    patron_temp_file (str): Path to a temporary file for saving the output in case of file permission issues.
    new_threshold (int): Threshold for identifying new patrons based on their first event date.
    returning_threshold (int): Threshold for identifying returning patrons based on their event attendance gaps.
    logger (logging.Logger): A logger object for logging debug information, execution progress, and results.

Process:
    1. **Data Preprocessing**: Filters events to include only relevant types (Live and completed/future events), 
        removes deleted transactions, and assigns missing account names to 'Walk Up Sales'.
    2. **Add Key Events**: Identifies the first, latest, and penultimate events for each account.
    3. **Bulk Buyers**: Identifies and flags bulk buyers and frequent bulk buyers based on ticket quantity.
    4. **Genre Scores**: Calculates genre preference scores for each patron based on event participation.
    5. **RFM Calculation**: Computes Recency, Frequency, and Monetary scores for each patron and assigns segments based on thresholds.
    6. **Address and ZIP Fixing**: Cleans city, state, address, and ZIP information to ensure consistency.
    7. **Geocode Information**: Updates missing latitude and longitude data for patrons based on their addresses, if enabled.
    8. **Region Assignment**: Maps patrons to geographical regions using the provided region file.
    9. **Saving Results**: Saves the final processed data to a specified output file or a temporary file in case of permission issues.

Returns:
    None: The function processes and saves data to the specified files, logging execution details and errors along the way.
"""
def get_patron_details(df,
                       RFMScoreThreshold,
                       GetLatLong,
                       regions_file,
                       patron_details_file,
                       anonymized,
                       new_threshold,
                       returning_threshold,
                       logger):

    # Change to working directory
    #os.chdir(data_dir)

    try:
        # Preprocess the data
        logger.debug('Preprocessing patron data...')

        df['EventDate'] = pd.to_datetime(df['EventDate'].copy(), errors='coerce')

        # only keep Live or Virtual that either completed or are planned. No subscriptions, cancelled, test, etc.
        df = df[df['EventType'].isin(['Live'])]
        df = df[df['EventStatus'].isin(['Complete', 'Future'])]
        logger.debug(f'initial patron Txn columns: {df.columns}')

        # # Set PII columns to Nan. We only do this now so that debugging modeling code doesn't include them.
        if anonymized == True:
            # Note Region is inherited from previous non-anonymized runs.
            PII_columns = ['AccountName','FirstName', 'LastName', 'Address', 'City', 'State', 'ZIP', 'OrderEmail']
            # Set the values in the PII columns to NaN where those columns exist in the DataFrame
            for col in PII_columns:
                if col not in df.columns:
                    df[col] = np.nan

        # Define the initial set of columns needed
        initial_columns = ['AccountName', 'ContactId',
                            'FirstName', 'LastName', 'Address', 'City', 'State', 'ZIP', 'OrderEmail',
                            'Quantity', 'ItemPrice', 'CreatedDate', 'EventDate', 'EventName',
                            'Subscriber', 'ChorusMember', 'DuesTxn', 'Season', 'Student',
                            'EventGenre', 'Choral', 'Brass', 'Classical', 'Contemporary', 'Dance']

        # Select only the relevant columns from the DataFrame
        df = df[initial_columns]

        df = df.sort_values(['ContactId', 'CreatedDate'])
        df = df.rename(columns={'Season':'LatestSeason'})

        logger.debug(f'Patron Txn columns: {df.columns}')

        first_latest_events = add_first_latest_events(df,logger)
        logger.debug(f'first_latest input shape: {df.shape}')
        df = df.merge(first_latest_events,on='ContactId',how='left')
        logger.debug(f'first_latest output shape: {df.shape}')

        #logger.debug('Identifying bulk buyers...')
        logger.debug(f'bulk input shape: {df.shape}')
        df = add_bulk_buyers(df, logger)
        logger.debug(f'bulk output shape: {df.shape}')
        logger.debug(f'bulk output columns: {df.columns}')

        # genre scores require aggregating full history
        logger.debug('Calculating genre scores...')
        genre_df = mod.calculate_genre_scores(df, logger)
        logger.debug(f'genre shape: {genre_df.shape}')
        logger.debug(f'genre scores columns: {df.columns}')

        df = df.merge(genre_df,on='ContactId',how='left')
        logger.debug(f'Genre columns: {df.columns}')
        del genre_df

        # Calculate RFM scores
        logger.info('Calculating RFM scores...')
        rfm_df = mod.calculate_rfm(df.copy(), 0, logger) #AAYAN
        logger.debug(f'RFM columns: {rfm_df.columns}')
        #logger.debug(f'Raw RFM shape: {rfm_df.shape}')
        logger.debug(f'Raw RFM: {rfm_df.head}')

        # plots
        #R = rfm_df['Recency']
        #F = np.log10(rfm_df['Frequency'] + .01)
        #M = np.log10(rfm_df['Monetary'] + .01)
        #plot_3D_scatter(R,'Recency',F,'Log Frequency',M,'Log Monetary')
        #plot_3D_scatter(rfm_df['RecencyScore'],'Recency Score',rfm_df['FrequencyScore'],'Frequency Score',rfm_df['MonetaryScore'],'Monetary Score')

        logger.debug('Calculating patron segments...')
        rfm_df['Segment'] = rfm_df.apply(mod.assign_segment, args=(new_threshold,returning_threshold), axis=1)

        logger.debug(f'final patron model shape: {rfm_df.shape}')
        logger.debug(f'final patron model columns: {rfm_df.columns}')
        ##logger.debug(f'Final rfm_df: {rfm_df.head}')

        # Modeling is complete so we can prune transaction columns.
        # Keep the most recent entry for each 'ContactId'
        logger.debug('Keeping only most recent Contact transaction address...')
        # belt and suspenders de-dup
        last_entry_df = df.drop_duplicates('ContactId', keep='last')

        logger.debug(f'Last Entry columns: {last_entry_df.columns}')
        logger.debug(f'Last Entry shape: {last_entry_df.shape}')

        #final prep
        keep_columns = ['AccountName','ContactId','FirstName', 'LastName', 'OrderEmail', 'Address', 'City', 'State', 'ZIP',
                        #'Subscriber','ChorusMember','DuesTxn','Student','BulkBuyer','FrequentBulkBuyer',
                        'ClassicalScore','ChoralScore','ContemporaryScore', 'DanceScore','BrassScore', 'PreferredGenre','Omni',
                        'FirstEvent', 'FirstEventDate','LatestEvent','LatestEventDate', 'PenultimateEvent', 'PenultimateEventDate', 'LatestSeason'
                        ]

        last_entry_df = last_entry_df[keep_columns]

        logger.debug('Merging RFM scores with the processed patron data...')
        logger.debug(f'last entry pre-merge shape: {last_entry_df.shape}')

        df = last_entry_df.merge(rfm_df, on='ContactId', how='left').sort_values(by=['MonetaryScore','FrequencyScore','RecencyScore'], ascending=False)
        logger.debug(f'pre-location shape: {df.shape}')
        logger.debug(f'pre-location columns: {df.columns}')
        logger.debug(f'pre-location df: {df.head}')

        # remove PII for anonymized output
        anon_df = df.copy()
        PII_columns = ['AccountName','FirstName', 'LastName', 'OrderEmail', 'Address', 'City', 'State']
        anon_df.drop(PII_columns, axis=1, inplace=True)
        # K-MeansClustering---------------------------------- #Aayan
        anon_df, wcss = mod.calculate_AUM_and_KMeans(anon_df)
        mod.KMeans_ElbowPlot(wcss)
        mod.analyze_silhouette_scores(anon_df, max_clusters=10)
        mod.snake_plot_rfm(anon_df, k_values=[4, 5, 6])
        mod.KMeans_RFMPlot(anon_df)
        # AgglomerativeClustering---------------------------------
        anon_df = mod.calculate_AUM_and_Agglomerative(anon_df)
        mod.analyze_silhouette_scores_agglomerative(anon_df, max_clusters=10)
        mod.Agglomerative_RFMPlot(anon_df)
        logger.debug(f'Anon Patron columns: {anon_df.columns}')
        logger.debug(f'Anon Patron shape: {anon_df.shape}')
        anon_output_file = 'anon_' + patron_details_file
        anon_df.to_csv(anon_output_file, index=False)
        logger.info(f'Anon Patron results written to file: {anon_output_file}')

        # PII-based location processing
        if anonymized == False: # then perform PII-based location processing.
            # Fix state and city user entry errors
            df = state_and_city_processing(df, logger)
            logger.debug(f'dataframe with state/city processing: {df}')

            # Fix address and ZIP consistency issues
            df  = address_and_ZIP_processing(df, logger)
            logger.debug(f'dataframe with address/ZIP processing: {df}')

            # Determine region assignments
            logger.debug('Applying Regions...')

            logger.debug(f'pre regions shape: {df.shape}')
            df = add_regions(df, regions_file, logger)

            logger.debug(f'post regions shape: {df.shape}')

            logger.debug('Add existing lat, long and ZIP+4 to the dataframe...')
            # first open the original file to see which lat/long details exist, so we don't re-generate them.
            orig_df = pd.read_csv(patron_details_file,low_memory=False)

            logger.debug(f'original: {orig_df.shape}')
            logger.debug(f'original: {orig_df.columns}')
            logger.debug(f'original: {orig_df.head}')

            # keep only most recent ContactId instance.
            orig_df = orig_df.sort_values(['ContactId','RFMScore'])
            orig_df = orig_df.groupby('ContactId').first().reset_index()

            # only run if the ZIP+4 format is wrong.
            #orig_df['ZIP+4'] = np.nan

            # Update existing lat, long and ZIP+4 from the previous run.
            df = df.merge(orig_df[['ContactId','Latitude','Longitude','ZIP+4']],on='ContactId', how='left').drop_duplicates()

            # Get the list of accounts missing lat/long that meet the RFM threshold criteria
            list_df = df[(df['Latitude'].isna() | df['Longitude'].isna()) & (df['RFMScore'] > RFMScoreThreshold)][['ContactId','Address','City','State','ZIP']]
            logger.debug(f'list with missing lat/long : {list_df}')

            count_missing_before = list_df['ContactId'].nunique()
            logger.debug(f'Patrons with missing Lat/Long before: {count_missing_before}')

            if GetLatLong:
                logger.info('Getting any new Lat/Long data...')
                start = timer()
                # this will only update ContactId's if lat and long are missing.
                df = df.apply(update_geocode_info, args=(RFMScoreThreshold,logger), axis=1)

                # We're not using ZIP+4, so bypass this.
                #logger.info('Get ZIP+4...')
                #df = df.apply(update_zip_plus4_info, args=(RFMScoreThreshold,), axis=1)

                end = timer()
                timing = timedelta(seconds=(end - start))
                formatted_timing = "{:.2f}".format(timing.total_seconds())

            else:
                logger.info('Bypassing Lat/Long and ZIP+4...')

            list_df = df[(df['Latitude'].isna() | df['Longitude'].isna()) & (df['RFMScore'] > RFMScoreThreshold)][['ContactId']]
            count_missing_after = list_df['ContactId'].nunique()
            new_counts = count_missing_before - count_missing_after

            logger.info(f'{count_missing_after} contacts are missing Lat/Long, likely to bad addresses')
            logger.info(f'{new_counts} new contacts had Lat/Long added.')

            logger.debug(f'Full non-anon shape: {df.shape}')
            logger.debug(f'Full non-anon columns: {df.columns}')
            logger.debug(f'The full non-anonymized df: {df.head}')

            # Save the full non-anonymized results
            logger.debug('Saving the full non-anon results...')
            df.to_csv(patron_details_file, index=False)

            logger.info(f'Full Patron results written to file: {patron_details_file}')

    except PermissionError:
        print(f'The output file is already open.')
    except FileNotFoundError:
        print("The output file was not found. Please check the file path.")
    except Exception as e:
        # This will catch any other exceptions
        print(f"An unexpected error occurred: {e}")


# General functions
def safe_divide(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(x, y)
        result[~np.isfinite(result)] = 0  # Set NaN, inf, -inf to 0
    return result
"""
Function: generate_identifier

Description:
    This function generates a unique identifier for a given account name by applying the SHA-256 hash function. 
    The result is a secure, consistent, and unique string (hash) that can be used as an anonymous identifier for each account.

Parameters:
    account_name (str): The account name (string) for which the unique identifier will be generated.

Process:
    1. Encodes the account name into bytes.
    2. Applies the SHA-256 hash function to the encoded account name.
    3. Returns the resulting hash as a hexadecimal string.

Returns:
    str: A unique SHA-256 hash string for the provided account name.
"""
def generate_identifier(account_name):
    # Use SHA-256 hash function
    return hashlib.sha256(account_name.encode()).hexdigest()
def plot_RFM(df, logger):
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
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
def plot_2D_scatter(x,x_label, y, y_label, logger):
    plt.figure(figsize=(20, 20))
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Distribution of {x_label} and {y_label}')
    plt.show()
    """
Function: get_geocode_info

Description:
    This function retrieves geocoding information (latitude and longitude) for a given address by making a request to the Google Geocoding API.

Parameters:
    address (str): A string containing the full address to be geocoded.

Process:
    1. Constructs the request URL using the Google Geocoding API and the provided address.
    2. Sends a GET request to the API and checks the response for a valid status.
    3. Parses the API response to extract the latitude and longitude of the address, if available.
    4. Handles errors such as request issues (HTTP errors), invalid JSON responses, or missing keys.

Returns:
    dict: A dictionary containing the latitude ('lat') and longitude ('lng') if the address is successfully geocoded.
    None: If there is an error or no geocoding result is found.
"""
def get_geocode_info(address):


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
"""
Function: update_geocode_info

Description:
    This function checks whether geocoding information (latitude and longitude) is missing for a record and attempts to update it using the Google Geocoding API, provided that the RFM score exceeds a threshold and the address is valid.

Parameters:
    df (pd.Series): A row of customer data, including 'RFMScore', 'Latitude', 'Longitude', 'Address', 'City', 'State', and 'ZIP'.
    RFMScoreThreshold (int): The minimum RFM score required for an account to trigger geocode lookup.
    logger (logging.Logger): A logger object for logging debug or information messages.

Process:
    1. Checks if the RFM score exceeds the given threshold and if latitude and longitude are missing.
    2. Validates that the address components (address, city, state, ZIP) are not null.
    3. Calls the `get_geocode_info()` function to retrieve latitude and longitude based on the full address.
    4. If geocoding is successful, updates the DataFrame with the obtained latitude and longitude values.

Returns:
    pd.Series: The updated row with new latitude and longitude if geocoding was successful.
"""
def update_geocode_info(df, RFMScoreThreshold, logger):
    # Check if the latitude and longitude exist and if conditions are met
    if df['RFMScore'] > RFMScoreThreshold and pd.isnull(df['Latitude']) and pd.isnull(df['Longitude']):
        # Additional checks for valid Address, City, State, and ZIP
        if pd.notnull(df['Address']) and pd.notnull(df['City']) and pd.notnull(df['State']) and pd.notnull(df['ZIP']):
            # Call get_geocode_info() and update latitude and longitude
            geocode_info = get_geocode_info(f"{df['Address']}, {df['City']}, {df['State']}, {df['ZIP']}")
            if geocode_info is not None:
                #start = timer()
                df['Latitude'] = geocode_info['lat']
                df['Longitude'] = geocode_info['lng']

    return df
def get_zip_plus4(street, city, state, zipcode):
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
    # Check if ZIP+4 exists and if conditions are met
    if df['RFMScore'] > RFMScoreThreshold and pd.isnull(df['ZIP+4']):
        # Call get_zip_plus4
        zip_plus4 = get_zip_plus4(df['Address'], df['City'], df['State'], df['ZIP'])
        # If get_zip_plus4 returns a value, assign it to df['ZIP+4']
        if zip_plus4 is not None:
            df['ZIP+4'] = df['ZIP'] + '-' + zip_plus4
    return df
#%%
