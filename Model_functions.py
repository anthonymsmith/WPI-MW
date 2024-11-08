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
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy, stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from datetime import datetime, timedelta
from timeit import default_timer as timer
import hashlib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import requests
from requests.exceptions import RequestException

def load_anonymized_dataset(anon_data_file, logger):
    start = timer()

    # Load event manifest file and fix column names
    event_df = pd.read_csv(anon_data_file)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Anon Dataset loaded. Execution Time: {formatted_timing}')

    return event_df

"""
Function: calculate_genre_scores

Description:
    This function calculates normalized genre preference scores for each account based on event attendance data. 
    It adjusts the counts of events attended per genre by accounting for the global frequency of each genre, 
    giving higher weight to less frequent genres. Additionally, it computes a variety of related metrics, such 
    as entropy to measure the diversity of genre preferences, preference strength, and confidence scores.

Parameters:
    df (pd.DataFrame): A DataFrame containing event data with columns:
        - 'AccountId': Unique identifier for each customer.
        - 'EventDate': The date of each event attended by the customer.
        - 'EventGenre': The genre of the event attended.
        - 'EventName': The name of the event, used for calculating total event attendance frequency.
    logger (logging.Logger): A logger object for logging execution details, debugging information, and execution time.

Process:
    1. **Unique Events**: Drop duplicates to obtain unique events per account, event date, and genre.
    2. **Global Genre Frequency**: Calculate the overall frequency of each genre across all events.
    3. **Genre Counts Per Account**: Compute the count of events attended per genre for each account.
    4. **Normalization**: Adjust the genre counts by normalizing with respect to the global genre frequency, giving less frequent genres a higher weight.
    5. **Total Normalized Count**: Calculate the total normalized event count for each account.
    6. **Normalized Genre Percentage**: For each account, calculate the percentage of events attended for each genre relative to the total normalized count.
    7. **Pivot Table**: Reshape the data into a pivot table where each row represents an account and each column represents a genre's normalized percentage score.
    8. **Frequency**: Recalculate the total number of distinct events attended by each account.
    9. **Entropy Calculation**: Calculate the entropy for each account to measure the spread of event attendance across different genres.
    10. **Preference Strength**: Invert the entropy to quantify the strength of genre preference (lower entropy indicates a stronger preference).
    11. **Preferred Genre**: Identify the genre with the highest normalized percentage for each account.
    12. **Omni Flag**: Set an 'Omni' flag based on an entropy threshold to identify accounts with broad genre preferences.
    13. **Confidence**: Apply a confidence metric based on the total number of events attended, using a logarithmic scaling.
    14. **Preference Confidence**: Adjust the preference strength using the confidence metric for a more reliable measure of preference.

Returns:
    pd.DataFrame: A DataFrame where each row represents an account and contains the following columns:
        - Normalized genre percentage scores for each genre (column names suffixed with 'Score').
        - 'Frequency': The total number of distinct events attended by the account.
        - 'Entropy': A measure of how spread the account's event attendance is across genres.
        - 'RawPreferenceStrength': A measure of how strongly an account prefers a specific genre (based on entropy).
        - 'PreferredGenre': The genre with the highest normalized percentage score for the account.
        - 'Omni': A flag indicating if the account has broad genre preferences (high entropy).
        - 'Confidence': A confidence score based on the total number of events attended.
        - 'PreferenceConfidence': The adjusted preference strength, factoring in both confidence and raw preference strength.
"""
def calculate_genre_scores(df, logger):
    start = timer()

    # Drop duplicates to get unique events per AccountId, Event Date, and genre
    unique_events_df = df.drop_duplicates(subset=['AccountId', 'EventDate', 'EventGenre'])

    # Calculate the global frequency for each genre
    global_genre_freq = unique_events_df['EventGenre'].value_counts() / len(unique_events_df)

    # Calculate the by-genre counts for each account
    genre_counts = unique_events_df.groupby(['AccountId', 'EventGenre']).size().reset_index(name='Count')

    # Adjusted normalization: Normalize counts based on global genre frequency
    genre_counts['NormalizedCount'] = genre_counts.apply(lambda row: row['Count'] / (1 + global_genre_freq[row['EventGenre']]), axis=1)

    # Calculate the total normalized count for each account
    total_counts = genre_counts.groupby('AccountId')['NormalizedCount'].sum().reset_index(name='TotalNormalized')

    # Calculate normalized percentage for each genre for each account
    genre_counts = genre_counts.merge(total_counts, on='AccountId')
    genre_counts['NormalizedPercentage'] = genre_counts['NormalizedCount'] / genre_counts['TotalNormalized']

    # Reshape and finalize DataFrame
    genre_df = genre_counts.pivot(index='AccountId', columns='EventGenre', values='NormalizedPercentage').fillna(0).reset_index()

    # Recalculate frequency as the number of distinct events attended by each AccountId
    # This frequency will be treated as the total number of events attended
    genre_df['Frequency'] = df.groupby('AccountId')['EventName'].nunique().reset_index(drop=True)

    # Define function to calculate genre entropy (to measure how evenly distributed attendance is)
    def calculate_entropy(row):
        proportions = row[row > 0]  # Only consider non-zero proportions
        return entropy(proportions)

    # Calculate entropy for each patron (indicating how spread their attendance is across genres)
    genre_df['Entropy'] = genre_df.drop(columns=['AccountId', 'Frequency']).apply(calculate_entropy, axis=1)

    # Invert entropy to quantify the strength of preference (higher entropy means weaker preference)
    genre_df['RawPreferenceStrength'] = 1 / (1 + genre_df['Entropy'])

    # Determine the preferred genre (the one with the highest normalized percentage)
    def get_preferred_genre(row):
        return row.idxmax()

    genre_df['PreferredGenre'] = genre_df.drop(columns=['AccountId', 'Entropy', 'RawPreferenceStrength', 'Frequency']).apply(get_preferred_genre, axis=1)

    # Add a suffix of 'Score' to EventGenre columns
    genre_df.columns = [col + 'Score' if col not in ['AccountId', 'PreferredGenre', 'Entropy', 'RawPreferenceStrength', 'Frequency'] else col for col in genre_df.columns]

    # Set 'Omni' flag based on entropy threshold
    entropy_threshold = 1  # Define your entropy threshold for omnivores
    genre_df['Omni'] = genre_df['Entropy'] > entropy_threshold

    # Linear scaling of preference strength based on the total number of events attended
    max_events = genre_df['Frequency'].max()

    genre_df['Confidence'] = np.log1p(genre_df['Frequency']) / np.log1p(max_events)

    # Adjust PreferenceStrength using the confidence metric
    genre_df['PreferenceConfidence'] = genre_df['RawPreferenceStrength'] * genre_df['Confidence']

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Genre Scores complete. Execution Time: {formatted_timing}')

    return genre_df

"""
Function: calculate_growth_score

Description:
    This function calculates a growth score for a given account based on the relationship between fiscal years and monetary 
    values (spending) using linear regression. The growth score is derived from the slope of the regression line, which 
    represents how the account's spending changes over time. A positive slope indicates increasing spending, while a 
    negative slope suggests a decline in spending.

Parameters:
    df (pd.DataFrame): A DataFrame containing the following columns:
        - 'FiscalYear': The fiscal year in which the spending occurred.
        - 'Monetary': The total monetary value (spending) for each fiscal year.

Process:
    1. **Data Preparation**: The function reshapes the 'FiscalYear' column to be used as the predictor (X) and the 'Monetary' 
       column as the response (y) for linear regression.
    2. **Linear Regression**: Fits a linear regression model to the fiscal year and monetary data, using fiscal years to 
       predict spending patterns over time.
    3. **Growth Score Calculation**: The slope of the regression line (regression coefficient) is used as the growth score. 
       This score indicates the rate of change in monetary value over time (positive for growth, negative for decline).
    4. **Handling Edge Cases**: If there is only one fiscal year (i.e., no variation in fiscal years), the growth score is 
       set to 0, as no trend can be established.

Returns:
    float: The growth score, which is the slope of the linear regression line. A positive value indicates increasing 
           spending over time, while a negative value indicates decreasing spending.
"""

def calculate_growth_score(df, current_year):
    """
    Calculate growth score using weighted linear regression on FiscalYear and Monetary values.
    Recent years are weighted higher to emphasize recent growth.
    """
    # Define weights favoring the most recent 5 years
    df['Weight'] = df['FiscalYear'].apply(lambda year:
                                          0.4 if year == current_year else
                                          0.3 if year == current_year - 1 else
                                          0.2 if year == current_year - 2 else
                                          0.1 if year == current_year - 3 else
                                          0.05 if year == current_year - 4 else 0)

    # Calculate weighted monetary values
    df['WeightedMonetary'] = df['Monetary'] * df['Weight']

    # Prepare the data for regression
    fiscal_years = df['FiscalYear'].values.reshape(-1, 1)
    weighted_monetary_values = df['WeightedMonetary'].values

    # Ensure at least 2 distinct fiscal years for regression
    if len(np.unique(fiscal_years)) > 1:
        reg = LinearRegression().fit(fiscal_years, weighted_monetary_values)
        growth_score = reg.coef_[0]  # Slope of the regression line as growth score
    else:
        growth_score = 0  # Growth score is 0 if all fiscal years are identical

    return growth_score

"""
def calculate_growth_score(df):
    
    #Calculate growth score using linear regression on FiscalYear and Monetary values.
    
    # Prepare the data
    fiscal_years = df['FiscalYear'].values.reshape(-1, 1)  # Fiscal years as the predictor
    monetary_values = df['Monetary'].values  # Monetary values as the response

    # Include years with zero monetary values in the regression
    if len(np.unique(fiscal_years)) > 1:  # Ensure at least 2 distinct fiscal years
        reg = LinearRegression().fit(fiscal_years, monetary_values)
        growth_score = reg.coef_[0]  # Slope of the regression line as growth score
    else:
        growth_score = 0  # If all fiscal years are the same, growth score is 0

    return growth_score
"""
def calculate_adjusted_overall_regularity(df, logger):
    today = datetime.today()

    # Step 1: Adjust recency based on seasons (already present in your code)
    df['FiscalYear'] = df['EventDate'].apply(lambda x: x.year if x.month > 6 else x.year - 1)
    df['SeasonRecency'] = today.year - df['FiscalYear']  # Season-based recency

    # Step 2: Sort by AccountId and EventDate to calculate the gap between consecutive events
    df = df.sort_values(by=['AccountId', 'EventDate'])

    # Calculate the difference in days between consecutive events for each patron
    df['EventGap'] = df.groupby('AccountId')['EventDate'].diff().dt.days

    # Step 3: Cluster concerts that occur within 4 consecutive days
    # Initialize the cluster id
    df['EventCluster'] = (df['EventGap'] > 4).cumsum()  # Start a new cluster if gap > 4 days

    # Step 4: Calculate the cluster frequency for each patron (number of unique clusters attended)
    df['ClusterFrequency'] = df.groupby('AccountId')['EventCluster'].transform('nunique')

    # Step 5: Handle the pandemic period (March 2020 to December 2021)
    pandemic_start = pd.to_datetime('2020-03-01')
    pandemic_end = pd.to_datetime('2021-12-31')
    df['DaysFromLatestEvent'] = (today - df['LatestEventDate']).dt.days
    df['DaysFromLatestEvent'] = np.where(df['LatestEventDate'].between(pandemic_start, pandemic_end), 0, df['DaysFromLatestEvent'])

    # Step 6: Calculate regularity score using recency and cluster frequency
    df['Regularity'] = (0.5 * df['SeasonRecency']) + (0.5 * df['ClusterFrequency'])

    # Continue with the rest of the RFM calculation or return the dataframe with regularity
    return df


"""
Function: calculate_rfm

Description:
    This function calculates Recency, Frequency, and Monetary (RFM) scores for customer accounts, based on their event participation 
    and purchasing behavior. In addition to the traditional RFM metrics, the function computes several related metrics, such as days 
    since the first, latest, and penultimate events, account lifespan, and growth score (derived from monetary spending trends over time). 
    It also incorporates the Average Yearly Monetary spend (AYM) and additional custom attributes like subscriber status and other 
    categorical features for deeper segmentation. Z-scores for key metrics are also calculated, allowing for a more normalized view of 
    customer behavior. 

Parameters:
    df (pd.DataFrame): A DataFrame containing customer event and purchase data. Key columns include:
        - 'AccountId': Unique identifier for each customer.
        - 'LatestEventDate', 'FirstEventDate', 'PenultimateEventDate': Dates of customer events.
        - 'EventName': The name of attended events, used for frequency calculation.
        - 'Quantity' and 'ItemPrice': Used to calculate the monetary value of purchases.
        - Additional categorical columns such as 'Subscriber', 'ChorusMember', etc.
    logger (logging.Logger): A logger object for logging execution details, debugging information, and completion time.

Process:
    1. **Recency Calculation**: Calculates days since the latest, first, and penultimate events. Handles future dates (recency set to 0) and NaN values.
    2. **Lifespan Calculation**: Computes the customer’s lifespan based on the difference between the first and latest event dates.
    3. **Frequency Calculation**: Determines the frequency of distinct events attended by each account.
    4. **Monetary Calculation**: Computes the monetary value as the product of quantity and item price, aggregated across fiscal years.
    5. **Fiscal Year Handling**: Defines a fiscal year (ending June 30) and groups monetary values by this period. A Cartesian join ensures zero spending years are accounted for.
    6. **Growth Score**: Uses a linear regression model to calculate growth in monetary spending over fiscal years, providing insight into spending trends.
    7. **Average Yearly Monetary (AYM)**: Computes the average yearly monetary spend for each account based on their transaction history.
    8. **Additional Metrics**: Includes custom features such as subscriber status, chorus membership, and dues transactions for more granular segmentation.
    9. **Z-Score Calculation**: Computes z-scores for Recency, Frequency, Monetary, GrowthScore, and AYM to normalize the metrics across customers.
    10. **Scoring**: Segments Recency, Frequency, and Monetary into predefined score bins (0 to 5). Total RFM score is calculated by summing these individual scores.
    11. **Handling NaN and Edge Cases**: Accounts for NaN values in monetary, event, and other fields, ensuring proper handling of customers who predate available sales history.
    12. **Logging**: Logs the final DataFrame shape and execution time.

Returns:
    pd.DataFrame: A DataFrame containing calculated RFM scores, z-scores, growth score, AYM, and other customer-level metrics. 
                  Key output columns include:
        - 'RecencyScore', 'FrequencyScore', 'MonetaryScore', 'RFMScore'
        - 'GrowthScore', 'AYM', 'RecencyZ', 'FrequencyZ', 'MonetaryZ', 'GrowthZ', 'AYMZ'
        - Additional categorical fields such as 'Subscriber', 'ChorusMember', etc.
"""
def calculate_rfm(df, logger):
    start = timer()
    today = datetime.today()  # Use today's date or a specific snapshot date

    # Calculate days from today for both CreatedDate and LatestEventDate
    df['DaysFromLatestEvent'] = (today - df['LatestEventDate']).dt.days

    # Calculate Recency as the maximum of DaysFromCreated or DaysFromLatestEvent,
    # limited to a minimum of 0. Future events are Recency = 0.
    df['Recency'] = df['DaysFromLatestEvent']
    df['Recency'] = np.where(df['Recency'] < 0, 0, df['Recency'])

    # Calculate days since first event, clipping to 0 if in the future
    df['DaysFromFirstEvent'] = (today - df['FirstEventDate']).dt.days
    df['DaysFromFirstEvent'] = np.where(df['FirstEventDate'] > today, 0, df['DaysFromFirstEvent'])

    df['Lifespan'] = (df['DaysFromFirstEvent'] - df['DaysFromLatestEvent']) / 365.0

    # Calculate days since penultimate event, setting to zero if in the future or NA
    df['DaysFromPenultimateEvent'] = (today - df['PenultimateEventDate']).dt.days
    df['DaysFromPenultimateEvent'] = np.where(df['PenultimateEventDate'] > today, 0, df['DaysFromPenultimateEvent'])
    df['DaysFromPenultimateEvent'].fillna(0, inplace=True)

    # Frequency = Count of distinct events attended
    df['Frequency'] = df.groupby('AccountId')['EventName'].transform('nunique')

    # Include ticket donations + ticket sales. Monetary = Quantity * ItemPrice
    df['Monetary'] = df['Quantity'] * df['ItemPrice'] # + df['DonationAmount']

    # Create a FiscalYear column based on fiscal year ending June 30
    df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])  # Ensure CreatedDate is datetime
    df['FiscalYear'] = df['CreatedDate'].apply(lambda x: x.year if x.month > 6 else x.year - 1)

    # Aggregate monetary value by Fiscal Year using the FiscalYear column
    monetary_by_fiscal_year = df.groupby(['AccountId', 'FiscalYear']).agg({'Monetary': 'sum'}).reset_index()

    # Get the unique AccountIds
    unique_accounts = df[['AccountId']].drop_duplicates()

    # Get the range of fiscal years from the minimum to today's year
    fiscal_years_range = pd.DataFrame({'FiscalYear': range(monetary_by_fiscal_year['FiscalYear'].min(), today.year + 1)})

    # Perform a Cartesian join while ensuring uniqueness in fiscal years
    all_years = unique_accounts.merge(fiscal_years_range, how='cross').drop_duplicates()

    # Merge with monetary_by_fiscal_year to include zero spending years
    all_years = all_years.merge(monetary_by_fiscal_year, on=['AccountId', 'FiscalYear'], how='left').fillna(0)

    # Calculate growth scores using the linear regression approach for each account using a 5-year weighted approach
    current_year = today.year
    growth_scores = all_years.groupby('AccountId').apply(calculate_growth_score, current_year).reset_index()
    #growth_scores = all_years.groupby('AccountId').apply(calculate_growth_score).reset_index()
    growth_scores.columns = ['AccountId', 'GrowthScore']

    # Calculate AYM (Average Yearly Monetary spend) based on active fiscal years
    aym_df = all_years.groupby('AccountId').agg(
        total_monetary=('Monetary', 'sum'),
        active_fiscal_years=('Monetary', lambda x: (x > 0).sum())  # Count only fiscal years with non-zero spend
    ).reset_index()

    # Calculate AYM based on active years
    aym_df['AYM'] = aym_df['total_monetary'] / aym_df['active_fiscal_years']

    # Call the function to calculate adjusted regularity (with clustering logic)
    df = calculate_adjusted_overall_regularity(df, logger)

    # Now merge back into the RFM table
    rfm_df = df.groupby('AccountId').agg({
        'Recency': 'min',
        'Frequency': 'max',
        'Monetary': 'sum',
        'Lifespan': 'max',
        'DaysFromFirstEvent': 'min',
        'DaysFromPenultimateEvent': 'min',
        'ClusterFrequency': 'max',  # Regularity will come from here
        'Regularity': 'max',  # Include the calculated regularity metric
        'Subscriber': 'last',
        'ChorusMember': 'last',
        'DuesTxn': 'last',
        'FrequentBulkBuyer': 'last',
        'Student': 'last'
    }).reset_index()
    logger.info(rfm_df.shape)

    rfm_df = rfm_df.merge(growth_scores, on='AccountId', how='left')
    rfm_df = rfm_df.merge(aym_df[['AccountId', 'AYM']], on='AccountId', how='left')

    # Fill NaN values
    rfm_df['Recency'].fillna(0, inplace=True)
    rfm_df['Frequency'].fillna(0, inplace=True)
    rfm_df['Monetary'].fillna(0, inplace=True)
    rfm_df['GrowthScore'].fillna(0, inplace=True)
    rfm_df['AYM'].fillna(0, inplace=True)
    rfm_df['Regularity'].fillna(0, inplace=True)
    rfm_df['DaysFromFirstEvent'].fillna(3600, inplace=True)  # If no DaysFromFirstEvent, then large value

    # Exclude $0 comp tickets for z-score calculations
    filtered_rfm_df = rfm_df[rfm_df['Monetary'] > 0]

    # Calculate z-scores for Recency, Frequency, Monetary, GrowthScore, AYM, and Regularity
    rfm_df['RecentEventGap'] = rfm_df['DaysFromPenultimateEvent'] - rfm_df['Recency']
    rfm_df['RecencyZ'] = stats.zscore(filtered_rfm_df['Recency'])
    rfm_df['FrequencyZ'] = stats.zscore(filtered_rfm_df['Frequency'])
    rfm_df['MonetaryZ'] = stats.zscore(filtered_rfm_df['Monetary'])
    rfm_df['GrowthZ'] = stats.zscore(filtered_rfm_df['GrowthScore'])
    rfm_df['AYMZ'] = stats.zscore(filtered_rfm_df['AYM'])
    rfm_df['RegularityZ'] = stats.zscore(filtered_rfm_df['Regularity'])

    # Replace NaN or infinite values with 0 for z-scores
    rfm_df[['RecencyZ', 'FrequencyZ', 'MonetaryZ', 'GrowthZ', 'AYMZ', 'RegularityZ']] = (
        rfm_df[['RecencyZ', 'FrequencyZ', 'MonetaryZ', 'GrowthZ', 'AYMZ', 'RegularityZ']]
        .replace([np.inf, -np.inf], np.nan).fillna(0))


    # Apply binning for Recency, Frequency, and Monetary Scores
    # these bins are not evenly distributed but tweaked to identify classes of patrons

    # Receny score is the hardest to define given seasonal variations.
    # "New" tends to be < 250 to catch an entire season, but a score of 5 is really
    # for recent in-season or subscriber purchases. The boundaries are then mostly
    # to catch seasons with multiples of years + a little bit.
    # Note: The pandemic started around 1500 days ago circa 2024 season start.
    bins = [0, 120, 400, 700, 1500, 2000, float('inf')]
    labels = [5, 4, 3, 2, 1, 0]
    rfm_df['RecencyScore'] = pd.cut(rfm_df['Recency'], bins=bins, labels=labels, right=False).astype(int)

    # These bins try to differentiate single from multiple from regular attendance.
    # It skews heavily toward one or two and then towards > 50 from loyalists.
    # Scores in the middle range of 10-40 are relatively rare.
    bins = [0, 1, 3, 5, 8, 11, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    rfm_df['FrequencyScore'] = pd.cut(rfm_df['Frequency'], bins=bins, labels=labels, right=False).astype(int)

    # These bins first separate Comp/TTO and then $7.50 youth tickets,
    # then single event ticket pairs, and then up from there.
    bins = [0, 10, 80, 200, 400, 1000, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    rfm_df['MonetaryScore'] = pd.cut(rfm_df['Monetary'], bins=bins, labels=labels, right=False).astype(int)

    # Extra step to handle outlier patrons who predate sales history
    rfm_df['MonetaryScore'] = rfm_df['MonetaryScore'].fillna(0).astype(int)

    """
    # This was an attempt to define bins uniformly, but the data just don't fit the goal
    # of Scores that fit patron journey stages. Frequency is particularly hard to fit 
    # because its distribution is so skewed to extremes. It doesn't execute successfuly 
    # because of skew and range problems became apparent and I abandoned it.

    # Calculate RFM Scores Using Quantiles for Granularity
    # Define quantile-based scoring function
    # Calculate quantiles and map to 1-5 scale for Recency, Frequency, and Monetary
    # Quantile-based scoring directly on raw values with 5 bins
    # Quantile-based scoring with duplicate edges handled
    
    def quantile_to_score(series):
        # Using `pd.qcut` with duplicates='drop' to ensure unique bin edges
        return pd.qcut(series, 5, labels=range(1, 6), duplicates='drop').astype(int)

    # Apply custom scoring logic
    # Recency: Higher quantiles mean more recent, so invert to match 1-5 scale (higher score = more recent)
    rfm_df['RecencyScore'] = 6 - quantile_to_score(rfm_df['Recency'])  # Inverting to make recent dates score higher

    # Frequency: Apply quantile scoring and handle Frequency == 1 as a special case
    rfm_df['FrequencyScore'] = np.where(rfm_df['Frequency'] == 1, 1, quantile_to_score(rfm_df['Frequency']))

    # Monetary: Apply quantile scoring and handle Monetary == 0 as a special case
    rfm_df['MonetaryScore'] = np.where(rfm_df['Monetary'] == 0, 0, quantile_to_score(rfm_df['Monetary']))

    # Combine RFM Scores
    rfm_df['RFMScore'] = rfm_df['RecencyScore'] + rfm_df['FrequencyScore'] + rfm_df['MonetaryScore']

    # Ensure scores are filled
    rfm_df[['RecencyScore', 'FrequencyScore', 'MonetaryScore', 'RFMScore']] = (
        rfm_df[['RecencyScore', 'FrequencyScore', 'MonetaryScore', 'RFMScore']]
        .fillna(0)
    )
    """
    # Aggregate RFM Scores
    rfm_df['RFMScore'] = rfm_df['RecencyScore'] + rfm_df['FrequencyScore'] + rfm_df['MonetaryScore']
    rfm_df['RFMScore'] = rfm_df['RFMScore'].fillna(0)

    end = timer()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'RFM scores complete. Execution Time: {formatted_timing}')

    return rfm_df

"""
Function: assign_segment

Description:
    This function assigns a customer to a specific segment based on their Recency, Frequency, and Monetary (RFM) scores, 
    along with other behavioral and engagement metrics. The segmentation helps in targeting customers for marketing, 
    retention, and engagement strategies by categorizing them into distinct groups such as 'New', 'Returning', 'Best', 
    'Upsell', 'Slipping', and others.

Parameters:
    df (pd.Series): A row of customer data containing relevant columns such as:
        - 'RecencyScore': Score indicating how recently the customer engaged.
        - 'FrequencyScore': Score based on the frequency of customer engagement.
        - 'MonetaryScore': Score representing the monetary value of customer purchases.
        - 'Subscriber': Boolean indicating if the customer is a subscriber.
        - 'DaysFromFirstEvent': Days since the customer's first event.
        - 'DaysFromPenultimateEvent': Days since the customer's second-most recent event.
        - Additional columns like 'FrequentBulkBuyer', 'ChorusMember', etc.
    new_threshold (int): Maximum number of days from the first event to qualify the customer as "New".
    returning_threshold (int): Maximum number of days since the penultimate event to qualify the customer as "Returning".

Process:
    1. **Initial Group Exclusion**: First, the function categorizes customers into specific groups that override other segmentations:
        - 'Comp': If the monetary score is 0, indicating no purchase activity.
        - 'Group Buyer': If the customer frequently buys in bulk (e.g., for groups).
        - 'New': If the customer’s first event was recent, based on the new_threshold.
        - 'Returning': If the customer attended an event recently, but with a long gap since their previous event.
        - 'Best': If the customer has the highest possible RFM score (15).

    2. **High Engagement Check**: If not excluded in the first step, the function checks for customers with high engagement:
        - 'Potential Chorus Subscriber': If the customer is highly engaged but not a subscriber, and is a chorus member.
        - 'Potential Subscriber': If the customer has a high recency score but is not currently a subscriber.
        - 'High': Customers with frequent and recent engagement (high recency and frequency scores).

    3. **Segment by Recency and Frequency**: If high engagement is
"""
def assign_segment(df, new_threshold, returning_threshold):
    # Initial checks for specific groups
    if df['MonetaryScore'] == 0:
        return 'Comp'
    if df['FrequentBulkBuyer']:
        return 'Group Buyer'

    # Check for recency and frequency conditions
    if df['RecencyScore'] < 2:
        return 'One&Done' if df['FrequencyScore'] <= 1 else 'Lapsed'

    if df['RecencyScore'] <= 3 and df['FrequencyScore'] >= 2:
        return 'Slipping'

    # Segment based on event timing and engagement
    if df['DaysFromFirstEvent'] <= new_threshold:
        return 'New'
    if df['RecentEventGap'] > returning_threshold:
        return 'Re-engaged'
    if df['RFMScore'] == 15:
        return 'Best'

    # High engagement and subscriber potential
    if df['RecencyScore'] >= 4 and df['FrequencyScore'] >= 4:
        return 'Potential Subscriber' if df['Subscriber'] != 'True' else 'High'

    # Recency and Frequency based segments for remaining cases
    if df['RecencyScore'] >= 3 and df['FrequencyScore'] >= 2:
        return 'Upsell'
    if df['RecencyScore'] >= 3:
        return 'Come Again'

    # Reminder segment for moderate recency and frequency
    if df['RecencyScore'] >= 2 and df['FrequencyScore'] >= 1:
        return 'Reminder'

    return 'Others'


# Example usage:
# df = pd.read_csv('your_data.csv')
# result_df = assign_segment(df, new_threshold=30, returning_threshold=365)


# Example usage:
# df = pd.read_csv('your_data.csv')
# result_df = assign_segment(df, new_threshold=30, returning_threshold=365)

# General functions
def safe_divide(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(x, y)
        result[~np.isfinite(result)] = 0  # Set NaN, inf, -inf to 0
    return result

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