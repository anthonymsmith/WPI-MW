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

from datetime import timedelta
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression

def load_anonymized_dataset(anon_data_file, logger):
    start = perf_counter()

    # Load event manifest file and fix column names
    event_df = pd.read_csv(anon_data_file)

    end = perf_counter()
    timing = timedelta(seconds=(end - start))
    formatted_timing = "{:.2f}".format(timing.total_seconds())
    logger.info(f'Anon Dataset loaded. Execution Time: {formatted_timing}')

    return event_df

def calculate_event_scores(df, logger, event_column, venue_threshold=6):
    """
    Generalized function to calculate scores for EventGenre, EventClass, or EventVenue.

    Parameters:
        df (DataFrame): The input DataFrame containing event data.
        logger (Logger): Logger instance for logging messages.
        event_column (str): The column to calculate scores for (e.g., 'EventGenre', 'EventClass', 'EventVenue').
        venue_threshold (int): Minimum occurrences required for an EventVenue to be considered (to remove one-offs).

    Returns:
        DataFrame: A DataFrame with scores, entropy, confidence, and preference calculations.
    """
    start = perf_counter()

    # Ensure we are not modifying the original DataFrame
    df = df.copy()

    # Replace missing values with 'None' and remove them from the calculation
    df[event_column] = df[event_column].fillna('None')
    df = df[df[event_column] != "None"]

    # Handle venue outliers: Remove venues with less than `venue_threshold` occurrences
    if event_column == 'EventVenue':
        venue_counts = df[event_column].value_counts()
        valid_venues = venue_counts[venue_counts >= venue_threshold].index
        df = df[df[event_column].isin(valid_venues)]
        logger.debug(f"Filtered out low-attendance venues. Kept {len(valid_venues)} venues.")

    # Drop duplicates to count unique events per AccountId, EventDate, and event_column
    unique_events_df = df.drop_duplicates(subset=['AccountId', 'EventDate', event_column])

    # Calculate the global frequency for each category
    global_event_freq = unique_events_df[event_column].value_counts(normalize=True)

    # Calculate event counts per account
    event_counts = unique_events_df.groupby(['AccountId', event_column]).size().reset_index(name='Count')

    # Normalize counts by global frequency
    event_counts['NormalizedCount'] = event_counts['Count'] / (1 + event_counts[event_column].map(global_event_freq))

    # Compute total normalized count per AccountId
    total_counts = event_counts.groupby('AccountId')['NormalizedCount'].sum().reset_index(name='TotalNormalized')

    # Calculate normalized percentage for each event type per AccountId
    event_counts = event_counts.merge(total_counts, on='AccountId')
    event_counts['NormalizedPercentage'] = event_counts['NormalizedCount'] / event_counts['TotalNormalized']

    # Reshape and finalize DataFrame
    event_df = event_counts.pivot(index='AccountId', columns=event_column, values='NormalizedPercentage').fillna(0).reset_index()

    # Compute frequency (distinct events attended by each AccountId)
    event_df['Frequency'] = df.groupby('AccountId')['EventName'].nunique().reset_index(drop=True)

    # Compute entropy (how evenly distributed attendance is across categories)
    def calculate_entropy(row):
        proportions = row[row > 0]  # Consider only non-zero proportions
        return entropy(proportions)

    event_df['Entropy'] = event_df.drop(columns=['AccountId', 'Frequency']).apply(calculate_entropy, axis=1)

    # Invert entropy to quantify strength of preference (higher entropy = weaker preference)
    event_df['RawPreferenceStrength'] = 1 / (0.8 + event_df['Entropy']).clip(upper=2)

    # Determine the preferred category for each AccountId
    event_df[f'Preferred{event_column}'] = event_df.drop(columns=['AccountId', 'Entropy', 'RawPreferenceStrength', 'Frequency']).idxmax(axis=1)

    # Add 'Score' suffix to relevant columns
    event_df.columns = [col + 'Score' if col not in ['AccountId', f'Preferred{event_column}', 'Entropy', 'RawPreferenceStrength', 'Frequency'] else col for col in event_df.columns]

    # Compute Event Count Weighting (more gradual scaling)
    max_events = event_df['Frequency'].max()
    # log smoothing to decrease weighting for lower frequency attendance
    event_df['EventCountWeighting'] = np.log1p(1 + event_df['Frequency']) / np.log1p(1 + max_events)

    # Reduce Preference Strength for low event counts.
    low_event_mask = event_df['Frequency'] <= 3
    event_df.loc[low_event_mask, 'RawPreferenceStrength'] *= 0.5  # Reduce weight for very few events

    # Adjust PreferenceConfidence with a controlled weighted sum instead of multiplication
    alpha = 0.4  # Controls balance between preference strength and event weighting
    event_df['PreferenceConfidence'] = np.clip(
        ((1 - alpha) * event_df['RawPreferenceStrength'] + alpha * event_df['EventCountWeighting']) * 100,
        a_min=0,  # Minimum allowed value
        a_max=100  # Maximum allowed value
    )

    # Set preference strength flag based on entropy threshold
    # Define conditions
    entropy_threshold = 1.1
    conditions = [
        event_df['Entropy'] > entropy_threshold,    # Omnivore
        event_df['PreferenceConfidence'] > 90,      # Focused
        event_df['PreferenceConfidence'] > 60,      # Favors
        event_df['PreferenceConfidence'] > 45,      # Mixed
        event_df['Frequency'] <= 3,                 # Unclear
        ]                                           # Unclear for remainder

    # Define corresponding labels.
    choices = ['Omnivore', 'Focused','Favors', 'Mixed','Unclear']

    # Apply vectorized conditional assignment
    event_df['Strength'] = np.select(conditions, choices, default='Unclear')

    # Rename key columns to avoid conflicts when merging multiple event scores
    event_df = event_df.rename(columns={
        'Entropy': f'{event_column}Entropy',
        'Strength': f'{event_column}Strength',
        'RawPreferenceStrength': f'{event_column}RawPreferenceStrength',
        'Frequency': f'{event_column}Frequency',
        'EventCountWeighting': f'{event_column}EventCountWeighting',
        'PreferenceConfidence': f'{event_column}PreferenceConfidence'
})


    # Execution time logging
    end = perf_counter()
    timing = timedelta(seconds=(end - start))
    logger.info(f'{event_column} Scores complete. Execution Time: {timing.total_seconds():.2f} seconds')

    return event_df

def calculate_genre_scores(df, logger):
    start = perf_counter()

    # Replace missing genres with 'None' and delete them from this calculation.
    df['EventGenre'] = df['EventGenre'].fillna('None')
    df = df[df['EventGenre'] != "None"]

    # Drop duplicates to get unique events per AccountId, Event Date, and genre
    unique_events_df = df.drop_duplicates(subset=['AccountId', 'EventDate', 'EventGenre'])

    # Calculate the global frequency for each genre
    #global_genre_freq = unique_events_df['EventGenre'].value_counts() / len(unique_events_df)
    global_genre_freq = unique_events_df['EventGenre'].value_counts(normalize=True)

    # Calculate the by-genre counts for each account
    genre_counts = unique_events_df.groupby(['AccountId', 'EventGenre']).size().reset_index(name='Count')

    # Adjusted normalization: Normalize counts based on global genre frequency
    #genre_counts['NormalizedCount'] = genre_counts.apply(lambda row: row['Count'] / (1 + global_genre_freq[row['EventGenre']]), axis=1)
    genre_counts['NormalizedCount'] = genre_counts['Count'] / (1 + genre_counts['EventGenre'].map(global_genre_freq))

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
    genre_df['RawPreferenceStrength'] = 1 / (.6 + genre_df['Entropy']).clip(upper=2) # peaks below 2

    # Determine the preferred genre (the one with the highest normalized percentage)
    genre_df['PreferredGenre'] = genre_df.drop(columns=['AccountId', 'Entropy', 'RawPreferenceStrength', 'Frequency']).idxmax(axis=1)

    # Add a suffix of 'Score' to EventGenre columns
    genre_df.columns = [col + 'Score' if col not in ['AccountId', 'PreferredGenre', 'Entropy', 'RawPreferenceStrength', 'Frequency'] else col for col in genre_df.columns]
    logger.debug(f'Genre columns: {genre_df.columns}')

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
    Estimate monetary growth over time using weighted linear regression.
    More recent years are weighted higher to emphasize recent growth.

    Args:
        df (pd.DataFrame): Dataframe with 'FiscalYear' and 'Monetary' columns.
        current_year (int): The current fiscal year.

    Returns:
        float: Growth score (slope of weighted regression line), or None if insufficient data.
    """
    # Define weights for the most recent 5 years
    weight_map = {
        0: 0.4,  # Current year
        -1: 0.3,  # Last year
        -2: 0.2,
        -3: 0.1,
        -4: 0.05
    }

    # Apply weights using a mapping
    df['Weight'] = df['FiscalYear'].apply(lambda y: weight_map.get(y - current_year, 0))

    # Remove years that received zero weight (older than 5 years)
    df = df[df['Weight'] > 0]

    # Ensure we have at least 2 different fiscal years
    if df['FiscalYear'].nunique() < 2:
        return None  # Not enough data for meaningful regression

    # Prepare regression variables
    fiscal_years = df['FiscalYear'].values.reshape(-1, 1)
    monetary_values = df['Monetary'].values

    # Apply weight scaling
    weighted_monetary = monetary_values * df['Weight'].values

    # Log transform monetary values to reduce impact of large values (optional)
    log_monetary = np.log1p(1 + weighted_monetary)  # log(1 + x) prevents log(0) errors

    # Fit weighted linear regression
    reg = LinearRegression().fit(fiscal_years, log_monetary)

    # Extract slope as growth score (adjusted for log scale)
    growth_score = reg.coef_[0]

    return growth_score

def old_calculate_growth_score(df, current_year,logger):
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

def calculate_regularity(df, logger=None):
    """
    Calculate Regularity based on their event attendance data. It combines inter-season,
    intra-season, average event frequency per season, and event "cluster attendance."
    """
    from datetime import datetime
    today = datetime.today()

    # Ensure EventDate is datetime64[ns]
    if not pd.api.types.is_datetime64_any_dtype(df['EventDate']):
        df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')

    # Drop rows with invalid EventDate values
    if df['EventDate'].isna().any():
        df = df.dropna(subset=['EventDate'])

    # Determine FiscalYear for season-based recency calculation
    logger.info("Calculating FiscalYear...")
    df['FiscalYear'] = df['EventDate'].apply(lambda x: x.year if x.month > 6 else x.year - 1)

    # Determine the first eligible event and fiscal year for each patron
    df['FirstEventDate'] = df.groupby('AccountId')['EventDate'].transform('min')
    df['FirstFiscalYear'] = df.groupby('AccountId')['FiscalYear'].transform('min')

    # Filter out events and seasons not eligible for each patron
    df = df[df['EventDate'] >= df['FirstEventDate']]

    # Calculate SeasonRecency as the difference between the current year and the fiscal year
    logger.info("Calculating SeasonRecency...")
    df['SeasonRecency'] = today.year - df['FiscalYear']

    # Sort by AccountId and EventDate to calculate the gap between consecutive events
    df = df.sort_values(by=['AccountId', 'EventDate'])
    df['EventGap'] = df.groupby('AccountId')['EventDate'].diff().dt.days

    # Cluster events that occur within 4 consecutive days (e.g., weekend festivals)
    logger.info("Calculating EventClusters...")
    df['EventCluster'] = (df.groupby('AccountId')['EventDate']
                          .transform(lambda x: (x.diff().dt.days > 3).cumsum()))

    # Calculate ClusterFrequency for each account by counting unique clusters within each season
    df['ClusterFrequency'] = df.groupby(['AccountId', 'FiscalYear'])['EventCluster'].transform('nunique')

    # Calculate SeasonCount (number of unique seasons attended) for inter-season regularity
    logger.info("Calculating SeasonCount...")
    df['SeasonCount'] = df.groupby('AccountId')['FiscalYear'].transform('nunique')

    # Calculate EventFrequencyPerSeason (average events attended per season)
    logger.info("Calculating EventFrequencyPerSeason...")
    total_events = df.groupby('AccountId')['EventCluster'].transform('count')
    df['EventFrequencyPerSeason'] = total_events / df['SeasonCount']

    # Define max values for normalization of metrics
    max_season_count = df['SeasonCount'].max()
    max_cluster_frequency = df['ClusterFrequency'].max()
    max_event_frequency_per_season = df['EventFrequencyPerSeason'].max()

    # Calculate the final Regularity score with weights for each metric
    w1, w2, w3 = 0.4, 0.2, 0.4
    logger.info("Calculating Regularity score...")
    df['Regularity'] = (w1 * (df['SeasonCount'] / max_season_count) +
                        w2 * (df['ClusterFrequency'] / max_cluster_frequency) +
                        w3 * (df['EventFrequencyPerSeason'] / max_event_frequency_per_season))

    # Return the dataframe with the Regularity score
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
def calculate_patron_metrics(df, logger):
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    start = perf_counter()
    today = datetime.today()

    logger.debug(f'Subscriber read into Calc: {df["Subscriber"].value_counts()}')

    # Convert date columns to datetime
    date_columns = ['FirstEventDate', 'LatestEventDate', 'PenultimateEventDate', 'SecondEventDate', 'CreatedDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Identify single-event patrons
    df['EventCount'] = df.groupby('AccountId')['EventName'].transform('count')
    single_event_mask = df['EventCount'] == 1
    multi_event_mask = df['EventCount'] > 1

    # Assign NaN to single-event patrons for second/penultimate-related fields
    df.loc[single_event_mask, ['SecondEvent', 'SecondEventDate', 'PenultimateEvent', 'PenultimateEventDate']] = np.nan
    df.loc[single_event_mask, ['DaysToReturn', 'DaysFromPenultimateEvent']] = np.nan

    # Calculate DaysToReturn for multi-event patrons
    if multi_event_mask.any():
        df.loc[multi_event_mask, 'DaysToReturn'] = (
            (df['SecondEventDate'] - df['FirstEventDate'])
            .where(df['SecondEventDate'].notna() & df['FirstEventDate'].notna(), np.nan)
            .dt.days
        )

    # Calculate DaysFromPenultimateEvent for multi-event patrons
    if multi_event_mask.any():
        df.loc[multi_event_mask, 'DaysFromPenultimateEvent'] = (
            (df['LatestEventDate'] - df['PenultimateEventDate'])
            .where(df['PenultimateEventDate'].notna(), np.nan)
            .dt.days
        )
        df['DaysFromPenultimateEvent'] = df['DaysFromPenultimateEvent'].clip(lower=0)

    # Calculate MonthsFromFirstEvent
    logger.info("Calculating MonthsFromFirstEvent...")
    if 'FirstEventDate' in df.columns:
        df['DaysFromFirstEvent'] = (
            (today - df['FirstEventDate'])
            .where(df['FirstEventDate'].notna(), np.nan)
            .dt.days
        )

    # Calculate Recency
    logger.info("Calculating Recency...")
    if 'LatestEventDate' in df.columns:
        df['DaysFromLatestEvent'] = (
            (today - df['LatestEventDate'])
            .where(df['LatestEventDate'].notna(), np.nan)  # Handle missing dates
            .dt.days  # Extract days
        )
        df['Recency'] = df['DaysFromLatestEvent']

    # Ensure Recency is numeric before further processing
    # Clip to between 4000 and 0 for binning.
    df['Recency'] = (
        pd.to_numeric(df['Recency'], errors='coerce')  # Ensure numeric
        .fillna(4000)                                  # Replace NaN with 4000
        .clip(lower=0, upper=4000)                     # Clip values between 0 and 4000
    )

    recency_stats = df['Recency'].describe()
    logger.debug(f'Recency stats raw: {recency_stats}')

    # Calculate Lifespan
    logger.info("Calculating Lifespan...")
    if 'DaysFromFirstEvent' in df.columns and 'DaysFromLatestEvent' in df.columns:
        df['Lifespan'] = (df['DaysFromFirstEvent'] - df['DaysFromLatestEvent']) / 365.0
        df['Lifespan'] = np.where(df['Lifespan'] == 0, 0.01, df['Lifespan'])

    # Frequency = Count of distinct events attended
    logger.info("Calculating Frequency...")
    df['Frequency'] = df.groupby('AccountId')['EventName'].transform('nunique')

    # Include ticket donations + ticket sales. Monetary = Quantity * ItemPrice
    logger.info("Calculating Monetary...")
    df['Monetary'] = pd.to_numeric(df['Quantity'], errors='coerce') * pd.to_numeric(df['ItemPrice'], errors='coerce')

    # Create a FiscalYear column
    df['FiscalYear'] = df['CreatedDate'].apply(lambda x: x.year if x.month > 6 else x.year - 1)

    # Aggregate monetary value by Fiscal Year
    monetary_by_fiscal_year = df.groupby(['AccountId', 'FiscalYear']).agg({'Monetary': 'sum'}).reset_index()

    # Get the range of fiscal years
    fiscal_years_range = pd.DataFrame({
        'FiscalYear': range(monetary_by_fiscal_year['FiscalYear'].min(), today.year + 1)
    })

    # Cartesian join for fiscal years
    unique_accounts = df[['AccountId']].drop_duplicates()
    all_years = unique_accounts.merge(fiscal_years_range, how='cross').drop_duplicates()
    all_years = all_years.merge(monetary_by_fiscal_year, on=['AccountId', 'FiscalYear'], how='left').fillna(0)

    # Calculate growth scores
    logger.info("Calculating GrowthScore...")
    current_year = today.year
    growth_scores = all_years.groupby('AccountId').apply(calculate_growth_score, current_year).reset_index()
    growth_scores.columns = ['AccountId', 'GrowthScore']

    # Calculate Average Yearly Monetary spend
    logger.info("Calculating Average Yearly Monetary spend...")
    aym_df = all_years.groupby('AccountId').agg(
        total_monetary=('Monetary', 'sum'),
        active_fiscal_years=('Monetary', lambda x: (x > 0).sum())
    ).reset_index()
    aym_df['AYM'] = aym_df['total_monetary'] / aym_df['active_fiscal_years']

    # Call the function to calculate adjusted regularity
    logger.info("Calculating Regularity...")
    df = calculate_regularity(df, logger)

    # Merge back into metrics
    metrics_df = df.groupby('AccountId').agg({
        'Recency': 'min',
        'Frequency': 'max',
        'Monetary': 'sum',
        'Lifespan': 'max',
        'DaysFromFirstEvent': 'min',
        'DaysToReturn': 'min',
        'DaysFromPenultimateEvent': 'min',
        'ClusterFrequency': 'max',
        'Regularity': 'max',
        'Subscriber': 'max',
        'ChorusMember': 'max',
        'DuesTxn': 'max',
        'FrequentBulkBuyer': 'max',
        'Student': 'max'
    }).reset_index()

    metrics_df = metrics_df.merge(growth_scores, on='AccountId', how='left')
    metrics_df = metrics_df.merge(aym_df[['AccountId', 'AYM']], on='AccountId', how='left')

    # Fill NaN values
    metrics_df['Recency'] = metrics_df['Recency'].fillna(0)
    metrics_df['Frequency'] = metrics_df['Frequency'].fillna(0)
    metrics_df['Monetary'] = metrics_df['Monetary'].fillna(0)
    metrics_df['GrowthScore'] = metrics_df['GrowthScore'].fillna(0)
    metrics_df['AYM'] = metrics_df['AYM'].fillna(0)
    metrics_df['Regularity'] = metrics_df['Regularity'].fillna(0)
    metrics_df['DaysFromFirstEvent'] = metrics_df['DaysFromFirstEvent'].fillna(3600)

    # Calculate additional metrics
    metrics_df['RecentEventYearsGap'] = metrics_df['DaysFromPenultimateEvent'] /365
    metrics_df['Engagement'] = safe_divide(metrics_df['Frequency'], metrics_df['DaysFromFirstEvent'])

    # Apply binning for RFM scores
    # Ensure Recency is numeric and clean
    # Apply Recency binning
    bins = [-1, 120, 400, 700, 1500, 2000, float('inf')]
    labels = [5, 4, 3, 2, 1, 0]
    metrics_df['RecencyScore'] = pd.cut(metrics_df['Recency'], bins=bins, labels=labels, right=False)

    # Convert to float first, then fill NaNs, and convert to int
    metrics_df['RecencyScore'] = metrics_df['RecencyScore'].astype(float).fillna(0).astype(int)

    # Apply Frequency binning
    bins = [-1, 1, 3, 5, 8, 11, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    metrics_df['FrequencyScore'] = pd.cut(metrics_df['Frequency'], bins=bins, labels=labels, right=False)
    metrics_df['FrequencyScore'] = metrics_df['FrequencyScore'].astype(float).fillna(0).astype(int)

    logger.info("Frequency Score done...")

    # Apply Monetary binning
    bins = [-1, 10, 80, 200, 400, 1000, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    metrics_df['MonetaryScore'] = pd.cut(metrics_df['Monetary'], bins=bins, labels=labels, right=False)
    metrics_df['MonetaryScore'] = metrics_df['MonetaryScore'].astype(float).fillna(0).astype(int)

    logger.info("Monetary Score done...")

    metrics_df['RFMScore'] = metrics_df['RecencyScore'] + metrics_df['FrequencyScore'] + metrics_df['MonetaryScore']

    end = perf_counter()
    timing = timedelta(seconds=(end - start))
    logger.info(f'Patron metrics complete. Execution Time: {timing}')

    return metrics_df

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
    reengaged_threshold (int): Maximum number of days since the penultimate event to qualify the customer as "Returning".

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
def assign_segment(df, new_threshold, reengaged_threshold):
    # Initial checks for specific groups
    if df['MonetaryScore'] == 0:
        return 'Comp'
    if df['RFMScore'] == 15:
        return 'Best'
    if df['FrequentBulkBuyer']:
        return 'Group Buyer'

    # Check for recency and frequency conditions
    if df['RecencyScore'] < 2:
        return 'One&Done' if df['Frequency'] <= 1 else 'Lapsed'

    if df['RecencyScore'] <= 3 and df['Frequency'] >= 2:
        return 'Slipping'

    # Segment based on event timing and engagement
    if df['DaysFromFirstEvent'] <= new_threshold:
        return 'New'
    if df['RecentEventYearsGap'] > reengaged_threshold:
        return 'Re-engaged'

    # High engagement and subscriber potential
    if df['RecencyScore'] >= 4 and df['FrequencyScore'] >= 4:
        return 'High'
    # Recency and Frequency based segments for remaining cases
    if df['RecencyScore'] >= 3 and df['FrequencyScore'] >= 2:
        return 'Upsell'
    if df['RecencyScore'] >= 3:
        return 'Come Again'
    # Reminder segment for moderate recency and frequency
    if df['RecencyScore'] >= 2:
        return 'Reminder'
    return 'Others'

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