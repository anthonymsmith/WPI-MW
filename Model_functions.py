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

def _elapsed(start):
    """Return elapsed seconds since `start` as a '%.2f' string."""
    return f'{timedelta(seconds=perf_counter() - start).total_seconds():.2f}'


def load_anonymized_dataset(anon_data_file, logger):
    """Load anonymized patron dataset from CSV."""
    start = perf_counter()

    # Load event manifest file and fix column names
    event_df = pd.read_csv(anon_data_file)

    logger.info(f'Anon Dataset loaded. Execution Time: {_elapsed(start)}')

    return event_df

def calculate_event_scores(df, logger, event_column, venue_threshold=6, burst_days=4):
    """
    Generalized function to calculate scores for EventGenre, EventClass, or EventVenue,
    with a 4-day burst collapse to reduce festival weekend bias (per Account x Category).
    """
    from time import perf_counter
    import numpy as np
    import pandas as pd
    from scipy.stats import entropy

    start = perf_counter()
    df = df.copy()

    # Hygiene
    df[event_column] = df[event_column].fillna('None')
    df = df[df[event_column] != "None"]
    df['EventDate'] = pd.to_datetime(df['EventDate'])

    # Handle venue filtering
    if event_column == 'EventVenue':
        venue_counts = df[event_column].value_counts()
        valid_venues = venue_counts[venue_counts >= venue_threshold].index
        df = df[df[event_column].isin(valid_venues)]

    # -------- 4-day burst collapse (per AccountId x Category) --------
    def _collapse_group(g, window_days):
        g = g.sort_values('EventDate')
        dates = g['EventDate'].to_numpy()
        keep = np.ones(len(g), dtype=bool)
        last_kept_idx = 0
        for i in range(1, len(g)):
            dt_days = (dates[i] - dates[last_kept_idx]) / np.timedelta64(1, 'D')
            if dt_days < window_days:
                keep[i] = False
            else:
                last_kept_idx = i
        return g.loc[keep]

    t = perf_counter()
    df = (
        df.groupby(['AccountId', event_column], group_keys=False)
        .apply(lambda g: _collapse_group(g, burst_days))
        .reset_index(drop=True)
    )
    logger.debug(f'{event_column}: burst collapse. Execution Time: {_elapsed(t)}')

    # Unique events after burst collapse
    t = perf_counter()
    unique_events_df = df.drop_duplicates(subset=['AccountId', 'EventDate', event_column])

    # Global frequencies (post-burst)
    global_event_freq = unique_events_df[event_column].value_counts(normalize=True)

    # Counts per account
    event_counts = (
        unique_events_df.groupby(['AccountId', event_column])
        .size()
        .reset_index(name='Count')
    )
    event_counts['NormalizedCount'] = event_counts['Count'] / (
            1 + event_counts[event_column].map(global_event_freq)
    )

    # Normalize within account
    total_counts = (
        event_counts.groupby('AccountId')['NormalizedCount']
        .sum()
        .reset_index(name='TotalNormalized')
    )
    event_counts = event_counts.merge(total_counts, on='AccountId')
    event_counts['NormalizedPercentage'] = np.where(
        event_counts['TotalNormalized'] > 0,
        event_counts['NormalizedCount'] / event_counts['TotalNormalized'],
        0.0
    )
    logger.debug(f'{event_column}: aggregation & normalization. Execution Time: {_elapsed(t)}')

    # Wide pivot
    t = perf_counter()
    event_df = (
        event_counts.pivot(index='AccountId', columns=event_column, values='NormalizedPercentage')
        .fillna(0)
        .reset_index()
    )

    # Frequency = distinct EventName per account (post-burst)
    freq_series = unique_events_df.groupby('AccountId')['EventName'].nunique()
    event_df['Frequency'] = event_df['AccountId'].map(freq_series).fillna(0).astype(int)
    logger.debug(f'{event_column}: pivot & frequency. Execution Time: {_elapsed(t)}')

    # Entropy & scoring
    t = perf_counter()
    def calculate_entropy(row):
        proportions = row[row > 0]
        return entropy(proportions) if len(proportions) else 0.0

    event_df['Entropy'] = event_df.drop(columns=['AccountId', 'Frequency']).apply(calculate_entropy, axis=1)

    # Preference strength
    event_df['RawPreferenceStrength'] = 1 / (0.8 + event_df['Entropy'])

    # Preferred category
    preferred_col = f'Preferred{event_column}'
    event_df[preferred_col] = (
        event_df.drop(columns=['AccountId', 'Entropy', 'RawPreferenceStrength', 'Frequency']).idxmax(axis=1)
    )

    event_df.columns = [
        (col + 'Score' if col not in ['AccountId', preferred_col, 'Entropy', 'RawPreferenceStrength', 'Frequency'] else col)
        for col in event_df.columns
    ]

    # Event count weighting
    max_events = event_df['Frequency'].max() if len(event_df) else 0
    event_df['EventCountWeighting'] = (
        np.log1p(1 + event_df['Frequency']) / np.log1p(1 + max_events) if max_events > 0 else 0.0
    )

    # Reduce preference for low counts
    event_df.loc[event_df['Frequency'] <= 3, 'RawPreferenceStrength'] *= 0.5

    # Confidence
    alpha = 0.4
    event_df['PreferenceConfidence'] = np.clip(
        ((1 - alpha) * event_df['RawPreferenceStrength'] + alpha * event_df['EventCountWeighting']) * 100,
        0, 100
    )

    # Strength label
    entropy_threshold = 1.1
    conditions = [
        event_df['Entropy'] > entropy_threshold,
        event_df['PreferenceConfidence'] > 90,
        event_df['PreferenceConfidence'] > 60,
        event_df['PreferenceConfidence'] > 45,
        event_df['Frequency'] <= 3,
        ]
    choices = ['Omnivore', 'Strong','Favors', 'Mixed','too few']

    event_df['Strength'] = np.select(conditions, choices, default='Unclear')

    # Rename final key columns
    event_df = event_df.rename(columns={
        'Entropy': f'{event_column}Entropy',
        'Strength': f'{event_column}Strength',
        'RawPreferenceStrength': f'{event_column}RawPreferenceStrength',
        'Frequency': f'{event_column}Frequency',
        'EventCountWeighting': f'{event_column}EventCountWeighting',
        'PreferenceConfidence': f'{event_column}PreferenceConfidence'
    })
    logger.debug(f'{event_column}: entropy & scoring. Execution Time: {_elapsed(t)}')

    logger.info(f'{event_column} Scores complete. Execution Time: {_elapsed(start)}')
    return event_df

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


def calculate_patron_metrics(df, logger):
    """
    Calculate per-patron RFM metrics, lifespan, growth score, AYM, and regularity.

        Recency, Frequency, and Monetary are each binned 0-5 and summed into RFMScore.
        Also computes DaysToReturn, DaysFromPenultimateEvent, ClusterFrequency, and Engagement.
    """
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    start = perf_counter()
    today = datetime.today()

    logger.debug('Subscriber read into Calc: %s', df["Subscriber"].value_counts())

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

    # Calculate Average Yearly Monetary — exponentially weighted toward recent years.
    # weight = AYM_DECAY ^ (current_year - FiscalYear), so last year counts fully,
    # 2 years ago at 70%, 3 years ago at 49%, etc. (~66% of weight on last 3 years).
    # History is trimmed to each patron's first active year so new patrons are not
    # diluted by years of zeros that predate their first transaction.
    logger.info("Calculating Average Yearly Monetary spend...")
    AYM_DECAY = 0.7
    _first_active = (
        all_years[all_years['Monetary'] > 0]
        .groupby('AccountId', as_index=False)['FiscalYear'].min()
        .rename(columns={'FiscalYear': 'FirstActiveYear'})
    )
    _aw = (all_years
           .merge(_first_active, on='AccountId')
           .query('FiscalYear >= FirstActiveYear')
           .copy())
    _aw['_w']  = AYM_DECAY ** (current_year - _aw['FiscalYear'])
    _aw['_wm'] = _aw['_w'] * _aw['Monetary']
    aym_df = (
        _aw.groupby('AccountId')
           .agg(weighted_sum=('_wm', 'sum'), weight_total=('_w', 'sum'))
           .reset_index()
    )
    aym_df['AYM'] = aym_df['weighted_sum'] / aym_df['weight_total']
    aym_df = aym_df[['AccountId', 'AYM']]

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
    # Recency: days since last event, season-aligned thresholds
    #   5 = within 6 months, 4 = 6–12 months, 3 = 1–2 years,
    #   2 = 2–4 years, 1 = 4–7 years, 0 = >7 years
    bins = [-1, 180, 365, 730, 1460, 2555, float('inf')]
    labels = [5, 4, 3, 2, 1, 0]
    metrics_df['RecencyScore'] = pd.cut(metrics_df['Recency'], bins=bins, labels=labels, right=False)
    metrics_df['RecencyScore'] = metrics_df['RecencyScore'].astype(float).fillna(0).astype(int)

    # Frequency: lifetime distinct events attended
    #   0 = 1 event (one-timer), 1 = 2–4, 2 = 5–9, 3 = 10–19, 4 = 20–29, 5 = 30+
    bins = [-1, 2, 5, 10, 20, 30, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    metrics_df['FrequencyScore'] = pd.cut(metrics_df['Frequency'], bins=bins, labels=labels, right=False)
    metrics_df['FrequencyScore'] = metrics_df['FrequencyScore'].astype(float).fillna(0).astype(int)

    logger.info("Frequency Score done...")

    # Monetary: uses AYM (decay-weighted annual spend) rather than lifetime total,
    # so current engagement drives the score rather than historical accumulation.
    #   0 = <$10/yr, 1 = $10–50, 2 = $50–120, 3 = $120–250, 4 = $250–500, 5 = $500+
    bins = [-1, 10, 50, 120, 250, 500, float('inf')]
    labels = [0, 1, 2, 3, 4, 5]
    metrics_df['MonetaryScore'] = pd.cut(metrics_df['AYM'], bins=bins, labels=labels, right=False)
    metrics_df['MonetaryScore'] = metrics_df['MonetaryScore'].astype(float).fillna(0).astype(int)

    logger.info("Monetary Score done...")

    metrics_df['RFMScore'] = metrics_df['RecencyScore'] + metrics_df['FrequencyScore'] + metrics_df['MonetaryScore']

    logger.info(f'Patron metrics complete. Execution Time: {_elapsed(start)}')

    return metrics_df

def assign_segment(df, new_threshold, reengaged_threshold):
    """
    Assign a patron segment label based on RFM scores and behavioral thresholds.

        Possible labels: Best, Comp, Group Buyer, New, Re-engaged, High, Upsell,
        Come Again, Slipping, Reminder, One&Done, Lapsed, Others.
    """
    # Initial checks for specific groups
    if df['MonetaryScore'] == 0:
        return 'Comp'
    if df['RFMScore'] == 15:
        return 'Best'
    if df['FrequentBulkBuyer']:
        return 'Group Buyer'

    # Lapsed / one-time — no realistic path to re-engagement without outreach
    if df['RecencyScore'] < 2:
        return 'One&Done' if df['Frequency'] <= 1 else 'Lapsed'

    # Segment based on event timing (New / Re-engaged checked before engagement tier)
    if df['DaysFromFirstEvent'] <= new_threshold:
        return 'New'
    if df['RecentEventYearsGap'] > reengaged_threshold:
        return 'Re-engaged'

    # High engagement — recently active with meaningful frequency
    if df['RecencyScore'] >= 4 and df['FrequencyScore'] >= 3:
        return 'High'

    # Upsell — recently active or active within 2 years, solid frequency history
    # Checked before Slipping so R=3/F>=3 patrons are not mis-labelled as drifting
    if df['RecencyScore'] >= 3 and df['FrequencyScore'] >= 3:
        return 'Upsell'

    # Slipping — meaningful history (5+ events, FrequencyScore >= 2) but attendance
    # has grown stale (R <= 3, i.e. last event 1+ year ago)
    if df['RecencyScore'] <= 3 and df['FrequencyScore'] >= 2:
        return 'Slipping'

    if df['RecencyScore'] >= 3:
        return 'Come Again'
    if df['RecencyScore'] >= 2:
        return 'Reminder'
    return 'Others'

# General functions
def safe_divide(x, y):
    """Element-wise division that replaces inf and NaN results with 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(x, y)
        result[~np.isfinite(result)] = 0  # Set NaN, inf, -inf to 0
    return result

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