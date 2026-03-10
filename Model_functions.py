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

def calculate_event_scores(df, logger, event_column, venue_threshold=6, use_tfidf=False):
    """
    Generalized function to calculate scores for EventGenre, EventClass, or EventVenue,
    with a 4-day burst collapse to reduce festival weekend bias (per Account x Category).

    use_tfidf: if True, applies TF-IDF normalization so categories that appear in
    fewer patrons' histories are weighted more heavily. Use for EventClass where
    Headliner is heavily over-represented in the catalog (33% of events but 65% of
    stated preferences on raw proportion). Genre and Venue use raw proportion.
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

    # Deduplicate: one attendance per patron per event name.
    # This collapses multiple ticket types/instances of the same concert
    # without suppressing legitimate back-to-back attendance of different events.
    t = perf_counter()
    unique_events_df = df.drop_duplicates(subset=['AccountId', 'EventName'])

    # Raw counts per patron per category
    event_counts = (
        unique_events_df.groupby(['AccountId', event_column])
        .size()
        .reset_index(name='Count')
    )

    # Raw proportion within patron.
    # Genre and Venue use raw proportion — the catalog is reasonably balanced
    # across genres (Classical 41%, Choral 23%, Contemporary 20%, Dance 8%, Bach 9%)
    # so TF-IDF distorts more than it helps.
    # EventClass uses TF-IDF because Headliner dominates the catalog (33% of events)
    # but captures 65% of patron preferences on raw proportion alone, masking genuine
    # preference for Standard, Mission, and Local Favorite events.
    total_counts = (
        event_counts.groupby('AccountId')['Count']
        .sum()
        .reset_index(name='Total')
    )
    event_counts = event_counts.merge(total_counts, on='AccountId')
    event_counts['Proportion'] = event_counts['Count'] / event_counts['Total']

    if use_tfidf:
        # IDF = log(total patrons / patrons who attended this category)
        n_patrons = event_counts['AccountId'].nunique()
        patrons_per_cat = event_counts.groupby(event_column)['AccountId'].nunique()
        idf = np.log(n_patrons / patrons_per_cat)
        event_counts['IDF'] = event_counts[event_column].map(idf)
        event_counts['Proportion'] = event_counts['Proportion'] * event_counts['IDF']
        # Re-normalize within patron so scores still sum to 1 and entropy is meaningful
        norm = event_counts.groupby('AccountId')['Proportion'].transform('sum')
        event_counts['Proportion'] = event_counts['Proportion'] / norm

    logger.debug(f'{event_column}: aggregation & normalization. Execution Time: {_elapsed(t)}')

    # Wide pivot — each cell is the raw proportion (0–1)
    t = perf_counter()
    event_df = (
        event_counts.pivot(index='AccountId', columns=event_column, values='Proportion')
        .fillna(0)
        .reset_index()
    )

    # Frequency = distinct EventName per account
    freq_series = unique_events_df.groupby('AccountId')['EventName'].nunique()
    event_df['Frequency'] = event_df['AccountId'].map(freq_series).fillna(0).astype(int)
    logger.debug(f'{event_column}: pivot & frequency. Execution Time: {_elapsed(t)}')

    # Entropy & scoring
    t = perf_counter()
    def calculate_entropy(row):
        proportions = row[row > 0]
        return entropy(proportions) if len(proportions) else 0.0

    score_cols = [c for c in event_df.columns if c not in ['AccountId', 'Frequency']]
    event_df['Entropy'] = event_df[score_cols].apply(calculate_entropy, axis=1)

    # Preferred category = highest raw proportion
    preferred_col = f'Preferred{event_column}'
    event_df[preferred_col] = event_df[score_cols].idxmax(axis=1)

    # Top score (highest proportion for this patron)
    event_df['TopScore'] = event_df[score_cols].max(axis=1)

    # Rename category columns to *Score
    event_df.columns = [
        (col + 'Score' if col not in ['AccountId', preferred_col, 'Entropy', 'Frequency', 'TopScore'] else col)
        for col in event_df.columns
    ]

    # Strength label — based on top score and event count
    conditions = [
        event_df['Frequency'] <= 1,                # single event — can't distinguish preference from chance
        event_df['TopScore'] >= 0.70,              # clear dominant preference
        event_df['TopScore'] >= 0.50,              # decided lean
        event_df['TopScore'] < 0.30,               # spread across many categories
    ]
    choices = ['too few', 'Strong', 'Favors', 'Omnivore']

    event_df['Strength'] = np.select(conditions, choices, default='Mixed')

    # Rename final key columns
    event_df = event_df.rename(columns={
        'Entropy':   f'{event_column}Entropy',
        'Strength':  f'{event_column}Strength',
        'Frequency': f'{event_column}Frequency',
        'TopScore':  f'{event_column}TopScore',
    })
    logger.debug(f'{event_column}: entropy & scoring. Execution Time: {_elapsed(t)}')

    logger.info(f'{event_column} Scores complete. Execution Time: {_elapsed(start)}')
    return event_df

def calculate_growth_score(df, current_year):
    """
    Measure engagement trend by comparing a patron's recent event attendance
    to their own historical average.

    Uses event counts per fiscal year (not dollars) so that stable long-term
    patrons score near zero rather than negative.

    Args:
        df (pd.DataFrame): Per-patron per-year rows with 'FiscalYear' and
                           'EventCount' columns (zeros for seasons with no attendance).
        current_year (int): The current calendar year, used to exclude the
                            in-progress fiscal year from the calculation.

    Returns:
        float: (recent_avg / historical_avg) - 1, where:
                 > 0  recently more active than their own average (growing)
                   0  stable, or fewer than 3 complete seasons of data
                 < 0  recently less active than their own average (declining)
    """
    df = df.sort_values('FiscalYear')

    # Trim history to the patron's first season with any events
    active = df[df['EventCount'] > 0]
    if active.empty:
        return 0.0
    df = df[df['FiscalYear'] >= active['FiscalYear'].min()]

    # Exclude the current (possibly partial) fiscal year.
    # Fiscal year N runs July N through June N+1; if today is in calendar year
    # current_year, the fiscal year current_year-1 may still be in progress.
    complete = df[df['FiscalYear'] < current_year - 1]

    if len(complete) < 3:
        return 0.0  # Too few complete seasons for a meaningful signal

    historical_avg = complete['EventCount'].mean()
    if historical_avg == 0:
        return 0.0

    recent_avg = complete.tail(2)['EventCount'].mean()
    return (recent_avg / historical_avg) - 1.0

def calculate_regularity(df, logger=None):
    """
    Compute a Regularity score in [0, 1] with two components:

        Inter-season consistency (70%): fraction of eligible seasons attended.
            EligibleSeasons = seasons from the patron's first season to the
            most recent complete season. A minimum denominator of 5 prevents
            new patrons (1-2 seasons of history) from scoring artificially high.

        Intra-season depth (30%): average unique events per attended season,
            log-scaled with a ceiling at 10 events/season.

    A long-term annual subscriber who attends once per year scores ~0.70.
    A Best patron attending 6-8 concerts per year scores ~0.90+.
    A One&Done patron scores near 0.
    """
    from datetime import datetime
    today = datetime.today()

    # Ensure EventDate is datetime64[ns]
    if not pd.api.types.is_datetime64_any_dtype(df['EventDate']):
        df['EventDate'] = pd.to_datetime(df['EventDate'], errors='coerce')

    # Drop rows with invalid EventDate values
    if df['EventDate'].isna().any():
        df = df.dropna(subset=['EventDate'])

    # Determine FiscalYear (July–June) for season-based calculations
    logger.info("Calculating FiscalYear...")
    df['FiscalYear'] = df['EventDate'].apply(lambda x: x.year if x.month > 6 else x.year - 1)

    # Determine the first eligible event and fiscal year for each patron
    df['FirstEventDate'] = df.groupby('AccountId')['EventDate'].transform('min')
    df['FirstFiscalYear'] = df.groupby('AccountId')['FiscalYear'].transform('min')

    # Filter out events before patron's first event
    df = df[df['EventDate'] >= df['FirstEventDate']]

    # SeasonRecency: seasons ago each event occurred
    logger.info("Calculating SeasonRecency...")
    df['SeasonRecency'] = today.year - df['FiscalYear']

    # Sort for gap calculations
    df = df.sort_values(by=['AccountId', 'EventDate'])
    df['EventGap'] = df.groupby('AccountId')['EventDate'].diff().dt.days

    # Keep EventCluster for export (retained for downstream use)
    logger.info("Calculating EventClusters...")
    df['EventCluster'] = (df.groupby('AccountId')['EventDate']
                          .transform(lambda x: (x.diff().dt.days > 3).cumsum()))

    # ClusterFrequency: visit occasions per season (retained for export)
    df['ClusterFrequency'] = df.groupby(['AccountId', 'FiscalYear'])['EventCluster'].transform('nunique')

    # SeasonCount: distinct fiscal years with at least one event
    logger.info("Calculating SeasonCount...")
    df['SeasonCount'] = df.groupby('AccountId')['FiscalYear'].transform('nunique')

    # EventFrequencyPerSeason: unique events per attended season
    logger.info("Calculating EventFrequencyPerSeason...")
    total_unique_events = df.groupby('AccountId')['EventName'].transform('nunique')
    df['EventFrequencyPerSeason'] = total_unique_events / df['SeasonCount']

    # --- Inter-season consistency ---
    # EligibleSeasons = seasons from patron's first to most recent complete season.
    # current_fiscal_year: the season currently in progress (may be partial).
    logger.info("Calculating Regularity score...")
    current_fiscal_year = today.year if today.month > 6 else today.year - 1
    df['EligibleSeasons'] = (current_fiscal_year - df['FirstFiscalYear'] + 1).clip(lower=1)

    # Cap at 16 (matching the 15-year data window) so that very old donation/
    # subscription records with ancient EventDates don't inflate the denominator.
    # Minimum of 5 so patrons with < 5 seasons of history don't score artificially high.
    df['InterSeasonConsistency'] = (
        df['SeasonCount'] / df['EligibleSeasons'].clip(lower=5, upper=16)
    ).clip(upper=1.0)

    # --- Intra-season depth ---
    # Log-scaled average unique events per attended season; ceiling at 10 events/season.
    df['IntraSeasonDepth'] = (
        np.log1p(df['EventFrequencyPerSeason']) / np.log1p(10)
    ).clip(upper=1.0)

    # --- Combined Regularity ---
    df['Regularity'] = (0.7 * df['InterSeasonConsistency'] +
                        0.3 * df['IntraSeasonDepth'])

    return df


def calculate_patron_metrics(df, logger):
    """
    Calculate per-patron RFM metrics, lifespan, growth score, AYM, and regularity.

        Recency, Frequency, and Monetary are each binned 0-5 and summed into RFMScore.
        Also computes DaysToReturn, DaysFromPenultimateEvent, ClusterFrequency, and Engagement.
    """
    from datetime import datetime

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

    if multi_event_mask.any():
        # DaysToReturn: gap between first and second event
        df.loc[multi_event_mask, 'DaysToReturn'] = (
            (df['SecondEventDate'] - df['FirstEventDate'])
            .where(df['SecondEventDate'].notna() & df['FirstEventDate'].notna(), np.nan)
            .dt.days
        )
        # DaysFromPenultimateEvent: gap between penultimate and latest event
        df.loc[multi_event_mask, 'DaysFromPenultimateEvent'] = (
            (df['LatestEventDate'] - df['PenultimateEventDate'])
            .where(df['PenultimateEventDate'].notna(), np.nan)
            .dt.days
        )
        df['DaysFromPenultimateEvent'] = df['DaysFromPenultimateEvent'].clip(lower=0)

    df['DaysFromFirstEvent'] = (
        (today - df['FirstEventDate']).where(df['FirstEventDate'].notna(), np.nan).dt.days
    )

    df['DaysFromLatestEvent'] = (
        (today - df['LatestEventDate']).where(df['LatestEventDate'].notna(), np.nan).dt.days
    )
    # Recency: clip to [0, 4000] for binning; 4000 = effectively never attended
    df['Recency'] = df['DaysFromLatestEvent'].clip(lower=0, upper=4000).fillna(4000)
    logger.debug('Recency stats: %s', df['Recency'].describe())

    df['Lifespan'] = (df['DaysFromFirstEvent'] - df['DaysFromLatestEvent']) / 365.0
    df['Lifespan'] = np.where(df['Lifespan'] == 0, 0.01, df['Lifespan'])

    # Frequency = distinct events attended; Monetary = Quantity * ItemPrice
    df['Frequency'] = df.groupby('AccountId')['EventName'].transform('nunique')
    df['Monetary'] = pd.to_numeric(df['Quantity'], errors='coerce') * pd.to_numeric(df['ItemPrice'], errors='coerce')

    # Create a FiscalYear column
    df['FiscalYear'] = df['CreatedDate'].apply(lambda x: x.year if x.month > 6 else x.year - 1)

    # Aggregate monetary value and event count by Fiscal Year
    yearly_stats = (
        df.groupby(['AccountId', 'FiscalYear'])
        .agg(Monetary=('Monetary', 'sum'), EventCount=('EventName', 'nunique'))
        .reset_index()
    )

    # Get the range of fiscal years
    fiscal_years_range = pd.DataFrame({
        'FiscalYear': range(yearly_stats['FiscalYear'].min(), today.year + 1)
    })

    # Cartesian join for fiscal years
    unique_accounts = df[['AccountId']].drop_duplicates()
    all_years = unique_accounts.merge(fiscal_years_range, how='cross').drop_duplicates()
    all_years = all_years.merge(yearly_stats, on=['AccountId', 'FiscalYear'], how='left').fillna(0)

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
        'Student': 'max',
        'FullPriceRate': 'max',
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
    metrics_df['RecentEventYearsGap'] = (metrics_df['DaysFromPenultimateEvent'] / 365).fillna(0)
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
    if df['Monetary'] == 0:
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