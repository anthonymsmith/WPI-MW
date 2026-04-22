"""
Compare forecasting approaches for 25-26 season to date:
  Model A: Current hierarchy — (Class,Venue,LoB,SubGenre) → (Class,Venue,SubGenre)
           → (Class,Venue,Genre) → (Class,Venue) → (SubGenre)
  Model B: Event name matching — use event's own history first, fall back to Model A

Training: all seasons through 24-25 (temporal holdout for 25-26).
"""

import pandas as pd
import numpy as np
import os
import warnings
from forecast_artist_adjustment import apply_artist_adjustment, print_adjustment_summary

warnings.filterwarnings("ignore", category=DeprecationWarning)

WORKING_DIR = '/Users/antho/Documents/WPI-MW'
INCLUDE_COMPS = True
FORECAST_SEASON = '25-26'

WEIGHTS = {'19-20': 1.0, '20-21': 0.3, '21-22': 0.7,
           '22-23': 2.0, '23-24': 3.0, '24-25': 3.0}

# Minimum observations for EventRepeat tag to fire the repeat level.
# Major artists rarely visit MW 3+ times, so a single prior visit should
# inform the forecast. Temporal holdout (below) prevents partial-sales
# future events from contaminating backward hindcasts.
MIN_REPEAT_OBS = 1

# Rolling window for the repeat layer. Recurring holiday/series events
# (Messiah, WC Orch) trend upward post-pandemic; averaging 10+ years of
# history dilutes the current trajectory. Take only the N most recent
# seasons per repeat tag.
REPEAT_ROLLING_WINDOW = 3

# Pandemic-era events are excluded from the repeat layer — attendance during
# shutdown / restricted-capacity periods doesn't reflect recurring-event draw.
# (Other hierarchy levels still use this data, weighted by season.)
PANDEMIC_START = pd.Timestamp('2020-03-10')
PANDEMIC_END   = pd.Timestamp('2022-06-30')

# Minimum observations for a (Class, Venue, LoB, SubGenre, SeatFormat) bucket
# to anchor its own prediction. Below this, fall through to the
# non-SeatFormat bucket, adjusted by the venue-wide Floor-vs-Full ratio.
MIN_SEATFORMAT_OBS = 3

# Empirical Bayes shrinkage strength for thin Primary / F1 buckets.
# Primary bucket mean is blended toward the broader (Class, Venue) fallback:
#   shrunk = (N * WA + K * WA_fallback) / (N + K)
# With K=3, a bucket with N=2 is ~40% its own mean + 60% fallback; a bucket
# with N=15 is ~83% own + 17% fallback. Motivation: thin buckets like
# (Standard, MH, Concert, World/Crossover) [n=2, Chinese Acrobats + Martial
# Artists] were dragging predictions toward 933 while the broader
# Standard-at-MH prior sits around 500.
SHRINKAGE_K = 3.0


def assign_season_weight(season):
    if pd.isnull(season): return 1.0
    if season < "19-20": return 1.0
    return WEIGHTS.get(season, 1.0)


def load_data():
    os.chdir(WORKING_DIR)
    df = pd.read_csv("anon_DataMerge.csv")
    em = pd.read_excel("EventManifest.xlsx", sheet_name="EventManifest")
    df.columns = df.columns.str.strip()
    em.columns = em.columns.str.strip()

    for c in ['EventType', 'EventStatus', 'TicketStatus']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()

    df['Quantity'] = pd.to_numeric(df.get('Quantity'), errors='coerce').fillna(0)
    df['TicketTotal'] = pd.to_numeric(df.get('TicketTotal'), errors='coerce').fillna(0)

    # Merge from manifest: EventLoB + EventRepeat + Pricing + SeatFormat + VenueType
    # (EventSubGenre stays from CSV to avoid collision)
    meta_cols = ['EventId', 'EventName', 'EventStatus', 'EventType', 'EventClass',
                 'EventVenue', 'EventGenre', 'EventLoB', 'EventRepeat', 'EventCapacity']
    if 'Pricing' in em.columns:
        meta_cols.append('Pricing')
    if 'Seat Format' in em.columns:
        em = em.rename(columns={'Seat Format': 'SeatFormat'})
    if 'SeatFormat' in em.columns:
        meta_cols.append('SeatFormat')
    if 'Venue Type' in em.columns:
        em = em.rename(columns={'Venue Type': 'VenueType'})
    if 'VenueType' in em.columns:
        meta_cols.append('VenueType')
    manifest_meta = em.drop_duplicates(subset='EventId')[meta_cols].copy()
    for c in ['EventStatus', 'EventType']:
        manifest_meta[c] = manifest_meta[c].astype(str).str.strip().str.title()

    merged = df.merge(manifest_meta, on='EventId', how='left', suffixes=('', '_m'))
    for c in ['EventStatus', 'EventType', 'EventClass', 'EventVenue', 'EventGenre', 'EventLoB']:
        col_m = c + '_m'
        if col_m in merged.columns:
            merged[c] = merged[c].combine_first(merged[col_m])
            merged.drop(columns=[col_m], inplace=True, errors='ignore')
    for c in ['EventType', 'EventStatus', 'TicketStatus']:
        if c in merged.columns:
            merged[c] = merged[c].astype(str).str.strip().str.title()

    merged['EventCapacity'] = pd.to_numeric(merged.get('EventCapacity'), errors='coerce')
    merged['EventLoB'] = merged['EventLoB'].fillna('Concert')
    if 'EventSubGenre' not in merged.columns:
        merged['EventSubGenre'] = np.nan
    if 'EventRepeat' in merged.columns:
        merged['EventRepeat'] = merged['EventRepeat'].astype(str).str.strip()
        merged.loc[merged['EventRepeat'].isin(['nan', 'None', '']), 'EventRepeat'] = pd.NA
    if 'Pricing' in merged.columns:
        merged['Pricing'] = merged['Pricing'].astype(str).str.strip()
        merged.loc[merged['Pricing'].isin(['nan', 'None', '']), 'Pricing'] = pd.NA
    if 'SeatFormat' in merged.columns:
        merged['SeatFormat'] = merged['SeatFormat'].astype(str).str.strip()
        merged.loc[merged['SeatFormat'].isin(['nan', 'None', '']), 'SeatFormat'] = pd.NA
    if 'VenueType' in merged.columns:
        merged['VenueType'] = merged['VenueType'].astype(str).str.strip()
        merged.loc[merged['VenueType'].isin(['nan', 'None', '']), 'VenueType'] = pd.NA

    return em, merged


def get_training_df(merged, training_seasons):
    # PWYW / free-admission events are excluded from bucket priors (their draw
    # reflects pricing, not artist pull), but kept separately for PWYW-lift
    # calibration — see build_pwyw_lift().
    mask = merged['Quantity'] > 0 if INCLUDE_COMPS else merged['TicketTotal'] > 0
    pricing = merged.get('Pricing', pd.Series(pd.NA, index=merged.index))
    is_paid = ~pricing.astype(str).str.upper().isin(['PWYW', 'FREE'])
    filtered = merged[
        (merged['Season'].isin(training_seasons)) &
        (merged['EventType'] == 'Live') &
        (merged['EventStatus'] == 'Complete') &
        (merged['TicketStatus'] == 'Active') &
        is_paid &
        mask
    ].copy()
    filtered['Weight'] = filtered['Season'].apply(assign_season_weight)
    return filtered


def build_pwyw_lift(merged, training_seasons, repeat_model, primary_sf, sf_ratio,
                    primary, f1, f2, f3, f3a, f3b, f4, f5):
    """Multiplicative PWYW lift: geometric mean of (actual / paid-bucket-pred)
    across completed PWYW events in training_seasons. Floored at 1.0 so PWYW
    is never predicted to draw less than a paid comp would.

    Returns (lift, samples_df) — samples_df holds per-event ratios for logging.
    """
    if 'Pricing' not in merged.columns:
        return 1.0, pd.DataFrame()
    pwyw = merged[
        (merged['Season'].isin(training_seasons)) &
        (merged['EventType'] == 'Live') &
        (merged['EventStatus'] == 'Complete') &
        (merged['TicketStatus'] == 'Active') &
        (merged['Pricing'].astype(str).str.upper() == 'PWYW') &
        (merged['Quantity'] > 0)
    ]
    if pwyw.empty:
        return 1.0, pd.DataFrame()
    gb = ['EventId', 'EventName', 'EventClass', 'EventVenue',
          'EventGenre', 'EventLoB', 'EventSubGenre', 'EventRepeat']
    if 'SeatFormat' in pwyw.columns:
        gb.append('SeatFormat')
    pwyw_events = (
        pwyw.groupby(gb, group_keys=False, dropna=False)
        .agg(Actual=('Quantity', 'sum'),
             EventCapacity=('EventCapacity', 'max'))
        .reset_index()
    )
    fc = predict_model_a(pwyw_events, repeat_model, primary_sf, sf_ratio,
                         primary, f1, f2, f3, f3a, f3b, f4, f5)
    # Use uncapped paid prediction as the denominator — actuals can be
    # capacity-bounded, so capping the denominator would understate the lift.
    valid = fc[(fc['Pred_A'] > 0) & fc['Actual'].notna()].copy()
    if valid.empty:
        return 1.0, valid
    valid['Ratio'] = valid['Actual'] / valid['Pred_A']
    lift_empirical = float(np.exp(np.log(valid['Ratio']).mean()))
    # Bayesian shrinkage toward 1.0 (no lift) with prior K.
    # Current calibration sample is n=1 (Dinnerstein 16-17 fully-free orchestra
    # headliner) — not representative of typical PWYW which has tiered pricing
    # with a $0-5 floor. Pull empirical toward 1.0 until more data accumulates.
    # With K=3 and N=1 empirical=2.88 → shrunk ≈ 1.47.
    K = 3.0
    n = len(valid)
    lift = (n * lift_empirical + K * 1.0) / (n + K)
    lift = max(lift, 1.0)
    return lift, valid[['EventName', 'Actual', 'Pred_A', 'Ratio']]


def build_hierarchy_models(filtered):
    """Hierarchy: EventRepeat (if tagged) → Primary_SF (if tagged, n≥MIN_SEATFORMAT_OBS)
       → (Class,Venue,LoB,SubGenre) × SeatFormatRatio (when thinly tagged)
       → (Class,Venue,LoB,SubGenre) → (Class,Venue,SubGenre)
       → (Class,Venue,Genre) → (Class,Venue) → (SubGenre) → (EventClass)"""
    gb_cols = ['EventId', 'EventName', 'EventClass', 'EventVenue',
               'EventGenre', 'EventLoB', 'EventSubGenre', 'Season']
    if 'EventRepeat' in filtered.columns:
        gb_cols = gb_cols + ['EventRepeat']
    if 'SeatFormat' in filtered.columns:
        gb_cols = gb_cols + ['SeatFormat']
    if 'VenueType' in filtered.columns:
        gb_cols = gb_cols + ['VenueType']
    ea = (
        filtered
        .groupby(gb_cols, group_keys=False, dropna=False)
        .agg(TotalTickets=('Quantity', 'sum'), Weight=('Weight', 'first'),
             EventCapacity=('EventCapacity', 'max'),
             EventDate=('EventDate', 'max'))
        .reset_index()
    )
    ea['EventDate'] = pd.to_datetime(ea['EventDate'], errors='coerce')

    def wg_avg(g):
        wt = (g['TotalTickets'] * g['Weight']).sum()
        tw = g['Weight'].sum()
        return wt / tw if tw > 0 else np.nan

    # Repeat-series bucket — uses manual EventRepeat tag. Same artist across
    # venues/subgenres shares this prior (e.g. Yo-Yo Ma, Vengerov, MacMaster).
    # Rolling window: take the REPEAT_ROLLING_WINDOW most recent seasons per tag
    # so upward trends (Messiah post-pandemic) aren't diluted by ancient history.
    def rolling_wg_avg(g):
        recent = g.sort_values('Season', ascending=False).head(REPEAT_ROLLING_WINDOW)
        return wg_avg(recent)

    if 'EventRepeat' in ea.columns:
        pandemic_mask = ea['EventDate'].between(PANDEMIC_START, PANDEMIC_END)
        ea_rep = ea[ea['EventRepeat'].notna() & (ea['EventRepeat'] != '') & ~pandemic_mask]
        repeat_model = (
            ea_rep.groupby(['EventRepeat'], group_keys=False)
            .apply(lambda g: pd.Series({'WA_rep': rolling_wg_avg(g),
                                         'RepeatCount': min(len(g), REPEAT_ROLLING_WINDOW)}))
            .reset_index()
        )
    else:
        repeat_model = pd.DataFrame(columns=['EventRepeat', 'WA_rep', 'RepeatCount'])

    # SeatFormat-aware primary bucket — same dims as Primary plus SeatFormat.
    # Fires when the event carries a SeatFormat tag AND the bucket has
    # enough observations to anchor its own mean. Sparse cases fall through
    # and get the venue-wide ratio treatment below.
    if 'SeatFormat' in ea.columns:
        ea_sf = ea[ea['SeatFormat'].notna() & (ea['SeatFormat'] != '')]
        primary_sf = (
            ea_sf.groupby(['EventClass', 'EventVenue', 'EventLoB',
                           'EventSubGenre', 'SeatFormat'], group_keys=False)
            .apply(lambda g: pd.Series({'WA_psf': wg_avg(g), 'N_psf': len(g)}))
            .reset_index()
        )
        # Per-venue Full/Floor/Intimate ratios vs Full (within-venue, n≥3).
        venue_sf_mean = (
            ea_sf.groupby(['EventVenue', 'SeatFormat'], group_keys=False)
            .apply(lambda g: pd.Series({'sf_mean': wg_avg(g), 'sf_n': len(g)}))
            .reset_index()
        )
        full_mean = venue_sf_mean[venue_sf_mean['SeatFormat'] == 'Full'] \
            .rename(columns={'sf_mean': 'full_mean'})[['EventVenue', 'full_mean']]
        sf_ratio = venue_sf_mean.merge(full_mean, on='EventVenue', how='left')
        sf_ratio['Ratio'] = sf_ratio['sf_mean'] / sf_ratio['full_mean']
        sf_ratio.loc[sf_ratio['sf_n'] < MIN_SEATFORMAT_OBS, 'Ratio'] = np.nan
        sf_ratio = sf_ratio[['EventVenue', 'SeatFormat', 'Ratio']]
    else:
        primary_sf = pd.DataFrame(columns=['EventClass', 'EventVenue',
                                            'EventLoB', 'EventSubGenre',
                                            'SeatFormat', 'WA_psf', 'N_psf'])
        sf_ratio = pd.DataFrame(columns=['EventVenue', 'SeatFormat', 'Ratio'])

    # A bucket is "recurring" if at least one EventName within it appears in
    # 2+ prior seasons. Those buckets are trusted as-is: the signal is a real
    # series, not noise. Non-recurring thin buckets (random spectacle events
    # that share a bucket only by happenstance) are the ones we shrink.
    def _has_recurring(g):
        return int((g.groupby('EventName')['Season'].nunique() >= 2).any())

    primary = (
        ea.groupby(['EventClass', 'EventVenue', 'EventLoB', 'EventSubGenre'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WA_p_raw': wg_avg(g),
            'N_p': len(g),
            'HasRecurring_p': _has_recurring(g),
        })).reset_index()
    )
    f1 = (
        ea.groupby(['EventClass', 'EventVenue', 'EventSubGenre'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WA_f1_raw': wg_avg(g),
            'N_f1': len(g),
            'HasRecurring_f1': _has_recurring(g),
        })).reset_index()
    )
    f2 = (
        ea.groupby(['EventClass', 'EventVenue', 'EventGenre'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WA_f2_raw': wg_avg(g),
            'N_f2': len(g),
        })).reset_index()
    )
    f3 = (
        ea.groupby(['EventClass', 'EventVenue'], group_keys=False)
        .apply(lambda g: pd.Series({
            'WA_f3_raw': wg_avg(g),
            'N_f3': len(g),
        })).reset_index()
    )

    # Venue-type pooled fallbacks — slotted between F3 and F4.
    # F3a: (Class, VenueType, SubGenre) — keeps genre specificity, pools venues
    # F3b: (Class, VenueType)           — pooled venue-tier prior
    # These rescue thin-venue events (Curtis Hall, JMAC, small churches) where
    # the venue-specific chain collapses to 1-2 observations.
    if 'VenueType' in ea.columns:
        ea_vt = ea[ea['VenueType'].notna()]
        f3a = (
            ea_vt.groupby(['EventClass', 'VenueType', 'EventSubGenre'], group_keys=False)
            .apply(lambda g: pd.Series({'WA_f3a': wg_avg(g)})).reset_index()
        )
        f3b = (
            ea_vt.groupby(['EventClass', 'VenueType'], group_keys=False)
            .apply(lambda g: pd.Series({'WA_f3b': wg_avg(g)})).reset_index()
        )
    else:
        f3a = pd.DataFrame(columns=['EventClass', 'VenueType', 'EventSubGenre', 'WA_f3a'])
        f3b = pd.DataFrame(columns=['EventClass', 'VenueType', 'WA_f3b'])

    # F3 shrinkage toward F3b (pooled venue-tier). Symmetric — unlike Primary/F1,
    # thin F3 buckets at small venues tend to sit *below* the venue-tier pool
    # (Curtis Hall Standard = 107 vs Standard-at-Small tier ~250), so asymmetric
    # wouldn't rescue them. With K=3, a thin venue with N=2 gets ~40% own mean +
    # 60% pool; a well-trafficked venue (MH, N=50) barely moves.
    if 'VenueType' in ea.columns:
        venue_vt = (ea[['EventVenue', 'VenueType']]
                    .dropna().drop_duplicates('EventVenue'))
        f3 = f3.merge(venue_vt, on='EventVenue', how='left')
        f3 = f3.merge(f3b, on=['EventClass', 'VenueType'], how='left')
        f3['WA_f3_shrunk'] = (
            (f3['N_f3'] * f3['WA_f3_raw'] + SHRINKAGE_K * f3['WA_f3b'])
            / (f3['N_f3'] + SHRINKAGE_K)
        )
        f3['WA_f3'] = f3['WA_f3_shrunk'].fillna(f3['WA_f3_raw'])
        f3 = f3.drop(columns=['WA_f3_raw', 'WA_f3_shrunk', 'N_f3',
                              'WA_f3b', 'VenueType'])
    else:
        f3 = f3.rename(columns={'WA_f3_raw': 'WA_f3'}).drop(columns=['N_f3'])

    # F2 shrinkage toward (already-shrunk) F3. Same motivation as F3 shrinkage:
    # F2 averages across subgenres at a venue, so thin buckets (Catherine Russell
    # at JMAC: one Jazz event) are noisy estimators. Pull toward the venue's
    # all-genre mean. Symmetric — thin-venue Jazz/Chamber buckets tend to sit
    # below the venue pool.
    f2 = f2.merge(
        f3[['EventClass', 'EventVenue', 'WA_f3']],
        on=['EventClass', 'EventVenue'], how='left')
    f2['WA_f2_shrunk'] = (
        (f2['N_f2'] * f2['WA_f2_raw'] + SHRINKAGE_K * f2['WA_f3'])
        / (f2['N_f2'] + SHRINKAGE_K)
    )
    f2['WA_f2'] = f2['WA_f2_shrunk'].fillna(f2['WA_f2_raw'])
    f2 = f2.drop(columns=['WA_f2_raw', 'WA_f2_shrunk', 'N_f2', 'WA_f3'])

    # Empirical Bayes shrinkage: thin Primary / F1 buckets get pulled toward
    # their (Class, Venue) fallback mean. Thick buckets stay near their
    # observed average.
    primary = primary.merge(
        f3[['EventClass', 'EventVenue', 'WA_f3']],
        on=['EventClass', 'EventVenue'], how='left')
    primary['WA_p_shrunk'] = (
        (primary['N_p'] * primary['WA_p_raw'] + SHRINKAGE_K * primary['WA_f3'])
        / (primary['N_p'] + SHRINKAGE_K)
    )
    # Asymmetric + recurring-series exemption: only pull toward fallback when
    # the raw mean exceeds the fallback AND the bucket has no recurring series.
    # Leaves both genuinely-low thin buckets (e.g. niche recitals) and
    # legitimately-high repeat series (e.g. Handel Messiah) alone.
    primary['WA_p'] = np.where(
        (primary['WA_p_raw'] > primary['WA_f3']) & (primary['HasRecurring_p'] == 0),
        primary['WA_p_shrunk'],
        primary['WA_p_raw'],
    )
    primary['WA_p'] = primary['WA_p'].fillna(primary['WA_p_raw'])
    primary = primary.drop(columns=['WA_p_raw', 'WA_p_shrunk', 'N_p', 'WA_f3',
                                      'HasRecurring_p'])

    f1 = f1.merge(
        f3[['EventClass', 'EventVenue', 'WA_f3']],
        on=['EventClass', 'EventVenue'], how='left')
    f1['WA_f1_shrunk'] = (
        (f1['N_f1'] * f1['WA_f1_raw'] + SHRINKAGE_K * f1['WA_f3'])
        / (f1['N_f1'] + SHRINKAGE_K)
    )
    f1['WA_f1'] = np.where(
        (f1['WA_f1_raw'] > f1['WA_f3']) & (f1['HasRecurring_f1'] == 0),
        f1['WA_f1_shrunk'],
        f1['WA_f1_raw'],
    )
    f1['WA_f1'] = f1['WA_f1'].fillna(f1['WA_f1_raw'])
    f1 = f1.drop(columns=['WA_f1_raw', 'WA_f1_shrunk', 'N_f1', 'WA_f3',
                            'HasRecurring_f1'])

    f4 = (
        ea.groupby(['EventSubGenre'], group_keys=False)
        .apply(lambda g: pd.Series({'WA_f4': wg_avg(g)})).reset_index()
    )
    # F5 — ultimate fallback: EventClass mean. Guarantees no NaN for
    # one-off subgenre × new-venue combinations (e.g. Fusion @ JMAC).
    f5 = (
        ea.groupby(['EventClass'], group_keys=False)
        .apply(lambda g: pd.Series({'WA_f5': wg_avg(g)})).reset_index()
    )
    return repeat_model, primary_sf, sf_ratio, primary, f1, f2, f3, f3a, f3b, f4, f5


def build_name_model(filtered):
    """Weighted avg attendance per EventName."""
    ea = (
        filtered
        .groupby(['EventId', 'EventName', 'Season'], group_keys=False)
        .agg(TotalTickets=('Quantity', 'sum'), Weight=('Weight', 'first'))
        .reset_index()
    )
    name_model = (
        ea.groupby('EventName', group_keys=False)
        .apply(lambda g: pd.Series({
            'NameWeightedAvg': (g['TotalTickets'] * g['Weight']).sum() / g['Weight'].sum(),
            'NameOccurrences': len(g),
            'NameSeasons': g['Season'].nunique()
        }))
        .reset_index()
    )
    return name_model


def predict_model_a(events_df, repeat_model, primary_sf, sf_ratio,
                    primary, f1, f2, f3, f3a, f3b, f4, f5=None, pwyw_lift=1.0):
    fc = events_df.copy()
    for col in ['EventClass', 'EventVenue', 'EventGenre', 'EventLoB', 'EventSubGenre']:
        fc[col] = fc[col].astype(str).str.strip()
    if 'VenueType' in fc.columns:
        fc['VenueType'] = fc['VenueType'].astype(str).str.strip()
        fc.loc[fc['VenueType'].isin(['nan', 'None', '']), 'VenueType'] = pd.NA

    has_repeat_col = 'EventRepeat' in fc.columns and repeat_model is not None \
                     and not repeat_model.empty
    if has_repeat_col:
        fc = fc.merge(repeat_model, on='EventRepeat', how='left')
        # Gate single-obs repeat tags — one observation is too thin a prior
        # and leaks into unrelated sub-events that share the same tag.
        fc.loc[fc['RepeatCount'] < MIN_REPEAT_OBS, 'WA_rep'] = np.nan
    else:
        fc['WA_rep'] = np.nan

    # SeatFormat layer: direct bucket match (n≥MIN_SEATFORMAT_OBS) OR
    # venue-wide ratio applied to the non-SF primary prediction.
    has_sf_col = 'SeatFormat' in fc.columns and sf_ratio is not None \
                 and not sf_ratio.empty
    if has_sf_col:
        fc = fc.merge(primary_sf,
                      on=['EventClass', 'EventVenue', 'EventLoB',
                          'EventSubGenre', 'SeatFormat'], how='left')
        fc.loc[fc['N_psf'].fillna(0) < MIN_SEATFORMAT_OBS, 'WA_psf'] = np.nan
        fc = fc.merge(sf_ratio, on=['EventVenue', 'SeatFormat'], how='left')
    else:
        fc['WA_psf'] = np.nan
        fc['Ratio'] = np.nan

    fc = fc.merge(primary, on=['EventClass', 'EventVenue', 'EventLoB', 'EventSubGenre'], how='left')
    fc = fc.merge(f1, on=['EventClass', 'EventVenue', 'EventSubGenre'], how='left')
    fc = fc.merge(f2, on=['EventClass', 'EventVenue', 'EventGenre'], how='left')
    fc = fc.merge(f3, on=['EventClass', 'EventVenue'], how='left')
    has_vt = 'VenueType' in fc.columns and f3a is not None and not f3a.empty
    if has_vt:
        fc = fc.merge(f3a, on=['EventClass', 'VenueType', 'EventSubGenre'], how='left')
        fc = fc.merge(f3b, on=['EventClass', 'VenueType'], how='left')
    else:
        fc['WA_f3a'] = np.nan
        fc['WA_f3b'] = np.nan
    fc = fc.merge(f4, on='EventSubGenre', how='left')
    if f5 is not None:
        fc = fc.merge(f5, on='EventClass', how='left')

    # Base (non-SF) prediction using the existing hierarchy.
    base = (
        fc['WA_rep']
        .combine_first(fc['WA_p'])
        .combine_first(fc['WA_f1'])
        .combine_first(fc['WA_f2'])
        .combine_first(fc['WA_f3'])
        .combine_first(fc['WA_f3a'])
        .combine_first(fc['WA_f3b'])
        .combine_first(fc['WA_f4'])
    )
    if f5 is not None:
        base = base.combine_first(fc['WA_f5'])

    # SeatFormat ratio adjustment — only when falling to F2 or coarser.
    # F1 (Class+Venue+SubGenre) is still SubGenre-specific and may already
    # be Floor-dominated (e.g. Piano at MH is mostly Hamelin-type recitals).
    # F2+ averages across SubGenres, so a venue-wide SF ratio is appropriate.
    apply_ratio = (
        fc['WA_psf'].isna() & fc['WA_p'].isna() & fc['WA_f1'].isna()
        & fc['Ratio'].notna()
    )
    base_adj = np.where(apply_ratio, base * fc['Ratio'], base)
    fc['Pred_A'] = np.where(fc['WA_psf'].notna(), fc['WA_psf'], base_adj)

    conditions = [
        fc['WA_psf'].notna(),
        fc['WA_rep'].notna(),
        fc['WA_p'].notna(),
        apply_ratio & fc['WA_f1'].notna(),
        apply_ratio & fc['WA_f1'].isna() & fc['WA_f2'].notna(),
        apply_ratio & fc['WA_f1'].isna() & fc['WA_f2'].isna() & fc['WA_f3'].notna(),
        fc['WA_f1'].notna(),
        fc['WA_f1'].isna() & fc['WA_f2'].notna(),
        fc['WA_f1'].isna() & fc['WA_f2'].isna() & fc['WA_f3'].notna(),
        fc['WA_f1'].isna() & fc['WA_f2'].isna() & fc['WA_f3'].isna() & fc['WA_f3a'].notna(),
        fc['WA_f1'].isna() & fc['WA_f2'].isna() & fc['WA_f3'].isna() & fc['WA_f3a'].isna() & fc['WA_f3b'].notna(),
        fc['WA_f1'].isna() & fc['WA_f2'].isna() & fc['WA_f3'].isna() & fc['WA_f3a'].isna() & fc['WA_f3b'].isna() & fc['WA_f4'].notna(),
    ]
    choices = ['Primary_SF (Class+Venue+LoB+SubGenre+SF)',
               'Repeat (EventRepeat)',
               'Primary (Class+Venue+LoB+SubGenre)',
               'F1 × SF ratio',
               'F2 × SF ratio',
               'F3 × SF ratio',
               'F1 (Class+Venue+SubGenre)',
               'F2 (Class+Venue+Genre)',
               'F3 (Class+Venue)',
               'F3a (Class+VenueType+SubGenre)',
               'F3b (Class+VenueType)',
               'F4 (SubGenre)']
    fc['FallbackLevel'] = np.select(conditions, choices, default='F5 (EventClass)')

    # PWYW lift — multiplicative boost for pay-what-you-will / free events
    # (calibrated from historical PWYW residuals vs paid-bucket predictions).
    if 'Pricing' in fc.columns and pwyw_lift != 1.0:
        is_pwyw = fc['Pricing'].astype(str).str.upper() == 'PWYW'
        fc.loc[is_pwyw, 'Pred_A'] = fc.loc[is_pwyw, 'Pred_A'] * pwyw_lift
        fc.loc[is_pwyw, 'FallbackLevel'] = (
            fc.loc[is_pwyw, 'FallbackLevel'] + f' × PWYW lift {pwyw_lift:.2f}'
        )
    return fc


def predict_model_b(events_df_with_a, name_model):
    """Model B: use event name history (≥2 seasons); otherwise fall back to Model A."""
    fc = events_df_with_a.copy()
    fc = fc.merge(name_model, on='EventName', how='left')

    has_name_history = fc['NameSeasons'] >= 2
    fc['Pred_B'] = np.where(has_name_history, fc['NameWeightedAvg'], fc['Pred_A'])
    fc['Model_B_Source'] = np.where(
        has_name_history,
        fc['NameSeasons'].apply(lambda n: f'Name ({int(n) if pd.notna(n) else 0} seasons)'),
        'Fallback → Model A'
    )
    return fc


def cap_at_capacity(pred_series, capacity_series):
    return pred_series.combine(capacity_series, lambda p, c:
        min(p, c) if pd.notna(p) and pd.notna(c) else p)


def print_metrics(label, comp, pred_col):
    valid = comp[(comp['Actual'] > 0) & comp[pred_col].notna()].copy()
    valid['AbsErr'] = (valid['Actual'] - valid[pred_col]).abs()
    valid['PctErr'] = valid['AbsErr'] / valid['Actual']
    valid['SignedPct'] = (valid[pred_col] - valid['Actual']) / valid['Actual']
    mape = valid['PctErr'].mean() * 100
    wape = valid['AbsErr'].sum() / valid['Actual'].sum() * 100
    bias = valid['SignedPct'].mean() * 100
    print(f"  {label:<40} MAPE={mape:.1f}%  WAPE={wape:.1f}%  Bias={bias:+.1f}%  (n={len(valid)})")


def main():
    em, merged = load_data()

    # Training: everything before 25-26
    all_prior = sorted([s for s in merged['Season'].dropna().unique() if s < FORECAST_SEASON])
    filtered_train = get_training_df(merged, all_prior)

    # Build models
    (repeat_model, primary_sf, sf_ratio, primary, f1, f2, f3, f3a, f3b, f4, f5
     ) = build_hierarchy_models(filtered_train)
    name_model = build_name_model(filtered_train)

    # PWYW lift calibrated on prior-season PWYW events only (no leakage)
    pwyw_lift, pwyw_samples = build_pwyw_lift(
        merged, all_prior, repeat_model, primary_sf, sf_ratio,
        primary, f1, f2, f3, f3a, f3b, f4, f5)
    print(f"\nPWYW lift: {pwyw_lift:.2f}x  (calibrated on n={len(pwyw_samples)} prior-season PWYW events)")
    if len(pwyw_samples):
        print(pwyw_samples.to_string(index=False))

    # Get 25-26 completed events with actuals
    actuals_gb = ['EventId', 'EventName', 'EventClass', 'EventVenue',
                  'EventGenre', 'EventLoB', 'EventSubGenre', 'EventRepeat']
    for c in ('SeatFormat', 'Pricing', 'VenueType'):
        if c in merged.columns:
            actuals_gb.append(c)
    actuals_2526 = (
        merged[
            (merged['Season'] == FORECAST_SEASON) &
            (merged['EventType'] == 'Live') &
            (merged['EventStatus'] == 'Complete') &
            (merged['TicketStatus'] == 'Active') &
            (merged['Quantity'] > 0)
        ]
        .groupby(actuals_gb, group_keys=False, dropna=False)
        .agg(Actual=('Quantity', 'sum'))
        .reset_index()
    )

    # Capacity from manifest
    cap = em.drop_duplicates('EventId')[['EventId', 'EventCapacity']].copy()
    cap['EventCapacity'] = pd.to_numeric(cap['EventCapacity'], errors='coerce')
    actuals_2526 = actuals_2526.merge(cap, on='EventId', how='left')

    # Generate predictions
    fc = predict_model_a(actuals_2526, repeat_model, primary_sf, sf_ratio,
                         primary, f1, f2, f3, f3a, f3b, f4, f5, pwyw_lift=pwyw_lift)
    fc = predict_model_b(fc, name_model)
    fc['Pred_A'] = cap_at_capacity(fc['Pred_A'], fc['EventCapacity'])
    fc['Pred_B'] = cap_at_capacity(fc['Pred_B'], fc['EventCapacity'])

    # Build labelled history for artist adjustment training
    # Event-level actuals from training seasons
    hist_gb = ['EventId', 'EventName', 'EventClass', 'EventVenue',
               'EventGenre', 'EventLoB', 'EventSubGenre', 'EventRepeat']
    for c in ('SeatFormat', 'Pricing', 'VenueType'):
        if c in filtered_train.columns:
            hist_gb.append(c)
    hist_actuals = (
        filtered_train
        .groupby(hist_gb, group_keys=False, dropna=False)
        .agg(Actual=('Quantity', 'sum'))
        .reset_index()
    )
    hist_actuals = hist_actuals.merge(cap, on='EventId', how='left')
    hist_fc = predict_model_a(hist_actuals, repeat_model, primary_sf, sf_ratio,
                               primary, f1, f2, f3, f3a, f3b, f4, f5)

    fc = apply_artist_adjustment(
        fc,
        merged_history=hist_fc,
        actuals_history=hist_fc['Actual'],
        bucket_preds_history=hist_fc['Pred_A'],
    )
    print_adjustment_summary(fc)

    has_history = fc['NameSeasons'] >= 2
    n_total = len(fc)
    n_history = has_history.sum()

    # ── Summary metrics ──────────────────────────────────────────────────
    print("=" * 75)
    print(f"25-26 FORECAST vs ACTUALS  (n={n_total} completed events)")
    print("=" * 75)
    print_metrics("Model A: Hierarchy (all events)", fc, 'Pred_A')
    print_metrics("Model B: Name history + fallback (all events)", fc, 'Pred_B')
    print_metrics("Model A + Artist Bayesian adj (all events)", fc, 'Pred_Adj')
    print()
    print_metrics(f"Model A: Hierarchy  (prior-season events only, n={n_history})",
                  fc[has_history], 'Pred_A')
    print_metrics(f"Model B: Name history  (prior-season events only, n={n_history})",
                  fc[has_history], 'Pred_B')

    # ── Event-level detail ───────────────────────────────────────────────
    print("\nEVENT-LEVEL DETAIL")
    hdr = f"{'EventName':<45} {'Class':<10} {'SubGenre':<18} {'Actual':>7} {'Pred_A':>7} {'Pred_Adj':>8} {'Err_A':>7} {'Err_Adj':>8}  AdjSource"
    print(hdr)
    print("-" * len(hdr))
    for _, row in fc.sort_values(['EventClass', 'EventName']).iterrows():
        err_a   = f"{(row['Pred_A']   - row['Actual']) / row['Actual'] * 100:+.0f}%" if pd.notna(row['Pred_A']) else "N/A"
        err_adj = f"{(row['Pred_Adj'] - row['Actual']) / row['Actual'] * 100:+.0f}%" if pd.notna(row.get('Pred_Adj')) else "N/A"
        marker = "★" if row['NameSeasons'] >= 2 else " "
        print(f"{marker}{str(row['EventName']):<44} {str(row['EventClass']):<10} "
              f"{str(row['EventSubGenre']):<18} "
              f"{row['Actual']:>7.0f} {row['Pred_A']:>7.0f} {row.get('Pred_Adj', float('nan')):>8.0f} "
              f"{err_a:>7} {err_adj:>8}  {row.get('Adj_Source', '')}")
    print("  ★ = event has ≥2 seasons of prior name history")

    # ── Name model coverage ──────────────────────────────────────────────
    print(f"\nNAME MODEL COVERAGE")
    print(f"  Events with ≥2 seasons of name history: {n_history} / {n_total}")
    print(f"  New events (fallback to Model A):        {n_total - n_history} / {n_total}")

    # ── Write output ─────────────────────────────────────────────────────
    out = fc[['EventName', 'EventClass', 'EventVenue', 'EventGenre', 'EventLoB', 'EventSubGenre',
              'EventCapacity', 'Actual', 'Pred_A', 'FallbackLevel',
              'Pred_Adj', 'Pred_Adj_Lo', 'Pred_Adj_Hi', 'Adj_LogFactor', 'Adj_Source',
              'Pred_B', 'Model_B_Source', 'NameSeasons', 'NameWeightedAvg']].copy()
    out['Error_A_Pct']   = ((out['Pred_A']   - out['Actual']) / out['Actual'] * 100).round(1)
    out['Error_Adj_Pct'] = ((out['Pred_Adj'] - out['Actual']) / out['Actual'] * 100).round(1)
    out['Error_B_Pct']   = ((out['Pred_B']   - out['Actual']) / out['Actual'] * 100).round(1)

    with pd.ExcelWriter("Forecast_2526_Comparison.xlsx", engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="2526_Comparison", index=False)
        name_model.sort_values('NameSeasons', ascending=False).to_excel(
            writer, sheet_name="Name_History", index=False)

    print(f"\n✅ Written to Forecast_2526_Comparison.xlsx")


if __name__ == "__main__":
    main()
