"""
Bayesian artist-popularity adjustment layer.

Sits on top of the base forecast model as an optional, easily-disabled
extension. When ARTIST_ADJUSTMENT_ENABLED = False (the default) the
function apply_artist_adjustment() is a no-op that returns predictions
unchanged.

When enabled:
  1. Loads cached artist signals (Spotify popularity, Wikipedia views).
  2. Trains a BayesianRidge regression on historical events where signals
     are available:  log(actual / bucket_prior) ~ f(signals)
  3. Applies the learned adjustment to new-event predictions.
  4. Returns both the adjusted prediction and a credible interval
     (5th / 95th percentile) useful for pessimistic/optimistic budgeting.

Requires at least MIN_TRAINING_EVENTS historical events with signals
before the adjustment fires; otherwise falls back to the bucket prior.
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# ── Master switch ─────────────────────────────────────────────────────────────
ARTIST_ADJUSTMENT_ENABLED = True    # flip to False to disable

MIN_TRAINING_EVENTS = 15            # minimum labelled history before adjusting
SIGNAL_COLS = [                     # features fed to the regression
    "spotify_popularity",
    "log_spotify_followers",
    "log_lastfm_listeners",
    "log_wikipedia_monthly_views",
]

# Genre-fit weights: how well an artist's global LFM audience
# maps to MW's classical/arts subscriber base (1.0 = full signal,
# 0.0 = global popularity irrelevant to MW draw).
# Applied as a multiplier on log_lastfm_listeners before regression.
GENRE_FIT = {
    "Orchestra":    1.0,
    "Recital":      1.0,
    "Chamber":      1.0,
    "Choral":       0.9,
    "Organ":        0.9,
    "Bach Choir":   0.9,
    "Ballet":       0.8,
    "Cantata":      0.9,
    "Jazz":         0.5,
    "Americana":    0.3,
    "World":        0.3,
    "Contemporary": 0.5,
    "Gospel":       0.3,
    "Folk":         0.3,
}
GENRE_FIT_DEFAULT = 0.7   # fallback for unmapped SubGenres

# For events where the lead artist is not the primary draw signal,
# map EventName (lowercase, partial match) → artist cache key.
# Used for orchestra events where the soloist is the real signal.
EVENT_TO_ARTIST = {
    "orchestre national de france":         "daniil trifonov",
    "orpheus - beethoven":                  "brad mehldau",
    "vienna orch. & entremont":             "philippe entremont",
    "yakushev & nhso":                      "boris yakushev",
    "the knights with aaron diehl":         "aaron diehl",
    "mso and garrick ohlsson":              "garrick ohlsson",
    "asmf & denk":                          "jeremy denk",
    "dinnerstein & orchestra":              "simone dinnerstein",
    "midori & festival strings":            "midori",
}

warnings.filterwarnings("ignore", category=UserWarning)


# ── Signal helpers ────────────────────────────────────────────────────────────

def _extract_lead_artist(event_name: str) -> str:
    """
    Heuristic: return the lead artist name from an event name string.
    Splits on ' & ', ' with ', ' and ', ','; takes the first token.
    Override by adding an 'ArtistLookup' column to EventManifest instead.
    """
    for sep in [" & ", " with ", " and ", ","]:
        if sep in event_name:
            return event_name.split(sep)[0].strip()
    return event_name.strip()


def _build_signal_features(names, subgenres=None):
    """
    Look up cached signals for a series of event names.
    subgenres: optional Series (same index) of EventSubGenre values for
               genre-fit weighting of log_lastfm_listeners.
    Returns a DataFrame with SIGNAL_COLS (NaN where not cached).
    """
    try:
        from artist_signals import _load_cache
    except ImportError:
        return pd.DataFrame(index=names.index,
                            columns=SIGNAL_COLS, dtype=float)

    cache = _load_cache()
    rows = []
    for idx, name in names.items():
        name_lower = str(name).strip().lower()
        entry = None

        # 1. Check EVENT_TO_ARTIST override (partial match on event name)
        for pattern, artist_key in EVENT_TO_ARTIST.items():
            if pattern in name_lower:
                entry = cache.get(artist_key)
                break

        # 2. Exact lead-artist match in cache
        if entry is None:
            lead = _extract_lead_artist(str(name)).strip().lower()
            entry = cache.get(lead)

        # 3. Fuzzy: cache key is substring of name, or vice versa
        if entry is None:
            for ck, cv in cache.items():
                if ck in name_lower or name_lower in ck:
                    entry = cv
                    break

        entry = entry or {}
        sp   = entry.get("spotify_popularity")
        fol  = entry.get("spotify_followers")
        wiki = entry.get("wikipedia_monthly_views")
        lfm  = entry.get("lastfm_listeners")

        # Genre-fit weight on LFM signal
        subgenre = str(subgenres.loc[idx]) if subgenres is not None and idx in subgenres.index else ""
        fit = GENRE_FIT.get(subgenre, GENRE_FIT_DEFAULT)
        lfm_fitted = lfm * fit if lfm is not None else None

        rows.append({
            "spotify_popularity":          float(sp)                if sp          is not None else np.nan,
            "log_spotify_followers":       np.log1p(fol)            if fol         is not None else np.nan,
            "log_lastfm_listeners":        np.log1p(lfm_fitted)     if lfm_fitted  is not None else np.nan,
            "log_wikipedia_monthly_views": np.log1p(wiki)           if wiki        is not None else np.nan,
        })
    return pd.DataFrame(rows, index=names.index)


# ── Training ──────────────────────────────────────────────────────────────────

def _train_adjustment_model(merged: pd.DataFrame, bucket_preds: pd.Series):
    """
    Train a BayesianRidge on historical events.

    merged       — full training DataFrame with EventName, Season, Actual
    bucket_preds — Series (same index as merged) of base-model predictions

    Returns (model, scaler, feature_cols) or None if insufficient data.
    """
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler

    df = merged.copy()
    df["bucket_pred"] = bucket_preds
    df = df[df["bucket_pred"] > 0].copy()

    subgenres = df["EventSubGenre"] if "EventSubGenre" in df.columns else None
    feats = _build_signal_features(df["EventName"], subgenres=subgenres)
    df = pd.concat([df, feats], axis=1)

    # Keep rows with at least one valid signal and a real actual
    valid_cols = [c for c in SIGNAL_COLS if c in df.columns]
    has_signal = df[valid_cols].notna().any(axis=1)
    has_actual = df["Actual"] > 0
    train = df[has_signal & has_actual].copy()

    if len(train) < MIN_TRAINING_EVENTS:
        return None

    # Drop columns that are entirely NaN, impute the rest with column median
    valid_cols = [c for c in valid_cols if train[c].notna().any()]
    if not valid_cols:
        return None
    for col in valid_cols:
        train[col] = train[col].fillna(train[col].median())

    X = train[valid_cols].values
    y = np.log(train["Actual"].values / train["bucket_pred"].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = BayesianRidge(compute_score=True)
    model.fit(X_scaled, y)

    return model, scaler, valid_cols


# ── Application ───────────────────────────────────────────────────────────────

def apply_artist_adjustment(
    fc,
    merged_history=None,
    actuals_history=None,
    bucket_preds_history=None,
    pred_col="Pred_A",
    ci_pct=0.90,
):
    """
    Apply (or skip) the artist-popularity adjustment.

    Parameters
    ----------
    fc                    : forecast DataFrame with pred_col already populated
    merged_history        : historical event rows (for training)
    actuals_history       : historical actuals aligned to merged_history
    bucket_preds_history  : historical bucket predictions aligned to merged_history
    pred_col              : column in fc to adjust (default 'Pred_A')
    ci_pct                : width of credible interval (default 90%)

    Returns fc with additional columns:
      Pred_Adj            — adjusted point estimate (= Pred_A when disabled)
      Pred_Adj_Lo         — lower credible interval bound
      Pred_Adj_Hi         — upper credible interval bound
      Adj_LogFactor       — log-scale adjustment applied (0.0 when disabled)
      Adj_Source          — 'ArtistSignal' | 'Fallback (no signal)' | 'Disabled'
    """
    result = fc.copy()

    # Default: passthrough
    result["Pred_Adj"]      = result[pred_col]
    result["Pred_Adj_Lo"]   = result[pred_col]
    result["Pred_Adj_Hi"]   = result[pred_col]
    result["Adj_LogFactor"] = 0.0
    result["Adj_Source"]    = "Disabled"

    if not ARTIST_ADJUSTMENT_ENABLED:
        return result

    # ── Training ──────────────────────────────────────────────────────────
    trained = None
    if (merged_history is not None and
            actuals_history is not None and
            bucket_preds_history is not None):
        train_df = merged_history.copy()
        train_df["Actual"] = actuals_history.values
        trained = _train_adjustment_model(train_df, bucket_preds_history)

    if trained is None:
        result["Adj_Source"] = "Fallback (insufficient training data)"
        return result

    model, scaler, feature_cols = trained

    # ── Prediction ────────────────────────────────────────────────────────
    subgenres = result["EventSubGenre"] if "EventSubGenre" in result.columns else None
    feats = _build_signal_features(result["EventName"], subgenres=subgenres)
    alpha = 1.0 - ci_pct
    z = _normal_ppf(1 - alpha / 2)          # e.g. 1.645 for 90% CI

    for idx, row in result.iterrows():
        base = row[pred_col]
        if pd.isna(base) or base <= 0:
            continue

        feat_row = feats.loc[idx, [c for c in feature_cols if c in feats.columns]] if idx in feats.index else pd.Series(dtype=float)
        has_any = feat_row.notna().any()

        if not has_any:
            result.at[idx, "Adj_Source"] = "Fallback (no signal)"
            continue

        x = feat_row.fillna(feat_row.median() if feat_row.notna().any() else 0).values.reshape(1, -1)
        x_scaled = scaler.transform(x)

        log_adj, log_std = model.predict(x_scaled, return_std=True)
        log_adj = float(log_adj[0])
        log_std = float(log_std[0])

        adj_pred = base * np.exp(log_adj)
        lo = base * np.exp(log_adj - z * log_std)
        hi = base * np.exp(log_adj + z * log_std)

        # Cap at capacity if available
        cap = row.get("EventCapacity")
        if pd.notna(cap) and cap > 0:
            adj_pred = min(adj_pred, cap)
            hi       = min(hi, cap)

        result.at[idx, "Pred_Adj"]      = round(adj_pred)
        result.at[idx, "Pred_Adj_Lo"]   = round(lo)
        result.at[idx, "Pred_Adj_Hi"]   = round(hi)
        result.at[idx, "Adj_LogFactor"] = round(log_adj, 3)
        result.at[idx, "Adj_Source"]    = "ArtistSignal"

    return result


# ── Utility ───────────────────────────────────────────────────────────────────

def _normal_ppf(p: float) -> float:
    """Approximate normal percent-point function (no scipy needed)."""
    # Abramowitz & Stegun rational approximation, max error < 4.5e-4
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    t = np.sqrt(-2.0 * np.log(1.0 - p if p > 0.5 else p))
    num = c[0] + c[1]*t + c[2]*t**2
    den = 1 + d[0]*t + d[1]*t**2 + d[2]*t**3
    x   = t - num / den
    return x if p > 0.5 else -x


def print_adjustment_summary(fc: pd.DataFrame, pred_col: str = "Pred_A") -> None:
    """Print a comparison of base vs. adjusted predictions."""
    if not ARTIST_ADJUSTMENT_ENABLED:
        print("Artist adjustment is DISABLED (ARTIST_ADJUSTMENT_ENABLED = False)")
        return

    cols = ["EventName", "EventClass", pred_col,
            "Pred_Adj", "Pred_Adj_Lo", "Pred_Adj_Hi", "Adj_LogFactor", "Adj_Source"]
    show = fc[[c for c in cols if c in fc.columns]].copy()

    print(f"\n{'EventName':<45} {'Base':>7} {'Adj':>7} {'Lo':>7} {'Hi':>7}  {'Factor':>7}  Source")
    print("-" * 100)
    for _, row in show.iterrows():
        factor_s = f"{row.get('Adj_LogFactor', 0):+.2f}" if row.get("Adj_Source") == "ArtistSignal" else "  n/a"
        print(f"{str(row['EventName']):<45} "
              f"{row.get(pred_col, float('nan')):>7.0f} "
              f"{row.get('Pred_Adj', float('nan')):>7.0f} "
              f"{row.get('Pred_Adj_Lo', float('nan')):>7.0f} "
              f"{row.get('Pred_Adj_Hi', float('nan')):>7.0f}  "
              f"{factor_s:>7}  {row.get('Adj_Source', '')}")


if __name__ == "__main__":
    print("Artist adjustment module loaded.")
    print(f"ARTIST_ADJUSTMENT_ENABLED = {ARTIST_ADJUSTMENT_ENABLED}")
    print(f"MIN_TRAINING_EVENTS       = {MIN_TRAINING_EVENTS}")
    print(f"Signal features           = {SIGNAL_COLS}")
