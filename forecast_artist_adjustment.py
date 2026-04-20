"""
Bayesian artist-popularity adjustment layer.

Uses an informative-prior Bayesian linear regression:
  log(actual / bucket_prior) ~ β₀ + β_lfm·log_lfm_fitted + β_wiki·log_wiki + ...

Prior means are calibrated from domain knowledge (not learned from scratch),
so the model produces sensible adjustments from day one — even with zero
MW training data. The posterior tightens as labelled seasons accumulate.

With ARTIST_ADJUSTMENT_ENABLED = False the module is a complete no-op.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ── Master switch ──────────────────────────────────────────────────────────────
ARTIST_ADJUSTMENT_ENABLED = True    # flip to False to disable entirely

# ── Signal columns fed to the regression ──────────────────────────────────────
SIGNAL_COLS = [
    "log_lastfm_listeners",        # genre-fit adjusted
    "log_wikipedia_monthly_views",
    "log_deezer_albums",           # discography depth — classical-friendly
]

# ── Informative priors ─────────────────────────────────────────────────────────
# Each entry: (prior_mean, prior_std) for the regression coefficient β.
#
# Calibration rationale (log feature space, log-ratio target):
#   Yo-Yo Ma  (LFM 503K, log=13.1) → typically ~1.5-1.7x bucket
#   Emi Ferguson (LFM 147, log=5.0) → typically ~0.75x bucket
#   Implied β_lfm ≈ log(1.6/0.77) / (13.1-5.0) ≈ 0.09
#
# Prior std is TIGHT (0.015, 0.010) — with only ~47 labelled obs and small
# within-bucket LFM variation, we're explicitly in the prior-dominated regime.
# Data can pull β toward 0 over many seasons, but not drag it there in one shot.
# Features are centered on training medians (see _build_model) so β acts on
# within-training deviation, not absolute log-listener counts.
#
# intercept: expect no systematic bias from signals alone (centered features).
# noise_std:  log-scale residual ≈ our observed ~27% WAPE → log(1.27)≈0.24
INFORMED_PRIORS = {
    "intercept":                   (0.00, 0.10),
    "log_lastfm_listeners":        (0.09, 0.015),
    "log_wikipedia_monthly_views": (0.05, 0.010),
    # Deezer album count: Yo-Yo Ma 183 (log≈5.2) vs. Fung 4 (log≈1.6). A ~4-unit
    # log spread mapping to ~1.25× draw range → β≈0.05. Prior-dominated (std 0.010).
    "log_deezer_albums":           (0.05, 0.010),
}
NOISE_STD = 0.27   # log-scale residual noise

# Cold-start centers used when no training history is available.
# Rough medians from the cache: typical classical artist ~log(8000)≈9.0 LFM,
# ~log(400)≈6.0 monthly Wikipedia views, ~log(20)≈3.0 Deezer albums.
DEFAULT_CENTERS = {
    "log_lastfm_listeners":        9.0,
    "log_wikipedia_monthly_views": 6.0,
    "log_deezer_albums":           3.0,
}

# ── Signal-strength gate ──────────────────────────────────────────────────────
# The web signals measure global reach, not MW classical subscriber loyalty.
# Firing the adjustment on weak-signal or wrong-genre artists produced large
# misses on historical seasons (e.g. Silkroad, Orchestre Métropolitain,
# Tchaikovsky Trio). Require:
#   1. Genre-fit ≥ 0.7 — classical/chamber/recital/choral/ballet
#   2. Substantive signal — Wiki ≥ 200/mo OR LFM (pre-discount) ≥ 2000
#      OR Deezer discography ≥ 3 albums (catches classical recitalists
#      like Zlatomir Fung, Hermitage Piano Trio — real pros w/ thin LFM).
# Each signal is gated individually; features that fail are NaN'd and
# imputed to the training center (neutral contribution). Events where
# no signal passes fall back to the bucket prior.
MIN_GENRE_FIT          = 0.7
MIN_WIKI_MONTHLY_VIEWS = 200
MIN_LFM_LISTENERS_RAW  = 2000
MIN_DEEZER_ALBUMS      = 3

# ── Genre-fit weights ──────────────────────────────────────────────────────────
# Discounts LFM signal where global audience ≠ MW classical subscriber base.
GENRE_FIT = {
    "Orchestra":    1.0,
    "Recital":      1.0,
    "Chamber":      1.0,
    "Choral":       0.9,
    "Organ":        0.9,
    "Bach Choir":   0.9,
    "Cantata":      0.9,
    "Ballet":       0.8,
    # World/Crossover — merged SubGenre for global/genre-bending acts at MW.
    # Averages the prior Crossover (0.8, classical-curious audience: MacMaster,
    # Silkroad, Sam Bush) and World (0.3, spectacle/ensemble: Bollywood,
    # Chinese Acrobats) priors, gated by MIN_GENRE_FIT=0.7 so it falls below
    # the signal threshold by default unless lifted by strong Wiki/Deezer.
    "World/Crossover": 0.5,
    "Jazz":         0.5,
    "Contemporary": 0.5,
    "Americana":    0.3,
    "Gospel":       0.3,
    "Folk":         0.3,
}
GENRE_FIT_DEFAULT = 0.7

# ── Orchestra soloist overrides ────────────────────────────────────────────────
# Maps EventName fragment (lowercase) → artist cache key for the featured soloist.
EVENT_TO_ARTIST = {
    "orchestre national de france":   "daniil trifonov",
    "orpheus - beethoven":            "brad mehldau",
    "vienna orch. & entremont":       "philippe entremont",
    "yakushev & nhso":                "boris yakushev",
    "the knights with aaron diehl":   "aaron diehl",
    "mso and garrick ohlsson":        "garrick ohlsson",
    "asmf & denk":                    "jeremy denk",
    "dinnerstein & orchestra":        "simone dinnerstein",
    "midori & festival strings":      "midori",
    # new entries — fix missing signals for top positive surprises
    "yo-yo ma":                       "yo-yo ma",
    "macmaster":                      "natalie macmaster",
    "silkroad":                       "silk road ensemble",
    "dance theater of harlem":        "dance theater of harlem",
}


# ── Bayesian linear regression ─────────────────────────────────────────────────

class InformedBayesianRegressor:
    """
    Conjugate Bayesian linear regression with explicit informative priors.

    Model:  y = X β + ε,  ε ~ N(0, noise_std²)
    Prior:  β ~ N(prior_mean, diag(prior_std²))
    Posterior updated analytically — closed-form normal-normal conjugate update.

    With zero training data the posterior equals the prior, giving sensible
    prior-based adjustments immediately. The posterior tightens as data arrives.
    """

    def __init__(self, feature_cols, priors, noise_std):
        self.feature_cols = feature_cols          # includes 'intercept'
        self.noise_std    = noise_std

        self.prior_mean = np.array([priors.get(f, (0.0, 0.10))[0] for f in feature_cols])
        self.prior_std  = np.array([priors.get(f, (0.0, 0.10))[1] for f in feature_cols])

        # Posterior initialised to prior
        self.post_mean = self.prior_mean.copy()
        self.post_cov  = np.diag(self.prior_std ** 2)
        self.n_obs     = 0

    def update(self, X, y):
        """Closed-form Bayesian update given design matrix X and targets y."""
        if len(X) == 0:
            return                              # no data → keep prior
        sigma2     = self.noise_std ** 2
        prior_prec = np.diag(1.0 / (self.prior_std ** 2))
        post_prec  = prior_prec + X.T @ X / sigma2
        self.post_cov  = np.linalg.inv(post_prec)
        self.post_mean = self.post_cov @ (
            prior_prec @ self.prior_mean + X.T @ y / sigma2
        )
        self.n_obs = len(y)

    def predict(self, x_vec):
        """
        Posterior predictive mean and std for feature vector x_vec.
        Variance = epistemic (parameter uncertainty) + aleatoric (noise).
        """
        mean         = float(x_vec @ self.post_mean)
        epistemic_var = float(x_vec @ self.post_cov @ x_vec)
        total_std    = np.sqrt(epistemic_var + self.noise_std ** 2)
        return mean, total_std

    def posterior_summary(self):
        """Return a dict of feature → (posterior_mean, posterior_std)."""
        return {
            f: (round(float(self.post_mean[i]), 4),
                round(float(np.sqrt(self.post_cov[i, i])), 4))
            for i, f in enumerate(self.feature_cols)
        }


# ── Signal lookup helpers ──────────────────────────────────────────────────────

def _extract_lead_artist(event_name):
    for sep in [" & ", " with ", " and ", ","]:
        if sep in event_name:
            return event_name.split(sep)[0].strip()
    return event_name.strip()


def _build_signal_features(names, subgenres=None):
    """
    Look up cached signals for a Series of event names.
    Returns DataFrame with SIGNAL_COLS (NaN where not cached).
    """
    try:
        from artist_signals import _load_cache
    except ImportError:
        return pd.DataFrame(index=names.index, columns=SIGNAL_COLS, dtype=float)

    cache = _load_cache()
    rows  = []
    for idx, name in names.items():
        name_lower = str(name).strip().lower()
        entry = None

        for pattern, artist_key in EVENT_TO_ARTIST.items():
            if pattern in name_lower:
                entry = cache.get(artist_key)
                break
        if entry is None:
            lead  = _extract_lead_artist(str(name)).strip().lower()
            entry = cache.get(lead)
        if entry is None:
            for ck, cv in cache.items():
                if ck in name_lower or name_lower in ck:
                    entry = cv
                    break
        entry = entry or {}

        wiki   = entry.get("wikipedia_monthly_views")
        lfm    = entry.get("lastfm_listeners")
        dz_alb = entry.get("deezer_albums")

        subgenre   = str(subgenres.loc[idx]) if subgenres is not None and idx in subgenres.index else ""
        fit        = GENRE_FIT.get(subgenre, GENRE_FIT_DEFAULT)
        fit_ok     = fit >= MIN_GENRE_FIT
        lfm_fitted = lfm * fit if lfm is not None else None

        # Per-feature gate — all require classical-leaning genre.
        wiki_ok = fit_ok and wiki   is not None and wiki   >= MIN_WIKI_MONTHLY_VIEWS
        lfm_ok  = fit_ok and lfm    is not None and lfm    >= MIN_LFM_LISTENERS_RAW
        dz_ok   = fit_ok and dz_alb is not None and dz_alb >= MIN_DEEZER_ALBUMS

        if not (wiki_ok or lfm_ok or dz_ok):
            rows.append({c: np.nan for c in
                         ["log_lastfm_listeners", "log_wikipedia_monthly_views",
                          "log_deezer_albums"]})
            continue

        rows.append({
            "log_lastfm_listeners":        np.log1p(lfm_fitted) if lfm_ok else np.nan,
            "log_wikipedia_monthly_views": np.log1p(wiki)       if wiki_ok else np.nan,
            "log_deezer_albums":           np.log1p(dz_alb)     if dz_ok   else np.nan,
        })
    return pd.DataFrame(rows, index=names.index)


def _make_feature_vector(feat_row, feature_cols, impute_vals):
    """
    Build a 1-D feature vector including intercept.
    Missing values imputed with training medians (or 0 for intercept).
    """
    vec = []
    for f in feature_cols:
        if f == "intercept":
            vec.append(1.0)
        else:
            val = feat_row.get(f, np.nan)
            if pd.notna(val):
                vec.append(float(val))
            else:
                imp = impute_vals.get(f, 0.0)
                vec.append(float(imp) if pd.notna(imp) else 0.0)
    return np.array(vec)


# ── Model build + update ───────────────────────────────────────────────────────

def _build_model(merged_history, actuals_history, bucket_preds_history):
    """
    Construct an InformedBayesianRegressor and update it with labelled history.
    Always returns a model (prior-only if no usable training data).

    Returns (model, active_sigs, impute, centers). Features are centered on
    training medians before fitting so β priors act on within-training
    deviation, not absolute log-signal levels — otherwise the intercept has
    to fight β×mean(feature) and the posterior collapses β toward 0.
    """
    # Determine which signal columns are actually present in the cache
    active_sigs = ["intercept"] + [c for c in SIGNAL_COLS
                                   if c in INFORMED_PRIORS]
    model    = InformedBayesianRegressor(active_sigs, INFORMED_PRIORS, NOISE_STD)
    sig_cols = [c for c in active_sigs if c != "intercept"]
    # Imputation in centered space is 0 (= median - median)
    zero_impute = {c: 0.0 for c in sig_cols}

    if merged_history is None or actuals_history is None or bucket_preds_history is None:
        return model, active_sigs, zero_impute, dict(DEFAULT_CENTERS)

    df = merged_history.copy()
    df["Actual"]      = actuals_history.values
    df["bucket_pred"] = bucket_preds_history.values
    df = df[(df["bucket_pred"] > 0) & (df["Actual"] > 0)].reset_index(drop=True)

    subgenres = df["EventSubGenre"] if "EventSubGenre" in df.columns else None
    feats     = _build_signal_features(df["EventName"], subgenres=subgenres)
    for col in feats.columns:
        df[col] = feats[col].values

    has_signal = df[sig_cols].notna().any(axis=1)
    train      = df[has_signal].copy()

    if train.empty:
        return model, active_sigs, zero_impute, dict(DEFAULT_CENTERS)

    # Training-median centers; features are subtracted before fitting so β
    # applies to "how much above/below a typical trained artist," not absolute log.
    centers = {c: float(train[c].median()) for c in sig_cols if c in train.columns}
    for c in sig_cols:
        if c in train.columns:
            train[c] = train[c] - centers[c]

    rows = []
    ys   = []
    for _, row in train.iterrows():
        feat_row = row[sig_cols] if all(c in row.index for c in sig_cols) else pd.Series(dtype=float)
        x = _make_feature_vector(feat_row, active_sigs, zero_impute)
        y = np.log(row["Actual"] / row["bucket_pred"])
        if np.isfinite(y) and np.isfinite(x).all():
            rows.append(x)
            ys.append(y)

    if rows:
        model.update(np.array(rows), np.array(ys))

    return model, active_sigs, zero_impute, centers


# ── Public API ─────────────────────────────────────────────────────────────────

def apply_artist_adjustment(
    fc,
    merged_history=None,
    actuals_history=None,
    bucket_preds_history=None,
    pred_col="Pred_A",
    ci_pct=0.90,
):
    """
    Apply the artist-popularity Bayesian adjustment.

    Always fires when ARTIST_ADJUSTMENT_ENABLED=True — even with no training
    data, the prior produces a sensible first-pass adjustment.

    Adds columns: Pred_Adj, Pred_Adj_Lo, Pred_Adj_Hi, Adj_LogFactor, Adj_Source.
    """
    result = fc.copy()
    result["Pred_Adj"]      = result[pred_col]
    result["Pred_Adj_Lo"]   = result[pred_col]
    result["Pred_Adj_Hi"]   = result[pred_col]
    result["Adj_LogFactor"] = 0.0
    result["Adj_Source"]    = "Disabled"

    if not ARTIST_ADJUSTMENT_ENABLED:
        return result

    model, feature_cols, impute, centers = _build_model(
        merged_history, actuals_history, bucket_preds_history
    )

    subgenres = result["EventSubGenre"] if "EventSubGenre" in result.columns else None
    feats     = _build_signal_features(result["EventName"], subgenres=subgenres)
    z         = _normal_ppf(1 - (1 - ci_pct) / 2)

    sig_cols = [c for c in feature_cols if c != "intercept"]

    for idx, row in result.iterrows():
        base = row[pred_col]
        if pd.isna(base) or base <= 0:
            continue

        feat_row  = feats.loc[idx].copy() if idx in feats.index else pd.Series(dtype=float)
        has_signal = feat_row[sig_cols].notna().any() if sig_cols else False

        if not has_signal:
            result.at[idx, "Adj_Source"] = "Fallback (no signal)"
            continue

        for c in sig_cols:
            if c in feat_row.index and pd.notna(feat_row[c]):
                feat_row[c] = feat_row[c] - centers.get(c, 0.0)

        x_vec   = _make_feature_vector(feat_row, feature_cols, impute)
        log_adj, log_std = model.predict(x_vec)

        adj_pred = base * np.exp(log_adj)
        lo       = base * np.exp(log_adj - z * log_std)
        hi       = base * np.exp(log_adj + z * log_std)

        cap = row.get("EventCapacity")
        if pd.notna(cap) and cap > 0:
            adj_pred = min(adj_pred, cap)
            hi       = min(hi, cap)

        n_label = f" ({model.n_obs} obs)" if model.n_obs else " (prior only)"
        if not (np.isfinite(adj_pred) and np.isfinite(lo) and np.isfinite(hi)):
            result.at[idx, "Adj_Source"] = "Fallback (no signal)"
            continue
        result.at[idx, "Pred_Adj"]      = round(adj_pred)
        result.at[idx, "Pred_Adj_Lo"]   = round(lo)
        result.at[idx, "Pred_Adj_Hi"]   = round(hi)
        result.at[idx, "Adj_LogFactor"] = round(log_adj, 3)
        result.at[idx, "Adj_Source"]    = f"ArtistSignal{n_label}"

    return result


def print_adjustment_summary(fc, pred_col="Pred_A"):
    if not ARTIST_ADJUSTMENT_ENABLED:
        print("Artist adjustment is DISABLED (ARTIST_ADJUSTMENT_ENABLED = False)")
        return

    print(f"\n{'EventName':<45} {'Base':>7} {'Adj':>7} {'Lo':>7} {'Hi':>7}  {'Factor':>7}  Source")
    print("-" * 105)
    for _, row in fc.iterrows():
        factor_s = f"{row.get('Adj_LogFactor', 0):+.2f}" if "ArtistSignal" in str(row.get("Adj_Source")) else "  n/a"
        print(f"{str(row['EventName']):<45} "
              f"{row.get(pred_col, float('nan')):>7.0f} "
              f"{row.get('Pred_Adj', float('nan')):>7.0f} "
              f"{row.get('Pred_Adj_Lo', float('nan')):>7.0f} "
              f"{row.get('Pred_Adj_Hi', float('nan')):>7.0f}  "
              f"{factor_s:>7}  {row.get('Adj_Source', '')}")


def _normal_ppf(p):
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    t = np.sqrt(-2.0 * np.log(1.0 - p if p > 0.5 else p))
    num = c[0] + c[1]*t + c[2]*t**2
    den = 1 + d[0]*t + d[1]*t**2 + d[2]*t**3
    x   = t - num / den
    return x if p > 0.5 else -x


if __name__ == "__main__":
    print(f"ARTIST_ADJUSTMENT_ENABLED = {ARTIST_ADJUSTMENT_ENABLED}")
    print(f"NOISE_STD                 = {NOISE_STD}")
    print("\nInformed priors:")
    for f, (m, s) in INFORMED_PRIORS.items():
        print(f"  {f:<35} mean={m:+.4f}  std={s:.4f}")
