"""
Artist popularity signal lookup with local JSON cache.

Fetches from Last.fm, Spotify (optional), and Wikipedia APIs.
Caches results so APIs are only called once per artist.
All lookups gracefully return None on failure so the adjustment
layer can fall back to the bucket prior.

Last.fm: set LASTFM_API_KEY (or hardcoded below).
Spotify: optional — set SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET.
Wikipedia: no auth required.
"""

import calendar
import json
import os
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta

CACHE_FILE = Path(__file__).parent / "artist_signals_cache.json"
CACHE_MAX_AGE_DAYS = 90   # re-fetch after 90 days

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "012c83319a3bcb27dd1420020a83ec55")


def _load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_fresh(entry):
    fetched = entry.get("fetched_at")
    if not fetched:
        return False
    age = datetime.utcnow() - datetime.fromisoformat(fetched)
    return age < timedelta(days=CACHE_MAX_AGE_DAYS)


# ── Spotify ──────────────────────────────────────────────────────────────────

def _spotify_token():
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    try:
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception:
        return None


def _spotify_lookup(artist_name, token):
    """Return {popularity: 0-100, monthly_listeners: int, followers: int} or None."""
    try:
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            params={"q": artist_name, "type": "artist", "limit": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("artists", {}).get("items", [])
        if not items:
            return None
        a = items[0]
        return {
            "spotify_popularity": a.get("popularity"),        # 0–100, genre-normalised
            "spotify_followers": a.get("followers", {}).get("total"),
        }
    except Exception:
        return None


# ── Last.fm ──────────────────────────────────────────────────────────────────

def _lastfm_lookup(artist_name):
    """Return {lastfm_listeners: int, lastfm_playcount: int} or None."""
    if not LASTFM_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={"method": "artist.getinfo", "artist": artist_name,
                    "api_key": LASTFM_API_KEY, "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        stats = resp.json().get("artist", {}).get("stats", {})
        listeners = int(stats.get("listeners", 0))
        playcount = int(stats.get("playcount", 0))
        return {"lastfm_listeners": listeners, "lastfm_playcount": playcount} if listeners else None
    except Exception:
        return None


# ── Wikipedia ────────────────────────────────────────────────────────────────

def _wikipedia_pageviews(artist_name):
    """
    Return average monthly page views (last 3 months), 0 if no Wikipedia page,
    or None if the fetch failed (caller should not cache None — retry later).
    """
    title = artist_name.replace(" ", "_")
    end = datetime.utcnow().replace(day=1)
    month_ranges = []
    for _ in range(3):
        end -= timedelta(days=1)
        y, m = end.year, end.month
        last_day = calendar.monthrange(y, m)[1]
        month_ranges.append((end.strftime("%Y%m01"), end.strftime(f"%Y%m{last_day:02d}")))
        end = end.replace(day=1)

    views = []
    fetch_errors = 0
    not_found = 0
    for ym_start, ym_end in month_ranges:
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"en.wikipedia.org/all-access/all-agents/{title}/monthly/{ym_start}/{ym_end}")
        try:
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "MW-forecast/1.0"})
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    views.append(items[0].get("views", 0))
            elif resp.status_code == 404:
                not_found += 1   # article doesn't exist
            time.sleep(0.2)      # polite to Wikimedia
        except Exception:
            fetch_errors += 1

    if views:
        return int(sum(views) / len(views))
    if not_found == 3:
        return 0    # confirmed no Wikipedia page — cache as 0, don't retry
    if fetch_errors > 0:
        return None  # transient failure — caller should not cache this


# ── Public API ───────────────────────────────────────────────────────────────

def get_signals(artist_name, force_refresh=False):
    """
    Return popularity signals for artist_name, using cache when fresh.

    Returns dict with keys:
      spotify_popularity   (0–100 or None)
      spotify_followers    (int or None)
      wikipedia_monthly_views (int or None)
      fetched_at           (ISO timestamp)

    Always returns a dict (never raises); missing signals are None.
    """
    cache = _load_cache()
    key = artist_name.strip().lower()

    if not force_refresh and key in cache and _cache_fresh(cache[key]):
        entry = cache[key]
        # If cache is fresh but wiki was never successfully fetched, retry wiki only
        if entry.get("wikipedia_monthly_views") is None:
            wiki = _wikipedia_pageviews(artist_name)
            if wiki is not None:   # 0 = confirmed no page; positive = got views
                entry["wikipedia_monthly_views"] = wiki
                cache[key] = entry
                _save_cache(cache)
        return cache[key]

    signals = {
        "artist_name": artist_name,
        "spotify_popularity": None,
        "spotify_followers": None,
        "lastfm_listeners": None,
        "lastfm_playcount": None,
        "wikipedia_monthly_views": None,
        "fetched_at": datetime.utcnow().isoformat(),
    }

    # Last.fm
    lfm = _lastfm_lookup(artist_name)
    if lfm:
        signals.update(lfm)

    # Spotify (optional)
    token = _spotify_token()
    if token:
        sp = _spotify_lookup(artist_name, token)
        if sp:
            signals.update(sp)

    # Wikipedia — only store if fetch succeeded (None = transient failure, don't cache)
    wiki = _wikipedia_pageviews(artist_name)
    if wiki is not None:
        signals["wikipedia_monthly_views"] = wiki

    cache[key] = signals
    _save_cache(cache)
    return signals


def bulk_fetch(artist_names, verbose=True):
    """Fetch and cache signals for a list of artist names."""
    results = {}
    for i, name in enumerate(artist_names):
        if verbose:
            print(f"  [{i+1}/{len(artist_names)}] {name} ...", end=" ", flush=True)
        sig = get_signals(name)
        results[name] = sig
        if verbose:
            sp = sig.get("spotify_popularity")
            wv = sig.get("wikipedia_monthly_views")
            print(f"Spotify={sp}  Wiki={wv:,}" if wv else f"Spotify={sp}  Wiki=–")
    return results


def cache_summary() -> None:
    """Print a summary of what's in the cache."""
    cache = _load_cache()
    if not cache:
        print("Cache is empty.")
        return
    print(f"{'Artist':<50} {'Spotify':>8} {'Followers':>10} {'LFM List':>10} {'Wiki/mo':>10}  {'Fetched'}")
    print("-" * 107)
    for entry in sorted(cache.values(), key=lambda x: x.get("artist_name", "")):
        sp  = entry.get("spotify_popularity", "–")
        fol = entry.get("spotify_followers")
        lfm = entry.get("lastfm_listeners")
        wv  = entry.get("wikipedia_monthly_views")
        ts  = entry.get("fetched_at", "")[:10]
        fol_s = f"{fol:,}" if fol else "–"
        lfm_s = f"{lfm:,}" if lfm else "–"
        wv_s  = f"{wv:,}" if wv else "–"
        print(f"{entry.get('artist_name',''):<50} {str(sp):>8} {fol_s:>10} {lfm_s:>10} {wv_s:>10}  {ts}")


if __name__ == "__main__":
    # Quick test / cache viewer
    import sys
    if len(sys.argv) > 1:
        name = " ".join(sys.argv[1:])
        print(f"Fetching signals for: {name}")
        sig = get_signals(name, force_refresh=True)
        print(json.dumps(sig, indent=2))
    else:
        cache_summary()
