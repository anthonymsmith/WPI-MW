"""
Artist popularity signal lookup with local JSON cache.

Fetches from Spotify and Wikipedia APIs; caches results so API is only
called once per artist. All lookups gracefully return None on failure
so the adjustment layer can fall back to the bucket prior.

Requires environment variables (or .env file):
  SPOTIFY_CLIENT_ID
  SPOTIFY_CLIENT_SECRET

Wikipedia requires no auth.
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


# ── Wikipedia ────────────────────────────────────────────────────────────────

def _wikipedia_pageviews(artist_name):
    """Return average monthly page views (last 3 months) or None."""
    # Normalise name to Wikipedia title format
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
    for ym_start, ym_end in month_ranges:
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"en.wikipedia.org/all-access/all-agents/{title}/monthly/{ym_start}/{ym_end}")
        try:
            resp = requests.get(url, timeout=10,
                                headers={"User-Agent": "MW-forecast/1.0"})
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    views.append(items[0].get("views", 0))
            time.sleep(0.1)   # be polite to Wikimedia
        except Exception:
            pass

    return int(sum(views) / len(views)) if views else None


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
        return cache[key]

    signals: dict = {
        "artist_name": artist_name,
        "spotify_popularity": None,
        "spotify_followers": None,
        "wikipedia_monthly_views": None,
        "fetched_at": datetime.utcnow().isoformat(),
    }

    # Spotify
    token = _spotify_token()
    if token:
        sp = _spotify_lookup(artist_name, token)
        if sp:
            signals.update(sp)

    # Wikipedia
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
    print(f"{'Artist':<50} {'Spotify':>8} {'Followers':>10} {'Wiki/mo':>10}  {'Fetched'}")
    print("-" * 95)
    for entry in sorted(cache.values(), key=lambda x: x.get("artist_name", "")):
        sp  = entry.get("spotify_popularity", "–")
        fol = entry.get("spotify_followers")
        wv  = entry.get("wikipedia_monthly_views")
        ts  = entry.get("fetched_at", "")[:10]
        fol_s = f"{fol:,}" if fol else "–"
        wv_s  = f"{wv:,}" if wv else "–"
        print(f"{entry.get('artist_name',''):<50} {str(sp):>8} {fol_s:>10} {wv_s:>10}  {ts}")


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
