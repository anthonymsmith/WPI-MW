#!/usr/bin/env python3
"""
generate_regions.py

Generates regions_computed.csv — a ZIP-to-region mapping for Music Worcester analytics.
Replaces the manually maintained final_regions.csv with an algorithmically derived version
based on haversine distance and compass bearing from Worcester, MA.

Source data: simplemaps US ZIP code database (free tier)
    https://simplemaps.com/data/us-zips  →  download and save as uszips.csv

Output format matches final_regions.csv for drop-in use in MWSalesSumm.

MA sub-regions are assigned by distance and bearing from Worcester.
Non-MA states are assigned by state, with a zip-based split for Connecticut.
"""

import math
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — adjust thresholds here without touching the logic below
# ---------------------------------------------------------------------------
WORCESTER_LAT = 42.2626
WORCESTER_LON = -71.8023

INPUT_ZIP  = 'simplemaps_uszips_basicv1.94.zip'  # simplemaps download (contains uszips.csv)
OUTPUT_CSV = 'regions_computed.csv'

# MA distance thresholds (miles from Worcester)
WOR_CENTRAL_MAX  =  8   # Wor Central inner radius
WOR_NEAR_MAX     = 40   # outer edge of close-in Worcester sub-regions
WOR_WEST_MAX     = 35   # Wor West outer edge (beyond this = West MA if westerly)
WOR_SOUTH_MAX    = 30   # Wor South outer edge

# MW/495 corridor
MW_MIN_DIST      =  8   # same as Wor Central radius — Wor Central handles < 8
MW_MAX_DIST      = 52   # outside this (easterly) → Boston Reg
MW_MAX_LON       = -71.25  # eastern boundary; towns east of this → Boston Reg
                            # (roughly the Route 128/95 ring)

# West MA
WEST_MA_MIN_DIST = 35   # beyond this, westerly bearing → West MA

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def compass_bearing(lat1, lon1, lat2, lon2):
    """Compass bearing in degrees (0=N, 90=E, 180=S, 270=W)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x  = math.sin(dl) * math.cos(p2)
    y  = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Region assignment
# ---------------------------------------------------------------------------
def assign_region(row):
    state = str(row['state_id']).upper().strip()
    zip5  = str(row['zip']).zfill(5)
    lat, lon = row['lat'], row['lng']

    if pd.isna(lat) or pd.isna(lon):
        return 'Far Away'

    dist = haversine_miles(WORCESTER_LAT, WORCESTER_LON, lat, lon)
    brng = compass_bearing(WORCESTER_LAT, WORCESTER_LON, lat, lon)

    # -----------------------------------------------------------------------
    # Massachusetts — distance + bearing from Worcester
    # -----------------------------------------------------------------------
    if state == 'MA':

        # North Shore — Salem, Beverly, Gloucester, Rockport, Newburyport, Ipswich
        if zip5[:3] == '019':
            return 'North Shore'

        # RI/Cape — Plymouth, Cape Cod, New Bedford/Wareham
        # (024=Newton/Brookline/Lexington — intentionally excluded)
        if zip5[:3] in ('023', '025', '026', '027'):
            return 'RI/Cape'

        # Wor Central
        if dist < WOR_CENTRAL_MAX:
            return 'Wor Central'

        # Wor North — Fitchburg, Leominster, Gardner (true north, close-in)
        if dist < WOR_NEAR_MAX and (brng >= 300 or brng <= 20):
            return 'Wor North'

        # Wor West — Spencer, Charlton, Sturbridge, Brookfield
        if dist < WOR_WEST_MAX and 200 <= brng < 300:
            return 'Wor West'

        # West MA — Springfield, Northampton, Pittsfield (far west/northwest)
        if dist > WEST_MA_MIN_DIST and 220 <= brng <= 340:
            return 'West MA'

        # Wor South — Grafton, Northbridge, Sutton, Uxbridge
        if dist < WOR_SOUTH_MAX and 130 <= brng < 200:
            return 'Wor South'

        # MW/495 — Westborough, Marlborough, Framingham, Natick (Pike),
        #           Concord (Rte 2), Westford (495)
        if (MW_MIN_DIST <= dist <= MW_MAX_DIST
                and 20 <= brng <= 135
                and lon <= MW_MAX_LON):
            return 'MW/495'

        # RI/Cape — remaining southeastern MA (Attleboro, Taunton, Brockton, S. Bridgewater)
        if 120 <= brng <= 220:
            return 'RI/Cape'

        # Boston Reg — everything else in MA
        return 'Boston Reg'

    # -----------------------------------------------------------------------
    # New England neighbors
    # -----------------------------------------------------------------------
    if state in ('NH', 'VT', 'ME'):
        return 'North'

    if state == 'RI':
        return 'RI/Cape'

    if state == 'CT':
        # Northern CT (Hartford and above) — close enough to Worcester → South
        # Southern CT (New Haven, Fairfield County) — NYC orbit → NY Area
        if zip5[:3] in ('060', '061', '062', '063'):
            return 'South'
        return 'NY Area'

    # -----------------------------------------------------------------------
    # NY Area — NY, NJ, PA
    # -----------------------------------------------------------------------
    if state in ('NY', 'NJ', 'PA'):
        return 'NY Area'

    # -----------------------------------------------------------------------
    # Everything else
    # -----------------------------------------------------------------------
    return 'Far Away'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not os.path.exists(INPUT_ZIP):
    print(f'ERROR: {INPUT_ZIP} not found in the working directory.')
    print()
    print('  1. Go to https://simplemaps.com/data/us-zips')
    print('  2. Download the free Basic database (zip file)')
    print(f'  3. Save as {INPUT_ZIP} in this directory')
    print('  4. Re-run this script')
    sys.exit(1)

import zipfile
print(f'Loading uszips.csv from {INPUT_ZIP}...')
with zipfile.ZipFile(INPUT_ZIP) as z:
    zips = pd.read_csv(z.open('uszips.csv'), dtype={'zip': str}, low_memory=False)
print(f'  {len(zips):,} ZIP codes loaded')

print('Assigning regions...')
zips['RegionAssignment'] = zips.apply(assign_region, axis=1)

out = pd.DataFrame({
    'PHYSICAL CITY':    zips['city'],
    'PHYSICAL STATE':   zips['state_id'],
    'PHYSICAL ZIP':     zips['zip'],
    'ZIP':              zips['zip'],
    'RegionAssignment': zips['RegionAssignment'],
})

out.to_csv(OUTPUT_CSV, index=False)
print(f'  {len(out):,} ZIP codes written to {OUTPUT_CSV}')
print()
print('Region breakdown:')
counts = zips['RegionAssignment'].value_counts()
for region, count in counts.items():
    print(f'  {region:<28} {count:>6,}')
