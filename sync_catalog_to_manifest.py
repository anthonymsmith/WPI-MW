"""
Sync MW staff edits from Event_Catalog.xlsx → EventManifest.xlsx.

Workflow:
  1. event_manifest_guide.py  — generate Event_Catalog.xlsx
  2. share with MW staff; they edit the Events sheet directly in Excel
  3. sync_catalog_to_manifest.py  — apply edits back to manifest
  4. event_manifest_guide.py  — regenerate Catalog from updated manifest

Editable columns in the Catalog (mapped to manifest columns):
    Venue              → EventVenue
    Capacity           → EventCapacity
    Programming Line   → EventLoB        (friendly label → code)
    Class              → EventClass
    Genre              → EventGenre
    Sub-genre          → EventSubGenre
    Pricing Tier       → PriceTier
    Festival Role      → FestivalRole
    Recurring Series   → EventRepeat
    Pricing Note       → Pricing         (friendly label → code)

Columns that are NOT synced back (computed or identity):
    Season, Date, Event, Instances, Status

Join key: (EventName, Season) — the Catalog doesn't expose EventId.
Edits apply to ALL instance rows of the matched event in the manifest.
A backup of the pre-sync manifest is saved with a `.bak.<ts>` suffix.
"""
import os
import sys
import shutil
from datetime import datetime
import pandas as pd

os.chdir("/Users/antho/Documents/WPI-MW")

CATALOG  = "Event_Catalog.xlsx"
MANIFEST = "EventManifest.xlsx"

# Reverse mappings (friendly label → manifest code).
# Anything not in these maps is passed through as-is, so staff can write
# either the friendly form or the raw code.
LOB_REVERSE = {
    "Regular Concert":                "Concert",
    "The Complete Bach festival":     "TCB",
    "Artist in Residence":            "AiR",
    "Bach Programming (legacy tag)":  "Bach",
}
PRICING_REVERSE = {
    "Pay What You Wish":      "PWYW",
    "Free / fully comped":    "Free",
}

# Catalog column → (manifest column, optional reverse map)
COLUMN_MAP = [
    ("Venue",            "EventVenue",     None),
    ("Capacity",         "EventCapacity",  None),
    ("Programming Line", "EventLoB",       LOB_REVERSE),
    ("Class",            "EventClass",     None),
    ("Genre",            "EventGenre",     None),
    ("Sub-genre",        "EventSubGenre",  None),
    ("Pricing Tier",     "PriceTier",      None),
    ("Festival Role",    "FestivalRole",   None),
    ("Recurring Series", "EventRepeat",    None),
    ("Pricing Note",     "Pricing",        PRICING_REVERSE),
]

# Validation enums — sync warns (but still applies) on out-of-set values
VALID_CLASS       = {"Headliner", "Prestige", "Standard", "Mission"}
VALID_PRICE_TIER  = {"Marquee", "Headliner", "Prestige", "Standard",
                     "Chorus", "TCB", "Mission", "AiR"}
VALID_LOB         = {"Concert", "TCB", "AiR", "Bach"}
VALID_FESTIVAL    = {"Main", "Side"}
VALID_PRICING     = {"PWYW", "Free"}


def to_code(val, reverse_map):
    if pd.isna(val) or val == "":
        return None
    if reverse_map is None:
        return val
    return reverse_map.get(val, val)


def values_match(a, b):
    """Pandas-safe equality with NaN treated as 'no value'."""
    a_blank = pd.isna(a) or a == "" or a is None
    b_blank = pd.isna(b) or b == "" or b is None
    if a_blank and b_blank:
        return True
    if a_blank or b_blank:
        return False
    # Normalize numeric vs string for EventCapacity
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def validate(cat_col, manifest_col, value, warnings):
    if value is None or pd.isna(value):
        return
    enum = {
        "EventClass":    VALID_CLASS,
        "PriceTier":     VALID_PRICE_TIER,
        "EventLoB":      VALID_LOB,
        "FestivalRole":  VALID_FESTIVAL,
        "Pricing":       VALID_PRICING,
    }.get(manifest_col)
    if enum and value not in enum:
        warnings.append(f"  ⚠ unrecognized {manifest_col} value '{value}' "
                        f"(catalog column: {cat_col}) — applying anyway")


def main():
    if not os.path.exists(CATALOG):
        print(f"✗ {CATALOG} not found")
        sys.exit(1)
    if not os.path.exists(MANIFEST):
        print(f"✗ {MANIFEST} not found")
        sys.exit(1)

    cat = pd.read_excel(CATALOG, sheet_name="Events")
    mfest = pd.read_excel(MANIFEST)

    # For change-detection, build a "what would the Catalog show right now"
    # snapshot from the live manifest, using the SAME logic as
    # event_manifest_guide.py (filter test/placeholder + restrict to Live/
    # Virtual/Pandemic, sort by date, groupby+first-non-null).
    # We compare the actual catalog vs this snapshot: any divergence is a
    # genuine user edit.
    mfest_snap = mfest[~mfest["EventName"].fillna("").str.contains(
        "test|TEST|Test|Placeholder", case=False, regex=True)]
    mfest_snap = mfest_snap[mfest_snap["EventType"].isin(
        ["Live", "Virtual", "Pandemic"])].copy()
    mfest_snap["__d"] = pd.to_datetime(mfest_snap["EventDate"], errors="coerce")
    mfest_snap = mfest_snap.sort_values("__d")
    snapshot = mfest_snap.groupby("EventId", dropna=False).agg(
        EventName     = ("EventName",     "first"),
        Season        = ("Season",        "first"),
        EventVenue    = ("EventVenue",    "first"),
        EventCapacity = ("EventCapacity", "first"),
        EventLoB      = ("EventLoB",      "first"),
        EventClass    = ("EventClass",    "first"),
        EventGenre    = ("EventGenre",    "first"),
        EventSubGenre = ("EventSubGenre", "first"),
        PriceTier     = ("PriceTier",     "first"),
        FestivalRole  = ("FestivalRole",  "first"),
        EventRepeat   = ("EventRepeat",   "first"),
        Pricing       = ("Pricing",       "first"),
    ).reset_index()
    snapshot_idx = {
        (r["EventName"], r["Season"]): r.to_dict()
        for _, r in snapshot.iterrows()
    }

    # Manifest row indices per (EventName, Season) for writing back
    mfest_idx = {}
    for i in mfest.index:
        key = (mfest.at[i, "EventName"], mfest.at[i, "Season"])
        mfest_idx.setdefault(key, []).append(i)

    n_events_changed = 0
    n_rows_changed   = 0
    n_unmatched      = 0
    changes_log      = []
    warnings         = []

    for _, cat_row in cat.iterrows():
        name = cat_row.get("Event")
        season = cat_row.get("Season")
        if pd.isna(name) or pd.isna(season):
            continue
        rows = mfest_idx.get((name, season))
        if not rows:
            n_unmatched += 1
            print(f"  ⚠ no manifest match for '{name}' / {season} — skipped")
            continue

        snap = snapshot_idx.get((name, season), {})

        # For each editable column: compare catalog value to the snapshot
        # value (= what the Catalog WOULD show from the current manifest).
        # Apply only if they differ — that's a true user edit.
        event_changes = []
        for cat_col, mfest_col, reverse in COLUMN_MAP:
            new_val = to_code(cat_row.get(cat_col), reverse)
            snap_val = snap.get(mfest_col)
            if values_match(snap_val, new_val):
                continue
            validate(cat_col, mfest_col, new_val, warnings)
            for ridx in rows:
                mfest.at[ridx, mfest_col] = new_val
                n_rows_changed += 1
            event_changes.append(f"{mfest_col}: '{snap_val}' → '{new_val}'")

        if event_changes:
            n_events_changed += 1
            changes_log.append(f"  {name} ({season})")
            for c in event_changes:
                changes_log.append(f"      {c}")

    if n_events_changed == 0:
        print(f"✓ Catalog matches manifest — no changes ({n_unmatched} unmatched Catalog rows)")
        return

    # Backup and write
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{MANIFEST}.bak.{ts}_sync"
    shutil.copy(MANIFEST, backup)
    mfest.to_excel(MANIFEST, sheet_name="EventManifest", index=False)

    print(f"✓ Synced {n_events_changed} events ({n_rows_changed} instance rows updated)")
    print(f"  backup: {backup}")
    print()
    print("Changes:")
    for line in changes_log:
        print(line)
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(w)
    if n_unmatched:
        print(f"\n{n_unmatched} Catalog row(s) had no manifest match (skipped above).")
    print()
    print("Next: run `python3 event_manifest_guide.py` to refresh the Catalog.")


if __name__ == "__main__":
    main()
