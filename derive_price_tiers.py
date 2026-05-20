"""
Derive PriceTier on EventManifest.xlsx using the tier policy.

Preserves existing Marquee tags (these are manual pre-season strategic
calls) and otherwise re-derives every row from the rule below.

Tier rule (in precedence order — first match wins):
  1. EventLoB == 'TCB'              → TCB
  2. EventLoB == 'AiR'              → AiR
  3. EventClass == 'Mission'        → Mission
  4. EventClass == 'Headliner'      → Headliner
  5. EventClass == 'Prestige'       → Prestige
  6. EventClass == 'Standard' AND any of:
        - EventRepeat in {WC Orch, Women, Bach M-minor}
        - EventName matches ^chorus:|worcester chorus|worc\\. chorus|
                            women's ensemble|wcwe|welcome yule
        - EventGenre == 'Choral' AND EventSubGenre in {Chor Accomp, Chor Orch, Chor Wom}
       → Chorus
  7. EventClass == 'Standard' otherwise → Standard
  8. EventType == 'Subscription'    → (null, not tiered)

Marquee tier is set manually on the manifest for pre-season top-draw
bookings and is preserved across re-runs by this script.
"""
import os
import re
import pandas as pd

MANIFEST = "/Users/antho/Documents/WPI-MW/EventManifest.xlsx"

CHORUS_REPEAT = {"WC Orch", "Women", "Bach M-minor"}
CHORUS_NAME_RE = re.compile(
    r"(?:^chorus:|worcester chorus|worc\. chorus|women's ensemble|wcwe|welcome yule)",
    re.IGNORECASE,
)
CHORUS_SUBGENRES = {"Chor Accomp", "Chor Orch", "Chor Wom"}


def derive_tier(row):
    cls   = row["EventClass"]
    lob   = row["EventLoB"]
    rep   = row["EventRepeat"]
    name  = str(row["EventName"]) if pd.notna(row["EventName"]) else ""
    genre = row.get("EventGenre")
    sub   = row.get("EventSubGenre")
    etype = row.get("EventType")

    if etype == "Subscription":
        return None
    if lob == "TCB":
        return "TCB"
    if lob == "AiR":
        return "AiR"
    if cls == "Mission":
        return "Mission"
    if cls == "Headliner":
        return "Headliner"
    if cls == "Prestige":
        return "Prestige"
    if cls == "Standard":
        if pd.notna(rep) and rep in CHORUS_REPEAT:
            return "Chorus"
        if CHORUS_NAME_RE.search(name):
            return "Chorus"
        if genre == "Choral" and sub in CHORUS_SUBGENRES:
            return "Chorus"
        return "Standard"
    return None


def main():
    m = pd.read_excel(MANIFEST)

    # Capture existing Marquee EventIds so we can preserve them
    existing_marquee = set(
        m.loc[m["PriceTier"] == "Marquee", "EventId"].dropna().unique()
    )

    # Re-derive every row
    new_tier = m.apply(derive_tier, axis=1)

    # Preserve Marquee
    new_tier = new_tier.where(~m["EventId"].isin(existing_marquee), other="Marquee")

    # Report changes
    old = m["PriceTier"].fillna("(null)")
    new = new_tier.fillna("(null)")
    changed_mask = (old != new)
    if changed_mask.any():
        print(f"Changes ({changed_mask.sum()} rows):")
        changes_df = m.loc[changed_mask, ["EventName", "Season", "EventClass",
                                          "EventLoB", "EventGenre", "EventSubGenre"]].copy()
        changes_df["PriceTier (was)"] = old[changed_mask].values
        changes_df["PriceTier (now)"] = new[changed_mask].values
        changes_df = changes_df.drop_duplicates(["EventName", "Season"])
        print(changes_df.to_string(index=False))
        print()

    m["PriceTier"] = new_tier
    m.to_excel(MANIFEST, sheet_name="EventManifest", index=False)

    print("Distribution after re-derivation (Live + Subscription rows):")
    print(m["PriceTier"].value_counts(dropna=False).to_string())
    print(f"\nMarquee preserved: {len(existing_marquee)} EventIds")
    print(f"Manifest saved: {MANIFEST}")


if __name__ == "__main__":
    main()
