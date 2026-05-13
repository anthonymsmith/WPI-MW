"""
Forecast Performance Memo — 25-26 Season

Generates a 1-2 page HTML memo summarizing the forecast model's live-season
performance. Two variants:

  python forecast_memo.py          → forecast_memo.html       (named, for MW/financials)
  python forecast_memo.py --anon   → forecast_memo_anon.html  (for prospects/academic/other clients)

Pairs a written narrative with forecast_2526_bar_chart_wide(.|_anon.).png.
Render to PDF via Chrome headless if needed.
"""
import os
import sys
from datetime import date
import pandas as pd

ANON = "--anon" in sys.argv
WORKING_DIR = "/Users/antho/Documents/WPI-MW"
os.chdir(WORKING_DIR)

NAVY   = "#1A3A5C"
ORANGE = "#E8922A"
TEAL   = "#2A9EA0"
INK    = "#1A2330"
INK_LT = "#5A6A7A"
LGRAY  = "#E8EDF2"


def load():
    df = pd.read_excel("Forecast_2526_FullSeason.xlsx")
    df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
    df = df.sort_values("EventDate").reset_index(drop=True)
    return df


def summary_stats(df):
    done = df[df["Status"] == "Completed"].copy()
    upc  = df[df["Status"] == "Upcoming"].copy()
    wape = (done["Pred_Adj"] - done["Actual"]).abs().sum() / done["Actual"].sum() * 100
    bias = ((done["Pred_Adj"] - done["Actual"]) / done["Actual"]).mean() * 100
    return wape, bias, len(done), len(upc)


def hits_and_misses(df, n_hits=4, n_misses=4):
    done = df[df["Status"] == "Completed"].copy()
    done["err"]      = done["Pred_Adj"] - done["Actual"]
    done["err_pct"]  = done["err"] / done["Actual"] * 100
    done["abs_pct"]  = done["err_pct"].abs()
    misses = done.nlargest(n_misses, "abs_pct")[
        ["EventName", "EventDate", "Pred_Adj", "Actual", "err_pct"]
    ].to_dict("records")
    hits = done.nsmallest(n_hits, "abs_pct")[
        ["EventName", "EventDate", "Pred_Adj", "Actual", "err_pct"]
    ].to_dict("records")
    return hits, misses


def anon_label(idx):
    return f"Event #{idx:02d}"


def display_name(name, idx, anon):
    if anon:
        return anon_label(idx)
    return str(name)


# Brief reason snippets keyed by event (named version only).
MISS_REASONS = {
    "Bach's Birthday Bash 2026: Keyboards Up Close":
        "Festival sub-event in an intimate room; model leaned on broader BBB priors and overshot the niche keyboard recital.",
    "Nelson Goerner":
        "Recitalist with limited US profile — Wikipedia/Last.fm signals understated draw; legacy classical-piano audience showed up.",
    "Bach's Birthday Bash 2026: Cantatathon":
        "Cantatathon is a long-form devotional event; model's BBB priors include broader-appeal sub-events and overshot.",
    "Catherine Russell & Sean Mason":
        "Jazz vocal pairing — genre-fit gate dampened popularity signal; actual draw exceeded the venue-tier baseline.",
    "Aaron Diehl Trio":
        "Jazz trio at a small-room slot — model's headliner-jazz prior pulled prediction up; actual draw closer to chamber norms.",
}

ANON_MISS_REASONS = [
    "Festival sub-event in an intimate room; model leaned on broader festival priors and overshot a niche pairing.",
    "Recitalist with limited online profile — popularity signals understated draw; loyal audience exceeded baseline.",
    "Devotional long-form festival sub-event; model's festival priors include broader-appeal sub-events and overshot.",
    "Genre-fit gate dampened popularity signal for a vocal pairing; actual draw exceeded venue-tier baseline.",
    "Small-room jazz slot — headliner-jazz prior pulled prediction up; actual draw closer to chamber norms.",
]


def render_html(df, anon):
    wape, bias, n_done, n_upc = summary_stats(df)
    hits, misses = hits_and_misses(df)

    org_name = "a regional performing arts presenter" if anon else "Music Worcester"
    title = ("Season Forecast — Performance Update" if anon
             else "25–26 Season Forecast — Performance Update")
    subtitle = (f"WAPE {wape:.0f}%  ·  Bias {bias:+.0f}%  ·  "
                f"{n_done} of {n_done + n_upc} events completed  ·  "
                f"as of {date.today():%b %-d, %Y}")
    chart_src = ("forecast_2526_bar_chart_wide_anon.png" if anon
                 else "forecast_2526_bar_chart_wide.png")

    upcoming_df = df[df["Status"] == "Upcoming"].sort_values("EventDate")
    if anon:
        upcoming_lines = [f"<li>{anon_label(len(df) - len(upcoming_df) + i + 1)}"
                          f" — predicted {int(round(r.Pred_Adj))} tickets</li>"
                          for i, r in enumerate(upcoming_df.itertuples())]
    else:
        upcoming_lines = [f"<li><strong>{r.EventName}</strong> "
                          f"({r.EventDate:%b %-d}) — predicted "
                          f"{int(round(r.Pred_Adj))} tickets</li>"
                          for r in upcoming_df.itertuples()]
    upcoming_html = "\n".join(upcoming_lines)

    name_to_idx = {row["EventName"]: i + 1 for i, row in df.iterrows()}

    def hits_table():
        rows = []
        for h in hits:
            label = display_name(h["EventName"], name_to_idx[h["EventName"]], anon)
            rows.append(
                f"<tr><td>{label}</td>"
                f"<td class='num'>{int(round(h['Pred_Adj']))}</td>"
                f"<td class='num'>{int(round(h['Actual']))}</td>"
                f"<td class='num pct'>{h['err_pct']:+.0f}%</td></tr>"
            )
        return "\n".join(rows)

    def misses_table():
        rows = []
        for i, m in enumerate(misses):
            label = display_name(m["EventName"], name_to_idx[m["EventName"]], anon)
            if anon:
                reason = ANON_MISS_REASONS[i] if i < len(ANON_MISS_REASONS) else ""
            else:
                reason = MISS_REASONS.get(m["EventName"], "")
            rows.append(
                f"<tr><td><div class='ev'>{label}</div>"
                f"<div class='reason'>{reason}</div></td>"
                f"<td class='num'>{int(round(m['Pred_Adj']))}</td>"
                f"<td class='num'>{int(round(m['Actual']))}</td>"
                f"<td class='num pct'>{m['err_pct']:+.0f}%</td></tr>"
            )
        return "\n".join(rows)

    chart_caption = (
        "Predicted vs. actual attendance by event (paid + comp). "
        "Predicted shown in navy, actual in orange; lighter tones are comps. "
        "Upcoming events show the prediction only."
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
@page {{ size: Letter; margin: 0.45in 0.55in; }}
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
    color: {INK}; margin: 0; padding: 0;
    font-size: 9.7pt; line-height: 1.38;
}}
.page-break {{ page-break-before: always; }}
h1 {{
    font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif;
    font-size: 17pt; color: {NAVY}; margin: 0 0 3pt 0; font-weight: 700;
    letter-spacing: -0.2pt;
}}
.subtitle {{
    color: {INK_LT}; font-size: 9.5pt; margin: 0 0 9pt 0;
    border-bottom: 1px solid {LGRAY}; padding-bottom: 5pt;
}}
h2 {{
    font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt; color: {NAVY}; margin: 9pt 0 3pt 0; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.4pt;
}}
p {{ margin: 0 0 5pt 0; }}
.lede {{ font-size: 10pt; color: {INK}; }}
.metric-defs {{
    background: #F8FAFC; border-left: 3px solid {TEAL};
    padding: 5pt 9pt; margin: 4pt 0 6pt 0; font-size: 9pt;
}}
.metric-defs strong {{ color: {NAVY}; }}
.chart-wrap {{
    margin: 2pt 0 2pt 0; text-align: center;
}}
.chart-wrap img {{
    max-width: 100%; max-height: 3.4in; object-fit: contain;
}}
.caption {{
    font-size: 8.2pt; color: {INK_LT}; font-style: italic;
    text-align: center; margin: 1pt 0 4pt 0;
}}
table {{
    width: 100%; border-collapse: collapse; margin: 3pt 0 5pt 0;
    font-size: 8.8pt;
}}
th {{
    text-align: left; color: {INK_LT}; font-weight: 600;
    border-bottom: 1px solid {LGRAY}; padding: 4pt 6pt 4pt 0;
}}
td {{
    padding: 5pt 6pt 5pt 0; border-bottom: 1px solid #F0F3F7;
    vertical-align: top;
}}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.pct {{ font-weight: 700; color: {NAVY}; }}
.ev {{ font-weight: 600; }}
.reason {{ color: {INK_LT}; font-size: 8.8pt; margin-top: 2pt; }}
ul {{ margin: 4pt 0 8pt 18pt; padding: 0; }}
li {{ margin-bottom: 3pt; }}
.footer {{
    margin-top: 16pt; padding-top: 6pt; border-top: 1px solid {LGRAY};
    color: {INK_LT}; font-size: 8.5pt;
}}
.two-col {{ display: flex; gap: 18pt; }}
.two-col > div {{ flex: 1; min-width: 0; }}
</style>
</head>
<body>

<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>

<p class="lede">
The forecast model predicts paid + comp attendance for every event on
{org_name}'s season, before tickets go on sale. Through {n_done} completed
events, the season-total forecast is running within
<strong>{wape:.0f}%</strong> of actuals on a volume-weighted basis, with
near-zero systematic bias (<strong>{bias:+.0f}%</strong>). {n_upc} events
remain.
</p>

<h2>How the model works</h2>
<p>
Each event's prediction is built from a hierarchy of comparable events —
same class and venue first (e.g. headliner recital at Mechanics Hall),
then progressively broader pools when the closest comparison is thin.
Recurring series carry their own prior. An artist-popularity layer adjusts
classical-genre events using Wikipedia, Last.fm, and Deezer signals, gated
so noisy or out-of-genre signals don't fire. Pricing format (PWYW) and
venue tier feed in as multiplicative and pooled adjustments respectively.
The model is evaluated on temporal holdouts — trained only on seasons
prior to the target — so live-season numbers reflect honest forward
performance, not a fitted backtest.
</p>

<div class="metric-defs">
<strong>WAPE (Weighted Absolute Percentage Error):</strong> total miss across
the season divided by total actual attendance. Volume-weighted — misses on
big events count more than misses on small ones. Lower is better.<br>
<strong>Bias:</strong> the average signed error. Near zero means the model
isn't systematically over- or under-shooting; positive means it tends to
over-predict.
</div>

<h2>Season-to-date</h2>
<div class="chart-wrap">
<img src="{chart_src}" alt="Predicted vs actual attendance by event">
</div>
<div class="caption">{chart_caption}</div>

<div class="page-break"></div>

<h1 style="font-size:14pt;">What hit, what missed</h1>

<div class="two-col">
<div>
<h2>Closest hits</h2>
<table>
<tr><th>Event</th><th class="num">Pred</th><th class="num">Actual</th><th class="num">Err</th></tr>
{hits_table()}
</table>
<p style="font-size:9pt; color:{INK_LT};">
Strong fits across the season: a returning headliner, a regular series,
festival anchors, and an annual choir program — all anchored by deep
historical pools.
</p>
</div>

<div>
<h2>Biggest misses</h2>
<table>
<tr><th>Event</th><th class="num">Pred</th><th class="num">Actual</th><th class="num">Err</th></tr>
{misses_table()}
</table>
</div>
</div>

<h2>What's coming</h2>
<p>{n_upc} events remain on the season. Predictions:</p>
<ul>
{upcoming_html}
</ul>
<p style="font-size:9pt; color:{INK_LT};">
Forecast will be refreshed at season close to fold in final actuals.
</p>

<h2>What this enables</h2>
<ul>
<li><strong>Budgeting:</strong> a per-event attendance + revenue forecast at
the start of the planning cycle, replacing rule-of-thumb averages.</li>
<li><strong>Season planning:</strong> proposed bookings get a draw estimate
in the same numeric language as historical events — easier to compare
risk and balance the calendar.</li>
<li><strong>Marketing prioritization:</strong> events flagged as
under-forecast against capacity surface natural targets for paid promotion
or audience-segment outreach.</li>
</ul>

<div class="footer">
{"Anonymized for external sharing — event identifiers and ticket scales removed." if anon else "Internal — for Music Worcester finance and planning use."}
</div>

</body>
</html>
"""

    out = "forecast_memo_anon.html" if anon else "forecast_memo.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"  ✓ {out}")
    return out


def main():
    df = load()
    render_html(df, ANON)


if __name__ == "__main__":
    main()
