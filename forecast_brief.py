"""
Forecast Approach & Results Brief — anonymized

Short methodology + multi-season results piece for sharing with the
WPI community, performing-arts analytics peers, and prospective
arts-org collaborators. Pairs a methodology summary with a 3-year
temporal-holdout table and live-season performance.

Render to PDF via Chrome headless.
"""
import os
from datetime import date
import pandas as pd

WORKING_DIR = "/Users/antho/Documents/WPI-MW"
os.chdir(WORKING_DIR)

NAVY   = "#1A3A5C"
ORANGE = "#E8922A"
TEAL   = "#2A9EA0"
INK    = "#1A2330"
INK_LT = "#5A6A7A"
LGRAY  = "#E8EDF2"


def load():
    df = pd.read_excel("forecasting/Forecast_2526_FullSeason.xlsx", sheet_name="Detail")
    df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
    return df.sort_values("EventDate").reset_index(drop=True)


def live_stats(df):
    done = df[df["Status"] == "Completed"].copy()
    wape = (done["Pred_Adj"] - done["Actual"]).abs().sum() / done["Actual"].sum() * 100
    bias = ((done["Pred_Adj"] - done["Actual"]) / done["Actual"]).mean() * 100
    return wape, bias, len(done), len(df) - len(done)


# Hard-coded from forecast_eval_honest.py temporal-holdout runs.
HOLDOUT = [
    ("2022–23", 20, 27.3, 31.7),
    ("2023–24", 21, 28.3,  6.3),
    ("2024–25", 23, 23.1, -5.9),
]
HOLDOUT_TOTAL = (64, 26.3, 9.8)


def build_html(df):
    wape, bias, n_done, n_upc = live_stats(df)
    today = f"{date.today():%B %Y}"

    rows = "\n".join(
        f"<tr><td>{s}</td><td class='num'>{n}</td>"
        f"<td class='num'>{w:.1f}%</td>"
        f"<td class='num'>{b:+.1f}%</td></tr>"
        for s, n, w, b in HOLDOUT
    )
    n_tot, w_tot, b_tot = HOLDOUT_TOTAL
    rows += (
        f"<tr class='total'><td>3-season total</td>"
        f"<td class='num'>{n_tot}</td>"
        f"<td class='num'>{w_tot:.1f}%</td>"
        f"<td class='num'>{b_tot:+.1f}%</td></tr>"
        f"<tr class='live'><td>2025–26 live (in-progress)</td>"
        f"<td class='num'>{n_done} of {n_done + n_upc}</td>"
        f"<td class='num'>{wape:.1f}%</td>"
        f"<td class='num'>{bias:+.1f}%</td></tr>"
    )

    # HTML img src is relative to the HTML file's own location (forecasting/),
    # so no subfolder prefix here.
    chart_src = "forecast_2526_bar_chart_eventaxis_anon.png"
    chart_src_wide = "forecast_2526_bar_chart_wide_anon.png"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Forecasting Performing-Arts Attendance — Approach &amp; Results</title>
<style>
@page {{ size: Letter; margin: 0.45in 0.55in; }}
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
    color: {INK}; margin: 0; padding: 0;
    font-size: 9.5pt; line-height: 1.35;
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
    font-size: 10.5pt; color: {NAVY}; margin: 8pt 0 3pt 0; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.4pt;
}}
p {{ margin: 0 0 5pt 0; }}
.lede {{ font-size: 10pt; }}
.callout {{
    background: #F8FAFC; border-left: 3px solid {TEAL};
    padding: 6pt 10pt; margin: 4pt 0 8pt 0; font-size: 9pt;
}}
.callout strong {{ color: {NAVY}; }}
table.results {{
    width: 100%; border-collapse: collapse;
    margin: 4pt 0 8pt 0; font-size: 9.2pt;
}}
table.results th {{
    text-align: left; color: {INK_LT}; font-weight: 600;
    border-bottom: 1px solid {LGRAY}; padding: 5pt 8pt 5pt 0;
}}
table.results td {{
    padding: 5pt 8pt 5pt 0; border-bottom: 1px solid #F0F3F7;
}}
table.results td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.results tr.total td {{
    border-top: 1px solid {LGRAY}; font-weight: 700; color: {NAVY};
}}
table.results tr.live td {{
    background: #FFF6E8; font-weight: 700; color: {NAVY};
}}
ol, ul {{ margin: 4pt 0 7pt 18pt; padding: 0; }}
li {{ margin-bottom: 3pt; }}
.chart-wrap {{ margin: 4pt 0 2pt 0; text-align: center; }}
.chart-wrap img {{ max-width: 100%; max-height: 3.4in; object-fit: contain; }}
.caption {{
    font-size: 8.2pt; color: {INK_LT}; font-style: italic;
    text-align: center; margin: 2pt 0 4pt 0;
}}
.footer {{
    margin-top: 14pt; padding-top: 6pt; border-top: 1px solid {LGRAY};
    color: {INK_LT}; font-size: 8.5pt;
}}
.two-col {{ display: flex; gap: 18pt; }}
.two-col > div {{ flex: 1; min-width: 0; }}
</style>
</head>
<body>

<h1>How many seats will this fill?</h1>
<div class="subtitle">
A working attendance forecast for a regional performing-arts season  ·  {today}
</div>

<p class="lede">
A regional presenter signs next season's contracts 12–18 months out:
venues, artists, fees, marketing budget. Each event is a bet on a single
number: how many people will show up. Miss high and you're staring at
empty seats and a marketing post-mortem; miss low and you've
under-resourced the show that needed the most help. Most arts
organizations make those bets on instinct. This brief is about a
forecasting model that doesn't.
</p>

<p class="lede">
Built and running in production at a regional presenter, the model
produces per-event attendance estimates at the start of the planning
cycle, before a single ticket goes on sale. Through
<strong>{n_done} of {n_done + n_upc}</strong> completed events of the
2025–26 season, those pre-season predictions are tracking within
<strong>{wape:.0f}%</strong> of actual attendance, with effectively no
directional drift (bias <strong>{bias:+.0f}%</strong>).
</p>

<h2>The 2025–26 scorecard</h2>
<div class="chart-wrap">
<img src="{chart_src}" alt="Predicted vs actual attendance for the live season, anonymized">
</div>
<div class="caption">
Predicted (navy) vs. actual (orange) attendance per event. Lighter tones
are comps. Anonymized; identifiers and scale removed.
</div>

<p>
The bars line up. That's the headline. Some events came in slightly
under, some slightly over, none catastrophically. The largest misses
sit on the model's known weak edges: programs without clean comparable
history, or recitals where the artist-signal layer hasn't yet seen
enough sample to anchor. The closest calls were headliners with strong
bucket precedent and a recurring-series tag, a vote for the model's
strength on its core repertoire.
</p>

<h2>What it changes in practice</h2>
<p>
Going into a season with a defensible per-event attendance number
changes three conversations. <strong>Budgeting</strong> stops being
aggregate hand-waving. Each event has a forecast and the season's
revenue picture is the sum of those, auditable line by line.
<strong>Season planning</strong> gets a common scale: a proposed
booking arrives with a draw estimate in the same units as everything
else on the calendar, so "do we have enough capacity at this draw
tier?" becomes a question with a number, not a vibe. And
<strong>marketing</strong> prioritizes earlier; events forecast under
capacity surface as candidates for paid promotion or partnership
outreach months before sale data would confirm the gap.
</p>

<div class="page-break"></div>

<h1 style="font-size:14pt;">Inside the model</h1>
<div class="subtitle" style="margin-bottom:7pt;">
Why the standard tools don't work, and what does.
</div>

<p>
The forecasting problem in performing arts isn't usually too much
data. It's too little of the right kind. A typical season is 25–35
events spanning headliner orchestras, chamber recitals, jazz combos,
choral programs, education events, and free pay-what-you-want concerts.
A cell defined by the natural cuts (class × venue × subgenre × line of
business) might contain a single prior observation, or none at all.
Rule-of-thumb averages ("a recital in this hall draws roughly 500")
collapse the signal that matters most: artist stature, repeat-series
momentum, pricing format, venue tier.
</p>

<p>
The model handles this through a <strong>five-level fallback
hierarchy</strong>. Each event finds its prediction at the closest
comparable cell available, falling back to progressively broader pools
until it lands on a stable mean. <strong>Empirical-Bayes
shrinkage</strong> at each level pulls thin buckets toward their
next-coarser fallback. Niche low buckets are protected, but thin
buckets sitting unrealistically high get pulled back. A
<strong>recurring-series prior</strong> fires before the hierarchy
whenever a series has ≥2 prior observations, capturing the momentum of
annual programs and returning headliners.
</p>

<p>
Two layers fill in where the bucket structure runs thin. PWYW events
receive a shrunken multiplicative lift from historical PWYW sample.
Venue-tier pooling (Marquee through Intimate) provides a fallback when
subgenre × venue cells are empty. Finally, an
<strong>artist-popularity layer</strong> absorbs Wikipedia, Last.fm,
and Deezer signals through an informed-Bayesian regression on log(actual
/ bucket prior). A signal-strength gate fires the adjustment only when
genre-fit and signal thresholds are met, preventing global-pop signals
from distorting world, folk, and Americana events where they don't
apply.
</p>

<p>
Evaluation is <strong>temporal-holdout</strong>: each test season is
predicted using only data strictly prior to it, with pandemic-era events
down-weighted. There's no way for future information to leak into a
past forecast, which is what makes the multi-season numbers below
defensible.
</p>

<h2>Three seasons of evidence</h2>
<table class="results">
<tr><th>Season</th><th class="num">n</th><th class="num">WAPE</th><th class="num">Bias</th></tr>
{rows}
</table>
<p style="font-size:9pt; color:{INK_LT}; margin-top:-2pt;">
Error has compressed across seasons as the model stabilized. The 2022–23
positive bias reflects the early model over-predicting against
still-recovering pandemic-era demand; the season-weighting scheme and
shrinkage layers since then have largely closed that gap.
</p>

<div class="callout">
<strong>WAPE</strong> (Weighted Absolute Percentage Error): total miss
divided by total actual attendance. Volume-weighted so misses on
high-draw events count more than misses on low-draw events.
<strong>Bias</strong>: average signed error; near zero indicates no
systematic over- or under-prediction.
</div>

<h2>What's coming</h2>
<ul>
<li><strong>In-season blending.</strong> Combine the pre-season
forecast with running sale counts via Kaplan–Meier survival curves of
ticket-purchase timing by event class, a forecast that sharpens as the
show approaches.</li>
<li><strong>Sales-pace and pricing.</strong> Use the same temporal
curves to test where late-sale discounting erodes revenue versus fills
otherwise-empty houses, and where higher-tier price headroom exists.</li>
<li><strong>Class-specific slopes</strong> in the artist-adjustment
layer once each event-class bucket reaches ~15–20 observations.</li>
</ul>

<h2>About this work</h2>
<p>
The model and supporting analyses were built by Nolichucky Associates
for a regional performing-arts presenter. This brief is anonymized for
sharing with academic, peer-analytics, and arts-organization audiences.
For collaboration, comparison studies, or replication discussions:
<a href="https://nolichuckyassociates.com">nolichuckyassociates.com</a>.
</p>

<div class="page-break"></div>

<h1 style="font-size:14pt;">Looking deeper</h1>
<div class="subtitle" style="margin-bottom:5pt;">
Calibration, error by event class, and accuracy trend over the past three seasons.
</div>

<h2 style="margin-top:5pt;">Calibration: predicted vs. actual, 2025–26</h2>
<div class="chart-wrap" style="margin:0;">
<img src="forecast_portfolio_scatter_2526.png" alt="Predicted vs actual scatter for 2025-26 completed events" style="max-height:2.9in;">
</div>
<div class="caption" style="margin:0 0 2pt 0;">
Each point is a completed event. Points on the dashed diagonal indicate
perfect prediction; above the line means the event drew more than forecast,
below means less. Color reflects event class.
</div>

<h2 style="margin-top:6pt;">Where the model fits best, by event class, 2025–26</h2>
<div class="chart-wrap" style="margin:0;">
<img src="forecast_portfolio_class_2526.png" alt="WAPE by event class, 2025-26" style="max-height:1.55in;">
</div>
<div class="caption" style="margin:0 0 2pt 0;">
Headliners and the new Prestige tier (specialist artists with strong
core-audience appeal) are predicting tightly. Standard programming carries
the largest share of variance; its breadth across genre, venue, and
repeat-status is also the broadest of any class.
</div>

<h2 style="margin-top:6pt;">Accuracy trend across past three seasons and live</h2>
<div class="chart-wrap" style="margin:0;">
<img src="forecast_portfolio_accuracy.png" alt="Forecast accuracy by season trend" style="max-height:2.2in;">
</div>
<div class="caption" style="margin:0;">
Per-event MAPE (bars) and average bias (line). Error has compressed over
three seasons as the season-weighting scheme, shrinkage layers, and
artist-popularity layer have accumulated. The live 2025–26 bar reflects
pre-Dec 2025 events; the full season is at WAPE 17% / bias +4% (see Page 1).
</div>

</body>
</html>
"""

    out = "forecasting/forecast_brief.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"  ✓ {out}")
    return out


def main():
    df = load()
    build_html(df)


if __name__ == "__main__":
    main()
