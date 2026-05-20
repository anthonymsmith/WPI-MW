"""
Finance-side intro brief — 1-page primer for new MW finance staff.

Pairs the attendance forecast and revenue projection into a single
page-and-a-half that explains: the goal (pre-season budget + revenue
planning), the attendance forecast and its accuracy, the revenue
projection and its accuracy, and how to use it in the budget cycle.

Output: forecasting/forecast_finance_intro.html (render to PDF via Chrome).
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


def load_attendance():
    df = pd.read_excel("forecasting/Forecast_2526_FullSeason.xlsx", sheet_name="Detail")
    done = df[df["Status"] == "Completed"].copy()
    wape = (done["Pred_Adj"] - done["Actual"]).abs().sum() / done["Actual"].sum() * 100
    bias = ((done["Pred_Adj"] - done["Actual"]) / done["Actual"]).mean() * 100
    return {
        "n":      len(done),
        "actual": int(done["Actual"].sum()),
        "pred":   int(done["Pred_Adj"].sum()),
        "wape":   wape,
        "bias":   bias,
    }


def load_revenue():
    per_ev = pd.read_excel(
        "forecasting/Forecast_2526_Revenue_Hindcast.xlsx", sheet_name="Per-Event")
    per_ev = per_ev[per_ev["Event"].notna() & (per_ev["Event"] != "Total")].copy()
    pred = int(pd.to_numeric(per_ev["Pred Revenue"], errors="coerce").sum())
    act  = int(pd.to_numeric(per_ev["Actual Revenue"], errors="coerce").sum())
    return {
        "n":     len(per_ev),
        "pred":  pred,
        "actual": act,
        "gap":   pred - act,
        "gap_pct": (pred - act) / act * 100,
    }


def build_html():
    att = load_attendance()
    rev = load_revenue()
    today = f"{date.today():%B %-d, %Y}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Music Worcester — Pre-Season Forecasting at a Glance</title>
<style>
@page {{ size: Letter; margin: 0.45in 0.55in; }}
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
    color: {INK}; margin: 0; padding: 0;
    font-size: 9.6pt; line-height: 1.40;
}}
h1 {{
    font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif;
    font-size: 16pt; color: {NAVY}; margin: 0 0 3pt 0; font-weight: 700;
    letter-spacing: -0.2pt;
}}
.subtitle {{
    color: {INK_LT}; font-size: 9.5pt; margin: 0 0 8pt 0;
    border-bottom: 1px solid {LGRAY}; padding-bottom: 5pt;
}}
h2 {{
    font-family: 'Montserrat', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt; color: {NAVY}; margin: 9pt 0 3pt 0; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.4pt;
}}
p {{ margin: 0 0 5pt 0; }}
.goal {{
    background: #F8FAFC; border-left: 3px solid {TEAL};
    padding: 7pt 11pt; margin: 5pt 0 8pt 0; font-size: 9.7pt;
}}
.goal strong {{ color: {NAVY}; }}
.kpi-row {{ display: flex; gap: 10pt; margin: 3pt 0 7pt 0; }}
.kpi {{
    flex: 1; background: #F8FAFC; padding: 7pt 9pt;
    border-radius: 3pt; border-left: 3px solid {NAVY};
}}
.kpi .label {{ color: {INK_LT}; font-size: 8.2pt; text-transform: uppercase;
    letter-spacing: 0.4pt; }}
.kpi .value {{ color: {NAVY}; font-size: 12.5pt; font-weight: 700;
    line-height: 1.1; margin-top: 1pt; }}
.kpi .sub   {{ color: {INK_LT}; font-size: 8.5pt; margin-top: 2pt; }}
.chart-wrap {{ margin: 4pt 0 1pt 0; text-align: center; }}
.chart-wrap img {{ max-width: 100%; max-height: 3.6in; object-fit: contain; }}
.caption {{
    font-size: 8.3pt; color: {INK_LT}; font-style: italic;
    text-align: center; margin: 1pt 0 5pt 0;
}}
.tier-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3pt 14pt;
    font-size: 9pt; margin-top: 3pt; }}
.tier-grid div {{ padding: 1pt 0; }}
.tier-grid .tier {{ color: {NAVY}; font-weight: 600; }}
ul {{ margin: 4pt 0 7pt 18pt; padding: 0; }}
li {{ margin-bottom: 3pt; }}
.footer {{
    margin-top: 10pt; padding-top: 5pt; border-top: 1px solid {LGRAY};
    color: {INK_LT}; font-size: 8.4pt;
}}
.page-break {{ page-break-before: always; }}
</style>
</head>
<body>

<h1>Music Worcester — Pre-Season Forecasting at a Glance</h1>
<div class="subtitle">
A finance-side primer  ·  Updated {today}
</div>

<div class="goal">
<strong>The goal:</strong> pre-season budget and revenue planning. Before
a single ticket goes on sale, the model produces an attendance and a
revenue estimate for every event in the upcoming season — built only
from the event manifest and prior-season patterns. Those estimates feed
budget templates, fundraising targets, marketing prioritization, and
capacity decisions.
</div>

<h2>The attendance forecast</h2>
<p>
For each event the model predicts paid + comp attendance using a
five-level hierarchy of historical comparables (event class, venue,
sub-genre, recurring series, artist popularity), trained only on data
from prior seasons. Predictions are made before the season starts and
locked in — they don't update as tickets sell.
</p>

<div class="kpi-row">
  <div class="kpi">
    <div class="label">25-26 Net Attendance Forecast</div>
    <div class="value">{att['pred']:,}</div>
    <div class="sub">vs. actual {att['actual']:,}  ·  {att['pred']-att['actual']:+,} ({(att['pred']-att['actual'])/att['actual']*100:+.2f}%)</div>
  </div>
  <div class="kpi">
    <div class="label">WAPE (volume-weighted error)</div>
    <div class="value">{att['wape']:.1f}%</div>
    <div class="sub">Across {att['n']} completed events</div>
  </div>
  <div class="kpi">
    <div class="label">Bias (per-event)</div>
    <div class="value">{att['bias']:+.1f}%</div>
    <div class="sub">≈0 means no over- or under-prediction trend</div>
  </div>
</div>

<div class="chart-wrap">
<img src="forecast_2526_bar_chart_eventaxis.png" alt="Forecast vs actual attendance for the 25-26 season">
</div>
<div class="caption">
Navy = predicted attendance, orange = actual. Lighter tones are comps.
</div>

<p>
<strong>For budgeting:</strong> aggregate attendance assumptions are
reliable — the season total came in 7 patrons off out of 14,411
({((att['pred']-att['actual'])/att['actual'])*100:+.2f}%). Per-event misses
concentrate on a handful of events with rising audience trends (Messiah)
or strong one-time demand that prior-season signal can't fully anticipate
(Trifonov, Savall). Use the per-event numbers as planning anchors with
the WAPE in mind: expect a 15-20% miss on any individual event, even
when the season total is on target.
</p>

<div class="page-break"></div>

<h2>The revenue forecast</h2>
<p>
Revenue = predicted paid attendance × historical ticket price for the
event's pricing tier. MW uses seven pricing tiers in the manifest that
reflect how we actually price each event type — set pre-season:
</p>
<div class="tier-grid">
  <div><span class="tier">Marquee</span> — top-draw bookings (+30% above bucket)</div>
  <div><span class="tier">Headliner &amp; Prestige</span> — +15%</div>
  <div><span class="tier">Standard</span> — +10%</div>
  <div><span class="tier">Chorus</span> — 0% (accessibly priced)</div>
  <div><span class="tier">TCB / Mission / AiR</span> — 0% (community / festival pricing)</div>
  <div>&nbsp;</div>
</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="label">25-26 Forecast Revenue</div>
    <div class="value">${rev['pred']:,}</div>
    <div class="sub">{rev['n']} events  ·  paid tickets only</div>
  </div>
  <div class="kpi">
    <div class="label">25-26 Actual Revenue</div>
    <div class="value">${rev['actual']:,}</div>
    <div class="sub">paid tickets only</div>
  </div>
  <div class="kpi" style="border-left-color: {ORANGE};">
    <div class="label">Gap (forecast − actual)</div>
    <div class="value" style="color: {ORANGE};">{rev['gap_pct']:+.1f}%</div>
    <div class="sub">${rev['gap']:+,}</div>
  </div>
</div>

<div class="chart-wrap">
<img src="forecast_revenue_chart.png" alt="Forecast vs actual revenue for the 25-26 season">
</div>
<div class="caption">
Navy = predicted revenue (pre-season), orange = actual. Tier-uplifted prior applied.
</div>

<p>
<strong>For budgeting:</strong> the season-net came within half a percent
of actual — but read that carefully. Part of the clean match is two real
errors offsetting at the aggregate level: the model under-forecast
attendance on the marquee headliners by about $46K worth, while the
tier-uplifted price prior happened to overshoot by about the same amount
in the other direction. Per-event revenue lines have visible misses on
the same events that miss on attendance (Trifonov, Messiah, Savall).
Treat the season revenue total as a credible planning number; treat
per-event revenue as a starting point that may shift by 10-20% for any
single show.
</p>

<h2>How this lands in the budget cycle</h2>
<ul>
<li><strong>Source of truth:</strong> <code>forecasting/Forecast_2526_FullSeason.xlsx</code>
(Summary + Forecast vs Actuals + Detail sheets) for attendance;
<code>forecasting/Forecast_2526_Revenue_Hindcast.xlsx</code> for revenue.
Both refresh whenever the event manifest or sales data changes.</li>
<li><strong>Pricing-tier tag is the main lever:</strong> when a new
booking is added to the manifest, the <code>PriceTier</code> column drives
the revenue projection. Marquee tag for known top-draw bookings;
Standard for typical commercial concerts; Chorus for Worcester Chorus events;
TCB for festival programming; etc. This is a strategic pre-season call,
not a model prediction.</li>
<li><strong>Refresh cadence:</strong> attendance + revenue forecasts
rerun after each Salesforce data pull (currently weekly during season,
ad-hoc off-season). All deliverables — workbooks, charts, briefs —
regenerate from the same scripts.</li>
</ul>

<div class="footer">
Built and maintained by Nolichucky Associates. Source scripts in the
repository root; outputs in <code>forecasting/</code>. For methodology
detail beyond this primer, see <code>forecast_brief.pdf</code> (peer
audience) or <code>forecast_revenue_brief.pdf</code>.
</div>

</body>
</html>
"""
    out = "forecasting/forecast_finance_intro.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"  ✓ {out}")
    return out


if __name__ == "__main__":
    build_html()
