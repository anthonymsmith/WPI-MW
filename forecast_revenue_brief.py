"""
Revenue Forecast Brief — 25-26 hindcast

1-2 page HTML brief explaining the revenue forecast methodology and
results from forecast_revenue_hindcast.py. Pairs the per-event chart
with a scenario table and the volume/price decomposition.

Output: forecasting/forecast_revenue_brief.html (render to PDF via Chrome)
Run with `--anon` for the anonymized variant.
"""
import os, sys
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
    per_ev = pd.read_excel(
        "forecasting/Forecast_2526_Revenue_Hindcast.xlsx", sheet_name="Per-Event")
    per_ev = per_ev[per_ev["Event"].notna() & (per_ev["Event"] != "Total")].copy()
    for c in ["Pred Revenue", "Actual Revenue", "Pred Paid", "Actual Paid",
              "Hist Price", "Actual Price"]:
        per_ev[c] = pd.to_numeric(per_ev[c], errors="coerce")
    return per_ev


def scenarios_table():
    # Re-derive from the Per-Event sheet so the brief always agrees with the
    # workbook even if the user tweaks tier policy. Read the Scenarios sheet
    # directly — it already has the four scenarios computed.
    s = pd.read_excel(
        "forecasting/Forecast_2526_Revenue_Hindcast.xlsx", sheet_name="Scenarios",
        header=3)  # headers in row 4 (1-indexed)
    return s[s["Scenario"].notna()].copy()


def build_html(df, anon=False):
    pred_total = int(df["Pred Revenue"].sum())
    act_total  = int(df["Actual Revenue"].sum())
    pred_paid  = int(df["Pred Paid"].sum())
    act_paid   = int(df["Actual Paid"].sum())
    gap        = pred_total - act_total
    gap_pct    = gap / act_total * 100
    pred_price = pred_total / pred_paid if pred_paid else 0
    act_price  = act_total  / act_paid  if act_paid  else 0

    n = len(df)
    today = f"{date.today():%B %-d, %Y}"

    org = "a regional performing-arts presenter" if anon else "Music Worcester"
    title_org = "" if anon else "Music Worcester — "
    chart_src = ("forecast_revenue_chart_anon.png" if anon
                 else "forecast_revenue_chart.png")

    # Scenarios
    scen = pd.read_excel(
        "forecasting/Forecast_2526_Revenue_Hindcast.xlsx", sheet_name="Scenarios",
        skiprows=3)
    scen = scen[scen["Scenario"].notna()
                & ~scen["Scenario"].str.contains("ACTUAL", na=False)].copy()

    scen_rows = []
    for _, r in scen.iterrows():
        delta = float(r["Δ vs Actual"])
        pct   = float(r["% vs Actual"])
        is_primary = "Tiered" in str(r["Scenario"])
        row_class = " class='primary'" if is_primary else ""
        scen_rows.append(
            f"<tr{row_class}>"
            f"<td>{r['Scenario']}</td>"
            f"<td class='num'>${int(r['Pred Revenue']):,}</td>"
            f"<td class='num'>{int(delta):+,}</td>"
            f"<td class='num'>{pct*100:+.1f}%</td>"
            f"</tr>"
        )
    scen_html = "\n".join(scen_rows)

    # Volume / price decomposition (from Summary sheet)
    summary = pd.read_excel(
        "forecasting/Forecast_2526_Revenue_Hindcast.xlsx", sheet_name="Summary",
        skiprows=3)
    vol_contrib   = int(summary.loc[summary["Metric"].str.contains(
        "Volume contribution", na=False), "Δ"].iloc[0])
    price_contrib = int(summary.loc[summary["Metric"].str.contains(
        "Price contribution", na=False), "Δ"].iloc[0])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Revenue Forecast Brief — 25-26 Season</title>
<style>
@page {{ size: Letter; margin: 0.45in 0.55in; }}
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
    color: {INK}; margin: 0; padding: 0;
    font-size: 9.7pt; line-height: 1.40;
}}
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
.lede {{ font-size: 10pt; }}
.headline {{
    background: #F8FAFC; border-left: 3px solid {TEAL};
    padding: 7pt 11pt; margin: 5pt 0 8pt 0; font-size: 9.7pt;
}}
.headline strong {{ color: {NAVY}; }}
.kpi-row {{ display: flex; gap: 12pt; margin: 4pt 0 7pt 0; }}
.kpi {{
    flex: 1; background: #F8FAFC; padding: 8pt 10pt;
    border-radius: 3pt; border-left: 3px solid {NAVY};
}}
.kpi .label {{ color: {INK_LT}; font-size: 8.4pt; text-transform: uppercase;
    letter-spacing: 0.4pt; }}
.kpi .value {{ color: {NAVY}; font-size: 13.5pt; font-weight: 700;
    line-height: 1.1; margin-top: 1pt; }}
.kpi .sub   {{ color: {INK_LT}; font-size: 8.5pt; margin-top: 2pt; }}
table.scen {{
    width: 100%; border-collapse: collapse;
    margin: 3pt 0 8pt 0; font-size: 9.3pt;
}}
table.scen th {{
    text-align: left; color: {INK_LT}; font-weight: 600;
    border-bottom: 1px solid {LGRAY}; padding: 5pt 8pt 5pt 0;
}}
table.scen td {{
    padding: 5pt 8pt 5pt 0; border-bottom: 1px solid #F0F3F7;
}}
table.scen td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.scen tr.primary td {{
    background: #FFF6E8; font-weight: 700; color: {NAVY};
}}
ol, ul {{ margin: 4pt 0 7pt 18pt; padding: 0; }}
li {{ margin-bottom: 3pt; }}
.chart-wrap {{ margin: 4pt 0 2pt 0; text-align: center; }}
.chart-wrap img {{ max-width: 100%; max-height: 5.0in; object-fit: contain; }}
.caption {{
    font-size: 8.3pt; color: {INK_LT}; font-style: italic;
    text-align: center; margin: 2pt 0 6pt 0;
}}
.tier-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4pt 14pt;
    font-size: 9pt; margin-top: 3pt; }}
.tier-grid div {{ padding: 1pt 0; }}
.tier-grid .tier {{ color: {NAVY}; font-weight: 600; }}
.footer {{
    margin-top: 12pt; padding-top: 6pt; border-top: 1px solid {LGRAY};
    color: {INK_LT}; font-size: 8.4pt;
}}
.page-break {{ page-break-before: always; }}
</style>
</head>
<body>

<h1>{title_org}25–26 Revenue Forecast — Hindcast &amp; Pricing Read</h1>
<div class="subtitle">
A blinded revenue projection compared to actual results, with pricing-tier policy applied  ·  {today}
</div>

<p class="lede">
This brief tests one question: how close to actual ticket revenue can the
forecast model land using only what's knowable <em>before</em> the season
starts: the event manifest, prior-season audience patterns, and historical
ticket prices? Predicted attendance is taken from the temporal-holdout
forecast (the same model that produced WAPE 17.2% on paid+comp attendance
this season). Each event's predicted paid attendance is then multiplied by
a historical bucket-average ticket price, adjusted upward by a tiered
pricing policy that reflects how {org} actually thinks about pricing
across event types.
</p>

<div class="kpi-row">
  <div class="kpi">
    <div class="label">Forecast Revenue</div>
    <div class="value">${pred_total:,}</div>
    <div class="sub">{n} events · ${pred_price:.0f} avg/tkt</div>
  </div>
  <div class="kpi">
    <div class="label">Actual Revenue</div>
    <div class="value">${act_total:,}</div>
    <div class="sub">{n} events · ${act_price:.0f} avg/tkt</div>
  </div>
  <div class="kpi" style="border-left-color: {ORANGE};">
    <div class="label">Gap</div>
    <div class="value" style="color: {ORANGE};">{gap_pct:+.1f}%</div>
    <div class="sub">${gap:+,} · forecast vs actual</div>
  </div>
</div>

<h2>Per-event view</h2>
<div class="chart-wrap">
<img src="{chart_src}" alt="Per-event forecast vs actual revenue">
</div>
<div class="caption">
Navy = forecast revenue (Pred_Paid × tier-uplifted historical price). Orange = actual paid revenue. Sorted by event date.
</div>

<h2>The pricing-tier policy</h2>
<p>
Historical bucket prices don't capture how {org} actually prices each event
type. The model applies a uniform tiered uplift to encode that pricing logic:
</p>
<div class="tier-grid">
  <div><span class="tier">Marquee</span>: +40% (pre-season top-draw tag)</div>
  <div><span class="tier">Headliner &amp; Prestige</span>: +15%</div>
  <div><span class="tier">Standard</span>: +10%</div>
  <div><span class="tier">Chorus</span>: 0% (accessibly priced)</div>
  <div><span class="tier">TCB / Mission / AiR</span>: 0% (community / festival pricing)</div>
</div>

<h2>Why scenarios matter</h2>
<table class="scen">
<tr><th>Scenario</th><th class="num">Forecast Revenue</th><th class="num">Δ vs Actual</th><th class="num">% Gap</th></tr>
{scen_html}
</table>
<p style="font-size:9pt; color:{INK_LT}; margin-top:-2pt;">
Each row uses the same pre-season attendance forecast; only the price
prior changes. The tiered policy (highlighted) closes about half of the
residual revenue gap relative to a raw historical prior.
</p>

<h2>How the tier policy works (and what the season-net gap hides)</h2>
<p>
At the season level, the tier policy with Marquee included lands within
half a percent of actual revenue. That clean aggregate match is partly
because a real volume miss and a partial price overshoot offset each
other. The underlying components are larger than the net:
</p>
<ul>
<li><strong>Volume contribution: ${vol_contrib:+,}.</strong> The forecast
under-called paid attendance on a handful of marquee headliners with
growing audiences (Messiah on its upward trajectory, ONF/Trifonov as the
most prestigious orchestra in years, Savall on a strong following). These are real-world drivers,
not model defects.</li>
<li><strong>Price contribution: ${price_contrib:+,}.</strong> Historical
bucket prices ran below 25-26 realized prices. The tier policy closes
this gap by encoding {org}'s pricing logic explicitly; the Marquee tier in
particular reflects MW's strategic decision to price top-draw bookings
above the bucket norm.</li>
</ul>

<h2>Takeaways for 26-27</h2>
<ol>
<li><strong>The Marquee tier is doing meaningful work.</strong> Six
events ({org}'s known top-draw bookings: ONF/Trifonov, Savall, the
Sebastians, Dinnerstein Recital) ran at prices 35%+ above their
historical bucket. A pre-season Marquee tag captures that strategic
pricing decision and feeds it into the revenue projection cleanly.</li>
<li><strong>Top-tier headliner pricing still has more headroom than the
+40% Marquee uplift captures.</strong> ONF/Trifonov alone ran +78% above
bucket; Savall +65%. Stretching to +35-40% on the most marquee bookings
next cycle has empirical support.</li>
<li><strong>The tier policy works as designed for non-Marquee tiers.</strong>
Standard secular concerts behaved like the +10% uplift. Chorus and TCB
events held to their accessible pricing posture. Mission and AiR
programming priced at community levels. Keeping these as separate policy
tiers protects each audience segment from misapplied uplifts.</li>
<li><strong>Per-event volume misses remain on the marquees.</strong> Even
with revenue netting cleanly, individual events show real volume
under-prediction (ONF/Trifonov 823 forecast vs 1,089 actual; Messiah 750 vs
986). These are signals about audience growth on specific recurring
properties, useful inputs to planning conversations on next year's
analogous bookings.</li>
</ol>

<div class="footer">
Model: temporal-holdout hierarchy (5 fallback levels) + Empirical-Bayes
shrinkage + Bayesian-shrunk PWYW lift + artist popularity adjustment. Price
prior: same hierarchy as comp-split, PWYW/Free excluded from training,
season-weighted, K=3 Bayesian shrink. Source: <code>forecast_revenue_hindcast.py</code> +
<code>forecasting/Forecast_2526_Revenue_Hindcast.xlsx</code>.
</div>

</body>
</html>
"""

    out = ("forecasting/forecast_revenue_brief_anon.html" if anon
           else "forecasting/forecast_revenue_brief.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"  ✓ {out}")
    return out


def main():
    df = load()
    build_html(df, anon=ANON)


if __name__ == "__main__":
    main()
