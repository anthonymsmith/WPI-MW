# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Patron and event analytics for Music Worcester (a WPI engagement, by Nolichucky Associates). Three concerns: a sales/patron processing pipeline, a donor-prospect classifier, and an attendance forecasting model. There is no application — everything is batch scripts and notebooks that read/write `.csv` and `.xlsx` files in the repo root.

## Environment

- Python 3.9, virtualenv at `.venv1`. Run scripts with `.venv1/bin/python <script>.py`.
- Install deps: `.venv1/bin/pip install -r requirements.txt`.
- No test suite, no linter, no build step.
- Most `forecast_*.py` scripts hardcode `os.chdir("/Users/antho/Documents/WPI-MW")` at the top, so they must be run from a machine with that path (or the chdir line updated). `MWSalesSumm_run.py` prompts for the working directory instead.
- IDE is DataSpell (JetBrains). For any file that exists as both `.ipynb` and `.py` (`MWSalesSumm`, `DonorClassifier`), the `.py` is the source of record. DataSpell auto-save will overwrite external edits to an open notebook — close the notebook in the IDE before editing it externally.

## Pipeline (run in order)

The three stages are chained by files on disk, not imports. Each stage's output is the next stage's input.

1. **Sales/patron processing** — `MWSalesSumm.ipynb` (or `MWSalesSumm_run.py`). Reads `SalesforceLatest.csv` + `EventManifest.xlsx` + `Budget/EventPnL.xlsx` + several roster CSVs. Writes `DataMerge.csv` (full processed transaction table), `Patrons.csv` (per-patron metrics: RFM, segment, growth, retention), `RetChurnRates.csv`, `summary_Patrons.xlsx`. Logic lives in `MW_functions.py` (load/merge/geocode/segment) and `Model_functions.py` (RFM, growth, segment assignment).
2. **Donor classification** — `DonorClassifier.py` (or `.ipynb`). Reads `Patrons.csv` + `DonationsLatest.xlsx`. Writes per-tranche sheets to `donors/`.
3. **Forecasting** — `forecast_*.py`. Reads `DataMerge.csv` + `EventManifest.xlsx`. Writes to `forecasting/`.

Before re-running stage 1 against a fresh Salesforce export, run `repair_salesforce_export.py` first — the SF export intermittently corrupts rows with extended-ASCII characters by eating a field separator, which silently drops transactions downstream. The repair rewrites `SalesforceLatest.csv` in place (original to `.bak`).

## Forecast model architecture

`forecast_2526_comparison.py` is the model library, not just a script — the other `forecast_*.py` files import `load_data`, `build_hierarchy_models`, `predict_model_a`, `build_pwyw_lift`, `build_side_lift` from it. Three further modules layer on top:

- `forecast_artist_adjustment.py` — Bayesian artist-popularity adjustment (Last.fm/Wikipedia/Deezer signals via `artist_signals.py`, cached in `artist_signals_cache.json`).
- `forecast_bucket_ci.py` — confidence intervals from bucket-level residuals.
- `forecast_comp_split.py` — splits a total-attendance prediction into paid vs. comp.

**Model A** is a fallback hierarchy: an event's predicted attendance comes from the most specific historical bucket with enough observations — `EventRepeat` series → `(Class, Venue, LoB, SubGenre)` → progressively coarser fallbacks → `EventClass`. Thin buckets are Bayesian-shrunk toward their next-coarser fallback. Multiplicative lifts (`build_pwyw_lift` for pay-what-you-will, `build_side_lift` for festival side events) are calibrated from prior-season residuals and applied inside `predict_model_a`.

**Honest vs. leaky evaluation:** hindcasts must train only on seasons *before* the target season (temporal holdout). The notebook default is a leaky leave-one-out for smoke-testing only — never quote it as real accuracy. When calibrating a new layer, verify the honest (prior-seasons-only) calibration; a signal that only shows up leaky is an artifact.

`forecast_2526_full_season.py` produces the headline deliverable (`forecasting/Forecast_2526_FullSeason.xlsx`); `forecast_2526_comparison.py` produces the per-event diagnostic table.

## EventManifest.xlsx

Hand-curated technical source of truth for event attributes — one row per `InstanceId`, raw codes. It is the only `.xlsx` tracked in git (whitelisted in `.gitignore`); all other `.xlsx`/`.csv` data files are gitignored. Edits go through this round-trip: `event_manifest_guide.py` exports a human-readable `Event_Catalog.xlsx` for staff review → staff edit the catalog → `sync_catalog_to_manifest.py` writes edits back to the manifest → `derive_price_tiers.py` re-derives computed columns. The manifest is backed up to `EventManifest.xlsx.bak.<date>_<reason>` before any scripted edit; preserve that convention.

## Conventions

- Logging: `logger.debug/info` with `%s`-style args (not f-strings) for expensive values. Timing via the `_elapsed(start)` helper.
- `Patrons.csv` uses native column names (`AccountName`, `Segment`, `RFMScore`); `summary_Patrons.xlsx` uses display names (`Account Name`, `Customer Segment`). The patron↔donation join key is `AccountName` (name-based, by design: Salesforce `ContactID` and `AccountID` sit at different hierarchy levels and can't be joined directly).
- Forecast scripts read chained inputs from and write outputs to `forecasting/<file>`; `DonorClassifier` writes to `donors/<file>`. HTML `img src` strings inside generated HTML stay as bare filenames (the HTML lives alongside its PNGs).
- When re-running an analysis script (`forecast_2526_comparison.py`, eval scripts), pipe stdout through `grep`/`head` to surface only the decision-relevant lines (e.g. `| grep -E "MAPE|WAPE|Bias"`). Full event-level dumps are large; only show them when asked or when a filter would hide something load-bearing.

## Writing style for generated documents

Several scripts generate prose deliverables (`forecast_brief.py`, `forecast_memo.py`, `forecast_finance_intro*.py`, the `generate_*slide*` / `generate_forecast_deck.py` scripts). Audience determines voice.

**External / peer briefs** (anything shared outside MW internal: peer analysts, prospects, other arts orgs). Narrative and stakes-driven, not academic. Open with concrete stakes, not an abstract problem statement. Lead with the result (chart plus one-line take), then the method. Compress methodology into a few prose paragraphs, not bulleted feature lists. Section labels are active ("Inside the model", not "Methodology"); the title poses a question or claim, not the artifact name.

**Internal MW finance memos** (`forecast_memo.py`) can stay clinical; that audience expects it.

**Both** use first-person plural ("we've developed", "we hindcasted"), define every technical term inline in plain language the first time it appears (hindcast, WAPE, bias), keep retrospective and forward-looking content under separate headers, aim short (cut hard, don't pad), and minimize em-dashes (prefer period, semicolon, parentheses, colon). Headers describe the takeaway or meaning, not the technique.

**Committee / board slides** use a plain white style (no cards, boxes, or icons), plain English with no analyst jargon ("headliner night", "about 300 fewer tickets", not "MAPE" or "per-event average"), and lead each slide with a one-sentence headline in plain numbers. The polished boxed/card deck style is reserved for client-facing decks and used only when explicitly asked.

**Editing workflow.** The user saves hand-edited versions of generated docs with an `-ams` filename suffix (e.g. `forecast_finance_intro-ams.docx` alongside the script-generated `forecast_finance_intro.docx`). The `-ams` file is the canonical text: read it before regenerating, never overwrite it, merge its wording back into the generator script. Once the user is actively editing a doc, deliver further prose tweaks as text in chat rather than regenerating the doc, which would overwrite their refinements. If a regenerate is genuinely needed (numbers changed), confirm first and warn that in-flight edits will be lost.
