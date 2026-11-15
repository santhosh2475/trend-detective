# Trend Detective

Streamlit dashboard + command-line pipeline for the NASA Space Apps "Earth Systems Trend Detective" challenge,
built on the Indonesia peat-fire dataset (2001-2025).

## Run

```bash
pip install -r requirements.txt
# optional: put gldas_grid.nc and imerg_grid.nc in data/ (aod_grid.nc is already there)
streamlit run app.py
```

The AI chat on the **AI advisor** page uses Claude. Give it a key in `.streamlit/secrets.toml`
(`ANTHROPIC_API_KEY = "sk-ant-..."`, already git-ignored), as an environment variable, or paste one into the page
for the session. Everything else, including the data-driven recommendations, works without a key.

Pipeline without the UI (any monthly CSV: a date column plus numeric columns):

```bash
python -m trendlab.pipeline data/central_kalimantan.csv --out results/central_kalimantan
python -m trendlab.pipeline my.csv --webhook https://hooks.slack.com/services/...   # push new alerts
python -m trendlab.models data/*.csv --lead 1 3 --save results                      # model leaderboards
```

Outputs: `report.md`, `trend_table.csv`, `outliers.csv`, `pca_loadings.csv`, `forecast.csv`, `alerts.json`,
`alerts_log.jsonl` (dedupes, so a re-run only notifies on new alerts), `summary.json`, and
`models/<data hash>/lead<L>_<variable>.csv` (per-year backtest predictions of every model).

## The app

| Page | What it does |
|---|---|
| Start here | Three questions (where, what worries you, which years) answered in plain language, plus one-click demos. |
| Explore trends | Rate of change and four significance tests for any variable; month-by-month trends; all variables on one axis. |
| Globe | Interactive 3D globe: year-by-year fire-season haze time-lapse with fire bubbles, or per-pixel trends with FDR control. Click a pixel to test it. |
| Compare and link regions | Lagged correlation and joint regression of Singapore's haze on upwind fires; the same variable across regions. |
| Patterns | PCA (which variables move together) and Elliptic Envelope (unusual months). |
| AI advisor | Variables to watch, early-warning signals, the model to trust, NASA datasets to add; optional Claude chat. |
| Models and forecast | Eight models compared on years they never saw; the recommended one makes the forecast. |
| Alerts, Data and export | Rule-based alerts (Slack, email, log) and downloads. |

## Pipeline

1. **Validate**: parse dates, average duplicate months, reindex to a full monthly calendar, fill Nino 3.4 / DMI from `climate_indices.csv`.
2. **Deseasonalise**: subtract each calendar month's mean. Fire counts use log(1 + count).
3. **Trend tests** (`trendlab/core.py`, written from scratch):
   - Seasonal Kendall (Hirsch 1982) with serial-correlation variance (Hirsch & Slack 1984) + seasonal Sen slope. This is the headline verdict.
   - Mann-Kendall on anomalies with Hamed-Rao autocorrelation correction; Theil-Sen slope with 95% CI.
   - t-test on the OLS slope with AR(1) effective sample size.
   - t-test on the trend with Nino 3.4 and DMI as lagged covariates, so El Nino spikes are not read as trend.
4. **Elliptic Envelope** (`trendlab/ml.py`): robust-covariance outlier months across variables.
5. **PCA**: dominant joint pattern and its correlation with ENSO / IOD.
6. **Model comparison** (`trendlab/models.py`): see below.
7. **Alerts** (`trendlab/alerts.py`): significant trends, latest-month extremes, recent outliers, forecast exceedance, El Nino watch. In-app toasts, JSONL log, Slack/Discord webhook, SMTP email.

Maps (`trendlab/grid.py`): per-pixel seasonal Kendall + Sen slope on the NetCDF grids, with Benjamini-Hochberg false-discovery control, and a draw-your-own-box region test.

## Forecast model comparison

Models: seasonal average, same month last year, persistence, ridge regression, negative binomial regression (counts),
gradient boosting (squared-error and Poisson), random forest, and a small neural network (3-seed MLP ensemble).

- **Inputs**: lagged anomalies of the target (6 months), of rain, soil moisture, VPD, temperature, ET, AOD and fires,
  lagged Nino 3.4 / DMI (also weighted by how volatile each calendar month is, so ENSO can matter mostly in the dry season),
  and the calendar month. A forecast `lead` months ahead uses only data at least `lead` months old: no future ENSO values.
- **Validation**: rolling origin. Each year from 2011 is predicted by models fitted only on the years before it, with
  climatologies recomputed on those years. Score = mean absolute error vs the seasonal average, overall and for Jul-Oct.
- **Recommendation**: the best model, unless a more explainable one is within 2 points; the seasonal average if nothing beats it by 5%.
- **Forecast**: one direct model per lead (1-12 months); the 80% range is the model's own rolling-origin error at each lead.
- Predictions are kept inside the range of anomalies seen in training, so a log-scale model cannot overshoot an El Nino year into an absurd count.

What the comparison found on the bundled regions (33 region-variable pairs):

- **1 month ahead**: ridge regression (13) or random forest (12) is recommended for most pairs, persistence for 6.
- **3 months ahead**: ridge regression wins for fires (+12-13% vs the seasonal average in both fire regions), VPD,
  temperature and evapotranspiration; for rainfall and soil moisture nothing beats the seasonal average (20 of 33 pairs).
- **6 months ahead** (Central Kalimantan fires): nothing beats the seasonal average.

No LSTM: about 300 months per region and five strong El Nino seasons are too few to train one; the MLP stands in for neural models.

## Calibration check

On 300 simulated trend-free series (25 years, monthly), share wrongly called significant at 0.05:

| Noise | Seasonal Kendall (corrected) | Mann-Kendall (corrected) | Uncorrected tests |
|---|---|---|---|
| White | 6% | 6% | 7% |
| AR(1) = 0.6 | 8% | 10% | 29-30% |

## Caveats

- GLDAS is a mineral-soil model: relative drying only, not the peat water table.
- 2001 to mid-2002 fire counts come from Terra only (Aqua launched in 2002), so early counts are low and the decline is understated.
- The ENSO scenario forecast used by the alerts is backtested with real Nino 3.4 / DMI values, so its skill is a best case;
  the model comparison does not have that problem.
- Correlations between regions and variables show what moves together, not what causes what. Most fires here are lit by people.
