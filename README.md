# Trend Detective

Streamlit dashboard + command-line pipeline for the NASA Space Apps "Earth Systems Trend Detective" challenge,
built on the Indonesia peat-fire dataset (2001-2025).

## Run

```bash
pip install -r requirements.txt
# put gldas_grid.nc and imerg_grid.nc in data/ (aod_grid.nc is already there)
streamlit run app.py
```

Pipeline without the UI (any monthly CSV: a date column plus numeric columns):

```bash
python -m trendlab.pipeline data/central_kalimantan.csv --out results/central_kalimantan
python -m trendlab.pipeline my.csv --webhook https://hooks.slack.com/services/...   # push new alerts
```

Outputs: `report.md`, `trend_table.csv`, `outliers.csv`, `pca_loadings.csv`, `forecast.csv`, `alerts.json`,
`alerts_log.jsonl` (dedupes, so a re-run only notifies on new alerts), `summary.json`.

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
6. **Forecast**: climatology + AR(2) + lagged, season-weighted Nino 3.4 / DMI (+ linear trend if it backtests better), 36-month hold-out, skill vs climatology.
7. **Alerts** (`trendlab/alerts.py`): significant trends, latest-month extremes, recent outliers, forecast exceedance, El Nino watch. In-app toasts, JSONL log, Slack/Discord webhook, SMTP email.

Maps (`trendlab/grid.py`): per-pixel seasonal Kendall + Sen slope on the NetCDF grids, with Benjamini-Hochberg false-discovery control, and a draw-your-own-box region test.

## Calibration check

On 300 simulated trend-free series (25 years, monthly), share wrongly called significant at 0.05:

| Noise | Seasonal Kendall (corrected) | Mann-Kendall (corrected) | Uncorrected tests |
|---|---|---|---|
| White | 6% | 6% | 7% |
| AR(1) = 0.6 | 8% | 10% | 29-30% |

## Caveats

- GLDAS is a mineral-soil model: relative drying only, not the peat water table.
- Forecasts for AOD and soil moisture do not beat climatology on the hold-out; the app labels them as scenarios.
- The forecast model choice (trend term on/off) is made on the same hold-out that reports skill, so skill is slightly optimistic.
