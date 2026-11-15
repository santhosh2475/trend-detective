# Trend report: singapore_receptor

Period 2001-01 to 2025-12 (25.0 years, 300 months). Rows inserted for missing months: 2.

## What is changing

Slope = seasonal Sen slope. p = seasonal Kendall test with serial-correlation correction. 'After ENSO/IOD' = t-test on the time trend with Nino 3.4 and DMI as covariates.

| Variable | Slope / decade | % / decade | Seasonal Kendall p | Mann-Kendall p | t-test p | After ENSO/IOD p | Verdict |
|---|---|---|---|---|---|---|---|
| Active fire detections (log1p count/month) | -0.225 | -41.6 | 0.000524 | 2.42e-07 | 8.13e-06 | 1.37e-05 | significant decreasing |
| Vapour pressure deficit (kPa) | +0.109 | +15.6 | 0.0157 | 0.122 | 0.00169 | 0.00187 | significant increasing |
| Aerosol optical depth (unitless) | -0.0101 | -6.2 | 0.0639 | 0.0204 | 0.115 | 0.0144 | weak decreasing (not significant) |
| Air temperature (deg C) | +0.157 | +0.6 | 0.211 | 0.428 | 0.162 | 0.122 | no significant trend |
| Deeper soil moisture (10-40 cm) (mm) | +0.348 | +0.5 | 0.276 | 0.291 | 0.485 | 0.668 | no significant trend |
| Deeper soil moisture (fraction) (m3/m3) | +0.00116 | +0.5 | 0.276 | 0.291 | 0.485 | 0.668 | no significant trend |
| Evapotranspiration (mm/month) | +1.7 | +2.1 | 0.285 | 0.345 | 0.0895 | 0.146 | no significant trend |
| Rainfall (mm/month) | +6.56 | +3.9 | 0.346 | 0.192 | 0.199 | 0.318 | no significant trend |
| Root-zone soil moisture (mm) | +2.65 | +0.4 | 0.42 | 0.535 | 0.65 | 0.639 | no significant trend |
| Topsoil moisture (fraction) (m3/m3) | +0.000907 | +0.3 | 0.483 | 0.475 | 0.649 | 0.629 | no significant trend |
| Topsoil moisture (0-10 cm) (mm) | +0.0907 | +0.3 | 0.483 | 0.475 | 0.649 | 0.629 | no significant trend |

## Most unusual months (Elliptic Envelope)

| Month | Distance | Main contributor | z |
|---|---|---|---|
| 2006-10 | 20.7 | Aerosol optical depth | +8.7 |
| 2015-09 | 16.0 | Aerosol optical depth | +6.6 |
| 2015-10 | 13.0 | Aerosol optical depth | +5.2 |
| 2019-09 | 10.1 | Deeper soil moisture (10-40 cm) | -4.2 |
| 2006-09 | 8.5 | Aerosol optical depth | +3.4 |
| 2006-12 | 8.3 | Rainfall | +5.2 |
| 2002-10 | 6.9 | Aerosol optical depth | +2.7 |
| 2014-03 | 6.9 | Deeper soil moisture (10-40 cm) | -3.9 |
| 2005-03 | 6.4 | Deeper soil moisture (10-40 cm) | -3.5 |
| 2014-02 | 6.4 | Active fire detections | +3.3 |

## Dominant pattern (PCA)

PC1 explains 42% of the variance. Loadings: sm_10_40_mm -0.49, precip_mm -0.44, tair_c +0.43, vpd_kpa +0.42, fire_count +0.32, aod +0.26, et_mm +0.17.
Correlation of PC1 with nino34: +0.27.
Correlation of PC1 with dmi: +0.07.

## Forecasts

| Variable | Model | Backtest skill vs climatology | Peak forecast | When |
|---|---|---|---|---|
| Active fire detections | climatology + trend + AR(2) + nino34/dmi | +22% | 4.39 | 2026-03 |
| Aerosol optical depth | climatology + trend + AR(2) + nino34/dmi | -42% | 0.532 | 2026-10 |
| Vapour pressure deficit | climatology + AR(2) + nino34/dmi | -5% | 0.839 | 2026-08 |
| Deeper soil moisture (10-40 cm) | climatology + AR(2) + nino34/dmi | -11% | 92.5 | 2026-11 |
| Rainfall | climatology + AR(2) + nino34/dmi | -1% | 330 | 2025-12 |

## Alerts

- **CRITICAL** El Nino watch: Nino 3.4 reaches +2.30 in Oct 2026. Past El Nino dry seasons (2002, 2006, 2015, 2019, 2023) produced the largest fire and haze spikes in this record. Expect elevated fire risk in the Aug-Oct dry season.
- **WARNING** Vapour pressure deficit: significant increasing. +0.109 kPa per decade (seasonal Kendall p=0.0157). Still significant after removing ENSO/IOD.
- **WARNING** Active fire detections: significant decreasing. -0.225 log1p count/month per decade (seasonal Kendall p=0.000524). Still significant after removing ENSO/IOD.
- **WARNING** Evapotranspiration: latest month is +2.2 sigma from normal. Dec 2025 value 107 mm/month.
- **WARNING** Aerosol optical depth forecast above the historical 90th percentile. 2 month(s), peak 0.532 unitless in Oct 2026 (threshold 0.295). Backtest skill vs climatology -42%: no better than climatology, read as a scenario not a prediction.
