# Trend report: central_kalimantan

Period 2001-01 to 2025-12 (25.0 years, 300 months). Rows inserted for missing months: 2.

## What is changing

Slope = seasonal Sen slope. p = seasonal Kendall test with serial-correlation correction. 'After ENSO/IOD' = t-test on the time trend with Nino 3.4 and DMI as covariates.

| Variable | Slope / decade | % / decade | Seasonal Kendall p | Mann-Kendall p | t-test p | After ENSO/IOD p | Verdict |
|---|---|---|---|---|---|---|---|
| Active fire detections (log1p count/month) | -0.994 | -30.0 | 0.000489 | 6.35e-06 | 8.43e-05 | 2.42e-09 | significant decreasing |
| Evapotranspiration (mm/month) | +7.5 | +6.2 | 0.000884 | 0.0153 | 1.24e-05 | 4.6e-05 | significant increasing |
| Vapour pressure deficit (kPa) | +0.159 | +24.0 | 0.00301 | 0.0313 | 0.000103 | 0.000111 | significant increasing |
| Air temperature (deg C) | +0.292 | +1.2 | 0.0509 | 0.207 | 0.0195 | 0.01 | weak increasing (not significant) |
| Aerosol optical depth (unitless) | -0.00857 | -9.5 | 0.0896 | 0.00416 | 0.0443 | 0.00945 | weak decreasing (not significant) |
| Rainfall (mm/month) | +6.32 | +2.4 | 0.593 | 0.377 | 0.475 | 0.111 | no significant trend |
| Deeper soil moisture (fraction) (m3/m3) | +0.000656 | +0.6 | 0.619 | 0.129 | 0.348 | 0.0508 | no significant trend |
| Deeper soil moisture (10-40 cm) (mm) | +0.197 | +0.6 | 0.619 | 0.155 | 0.348 | 0.0508 | no significant trend |
| Root-zone soil moisture (mm) | +1.56 | +0.8 | 0.684 | 0.129 | 0.4 | 0.0593 | no significant trend |
| Topsoil moisture (fraction) (m3/m3) | +0.000444 | +0.5 | 0.732 | 0.299 | 0.37 | 0.0576 | no significant trend |
| Topsoil moisture (0-10 cm) (mm) | +0.0444 | +0.5 | 0.732 | 0.299 | 0.37 | 0.0576 | no significant trend |

## Most unusual months (Elliptic Envelope)

| Month | Distance | Main contributor | z |
|---|---|---|---|
| 2002-09 | 92.8 | Aerosol optical depth | +9.7 |
| 2006-10 | 59.3 | Aerosol optical depth | +6.3 |
| 2002-10 | 51.4 | Aerosol optical depth | +5.4 |
| 2002-08 | 45.9 | Aerosol optical depth | +4.8 |
| 2006-11 | 43.6 | Aerosol optical depth | +4.7 |
| 2015-10 | 34.2 | Deeper soil moisture (10-40 cm) | -5.4 |
| 2015-09 | 22.6 | Deeper soil moisture (10-40 cm) | -4.6 |
| 2006-09 | 21.3 | Aerosol optical depth | +2.3 |
| 2019-09 | 16.6 | Deeper soil moisture (10-40 cm) | -3.5 |
| 2006-08 | 16.4 | Aerosol optical depth | +1.8 |

## Dominant pattern (PCA)

PC1 explains 48% of the variance. Loadings: sm_10_40_mm -0.51, precip_mm -0.45, fire_count +0.41, vpd_kpa +0.38, aod +0.35, tair_c +0.31, et_mm -0.03.
Correlation of PC1 with nino34: +0.39.
Correlation of PC1 with dmi: +0.35.

## Forecasts

| Variable | Model | Backtest skill vs climatology | Peak forecast | When |
|---|---|---|---|---|
| Active fire detections | climatology + trend + AR(2) + nino34/dmi | +53% | 2.73e+03 | 2026-09 |
| Aerosol optical depth | climatology + AR(2) + nino34/dmi | -49% | 0.932 | 2026-09 |
| Vapour pressure deficit | climatology + AR(2) + nino34/dmi | +11% | 0.974 | 2026-09 |
| Deeper soil moisture (10-40 cm) | climatology + trend + AR(2) + nino34/dmi | -10% | 92.3 | 2026-03 |
| Rainfall | climatology + AR(2) + nino34/dmi | +9% | 372 | 2025-12 |

## Alerts

- **CRITICAL** Active fire detections forecast above the historical 90th percentile. 2 month(s), peak 2.73e+03 count/month in Sep 2026 (threshold 1.4e+03). Backtest skill vs climatology +53%.
- **CRITICAL** El Nino watch: Nino 3.4 reaches +2.30 in Oct 2026. Past El Nino dry seasons (2002, 2006, 2015, 2019, 2023) produced the largest fire and haze spikes in this record. Expect elevated fire risk in the Aug-Oct dry season.
- **WARNING** Evapotranspiration: significant increasing. +7.5 mm/month per decade (seasonal Kendall p=0.000884). Still significant after removing ENSO/IOD.
- **WARNING** Vapour pressure deficit: significant increasing. +0.159 kPa per decade (seasonal Kendall p=0.00301). Still significant after removing ENSO/IOD.
- **WARNING** Active fire detections: significant decreasing. -0.994 log1p count/month per decade (seasonal Kendall p=0.000489). Still significant after removing ENSO/IOD.
- **WARNING** Aerosol optical depth forecast above the historical 90th percentile. 4 month(s), peak 0.932 unitless in Sep 2026 (threshold 0.357). Backtest skill vs climatology -49%: no better than climatology, read as a scenario not a prediction.
- **WARNING** Vapour pressure deficit forecast above the historical 90th percentile. 2 month(s), peak 0.974 kPa in Sep 2026 (threshold 0.94). Backtest skill vs climatology +11%.
