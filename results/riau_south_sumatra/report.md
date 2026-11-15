# Trend report: riau_south_sumatra

Period 2001-01 to 2025-12 (25.0 years, 300 months). Rows inserted for missing months: 2.

## What is changing

Slope = seasonal Sen slope. p = seasonal Kendall test with serial-correlation correction. 'After ENSO/IOD' = t-test on the time trend with Nino 3.4 and DMI as covariates.

| Variable | Slope / decade | % / decade | Seasonal Kendall p | Mann-Kendall p | t-test p | After ENSO/IOD p | Verdict |
|---|---|---|---|---|---|---|---|
| Vapour pressure deficit (kPa) | +0.147 | +26.0 | 0.00122 | 0.0301 | 1.82e-05 | 3.06e-05 | significant increasing |
| Active fire detections (log1p count/month) | -0.755 | -12.5 | 0.0046 | 0.0017 | 0.000415 | 3.28e-05 | significant decreasing |
| Evapotranspiration (mm/month) | +4.59 | +4.2 | 0.0137 | 0.0677 | 0.00523 | 0.0112 | significant increasing |
| Aerosol optical depth (unitless) | -0.0131 | -8.3 | 0.0272 | 0.00112 | 0.107 | 0.0109 | significant decreasing |
| Air temperature (deg C) | +0.183 | +0.8 | 0.181 | 0.328 | 0.112 | 0.0835 | no significant trend |
| Rainfall (mm/month) | +4.04 | +1.6 | 0.492 | 0.445 | 0.58 | 0.13 | no significant trend |
| Deeper soil moisture (fraction) (m3/m3) | +0.000933 | +0.3 | 0.508 | 0.534 | 0.935 | 0.346 | no significant trend |
| Deeper soil moisture (10-40 cm) (mm) | +0.28 | +0.3 | 0.508 | 0.534 | 0.935 | 0.346 | no significant trend |
| Root-zone soil moisture (mm) | +1.46 | +0.2 | 0.596 | 0.664 | 0.984 | 0.456 | no significant trend |
| Topsoil moisture (fraction) (m3/m3) | +0.000669 | +0.2 | 0.66 | 0.624 | 0.967 | 0.361 | no significant trend |
| Topsoil moisture (0-10 cm) (mm) | +0.0669 | +0.2 | 0.66 | 0.637 | 0.967 | 0.361 | no significant trend |

## Most unusual months (Elliptic Envelope)

| Month | Distance | Main contributor | z |
|---|---|---|---|
| 2006-10 | 43.6 | Aerosol optical depth | +9.3 |
| 2015-10 | 27.2 | Aerosol optical depth | +6.1 |
| 2015-09 | 20.1 | Aerosol optical depth | +4.5 |
| 2006-09 | 17.5 | Aerosol optical depth | +3.6 |
| 2002-10 | 15.1 | Aerosol optical depth | +3.1 |
| 2005-02 | 13.6 | Aerosol optical depth | +2.9 |
| 2002-09 | 13.5 | Aerosol optical depth | +2.7 |
| 2006-11 | 12.9 | Aerosol optical depth | +3.1 |
| 2019-09 | 12.2 | Deeper soil moisture (10-40 cm) | -4.1 |
| 2014-02 | 11.1 | Aerosol optical depth | +2.4 |

## Dominant pattern (PCA)

PC1 explains 48% of the variance. Loadings: sm_10_40_mm -0.50, precip_mm -0.47, aod +0.39, fire_count +0.38, vpd_kpa +0.35, tair_c +0.33, et_mm +0.07.
Correlation of PC1 with nino34: +0.32.
Correlation of PC1 with dmi: +0.32.

## Forecasts

| Variable | Model | Backtest skill vs climatology | Peak forecast | When |
|---|---|---|---|---|
| Active fire detections | climatology + trend + AR(2) + nino34/dmi | +23% | 1.46e+03 | 2026-09 |
| Aerosol optical depth | climatology + trend + AR(2) + nino34/dmi | -9% | 0.684 | 2026-10 |
| Vapour pressure deficit | climatology + AR(2) + nino34/dmi | -2% | 0.778 | 2026-09 |
| Deeper soil moisture (10-40 cm) | climatology + AR(2) + nino34/dmi | -15% | 104 | 2026-01 |
| Rainfall | climatology + trend + AR(2) + nino34/dmi | +0% | 347 | 2025-11 |

## Alerts

- **CRITICAL** El Nino watch: Nino 3.4 reaches +2.30 in Oct 2026. Past El Nino dry seasons (2002, 2006, 2015, 2019, 2023) produced the largest fire and haze spikes in this record. Expect elevated fire risk in the Aug-Oct dry season.
- **WARNING** Evapotranspiration: significant increasing. +4.59 mm/month per decade (seasonal Kendall p=0.0137). Still significant after removing ENSO/IOD.
- **WARNING** Vapour pressure deficit: significant increasing. +0.147 kPa per decade (seasonal Kendall p=0.00122). Still significant after removing ENSO/IOD.
- **WARNING** Aerosol optical depth: significant decreasing. -0.0131 unitless per decade (seasonal Kendall p=0.0272). Still significant after removing ENSO/IOD.
- **WARNING** Active fire detections: significant decreasing. -0.755 log1p count/month per decade (seasonal Kendall p=0.0046). Still significant after removing ENSO/IOD.
- **WARNING** Aerosol optical depth forecast above the historical 90th percentile. 2 month(s), peak 0.684 unitless in Oct 2026 (threshold 0.328). Backtest skill vs climatology -9%: no better than climatology, read as a scenario not a prediction.
