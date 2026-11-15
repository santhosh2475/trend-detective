# Indonesia peat-fire dataset: processed data

Monthly hydrology, fire-weather, fire and haze data for 2001–2025, built for our hackathon
case study (Indonesian forest/peat fires → haze over Singapore). Everything here is already cleaned,
in readable units, and averaged per region. Start with the CSVs.

## Files

| File | What it is |
|---|---|
| `central_kalimantan.csv` | Monthly time series, averaged over Central Kalimantan (peat-fire hotspot) |
| `riau_south_sumatra.csv` | Same, averaged over Riau / Jambi / South Sumatra (peat-fire hotspot) |
| `singapore_receptor.csv` | Same, averaged over Singapore and southern Johor (where haze arrives) |
| `climate_indices.csv` | Niño 3.4 and DMI on their own (full length) |
| `gldas_grid.nc` | Gridded GLDAS variables, 0.25°, for per-pixel trend maps |
| `imerg_grid.nc` | Gridded rainfall, 0.1° |
| `aod_grid.nc` | Gridded aerosol optical depth (MERRA-2), 0.5° × 0.625° |

Grids cover the whole study area: **95°E–119.5°E, 6°S–7.5°N** (all of Sumatra and Borneo plus the Malay Peninsula).

## Regions (bounding boxes used for the CSVs)

| Region | Latitude | Longitude |
|---|---|---|
| central_kalimantan | 3.5°S – 1.5°S | 112.0°E – 115.0°E |
| riau_south_sumatra | 4.0°S – 2.5°N | 100.5°E – 106.0°E |
| singapore_receptor | 1.0°N – 2.0°N | 103.0°E – 105.0°E |

Regional values are area-weighted means (cos latitude) over land cells, except AOD, which also includes
sea cells because haze doesn't stop at the coast.

## Columns in the regional CSVs

One row per month; `date` is the first day of the month.

| Column | Meaning | Units | Source | Role in the fire story |
|---|---|---|---|---|
| `precip_mm` | Total rainfall that month | mm/month | GPM IMERG Final V07 | Driver |
| `sm_0_10_mm` | Topsoil moisture (0–10 cm) | mm of water | GLDAS-2.1 Noah | Water state (fast) |
| `sm_10_40_mm` | Deeper soil moisture (10–40 cm) | mm of water | GLDAS-2.1 Noah | Water state (slow; peat burns underground) |
| `rootzone_mm` | Root-zone soil moisture | mm of water | GLDAS-2.1 Noah | Water state |
| `sm_0_10_vwc` | Topsoil moisture as volume fraction | m³/m³ (0–1) | derived | Same as above, easier to read |
| `sm_10_40_vwc` | Deeper soil moisture as volume fraction | m³/m³ (0–1) | derived | Same as above |
| `et_mm` | Evapotranspiration (water returned to the air) | mm/month | GLDAS-2.1 Noah | Water leaving the land |
| `tair_c` | Near-surface air temperature | °C | GLDAS-2.1 Noah | Amplifier |
| `vpd_kpa` | Vapour pressure deficit: how "thirsty" the air is | kPa | derived from GLDAS T, humidity, pressure | Amplifier / fire weather |
| `aod` | Total aerosol optical depth at 550 nm (haze/smoke) | unitless | MERRA-2 (M2TMNXAER, TOTEXTTAU) | Impact |
| `nino34` | Niño 3.4 index (El Niño > ~+0.5, La Niña < ~−0.5) | °C anomaly | KNMI / NOAA | Climate driver |
| `dmi` | Indian Ocean Dipole index (positive = drier Indonesia) | °C anomaly | KNMI / NOAA | Climate driver |
| `fire_count` | MODIS active-fire detections in the region that month | count | NASA FIRMS (MODIS C6.1) | Outcome |

## Processing notes

- **Fire counts:** only vegetation fires (`type == 0`) with confidence ≥ 30% are counted. Fire detections are
  only counted for the two Indonesian regions; Singapore's fire count is ~0 by design.
- **VPD** is computed from monthly-mean temperature and humidity, so it slightly underestimates the true mean
  VPD. That's fine for trends and relative comparisons.
- **Missing data:** a few monthly files failed to download (GLDAS 5, IMERG ~8). Those months are blank (NaN)
  for the affected variables; 298 of 300 months have at least some data. Two months are absent as rows
  entirely, so **reindex to a full monthly calendar before any time-series work**:
  `df = df.reindex(pd.date_range("2001-01-01", "2025-12-01", freq="MS"))`.
- **Seasonal cycle not removed.** The data is raw monthly values. Subtract each calendar month's mean
  (anomalies) or use seasonal methods before trend testing, otherwise the wet/dry season dominates.
- **2026 not included.** IMERG Final arrives ~4 months late, so 2001–2025 is the clean trend period.
  The 2026 fire season can be added later as a separate "current event" overlay.

## Caveats worth knowing

- GLDAS is a land-surface model at 0.25° using mineral-soil parameters, so it shows **relative** drying well
  but not the actual peat water table.
- Many Indonesian fires are started by people (land clearing). The hydrology explains when land *can* burn
  badly, not where ignitions happen.
- Region boxes are approximate. A peat-extent map could refine them later.
- El Niño years (2015, 2019, 2023) produce huge spikes. Account for `nino34`/`dmi` so they aren't mistaken
  for a long-term trend.

## Coming later

- **NDVI / EVI** (vegetation greenness, MODIS MOD13A3, 1 km): requested from AppEEARS, to be added as extra
  columns and a grid file.

## Dataset citations

- GLDAS-2.1 Noah monthly 0.25°: Beaudoing, H. and M. Rodell, NASA/GSFC/HSL, GES DISC. doi:10.5067/SXAVCZFAQLNO
- GPM IMERG Final Precipitation L3 1 month 0.1° V07: Huffman, G.J. et al., GES DISC. doi:10.5067/GPM/IMERG/3B-MONTH/07
- MERRA-2 tavgM_2d_aer_Nx (M2TMNXAER), GMAO, GES DISC
- MODIS Collection 6.1 active fires, NASA FIRMS
- Niño 3.4 and DMI indices: KNMI Climate Explorer / NOAA PSL
