"""Per-pixel seasonal Kendall trend maps and region extraction from NetCDF grids."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from .core import DATA_DIR

REGIONS = {  # name: (lat_min, lat_max, lon_min, lon_max)
    "central_kalimantan": (-3.5, -1.5, 112.0, 115.0),
    "riau_south_sumatra": (-4.0, 2.5, 100.5, 106.0),
    "singapore_receptor": (1.0, 2.0, 103.0, 105.0),
}


def find_grids(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Map variable name -> NetCDF file, for every *.nc in the data folder."""
    out = {}
    for f in sorted(data_dir.glob("*.nc")):
        try:
            with xr.open_dataset(f) as ds:
                for v in ds.data_vars:
                    if {"time", "lat", "lon"} <= set(ds[v].dims):
                        out[v] = f
        except Exception:
            continue
    return out


def load_var(path: Path, var: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        return ds[var].transpose("time", "lat", "lon").load()


def pixel_trends(da: xr.DataArray, min_years: int = 10) -> dict:
    """Seasonal Kendall test + seasonal Sen slope for every pixel (vectorised).

    Returns 2-D arrays: slope per decade, z, p, and BH-FDR adjusted significance.
    """
    t = pd.DatetimeIndex(da["time"].values)
    years = np.arange(t.year.min(), t.year.max() + 1)
    ny, (nlat, nlon) = len(years), da.shape[1:]
    P = nlat * nlon
    cube = np.full((ny, 12, P), np.nan, dtype=np.float32)
    cube[t.year - years[0], t.month - 1] = da.values.reshape(len(t), P)
    iu, ju = np.triu_indices(ny, 1)
    yd = (years[ju] - years[iu]).astype(np.float32)
    S = np.zeros(P)
    var = np.zeros(P)
    slope = np.full(P, np.nan, dtype=np.float32)
    for g in range(12):
        n = np.sum(~np.isnan(cube[:, g]), axis=0).astype(float)
        var += n * (n - 1) * (2 * n + 5) / 18
    step = 6000
    for a in range(0, P, step):
        sl_all = []
        for g in range(12):
            x = cube[:, g, a:a + step]
            d = x[ju] - x[iu]
            S[a:a + step] += np.nansum(np.sign(d), axis=0)
            sl_all.append(d / yd[:, None])
        sl = np.concatenate(sl_all, axis=0)
        enough = np.sum(~np.isnan(sl), axis=0) > 0
        with np.errstate(all="ignore"):
            slope[a:a + step][enough] = np.nanmedian(sl[:, enough], axis=0) * 10
    with np.errstate(all="ignore"):
        z = np.where(var > 0, (S - np.sign(S)) / np.sqrt(var), np.nan)
    n_valid = np.sum(~np.isnan(cube), axis=(0, 1))
    z[n_valid < min_years * 12 * 0.7] = np.nan
    slope[np.isnan(z)] = np.nan
    p = 2 * stats.norm.sf(np.abs(z))
    # Benjamini-Hochberg across pixels (controls false discoveries over the map)
    ok = ~np.isnan(p)
    sig_fdr = np.zeros(P, bool)
    if ok.any():
        ps = np.sort(p[ok])
        m = len(ps)
        below = ps <= 0.05 * np.arange(1, m + 1) / m
        if below.any():
            sig_fdr = ok & (p <= ps[np.where(below)[0].max()])
    shp = (nlat, nlon)
    return {"lat": da["lat"].values, "lon": da["lon"].values,
            "slope": slope.reshape(shp), "z": z.reshape(shp), "p": p.reshape(shp),
            "sig": (p < 0.05).reshape(shp), "sig_fdr": sig_fdr.reshape(shp),
            }


def region_series(da: xr.DataArray, box: tuple[float, float, float, float]) -> pd.Series:
    """cos(lat)-weighted mean over a lat/lon box, as a monthly Series."""
    la0, la1, lo0, lo1 = box
    sub = da.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))
    if sub.size == 0:
        return pd.Series(dtype=float)
    w = np.cos(np.deg2rad(sub["lat"]))
    s = sub.weighted(w).mean(("lat", "lon")).to_series()
    s.index = pd.DatetimeIndex(s.index)
    return s.asfreq("MS")


def land_mask(data_dir: Path = DATA_DIR):
    """Land/sea mask from GLDAS (land-only model), for drawing coastlines."""
    grids = find_grids(data_dir)
    if "tair_c" not in grids:
        return None
    with xr.open_dataset(grids["tair_c"]) as ds:
        m = ds["tair_c"].isel(time=0).load()
    return m["lat"].values, m["lon"].values, (~np.isnan(m.values)).astype(float)
