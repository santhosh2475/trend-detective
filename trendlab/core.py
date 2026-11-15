"""Loading, deseasonalising and trend statistics.

Everything here works on a monthly pandas Series with a DatetimeIndex.
Tests implemented from scratch so the maths is inspectable:
  - Mann-Kendall (tie-corrected, optional Hamed-Rao autocorrelation correction)
  - Seasonal Kendall (Hirsch et al. 1982) + serial-correlation variance
    (Hirsch & Slack 1984) + seasonal Sen slope
  - OLS slope t-test with AR(1) effective sample size (Santer et al. 2008)
  - Trend t-test controlling for ENSO / IOD (OLS, AR(1)-adjusted errors)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# name -> (label, unit, plain-language definition, analyse on log1p scale?)
VARIABLES = {
    "precip_mm": ("Rainfall", "mm/month", "Total rain that fell in the month (GPM IMERG satellite).", False),
    "sm_0_10_mm": ("Topsoil moisture (0-10 cm)", "mm", "Water held in the top 10 cm of soil. Reacts within days to rain.", False),
    "sm_10_40_mm": ("Deeper soil moisture (10-40 cm)", "mm", "Water held 10-40 cm down. Dries slowly; a proxy for how burnable peat is.", False),
    "rootzone_mm": ("Root-zone soil moisture", "mm", "Water available to plant roots.", False),
    "sm_0_10_vwc": ("Topsoil moisture (fraction)", "m3/m3", "Topsoil water as a fraction of soil volume.", False),
    "sm_10_40_vwc": ("Deeper soil moisture (fraction)", "m3/m3", "Deeper soil water as a fraction of soil volume.", False),
    "et_mm": ("Evapotranspiration", "mm/month", "Water returned to the air by evaporation and plants.", False),
    "tair_c": ("Air temperature", "deg C", "Near-surface air temperature.", False),
    "vpd_kpa": ("Vapour pressure deficit", "kPa", "How 'thirsty' the air is. Higher VPD dries fuel faster.", False),
    "aod": ("Aerosol optical depth", "unitless", "How much haze/smoke blocks sunlight (550 nm). Above ~0.5 is visibly hazy.", False),
    "fire_count": ("Active fire detections", "count/month", "MODIS satellite fire detections in the region.", True),
    "nino34": ("Nino 3.4 index", "deg C anomaly", "Pacific sea-surface temperature anomaly. Above +0.5 = El Nino (drier Indonesia).", False),
    "dmi": ("Indian Ocean Dipole", "deg C anomaly", "Positive values mean a drier Indonesia.", False),
}
DRIVERS = ["nino34", "dmi"]


def meta(col: str):
    return VARIABLES.get(col, (col, "", "User-supplied variable.", False))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_indices(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "climate_indices.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df[[c for c in DRIVERS if c in df.columns]]


def load_csv(src, indices: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Read a monthly CSV, validate it and put it on a complete monthly calendar.

    Returns (frame, quality_report). The date column is `date` or the first column.
    """
    raw = pd.read_csv(src)
    date_col = "date" if "date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    bad_dates = int(raw[date_col].isna().sum())
    raw = raw.dropna(subset=[date_col])
    raw[date_col] = raw[date_col].dt.to_period("M").dt.to_timestamp()
    dups = int(raw[date_col].duplicated().sum())
    df = raw.groupby(date_col).mean(numeric_only=True)
    num = df.select_dtypes("number")
    full = pd.date_range(num.index.min(), num.index.max(), freq="MS")
    missing_rows = [d.strftime("%Y-%m") for d in full.difference(num.index)]
    num = num.reindex(full)
    num.index.name = "date"
    # drivers: fill gaps (and add if absent) from the bundled index file
    if indices is not None:
        for c in DRIVERS:
            if c in indices.columns:
                fill = indices[c].reindex(num.index)
                num[c] = num[c].fillna(fill) if c in num.columns else fill
    report = {
        "start": full[0].strftime("%Y-%m"),
        "end": full[-1].strftime("%Y-%m"),
        "n_months": len(full),
        "n_years": round(len(full) / 12, 1),
        "unparseable_dates": bad_dates,
        "duplicate_months_averaged": dups,
        "missing_rows_inserted": missing_rows,
        "missing_per_column": {c: int(num[c].isna().sum()) for c in num.columns},
        "warnings": [],
    }
    if len(full) < 120:
        report["warnings"].append("Under 10 years of data: trend tests have little power.")
    for c, n in report["missing_per_column"].items():
        if n / len(full) > 0.2:
            report["warnings"].append(f"{c}: {n} of {len(full)} months missing (>20%).")
    return num, report


def analysis_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DRIVERS and df[c].notna().sum() >= 36 and df[c].nunique() > 5]


def transform(s: pd.Series, col: str) -> pd.Series:
    return np.log1p(s) if meta(col)[3] else s


def deseasonalise(s: pd.Series) -> pd.Series:
    """Anomaly = value minus the mean of that calendar month."""
    return s - s.groupby(s.index.month).transform("mean")


# ----------------------------------------------------------------------------
# Mann-Kendall
# ----------------------------------------------------------------------------
def _tie_term(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    t = counts[counts > 1]
    return float(np.sum(t * (t - 1) * (2 * t + 5)))


def _z_p(S: float, var: float) -> tuple[float, float]:
    if var <= 0:
        return 0.0, 1.0
    z = (S - np.sign(S)) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def mann_kendall(s: pd.Series, autocorr_correction: bool = True) -> dict:
    x = s.dropna().to_numpy(float)
    n = len(x)
    if n < 8:
        return {"n": n, "S": np.nan, "z": np.nan, "p": np.nan, "tau": np.nan}
    d = x[None, :] - x[:, None]
    iu = np.triu_indices(n, 1)
    S = float(np.sign(d[iu]).sum())
    var = (n * (n - 1) * (2 * n + 5) - _tie_term(x)) / 18
    factor = 1.0
    if autocorr_correction:
        # Hamed & Rao (1998): inflate the variance using significant
        # autocorrelations of the ranks of the detrended series.
        t = np.arange(n)
        slope = np.median(d[iu] / (t[None, :] - t[:, None])[iu])
        r = stats.rankdata(x - slope * t)
        r = r - r.mean()
        denom = np.sum(r * r)
        lim = 1.96 / np.sqrt(n)
        acc = 0.0
        for k in range(1, min(n - 3, 24) + 1):
            rho = np.sum(r[:-k] * r[k:]) / denom
            if abs(rho) > lim:
                acc += (n - k) * (n - k - 1) * (n - k - 2) * rho
        factor = max(1.0, 1 + 2 / (n * (n - 1) * (n - 2)) * acc)
    z, p = _z_p(S, var * factor)
    return {"n": n, "S": S, "z": z, "p": p, "tau": S / (n * (n - 1) / 2), "var_inflation": factor}


# ----------------------------------------------------------------------------
# Seasonal Kendall
# ----------------------------------------------------------------------------
def _year_month_matrix(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    years = np.arange(s.index.year.min(), s.index.year.max() + 1)
    M = np.full((len(years), 12), np.nan)
    M[s.index.year - years[0], s.index.month - 1] = s.to_numpy(float)
    return years, M


def seasonal_kendall(s: pd.Series) -> dict:
    """Seasonal Kendall test on raw (not deseasonalised) monthly data.

    Each calendar month is only compared with the same month in other years, so
    the seasonal cycle cannot create or hide a trend. `p` assumes months are
    independent; `p_adj` adds the between-month covariance (Hirsch & Slack 1984)
    and is the one to quote when the series is autocorrelated.
    """
    years, M = _year_month_matrix(s)
    ny = len(years)
    S_tot, var_tot, slopes = 0.0, 0.0, []
    sgn, ranks, ns, per_month = [], [], [], []
    for g in range(12):
        x = M[:, g]
        ok = ~np.isnan(x)
        n = int(ok.sum())
        d = x[None, :] - x[:, None]          # d[i, j] = x[j] - x[i]
        sg = np.nan_to_num(np.sign(d))        # pairs with a missing value -> 0
        sgn.append(sg)
        ns.append(n)
        R = np.where(ok, (n + 1 - sg.sum(axis=1)) / 2, (n + 1) / 2)
        ranks.append(R)
        iu = np.triu_indices(ny, 1)
        S_g = float(sg[iu].sum())
        var_g = (n * (n - 1) * (2 * n + 5) - _tie_term(x[ok])) / 18 if n > 1 else 0.0
        S_tot += S_g
        var_tot += var_g
        yd = (years[None, :] - years[:, None])[iu]
        sl = d[iu] / yd
        slopes.append(sl[~np.isnan(sl)])
        zg, pg = _z_p(S_g, var_g)
        per_month.append({"month": g + 1, "S": S_g, "z": zg, "p": pg,
                          "slope_per_decade": float(np.median(slopes[-1]) * 10) if len(slopes[-1]) else np.nan})
    cov = 0.0
    iu = np.triu_indices(ny, 1)
    for g in range(12):
        for h in range(g + 1, 12):
            K = float((sgn[g][iu] * sgn[h][iu]).sum())
            cov += (K + 4 * float(np.sum(ranks[g] * ranks[h])) - ny * (ns[g] + 1) * (ns[h] + 1)) / 3
    var_adj = max(var_tot + 2 * cov, 1e-9)
    z, p = _z_p(S_tot, var_tot)
    z_adj, p_adj = _z_p(S_tot, var_adj)
    all_slopes = np.concatenate(slopes) if slopes else np.array([np.nan])
    return {"S": S_tot, "z": z, "p": p, "z_adj": z_adj, "p_adj": p_adj,
            "slope_per_decade": float(np.median(all_slopes) * 10), "per_month": per_month}


# ----------------------------------------------------------------------------
# Slopes and t-tests
# ----------------------------------------------------------------------------
def _years(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.year + (index.month - 0.5) / 12).to_numpy(float)


def sen_slope(anom: pd.Series) -> dict:
    a = anom.dropna()
    if len(a) < 8:
        return {"slope_per_decade": np.nan, "lo": np.nan, "hi": np.nan, "intercept": np.nan}
    t = _years(a.index)
    slope, intercept, lo, hi = stats.theilslopes(a.to_numpy(), t, 0.95)
    return {"slope_per_decade": slope * 10, "lo": lo * 10, "hi": hi * 10, "intercept": intercept}


def linear_trend(anom: pd.Series) -> dict:
    """OLS slope with a t-test whose degrees of freedom are reduced for
    lag-1 autocorrelation of the residuals."""
    a = anom.dropna()
    n = len(a)
    if n < 8:
        return {"slope_per_decade": np.nan, "p": np.nan, "lo": np.nan, "hi": np.nan, "t": np.nan, "n_eff": n, "r1": np.nan}
    t = _years(a.index)
    y = a.to_numpy()
    b, c = np.polyfit(t, y, 1)
    res = pd.Series(y - (b * t + c), index=a.index).asfreq("MS")
    r1 = float(res.autocorr(1)) if res.notna().sum() > 3 else 0.0
    r1 = 0.0 if np.isnan(r1) else min(max(r1, 0.0), 0.95)
    n_eff = max(n * (1 - r1) / (1 + r1), 3.0)
    se = np.sqrt(np.sum(res.dropna() ** 2) / (n_eff - 2) / np.sum((t - t.mean()) ** 2))
    tstat = b / se
    p = 2 * stats.t.sf(abs(tstat), n_eff - 2)
    half = stats.t.ppf(0.975, n_eff - 2) * se
    return {"slope_per_decade": b * 10, "lo": (b - half) * 10, "hi": (b + half) * 10,
            "t": float(tstat), "p": float(p), "n_eff": float(n_eff), "r1": r1}


def best_lag(anom: pd.Series, driver: pd.Series, max_lag: int = 6) -> tuple[int, float]:
    """Lag (driver leads by k months) with the strongest correlation."""
    best = (0, 0.0)
    for k in range(max_lag + 1):
        r = anom.corr(driver.shift(k))
        if not np.isnan(r) and abs(r) > abs(best[1]):
            best = (k, float(r))
    return best


def driver_adjusted_trend(anom: pd.Series, drivers: pd.DataFrame) -> dict:
    """t-test of the time trend after controlling for ENSO / IOD:
    anomaly ~ time + nino34(lag) + dmi(lag), errors widened for AR(1) residuals."""
    import statsmodels.api as sm

    X = pd.DataFrame({"time": _years(anom.index)}, index=anom.index)
    lags = {}
    for c in drivers.columns:
        k, r = best_lag(anom, drivers[c])
        lags[c] = {"lag": k, "r": r}
        X[c] = drivers[c].shift(k)
    d = pd.concat([anom.rename("y"), X], axis=1).dropna()
    if len(d) < 36:
        return {"slope_per_decade": np.nan, "p": np.nan, "lags": lags}
    fit = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit()
    # autocorrelated residuals -> fewer independent months; widen the errors accordingly
    r1 = fit.resid.asfreq("MS").autocorr(1)
    r1 = 0.0 if np.isnan(r1) else min(max(float(r1), 0.0), 0.95)
    infl = np.sqrt((1 + r1) / (1 - r1))
    dof = max(len(d) / infl ** 2 - len(fit.params), 3.0)
    se = fit.bse * infl
    pvals = pd.Series(2 * stats.t.sf(np.abs(fit.params / se), dof), index=fit.params.index)
    half = stats.t.ppf(0.975, dof) * se["time"]
    ci = (fit.params["time"] - half, fit.params["time"] + half)
    resid_plus_trend = d["y"] - sum(fit.params[c] * d[c] for c in drivers.columns)
    return {"slope_per_decade": float(fit.params["time"] * 10), "p": float(pvals["time"]),
            "lo": float(ci[0] * 10), "hi": float(ci[1] * 10), "r2": float(fit.rsquared), "lags": lags,
            "coef": {c: float(fit.params[c]) for c in drivers.columns},
            "coef_p": {c: float(pvals[c]) for c in drivers.columns},
            "adjusted_series": resid_plus_trend}


def verdict(p: float, slope: float, alpha: float = 0.05) -> str:
    if np.isnan(p):
        return "not enough data"
    direction = "increasing" if slope > 0 else "decreasing"
    if p < alpha:
        return f"significant {direction}"
    if p < 0.10:
        return f"weak {direction} (not significant)"
    return "no significant trend"


def analyse_variable(df: pd.DataFrame, col: str, alpha: float = 0.05) -> dict:
    """Full trend battery for one column. Returns numbers plus the series used."""
    y = transform(df[col], col)
    anom = deseasonalise(y)
    drivers = df[[c for c in DRIVERS if c in df.columns]]
    mk = mann_kendall(anom)
    sk = seasonal_kendall(y.dropna())
    sen = sen_slope(anom)
    ols = linear_trend(anom)
    adj = driver_adjusted_trend(anom, drivers) if len(drivers.columns) else {"slope_per_decade": np.nan, "p": np.nan, "lags": {}}
    mean = float(y.mean())
    span = (anom.dropna().index[-1] - anom.dropna().index[0]).days / 3652.5
    return {
        "variable": col, "label": meta(col)[0], "unit": ("log1p " if meta(col)[3] else "") + meta(col)[1],
        "n": int(anom.notna().sum()), "mean": mean, "anom_std": float(anom.std()),
        "sen_slope_per_decade": sen["slope_per_decade"], "sen_lo": sen["lo"], "sen_hi": sen["hi"],
        "total_change": sen["slope_per_decade"] * span,
        "pct_per_decade": 100 * sen["slope_per_decade"] / abs(mean) if mean else np.nan,
        "mk_z": mk["z"], "mk_p": mk["p"], "mk_tau": mk["tau"],
        "sk_z": sk["z_adj"], "sk_p": sk["p_adj"], "sk_p_indep": sk["p"], "sk_slope_per_decade": sk["slope_per_decade"],
        "ols_slope_per_decade": ols["slope_per_decade"], "ols_lo": ols["lo"], "ols_hi": ols["hi"],
        "ols_t": ols["t"], "ols_p": ols["p"], "ols_n_eff": ols["n_eff"], "ar1": ols["r1"],
        "adj_slope_per_decade": adj["slope_per_decade"], "adj_p": adj["p"],
        "adj_r2": adj.get("r2", np.nan), "driver_lags": adj.get("lags", {}),
        "verdict": verdict(sk["p_adj"], sk["slope_per_decade"], alpha),
        "verdict_adjusted": verdict(adj["p"], adj["slope_per_decade"], alpha),
        "_anom": anom, "_sen_intercept": sen["intercept"], "_per_month": sk["per_month"],
        "_adjusted_series": adj.get("adjusted_series"),
    }


def trend_table(df: pd.DataFrame, cols: list[str] | None = None, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for c in cols or analysis_columns(df):
        r = analyse_variable(df, c, alpha)
        rows.append({k: v for k, v in r.items() if not k.startswith("_") and k != "driver_lags"})
    return pd.DataFrame(rows)
