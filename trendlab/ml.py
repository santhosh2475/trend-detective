"""Multivariate tools: Elliptic Envelope outliers, PCA, and forecasting."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .core import DRIVERS, best_lag, deseasonalise, meta, transform


def anomaly_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Deseasonalised, standardised anomalies (z-scores) for the chosen columns."""
    Z = pd.DataFrame({c: deseasonalise(transform(df[c], c)) for c in cols})
    return (Z - Z.mean()) / Z.std()


# ----------------------------------------------------------------------------
# Elliptic Envelope
# ----------------------------------------------------------------------------
def elliptic_outliers(df: pd.DataFrame, cols: list[str], contamination: float = 0.05) -> pd.DataFrame:
    """Flag months whose combination of anomalies is unusual.

    A robust covariance ellipse (Minimum Covariance Determinant) is fitted to the
    z-scored anomalies; months far outside it (large Mahalanobis distance) are
    outliers. Unlike a per-variable threshold this catches odd *combinations*,
    e.g. normal rain but very dry soil and high VPD.
    """
    from sklearn.covariance import EllipticEnvelope

    Z = anomaly_matrix(df, cols)
    Zc = Z.dropna()
    if len(Zc) < max(30, 5 * len(cols)):
        return pd.DataFrame()
    env = EllipticEnvelope(contamination=contamination, random_state=0).fit(Zc.to_numpy())
    out = Zc.copy()
    out["distance"] = np.sqrt(env.mahalanobis(Zc.to_numpy()))
    out["outlier"] = env.predict(Zc.to_numpy()) == -1
    out["threshold"] = float(out.loc[out["outlier"], "distance"].min()) if out["outlier"].any() else np.nan
    out["main_driver"] = Zc.abs().idxmax(axis=1)
    out["main_z"] = [Zc.loc[i, c] for i, c in out["main_driver"].items()]
    return out


# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------
def pca_modes(df: pd.DataFrame, cols: list[str], n: int = 3) -> dict:
    """Principal components of the anomaly matrix: which variables move together."""
    from sklearn.decomposition import PCA

    Zc = anomaly_matrix(df, cols).dropna()
    n = min(n, len(cols))
    if len(Zc) < 30 or n < 2:
        return {}
    pca = PCA(n_components=n).fit(Zc.to_numpy())
    names = [f"PC{i + 1}" for i in range(n)]
    scores = pd.DataFrame(pca.transform(Zc.to_numpy()), index=Zc.index, columns=names)
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=names)
    # orient PC1 so that positive = the fire/haze/dry side when identifiable
    for ref in ("fire_count", "aod", "vpd_kpa"):
        if ref in cols:
            if loadings.loc[ref, "PC1"] < 0:
                loadings["PC1"] *= -1
                scores["PC1"] *= -1
            break
    corr = {d: {p: float(scores[p].corr(df[d].reindex(scores.index))) for p in names}
            for d in DRIVERS if d in df.columns}
    return {"explained": pca.explained_variance_ratio_.tolist(), "loadings": loadings,
            "scores": scores, "driver_corr": corr}


# ----------------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------------
def _future_drivers(indices: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Driver values on `index`; months past the last known value decay to zero."""
    out = indices.reindex(index)
    for c in out.columns:
        last = out[c].last_valid_index()
        if last is None:
            out[c] = 0.0
            continue
        tail = out.index > last
        out.loc[tail, c] = out.loc[last, c] * 0.8 ** np.arange(1, tail.sum() + 1)
        out[c] = out[c].interpolate().bfill()
    return out


def _fit_predict(anom, X, steps, X_future):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(anom, exog=X, order=(2, 0, 0), trend="n",
                      enforce_stationarity=True).fit(disp=False, maxiter=200)
        fc = fit.get_forecast(steps=steps, exog=X_future)
    return fc.predicted_mean, fc.conf_int(alpha=0.2), fit


def _forecast_one(df: pd.DataFrame, col: str, indices: pd.DataFrame | None = None,
             horizon: int = 12, holdout: int = 36,
             seasonal_drivers: bool = True, include_trend: bool = True) -> dict:
    """Forecast = monthly climatology + linear trend + AR(2) on the anomaly + lagged ENSO/IOD.

    The seasonal cycle is handled by the climatology, persistence by the AR terms,
    and the climate drivers enter as regressors at their best-correlated lag.
    A hold-out backtest reports skill against the 'just use climatology' baseline.
    """
    y = transform(df[col], col).copy()
    last = y.last_valid_index()
    y = y.loc[:last]
    idx_all = pd.date_range(y.index[0], periods=len(y) + horizon, freq="MS")
    drv = indices if indices is not None else df[[c for c in DRIVERS if c in df.columns]]
    use = [c for c in DRIVERS if c in drv.columns and c != col]
    anom_full = deseasonalise(y)
    X_all = None
    lags = {}
    if use:
        D = _future_drivers(drv[use], pd.date_range(idx_all[0] - pd.DateOffset(months=6), idx_all[-1], freq="MS"))
        cols = {}
        for c in use:
            k, r = best_lag(anom_full, D[c].reindex(y.index))
            lags[c] = {"lag": k, "r": r}
            cols[c] = D[c].shift(k)
        X_all = pd.DataFrame(cols).reindex(idx_all).bfill()
        if seasonal_drivers:
            # drivers matter most in the months when the variable is most variable
            # (e.g. ENSO only moves fire counts in the dry season)
            w = anom_full.groupby(anom_full.index.month).std() / anom_full.std()
            X_all = X_all.mul(w.reindex(idx_all.month).to_numpy(), axis=0)

    if include_trend:
        tt = pd.Series((np.arange(len(idx_all)) - len(y) / 2) / 120.0, index=idx_all, name="trend")
        X_all = tt.to_frame() if X_all is None else X_all.assign(trend=tt)

    def run(train_end_pos: int, steps: int):
        tr = y.iloc[:train_end_pos]
        clim = tr.groupby(tr.index.month).mean()
        anom = (tr - clim.reindex(tr.index.month).to_numpy()).asfreq("MS")
        f_idx = idx_all[train_end_pos:train_end_pos + steps]
        Xtr = X_all.iloc[:train_end_pos] if X_all is not None else None
        Xf = X_all.iloc[train_end_pos:train_end_pos + steps] if X_all is not None else None
        mean, ci, fit = _fit_predict(anom, Xtr, steps, Xf)
        base = clim.reindex(f_idx.month).to_numpy()
        out = pd.DataFrame({"forecast": mean.to_numpy() + base, "lo": ci.iloc[:, 0].to_numpy() + base,
                            "hi": ci.iloc[:, 1].to_numpy() + base, "climatology": base}, index=f_idx)
        return out, fit

    # backtest
    bt, skill = None, {}
    if len(y) > holdout + 96:
        bt, _ = run(len(y) - holdout, holdout)
        bt["actual"] = y.iloc[-holdout:].to_numpy()
        ok = bt["actual"].notna()
        mae = float((bt["forecast"] - bt["actual"])[ok].abs().mean())
        mae_c = float((bt["climatology"] - bt["actual"])[ok].abs().mean())
        skill = {"mae": mae, "mae_climatology": mae_c, "skill": 1 - mae / mae_c if mae_c else np.nan,
                 "months": int(ok.sum())}
    fc, fit = run(len(y), horizon)
    hist = y
    if meta(col)[3]:  # back to counts
        for d in (fc, bt):
            if d is not None:
                d[d.columns] = np.expm1(d).clip(lower=0)
        hist = np.expm1(y)
    p90 = float(hist.quantile(0.9))
    coefs = {c: float(fit.params[c]) for c in use} if use else {}
    return {"forecast": fc, "backtest": bt, "skill": skill, "history": hist, "p90": p90,
            "driver_lags": lags, "driver_coef": coefs, "log_scale": meta(col)[3],
            "model": "climatology" + (" + trend" if include_trend else "") + " + AR(2)" + (" + " + "/".join(use) if use else "")}


def forecast(df: pd.DataFrame, col: str, indices: pd.DataFrame | None = None,
             horizon: int = 12, holdout: int = 36) -> dict:
    """Fit with and without a linear trend term and keep whichever backtests better.

    A trend term helps when the change is steady (fire counts here) and hurts when
    it is not (VPD rose then fell), so the choice is made per variable on the hold-out.
    """
    cands = [_forecast_one(df, col, indices, horizon, holdout, include_trend=t) for t in (False, True)]
    return max(cands, key=lambda f: f["skill"].get("skill", -np.inf))
