"""Forecast model comparison with rolling-origin validation, and forecasts from the winner.

Every model predicts month t using only what was known at month t - lead:
lagged anomalies of the target and of the other variables, lagged Nino 3.4 / DMI,
and the calendar month. No future driver values are used, so the scores are what a
forecast issued `lead` months ahead would really have achieved.

Validation (rolling origin): for each test year Y, fit on every month before
January Y and predict the 12 months of Y. The climatology and anomalies are
recomputed from the training months only, so nothing from the test year leaks in.
Each strong El Nino year (2015, 2019, 2023) is therefore predicted unseen.

    python -m trendlab.models data/central_kalimantan.csv --target fire_count
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .core import DRIVERS, meta, transform

def single_thread(fn):
    """Run model fits on one thread. On ~300 rows, thread start-up costs far more than it saves
    (gradient boosting is ~12x faster single-threaded here); results are identical."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        import sklearn.ensemble  # noqa: F401  load OpenMP first, or the limit misses it
        import sklearn.neural_network  # noqa: F401
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            return fn(*a, **kw)
    return wrapper


COVARIATES = ["precip_mm", "sm_10_40_mm", "vpd_kpa", "tair_c", "et_mm", "aod", "fire_count"]
DRY_SEASON = (7, 8, 9, 10)
TARGET_LAGS = 6


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
@dataclass
class Design:
    X: pd.DataFrame          # features, full index
    y: pd.Series             # target on the analysis scale (log1p for counts)
    base: pd.Series          # training climatology for each month (analysis scale)
    anom: pd.Series          # y - base
    counts: pd.Series        # raw target (for count models)


def _clim(s: pd.Series) -> pd.Series:
    return s.dropna().groupby(s.dropna().index.month).mean()


def _covariates(df: pd.DataFrame, target: str) -> list[str]:
    return [c for c in COVARIATES if c in df.columns and c != target and df[c].notna().sum() >= 60]


def design(df: pd.DataFrame, target: str, lead: int, test_start: pd.Timestamp) -> Design:
    """Lagged features for predicting `target` `lead` months ahead.

    Climatologies come from months before `test_start` only.
    """
    train = df.index < test_start
    y = transform(df[target], target)
    clim = _clim(y[train])
    base = pd.Series(clim.reindex(df.index.month).to_numpy(), index=df.index)
    anom = y - base
    # how much each calendar month varies: ENSO matters most in the volatile (dry) months
    a_tr = anom[train]
    w = (a_tr.groupby(a_tr.index.month).std() / a_tr.std()).reindex(df.index.month).to_numpy()

    X = {}
    for k in range(lead, lead + TARGET_LAGS):
        X[f"{target} anomaly, {k} mo earlier"] = anom.shift(k)
    for d in DRIVERS:
        if d in df.columns:
            for k in (lead, lead + 2, lead + 4):
                X[f"{d}, {k} mo earlier"] = df[d].shift(k)
            for k in (lead, lead + 2):
                X[f"{d}, {k} mo earlier x season"] = df[d].shift(k) * w
    for c in _covariates(df, target):
        yc = transform(df[c], c)
        ac = yc - _clim(yc[train]).reindex(df.index.month).to_numpy()
        X[f"{c} anomaly, {lead} mo earlier"] = ac.shift(lead)
    m = df.index.month.to_numpy()
    X["month (sin)"] = np.sin(2 * np.pi * m / 12)
    X["month (cos)"] = np.cos(2 * np.pi * m / 12)
    X["normal for the month"] = base
    return Design(pd.DataFrame(X, index=df.index), y, base, anom, df[target])


# ----------------------------------------------------------------------------
# Models. Each returns predictions on the analysis scale (log1p for counts).
# ----------------------------------------------------------------------------
@dataclass
class Spec:
    key: str
    name: str
    family: str          # baseline | linear | tree | neural | count
    plain: str           # one-line description for the UI
    explainable: str     # how easy it is to explain
    fit_predict: Callable
    counts_only: bool = False


def _climatology(D, tr, te, lead):
    return D.base[te].to_numpy()


def _last_year(D, tr, te, lead):
    if lead > 12:
        return D.base[te].to_numpy()
    v = D.y.shift(12)[te]
    return v.fillna(D.base[te]).to_numpy()


def _persistence(D, tr, te, lead):
    v = D.anom.shift(lead)[te].fillna(0.0)
    return (D.base[te] + v).to_numpy()


def _sk_pipeline(est, scale=True):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else []) + [est]
    return make_pipeline(*steps)


def _fit_anom(model, D, tr, te):
    ok = tr & D.anom.notna()
    model.fit(D.X[ok].to_numpy(), D.anom[ok].to_numpy())
    return D.base[te].to_numpy() + model.predict(D.X[te].to_numpy())


def _ridge(D, tr, te, lead):
    from sklearn.linear_model import RidgeCV

    return _fit_anom(_sk_pipeline(RidgeCV(alphas=np.logspace(-2, 3.5, 24))), D, tr, te)


def _boosting(D, tr, te, lead):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=10,
                                      l2_regularization=1.0, random_state=0)
    ok = tr & D.anom.notna()
    m.fit(D.X[ok], D.anom[ok])  # handles missing values natively
    return D.base[te].to_numpy() + m.predict(D.X[te])


def _boosting_poisson(D, tr, te, lead):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.04, max_depth=3,
                                      min_samples_leaf=10, l2_regularization=1.0, random_state=0)
    ok = tr & D.counts.notna()
    m.fit(D.X[ok], D.counts[ok].clip(lower=0))
    return np.log1p(m.predict(D.X[te]))


def _forest(D, tr, te, lead):
    from sklearn.ensemble import RandomForestRegressor

    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=0)
    return _fit_anom(_sk_pipeline(m, scale=False), D, tr, te)


def _mlp(D, tr, te, lead):
    from sklearn.neural_network import MLPRegressor

    preds = []
    for seed in (0, 1, 2):  # small ensemble: one net on 200 points is noisy
        m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=3.0, learning_rate_init=0.003,
                         max_iter=3000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds.append(_fit_anom(_sk_pipeline(m), D, tr, te))
    return np.mean(preds, axis=0)


def _negbin(D, tr, te, lead):
    """Negative binomial GLM on counts: month effects + lagged fire, drivers and dryness."""
    import statsmodels.api as sm

    keep = [c for c in D.X.columns
            if (" x season" in c) or c.endswith(f", {lead} mo earlier") and "anomaly" in c]
    Z = D.X[keep].copy()
    mu, sd = Z[tr].mean(), Z[tr].std().replace(0, 1)
    Z = ((Z - mu) / sd).fillna(0.0)
    months = pd.get_dummies(pd.Categorical(D.X.index.month, categories=range(1, 13)), prefix="m",
                            drop_first=True, dtype=float)
    months.index = D.X.index
    Z = sm.add_constant(pd.concat([Z, months], axis=1))
    ok = tr & D.counts.notna()
    yc = D.counts[ok].clip(lower=0).round()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois = sm.GLM(yc, Z[ok], family=sm.families.Poisson()).fit()
        m = pois.mu
        alpha = float(np.clip(np.mean(((yc - m) ** 2 - yc) / np.maximum(m, 1e-6) ** 2), 0.05, 5.0))
        fit = sm.GLM(yc, Z[ok], family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    return np.log1p(np.clip(fit.predict(Z[te]).to_numpy(), 0, None))


MODELS = [
    Spec("climatology", "Seasonal average", "baseline",
         "Predicts the normal value for that calendar month. The bar every model must clear.", "trivial", _climatology),
    Spec("last_year", "Same month last year", "baseline",
         "Predicts whatever happened in the same month a year earlier.", "trivial", _last_year),
    Spec("persistence", "Persistence", "baseline",
         "Seasonal average plus the most recent known anomaly, carried forward.", "trivial", _persistence),
    Spec("ridge", "Linear regression (ridge)", "linear",
         "A weighted sum of recent anomalies, El Nino / IOD and dryness, with penalised weights.", "high", _ridge),
    Spec("negbin", "Negative binomial regression", "count",
         "A regression built for skewed counts: month effects plus lagged fire, drivers and dryness.", "high", _negbin,
         counts_only=True),
    Spec("boosting", "Gradient boosting", "tree",
         "Hundreds of small decision trees, each fixing the last one's errors. Finds thresholds and interactions.",
         "medium", _boosting),
    Spec("boosting_poisson", "Gradient boosting (Poisson)", "tree",
         "Gradient boosting with a loss designed for counts.", "medium", _boosting_poisson, counts_only=True),
    Spec("forest", "Random forest", "tree",
         "Averages many decision trees grown on random subsets of the data.", "medium", _forest),
    Spec("mlp", "Neural network (MLP)", "neural",
         "A small feed-forward neural network (3-seed ensemble). Stands in for deep learning; "
         "an LSTM needs far more than 300 months to train.", "low", _mlp),
]
BY_KEY = {m.key: m for m in MODELS}


def models_for(col: str) -> list[Spec]:
    return [m for m in MODELS if not m.counts_only or meta(col)[3]]


# ----------------------------------------------------------------------------
# Rolling-origin leaderboard
# ----------------------------------------------------------------------------
def _back(v, col):
    v = np.asarray(v, dtype=float)
    return np.clip(np.expm1(v), 0, None) if meta(col)[3] else v


def predict(spec: Spec, D: Design, tr: pd.Series, te: pd.Series, lead: int) -> np.ndarray:
    """Run one model and keep its anomaly inside the range seen in training.

    Without this, a linear model on log counts can overshoot an El Nino year and
    the back-transform turns a modest log error into a wildly wrong count.
    """
    v = np.asarray(spec.fit_predict(D, tr, te, lead), dtype=float)
    a_tr = D.anom[tr].dropna()
    base = D.base[te].to_numpy()
    return base + np.clip(v - base, a_tr.min(), a_tr.max())


@single_thread
def leaderboard(df: pd.DataFrame, target: str, lead: int = 1, min_train_years: int = 10,
                keys: list[str] | None = None, progress: Callable | None = None) -> dict:
    """Score every model by predicting each year from data before it."""
    y_raw = df[target]
    first = y_raw.first_valid_index()
    last = y_raw.last_valid_index()
    years = list(range(first.year + min_train_years, last.year + 1))
    specs = [m for m in models_for(target) if keys is None or m.key in keys]
    rows = []
    for i, Y in enumerate(years):
        start = pd.Timestamp(f"{Y}-01-01")
        D = design(df, target, lead, start)
        tr = pd.Series(D.X.index < start, index=D.X.index)
        te = pd.Series((D.X.index >= start) & (D.X.index.year == Y), index=D.X.index) & y_raw.notna()
        if te.sum() == 0:
            continue
        fold = pd.DataFrame({"actual": y_raw[te].to_numpy()}, index=D.X.index[te])
        fold["year"] = Y
        for m in specs:
            try:
                fold[m.key] = _back(predict(m, D, tr, te, lead), target)
            except Exception:
                fold[m.key] = np.nan
        rows.append(fold)
        if progress:
            progress((i + 1) / len(years), Y)
    preds = pd.concat(rows)
    return {"target": target, "lead": lead, "years": years, "preds": preds,
            "table": score(preds, target, specs), "specs": [m.key for m in specs]}


def score(preds: pd.DataFrame, target: str, specs: list[Spec] | None = None) -> pd.DataFrame:
    specs = specs or [BY_KEY[c] for c in preds.columns if c in BY_KEY]
    a = preds["actual"]
    dry = preds.index.month.isin(DRY_SEASON)
    err_c = (preds["climatology"] - a).abs()
    by_year_c = err_c.groupby(preds["year"]).mean()
    out = []
    for m in specs:
        e = (preds[m.key] - a).abs()
        if e.isna().all():
            continue
        by_year = e.groupby(preds["year"]).mean()
        out.append({
            "key": m.key, "Model": m.name, "Type": m.family,
            "MAE": e.mean(), "RMSE": float(np.sqrt(((preds[m.key] - a) ** 2).mean())),
            "Skill vs seasonal average": 1 - e.mean() / err_c.mean(),
            "Dry-season skill": 1 - e[dry].mean() / err_c[dry].mean() if dry.any() else np.nan,
            "Years beating seasonal average": int((by_year < by_year_c - 1e-12).sum()),
            "Years tested": int(by_year.size),
            "Explainability": m.explainable, "How it works": m.plain,
        })
    return pd.DataFrame(out).sort_values("Skill vs seasonal average", ascending=False).reset_index(drop=True)


def recommend(table: pd.DataFrame) -> dict:
    """Pick the model to trust, preferring simple models when the gain is marginal."""
    t = table.set_index("key")
    best = t["Skill vs seasonal average"].idxmax()
    sk = float(t.loc[best, "Skill vs seasonal average"])
    if sk < 0.05:
        best_txt = ("none of them beat it at all" if best == "climatology" else
                    f"the closest was {t.loc[best, 'Model']} at {sk:+.0%}")
        return {"key": "climatology", "skill": float(t.loc["climatology", "Skill vs seasonal average"]),
                "reason": f"No model beats the seasonal average by more than 5% at this lead; {best_txt}. "
                          "For this variable and lead the honest forecast is the normal seasonal cycle."}
    # within 2 points of the best, take the most explainable
    rank = {"trivial": 0, "high": 1, "medium": 2, "low": 3}
    close = t[t["Skill vs seasonal average"] >= sk - 0.02].copy()
    close["r"] = close["Explainability"].map(rank)
    pick = close.sort_values(["r", "Skill vs seasonal average"], ascending=[True, False]).index[0]
    psk = float(t.loc[pick, "Skill vs seasonal average"])
    yrs = f"{int(t.loc[pick, 'Years beating seasonal average'])} of {int(t.loc[pick, 'Years tested'])} test years"
    if pick == best:
        reason = f"{t.loc[pick, 'Model']} had the lowest error ({psk:+.0%} vs the seasonal average) and won in {yrs}."
    else:
        reason = (f"{t.loc[pick, 'Model']} ({psk:+.0%}) is within 2 points of the top scorer, "
                  f"{t.loc[best, 'Model']} ({sk:+.0%}), and is easier to explain. It won in {yrs}.")
    return {"key": pick, "skill": psk, "reason": reason, "top": best}


# ----------------------------------------------------------------------------
# Forecast with a chosen model (one direct model per lead time)
# ----------------------------------------------------------------------------
@single_thread
def forecast(df: pd.DataFrame, target: str, key: str, horizon: int = 12, error_years: int = 8,
             progress: Callable | None = None) -> dict:
    """Direct multi-step forecast: a separate fit for each lead 1..horizon.

    The 80% band comes from this model's own rolling-origin errors at each lead
    over the last `error_years` years, so it is as wide as the model has earned.
    """
    spec = BY_KEY[key]
    last = df[target].last_valid_index()
    idx = pd.date_range(df.index[0], last + pd.DateOffset(months=horizon), freq="MS")
    ext = df.reindex(idx)
    future = idx[idx > last]
    err_leads = sorted({1, 2, 3, 6, 9, 12, horizon} & set(range(1, horizon + 1)))
    q = {}
    years = list(range(last.year - error_years + 1, last.year + 1))
    total = len(err_leads) * len(years) + horizon
    done = 0
    for h in err_leads:
        res = []
        for Y in years:
            start = pd.Timestamp(f"{Y}-01-01")
            D = design(ext, target, h, start)
            tr = pd.Series(D.X.index < start, index=idx)
            te = pd.Series((D.X.index.year == Y) & (D.X.index <= last), index=idx) & D.y.notna()
            if te.sum():
                try:
                    res.append(D.y[te].to_numpy() - predict(spec, D, tr, te, h))
                except Exception:
                    pass
            done += 1
            if progress:
                progress(done / total)
        r = np.concatenate(res) if res else np.array([0.0])
        q[h] = (float(np.nanquantile(r, 0.1)), float(np.nanquantile(r, 0.9)))
    qlo = np.interp(range(1, horizon + 1), err_leads, [q[h][0] for h in err_leads])
    qhi = np.interp(range(1, horizon + 1), err_leads, [q[h][1] for h in err_leads])

    rows = []
    for h in range(1, horizon + 1):
        start = last + pd.DateOffset(months=1)
        D = design(ext, target, h, start)
        tr = pd.Series(D.X.index <= last, index=idx)
        te = pd.Series(D.X.index == last + pd.DateOffset(months=h), index=idx)
        v = float(predict(spec, D, tr, te, h)[0])
        rows.append({"date": future[h - 1], "forecast": v, "lo": v + qlo[h - 1], "hi": v + qhi[h - 1],
                     "climatology": float(D.base[te].iloc[0])})
        done += 1
        if progress:
            progress(min(done / total, 1.0))
    fc = pd.DataFrame(rows).set_index("date")
    fc = fc.apply(lambda c: _back(c.to_numpy(), target)).set_axis(fc.index)
    hist = df[target].loc[:last]
    return {"forecast": fc, "history": hist, "model": spec.name, "key": key,
            "p90": float(hist.quantile(0.9)), "error_years": years}


@single_thread
def feature_importance(df: pd.DataFrame, target: str, lead: int = 1, top: int = 10) -> pd.Series:
    """Permutation importance of the gradient-boosting model, fitted on all data.

    Shows which inputs the model leans on, in plain names, for the advisor page.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    last = df[target].last_valid_index()
    D = design(df, target, lead, last + pd.DateOffset(months=1))
    ok = D.anom.notna()
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_depth=3, min_samples_leaf=10,
                                      l2_regularization=1.0, random_state=0).fit(D.X[ok], D.anom[ok])
    pi = permutation_importance(m, D.X[ok], D.anom[ok], n_repeats=8, random_state=0)
    return pd.Series(pi.importances_mean, index=D.X.columns).sort_values(ascending=False).head(top)


def cached_leaderboard(df: pd.DataFrame, target: str, lead: int, cache_dir: Path | None = None,
                       progress: Callable | None = None) -> dict:
    """leaderboard(), with the per-year predictions saved to `cache_dir` and reused."""
    f = cache_dir / f"lead{lead}_{target}.csv" if cache_dir else None
    if f is not None and f.exists():
        preds = pd.read_csv(f, index_col=0, parse_dates=True)
        keys = [m.key for m in models_for(target) if m.key in preds.columns]
        return {"target": target, "lead": lead, "years": sorted(preds["year"].unique().tolist()), "preds": preds,
                "table": score(preds, target, [BY_KEY[k] for k in keys]), "specs": keys}
    lb = leaderboard(df, target, lead, progress=progress)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        lb["preds"].to_csv(f)
    return lb


def cache_dir_for(results: Path, name: str, raw: bytes) -> Path:
    import hashlib

    return Path(results) / name / "models" / hashlib.sha1(raw).hexdigest()[:10]


def main():
    import argparse
    import time

    from .core import analysis_columns, load_csv, load_indices

    ap = argparse.ArgumentParser(description="Compare forecast models with rolling-origin validation.")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--target", default=None, help="one column; default = every analysable column")
    ap.add_argument("--lead", type=int, nargs="+", default=[1])
    ap.add_argument("--save", default=None, help="results folder to cache predictions in (e.g. results)")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    for path in a.csv:
        raw = Path(path).read_bytes()
        df, _ = load_csv(path, load_indices())
        targets = [a.target] if a.target else analysis_columns(df)
        cdir = cache_dir_for(Path(a.save), Path(path).stem, raw) if a.save else None
        for lead in a.lead:
            for t in targets:
                t0 = time.time()
                lb = cached_leaderboard(df, t, lead, cdir)
                rec = recommend(lb["table"])
                print(f"\n== {Path(path).stem} / {t} / lead {lead}  ({time.time() - t0:.1f}s)")
                print(lb["table"][["Model", "MAE", "Skill vs seasonal average", "Dry-season skill",
                                   "Years beating seasonal average"]].to_string(index=False))
                print("->", rec["reason"])


if __name__ == "__main__":
    main()
