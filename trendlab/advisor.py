"""The advisor: which variables to look at, which model to trust, in plain words.

Two layers:
  1. A data-driven recommender that needs no API key. It maps a concern
     (fires, haze, drying, heat, rain) to target variables, ranks every other
     variable by how strongly it leads the target, links regions (fires upwind ->
     haze downwind), and suggests NASA datasets the file does not have yet.
  2. An optional Claude chat that receives those numbers as context and answers
     free-form questions. It only sees the summary, never the raw file.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from .core import DRIVERS, deseasonalise, meta, transform

CONCERNS: dict[str, dict[str, Any]] = {
    "fires": {"label": "Fires", "icon": ":material/local_fire_department:",
              "targets": ["fire_count"], "related": ["vpd_kpa", "sm_10_40_mm", "precip_mm", "tair_c"],
              "why": "Fire detections are the outcome; dry deep soil, thirsty air and low rain are the conditions that let peat burn."},
    "haze": {"label": "Haze and air quality", "icon": ":material/foggy:",
             "targets": ["aod"], "related": ["fire_count", "precip_mm", "vpd_kpa"],
             "why": "Aerosol optical depth measures smoke and haze; most of it comes from fires, and rain washes it out."},
    "drying": {"label": "Drying land", "icon": ":material/water_drop:",
               "targets": ["sm_10_40_mm", "rootzone_mm", "sm_0_10_mm"], "related": ["precip_mm", "et_mm", "vpd_kpa"],
               "why": "Soil moisture is the water the land holds; rain fills it, evaporation and thirsty air drain it."},
    "heat": {"label": "Heat and thirsty air", "icon": ":material/thermostat:",
             "targets": ["tair_c", "vpd_kpa"], "related": ["precip_mm", "et_mm", "sm_0_10_mm"],
             "why": "Temperature and vapour pressure deficit (how thirsty the air is) set how fast fuel dries."},
    "rain": {"label": "Rainfall", "icon": ":material/rainy:",
             "targets": ["precip_mm"], "related": ["sm_0_10_mm", "et_mm", "vpd_kpa"],
             "why": "Rainfall drives everything downstream: soil water, fire risk and haze wash-out."},
}

# column pairs that are the same quantity in different units, so not "new" information
SAME_AS = {"sm_0_10_vwc": "sm_0_10_mm", "sm_10_40_vwc": "sm_10_40_mm"}
# layers of the same quantity: ranking one against another says nothing new
FAMILY = {c: "soil" for c in ("sm_0_10_mm", "sm_10_40_mm", "rootzone_mm", "sm_0_10_vwc", "sm_10_40_vwc")}

# NASA datasets a user could add for each concern. Earthdata Search links by short name.
NASA_CATALOG: list[dict[str, Any]] = [
    {"name": "Burned area", "product": "MCD64A1", "source": "MODIS (Terra + Aqua), 500 m, monthly",
     "concerns": ["fires"], "why": "Area actually burned, not just hot spots. Separates many small fires from a few huge ones."},
    {"name": "VIIRS active fires (375 m)", "product": "VNP14IMG", "source": "Suomi NPP VIIRS via NASA FIRMS, 2012+",
     "concerns": ["fires", "haze"], "why": "Finer than MODIS, so it catches the small smouldering peat fires that MODIS misses."},
    {"name": "Vegetation greenness (NDVI / EVI)", "product": "MOD13A3", "source": "MODIS Terra, 1 km, monthly",
     "concerns": ["fires", "drying", "heat"], "why": "Shows plant stress and land clearing before and after fire seasons."},
    {"name": "Land surface temperature", "product": "MOD11C3", "source": "MODIS Terra, 0.05 deg, monthly",
     "concerns": ["heat", "drying", "fires"], "why": "Skin temperature of the ground; rises sharply when peat dries out."},
    {"name": "SMAP soil moisture (Level 4)", "product": "SPL4SMGP", "source": "SMAP, 9 km, 3-hourly, 2015+",
     "concerns": ["drying", "fires"], "why": "Satellite-constrained soil moisture, an independent check on the GLDAS model."},
    {"name": "Black and organic carbon AOD", "product": "M2TMNXAER (BCEXTTAU, OCEXTTAU)", "source": "MERRA-2, monthly",
     "concerns": ["haze"], "why": "Splits total haze into smoke (carbon) and other aerosols, so fire haze is isolated."},
    {"name": "MODIS aerosol optical depth", "product": "MOD08_M3", "source": "MODIS Terra, 1 deg, monthly",
     "concerns": ["haze"], "why": "Observed AOD to compare with the MERRA-2 reanalysis already in the file."},
    {"name": "Carbon monoxide", "product": "S5P_L2__CO_____", "source": "Sentinel-5P TROPOMI, 2018+",
     "concerns": ["haze", "fires"], "why": "A gas tracer of burning that travels with the smoke plume."},
    {"name": "Near-surface humidity and wind", "product": "M2TMNXSLV", "source": "MERRA-2, monthly",
     "concerns": ["heat", "haze", "drying"], "why": "Wind direction decides whether haze reaches Singapore; humidity feeds VPD."},
    {"name": "Terrestrial water storage", "product": "GRACE / GRACE-FO mascons", "source": "JPL, monthly, 2002+",
     "concerns": ["drying", "rain"], "why": "Total water in the ground, including deep layers the soil model cannot see."},
]


def earthdata_url(product: str) -> str:
    q = product.split(" ")[0]
    return f"https://search.earthdata.nasa.gov/search?q={q}"


# ----------------------------------------------------------------------------
# Plain-language conclusions
# ----------------------------------------------------------------------------
def pct_per_decade(r: dict) -> float:
    """Percent change per decade; for log-scale counts use the multiplicative change."""
    if meta(r["variable"])[3]:
        return float((np.exp(r["sen_slope_per_decade"]) - 1) * 100)
    return float(r["pct_per_decade"])


def _p_words(p: float) -> str:
    if np.isnan(p):
        return "there is not enough data to tell"
    if p < 0.001:
        return "this is very unlikely to be chance (p < 0.001)"
    if p < 0.01:
        return f"this is unlikely to be chance (p = {p:.3f})"
    if p < 0.05:
        return f"this is probably not chance (p = {p:.2f})"
    if p < 0.10:
        return f"but it could still be chance (p = {p:.2f})"
    return f"the ups and downs look like natural variability (p = {p:.2f})"


def plain_conclusion(r: dict, where: str, period: str, alpha: float = 0.05) -> str:
    """One or two sentences a student can read aloud."""
    name = meta(r["variable"])[0].lower()
    if r["variable"] == "fire_count":
        name = "fire detections"
    p, slope = r["sk_p"], r["sk_slope_per_decade"]
    pct = pct_per_decade(r)
    unit = meta(r["variable"])[1]
    if np.isnan(p):
        return f"There is not enough data to judge {name} in {where}."
    if meta(r["variable"])[3]:
        name = f"the typical month's {name}"
        amount = f"about {abs(pct):.0f}% per decade"
    else:
        u = "" if unit == "unitless" else f" {unit}"
        amount = f"{abs(slope):.3g}{u} per decade ({abs(pct):.1f}% of the average)"
    if p < alpha:
        verb = "risen" if slope > 0 else "fallen"
        s = f"In {where}, {name} {'have' if name.endswith('s') else 'has'} {verb} by {amount} over {period}, and {_p_words(p)}."
    elif p < 0.10:
        verb = "rise" if slope > 0 else "fall"
        s = f"In {where}, {name} show{'' if name.endswith('s') else 's'} a hint of a {verb} of {amount} over {period}, {_p_words(p)}."
    else:
        s = f"In {where}, {name} show{'' if name.endswith('s') else 's'} no clear long-term change over {period}: {_p_words(p)}."
    ap = r.get("adj_p", np.nan)
    if not np.isnan(ap) and p < alpha:
        if ap < alpha:
            s += " The change still holds after allowing for El Nino and the Indian Ocean Dipole."
        else:
            s += " Once El Nino and the Indian Ocean Dipole are allowed for, it is no longer significant, so the climate swings explain much of it."
    return s


# ----------------------------------------------------------------------------
# Which variables lead the target?
# ----------------------------------------------------------------------------
def _anom(df: pd.DataFrame, c: str) -> pd.Series:
    return deseasonalise(transform(df[c], c)) if c not in DRIVERS else df[c]


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int = 6) -> dict:
    """Correlation of b (leading by k months) with a, for k = 0..max_lag."""
    rs = {k: a.corr(b.shift(k)) for k in range(max_lag + 1)}
    rs = {k: (0.0 if np.isnan(v) else float(v)) for k, v in rs.items()}
    k_best = max(rs, key=lambda k: abs(rs[k]))
    k_warn = max((k for k in rs if k >= 1), key=lambda k: abs(rs[k]))
    return {"lag": k_best, "r": rs[k_best], "warn_lag": k_warn, "warn_r": rs[k_warn], "by_lag": rs}


def rank_drivers(df: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    """Rank every other variable by how strongly its anomaly tracks the target's."""
    a = _anom(df, target)
    rows = []
    for c in list(columns) + [d for d in DRIVERS if d in df.columns]:
        if c == target or c in SAME_AS or SAME_AS.get(target) == c:
            continue
        if target in FAMILY and FAMILY.get(c) == FAMILY[target]:
            continue
        ll = lead_lag(a, _anom(df, c))
        rows.append({"variable": c, "Variable": meta(c)[0], "Best lag (months)": ll["lag"], "r": ll["r"],
                     "Explains (r2)": ll["r"] ** 2, "Early-warning lag": ll["warn_lag"], "Early-warning r": ll["warn_r"],
                     "Direction": "same way" if ll["r"] > 0 else "opposite way"})
    out = pd.DataFrame(rows)
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True) if len(out) else out


def driver_sentence(row: pd.Series, target: str) -> str:
    t = _short_name(target)
    v = _short_name(row["variable"])
    lag = int(row["Best lag (months)"])
    when = "in the same month" if lag == 0 else f"{lag} month{'s' if lag > 1 else ''} earlier"
    move = ("rise" if t.endswith("s") else "rises") if row["r"] > 0 else ("fall" if t.endswith("s") else "falls")
    return (f"When {v} {'are' if v.endswith('s') else 'is'} above normal {when}, {t} usually {move}"
            + (" too" if row["r"] > 0 else "") + f" (r = {row['r']:+.2f}).")


def _short_name(c: str) -> str:
    return "fire detections" if c == "fire_count" else meta(c)[0].lower()


# ----------------------------------------------------------------------------
# Linked regions: fires upwind -> haze downwind
# ----------------------------------------------------------------------------
def linked_regions(receptor: pd.DataFrame, sources: dict[str, pd.DataFrame], target: str = "aod",
                   source_col: str = "fire_count", max_lag: int = 3) -> dict:
    """Lagged correlation and joint regression of the receptor's haze on each source's fires."""
    import statsmodels.api as sm

    a = _anom(receptor, target)
    out = {"by_source": {}, "regression": None}
    X = {}
    for name, sdf in sources.items():
        if source_col not in sdf.columns or sdf[source_col].notna().sum() < 36:
            continue
        b = _anom(sdf, source_col).reindex(a.index)
        ll = lead_lag(a, b, max_lag)
        out["by_source"][name] = ll
        X[name] = b.shift(ll["lag"])
    if X:
        d = pd.concat([a.rename("y"), pd.DataFrame(X)], axis=1).dropna()
        if len(d) > 36:
            fit = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            out["regression"] = {"r2": float(fit.rsquared), "n": int(len(d)),
                                 "coef": {k: float(v) for k, v in fit.params.drop("const").items()},
                                 "p": {k: float(v) for k, v in fit.pvalues.drop("const").items()},
                                 "fitted": fit.fittedvalues, "actual": d["y"]}
    return out


# ----------------------------------------------------------------------------
# Model hint (before the leaderboard has run)
# ----------------------------------------------------------------------------
def model_hint(df: pd.DataFrame, target: str, drivers_rank: pd.DataFrame | None = None) -> str:
    a = _anom(df, target).dropna()
    ac1 = float(a.autocorr(1)) if len(a) > 24 else np.nan
    is_count = meta(target)[3]
    parts = []
    if is_count:
        parts.append("It is a skewed count, so a model built for counts (negative binomial, or boosting with a "
                     "Poisson loss) or a model on log counts is appropriate.")
    if not np.isnan(ac1) and ac1 > 0.5:
        parts.append(f"Anomalies persist from month to month (lag-1 correlation {ac1:.2f}), so simple persistence "
                     "will be hard to beat one month ahead.")
    if drivers_rank is not None and len(drivers_rank):
        enso = drivers_rank[drivers_rank["variable"].isin(DRIVERS)]
        if len(enso) and enso["Early-warning r"].abs().max() > 0.3:
            parts.append("El Nino / IOD lead it by a few months, so models that use them (ridge, boosting) "
                         "have something to work with at longer leads.")
    parts.append("Run the model comparison to see which one actually wins on years it never saw.")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Claude chat (optional)
# ----------------------------------------------------------------------------
MODEL_ID = "claude-opus-5-5"

SYSTEM = """You are the AI advisor inside Trend Detective, a dashboard that tests NASA Earth-observation \
time series for trends and forecasts them. Users are students, teachers and members of the public; \
hackathon judges may also ask questions.

Your job:
1. Recommend which variables to examine for the user's question, using the driver rankings, trend \
results and linked-region results in the context.
2. Recommend which forecasting model to trust for a variable, using the model leaderboard in the \
context. Prefer the simplest model within about 2 points of the best skill, and say plainly when \
nothing beats the seasonal average.
3. When the user's question needs data the file does not have, suggest datasets from the NASA catalog \
in the context, with the product short name.

Rules:
- Use only the numbers in the context. If something needed is missing (for example the leaderboard \
has not been run for that variable), say which page of the app to open: Explore trends, Globe, \
Compare and link regions, Models and forecast, Patterns.
- Correlation is not causation; say so when it matters. Fires here are mostly started by people; the \
weather decides how badly they spread.
- Name variables by their plain names, with the column name in backticks the first time.
- Keep answers under about 180 words unless asked for more. Use short paragraphs or a short list."""


def build_context(where: str, period: str, table: pd.DataFrame, rankings: dict[str, pd.DataFrame],
                  leaderboards: dict[str, pd.DataFrame], links: dict | None, columns: list[str]) -> str:
    """Compact JSON summary of everything the advisor knows. No raw data rows."""
    trends = [{"column": r["variable"], "name": r["label"], "unit": r["unit"],
               "slope_per_decade": round(float(r["sk_slope_per_decade"]), 4),
               "seasonal_kendall_p": round(float(r["sk_p"]), 4),
               "enso_adjusted_p": None if pd.isna(r["adj_p"]) else round(float(r["adj_p"]), 4),
               "verdict": r["verdict"]} for _, r in table.iterrows()]
    ctx = {
        "region": where, "period": period,
        "variables": {c: {"name": meta(c)[0], "unit": meta(c)[1], "meaning": meta(c)[2]} for c in columns + DRIVERS},
        "trends": trends,
        "driver_rankings": {t: df.head(6)[["variable", "Best lag (months)", "r", "Early-warning lag", "Early-warning r"]]
                            .round(3).to_dict("records") for t, df in rankings.items()},
        "model_leaderboards": {k: v[["Model", "Skill vs seasonal average", "Dry-season skill",
                                     "Years beating seasonal average", "Years tested", "Explainability"]]
                               .round(3).to_dict("records") for k, v in leaderboards.items()},
        "linked_regions": None if not links else {
            "haze_vs_upwind_fires": {k: {"best_lag_months": v["lag"], "r": round(v["r"], 3)} for k, v in links["by_source"].items()},
            "joint_regression_r2": None if not links.get("regression") else round(links["regression"]["r2"], 3)},
        "nasa_catalog": [{k: d[k] for k in ("name", "product", "source", "why")} for d in NASA_CATALOG],
    }
    return json.dumps(ctx, default=float)


def stream_answer(messages: list[dict], context: str, api_key: str | None = None) -> Iterator[str]:
    """Stream Claude's reply as text chunks. Raises anthropic errors to the caller."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    with client.beta.messages.stream(
        model=MODEL_ID,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        system=[{"type": "text", "text": SYSTEM},
                {"type": "text", "text": "Context for this session (JSON):\n" + context}],
        messages=messages,
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n_The model declined to answer this one. Try rephrasing the question._"
    elif final.stop_reason == "max_tokens":
        yield "\n\n_(Answer cut off at the length limit.)_"
