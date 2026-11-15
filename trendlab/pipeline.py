"""CSV in, results out.

    python -m trendlab.pipeline data/central_kalimantan.csv --out results/central_kalimantan

Steps: validate -> monthly calendar -> anomalies -> trend tests -> Elliptic Envelope
-> PCA -> forecasts -> alerts -> files (trend_table.csv, outliers.csv, forecast.csv,
pca_loadings.csv, alerts.json, summary.json, report.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import alerts as al
from . import core, ml

FORECAST_DEFAULT = ["fire_count", "aod", "vpd_kpa", "sm_10_40_mm", "precip_mm"]
FEATURE_DEFAULT = ["precip_mm", "sm_10_40_mm", "et_mm", "tair_c", "vpd_kpa", "aod", "fire_count"]


def run(src, name: str = "dataset", alpha: float = 0.05, contamination: float = 0.05,
        horizon: int = 12, forecast_cols: list[str] | None = None, rules: dict | None = None,
        indices: pd.DataFrame | None = None) -> dict:
    indices = indices if indices is not None else core.load_indices()
    df, quality = core.load_csv(src, indices)
    cols = core.analysis_columns(df)
    table = core.trend_table(df, cols, alpha)
    feats = [c for c in FEATURE_DEFAULT if c in cols] or cols[:8]
    outliers = ml.elliptic_outliers(df, feats, contamination) if len(feats) >= 2 else pd.DataFrame()
    pca = ml.pca_modes(df, feats) if len(feats) >= 2 else {}
    fcols = forecast_cols if forecast_cols is not None else ([c for c in FORECAST_DEFAULT if c in cols] or cols[:3])
    forecasts = {}
    for c in fcols:
        try:
            forecasts[c] = ml.forecast(df, c, indices, horizon)
        except Exception as e:  # one bad column should not sink the run
            quality["warnings"].append(f"forecast failed for {c}: {e}")
    alerts = al.build_alerts(df, table, outliers, forecasts, indices, {**(rules or {}), "alpha": alpha}, name)
    return {"name": name, "data": df, "quality": quality, "columns": cols, "table": table,
            "features": feats, "outliers": outliers, "pca": pca, "forecasts": forecasts, "alerts": alerts}


def report_markdown(res: dict) -> str:
    q, t = res["quality"], res["table"]
    L = [f"# Trend report: {res['name']}", "",
         f"Period {q['start']} to {q['end']} ({q['n_years']} years, {q['n_months']} months). "
         f"Rows inserted for missing months: {len(q['missing_rows_inserted'])}.", ""]
    L += [f"- Warning: {w}" for w in q["warnings"]] + ([""] if q["warnings"] else [])
    L += ["## What is changing", "",
          "Slope = seasonal Sen slope. p = seasonal Kendall test with serial-correlation correction. "
          "'After ENSO/IOD' = t-test on the time trend with Nino 3.4 and DMI as covariates.", "",
          "| Variable | Slope / decade | % / decade | Seasonal Kendall p | Mann-Kendall p | t-test p | After ENSO/IOD p | Verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in t.sort_values("sk_p").iterrows():
        L.append(f"| {r['label']} ({r['unit']}) | {r['sk_slope_per_decade']:+.3g} | {r['pct_per_decade']:+.1f} | "
                 f"{r['sk_p']:.3g} | {r['mk_p']:.3g} | {r['ols_p']:.3g} | {r['adj_p']:.3g} | {r['verdict']} |")
    o = res["outliers"]
    if len(o):
        top = o[o["outlier"]].sort_values("distance", ascending=False).head(10)
        L += ["", "## Most unusual months (Elliptic Envelope)", "", "| Month | Distance | Main contributor | z |", "|---|---|---|---|"]
        L += [f"| {i:%Y-%m} | {r['distance']:.1f} | {core.meta(r['main_driver'])[0]} | {r['main_z']:+.1f} |" for i, r in top.iterrows()]
    if res["pca"]:
        ev = res["pca"]["explained"]
        ld = res["pca"]["loadings"]["PC1"].sort_values(key=abs, ascending=False)
        L += ["", "## Dominant pattern (PCA)", "",
              f"PC1 explains {ev[0]:.0%} of the variance. Loadings: " + ", ".join(f"{k} {v:+.2f}" for k, v in ld.items()) + "."]
        for d, c in res["pca"]["driver_corr"].items():
            L.append(f"Correlation of PC1 with {d}: {c['PC1']:+.2f}.")
    if res["forecasts"]:
        L += ["", "## Forecasts", "", "| Variable | Model | Backtest skill vs climatology | Peak forecast | When |", "|---|---|---|---|---|"]
        for c, f in res["forecasts"].items():
            fc = f["forecast"]["forecast"]
            sk = f["skill"].get("skill", np.nan)
            L.append(f"| {core.meta(c)[0]} | {f['model']} | {sk:+.0%} | {fc.max():.3g} | {fc.idxmax():%Y-%m} |")
    L += ["", "## Alerts", ""] + ([f"- **{a['severity'].upper()}** {a['title']}. {a['detail']}" for a in res["alerts"]] or ["None."])
    return "\n".join(L) + "\n"


def save(res: dict, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / "trend_table.csv", index=False)
    if len(res["outliers"]):
        res["outliers"].to_csv(out / "outliers.csv")
    if res["pca"]:
        res["pca"]["loadings"].to_csv(out / "pca_loadings.csv")
    if res["forecasts"]:
        pd.concat({c: f["forecast"] for c, f in res["forecasts"].items()}, names=["variable", "date"]).to_csv(out / "forecast.csv")
    (out / "alerts.json").write_text(json.dumps(res["alerts"], indent=2))
    summary = {"name": res["name"], "quality": res["quality"],
               "significant": res["table"].loc[res["table"]["sk_p"] < 0.05, "variable"].tolist(),
               "forecast_skill": {c: f["skill"] for c, f in res["forecasts"].items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "report.md").write_text(report_markdown(res))
    return out


def main():
    ap = argparse.ArgumentParser(description="Run the trend pipeline on a monthly CSV.")
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--webhook", default=None, help="Slack/Discord webhook URL for new alerts")
    ap.add_argument("--email", default=None, help="email address for new alerts (needs SMTP_* env vars)")
    a = ap.parse_args()
    name = Path(a.csv).stem
    res = run(a.csv, name, a.alpha, a.contamination, a.horizon)
    out = save(res, a.out or Path("results") / name)
    new = al.log_alerts(res["alerts"], out / "alerts_log.jsonl")
    print(report_markdown(res))
    print(f"Saved to {out}/  ({len(new)} new alert(s))")
    if new and a.webhook:
        print("webhook:", al.send_webhook(new, a.webhook, name))
    if new and a.email:
        print("email:", al.send_email(new, a.email, name))


if __name__ == "__main__":
    main()
