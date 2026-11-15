"""Notification rules and delivery (in-app list, log file, webhook, email)."""
from __future__ import annotations

import hashlib
import json
import os
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

from .core import DRIVERS, deseasonalise, meta, transform

DEFAULT_RULES = {
    "alpha": 0.05,            # significance level for trend alerts
    "z_threshold": 2.0,       # |anomaly z| in the latest month
    "recent_months": 3,       # window for outlier alerts
    "forecast_quantile": 0.9, # forecast above this historical quantile
    "nino_watch": 1.0,        # Nino 3.4 at/above this in the next months
}
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _alert(severity, kind, variable, title, detail, when=None):
    key = f"{kind}|{variable}|{when}|{title}"
    return {"id": hashlib.sha1(key.encode()).hexdigest()[:12], "severity": severity, "kind": kind,
            "variable": variable, "title": title, "detail": detail, "when": when}


def build_alerts(df, table, outliers=None, forecasts=None, indices=None, rules=None, dataset="dataset") -> list[dict]:
    r = {**DEFAULT_RULES, **(rules or {})}
    alerts = []
    # 1. significant trends (seasonal Kendall, autocorrelation-adjusted)
    for _, row in table.iterrows():
        if row["sk_p"] < r["alpha"]:
            robust = row["adj_p"] < r["alpha"] and np.sign(row["adj_slope_per_decade"]) == np.sign(row["sk_slope_per_decade"])
            alerts.append(_alert(
                "warning" if robust else "info", "trend", row["variable"],
                f"{row['label']}: {row['verdict']}",
                f"{row['sk_slope_per_decade']:+.3g} {row['unit']} per decade (seasonal Kendall p={row['sk_p']:.3g}). "
                + ("Still significant after removing ENSO/IOD." if robust else
                   f"Not significant once ENSO/IOD are controlled for (p={row['adj_p']:.2g}); treat as climate-driven variability."),
                dataset))
    # 2. latest-month extremes
    for c in table["variable"]:
        a = deseasonalise(transform(df[c], c)).dropna()
        if len(a) < 36:
            continue
        z = (a.iloc[-1] - a.mean()) / a.std()
        if abs(z) >= r["z_threshold"]:
            alerts.append(_alert("warning" if abs(z) < 3 else "critical", "extreme", c,
                                 f"{meta(c)[0]}: latest month is {z:+.1f} sigma from normal",
                                 f"{a.index[-1]:%b %Y} value {df[c].loc[a.index[-1]]:.3g} {meta(c)[1]}.",
                                 f"{a.index[-1]:%Y-%m}"))
    # 3. recent multivariate outliers
    if outliers is not None and len(outliers):
        recent = outliers.loc[outliers.index >= df.index[-1] - pd.DateOffset(months=r["recent_months"] - 1)]
        for when, row in recent[recent["outlier"]].iterrows():
            alerts.append(_alert("warning", "outlier", row["main_driver"],
                                 f"Unusual month: {when:%b %Y}",
                                 f"Elliptic Envelope distance {row['distance']:.1f}; largest contributor "
                                 f"{meta(row['main_driver'])[0]} at {row['main_z']:+.1f} sigma.", f"{when:%Y-%m}"))
    # 4. forecast exceedance
    for c, fc in (forecasts or {}).items():
        thr = float(fc["history"].quantile(r["forecast_quantile"]))
        hot = fc["forecast"][fc["forecast"]["forecast"] > thr]
        if len(hot) and c in ("fire_count", "aod", "vpd_kpa"):
            peak = hot["forecast"].idxmax()
            sk = fc["skill"].get("skill", np.nan)
            trusted = sk > 0.1
            alerts.append(_alert("critical" if trusted and c in ("fire_count", "aod") else "warning", "forecast", c,
                                 f"{meta(c)[0]} forecast above the historical {int(r['forecast_quantile'] * 100)}th percentile",
                                 f"{len(hot)} month(s), peak {hot['forecast'].max():.3g} {meta(c)[1]} in {peak:%b %Y} "
                                 f"(threshold {thr:.3g}). Backtest skill vs climatology {sk:+.0%}"
                                 + ("." if trusted else ": no better than climatology, read as a scenario not a prediction."),
                                 f"{peak:%Y-%m}"))
    # 5. El Nino watch from the index file
    if indices is not None and "nino34" in indices:
        ahead = indices["nino34"].loc[df.index[-1] + pd.DateOffset(months=1):].dropna()
        if len(ahead) and ahead.max() >= r["nino_watch"]:
            alerts.append(_alert("critical", "enso", "nino34",
                                 f"El Nino watch: Nino 3.4 reaches {ahead.max():+.2f} in {ahead.idxmax():%b %Y}",
                                 "Past El Nino dry seasons (2002, 2006, 2015, 2019, 2023) produced the largest fire and haze "
                                 "spikes in this record. Expect elevated fire risk in the Aug-Oct dry season.",
                                 f"{ahead.idxmax():%Y-%m}"))
    return sorted(alerts, key=lambda a: SEVERITY_ORDER[a["severity"]])


# ----------------------------------------------------------------------------
# Delivery
# ----------------------------------------------------------------------------
def log_alerts(alerts: list[dict], path: Path) -> list[dict]:
    """Append unseen alerts to a JSONL log. Returns only the new ones."""
    path = Path(path)
    seen = set()
    if path.exists():
        seen = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}
    new = [a for a in alerts if a["id"] not in seen]
    if new:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with path.open("a") as f:
            for a in new:
                f.write(json.dumps({**a, "logged_at": stamp}) + "\n")
    return new


def format_text(alerts: list[dict], dataset: str) -> str:
    lines = [f"Trend Detective: {len(alerts)} alert(s) for {dataset}"]
    lines += [f"[{a['severity'].upper()}] {a['title']}\n    {a['detail']}" for a in alerts]
    return "\n".join(lines)


def send_webhook(alerts: list[dict], url: str, dataset: str = "dataset") -> str:
    """POST to a Slack / Discord / Teams style incoming webhook."""
    text = format_text(alerts, dataset)
    body = json.dumps({"text": text, "content": text[:1900]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return f"HTTP {resp.status}"


def send_email(alerts: list[dict], to: str, dataset: str = "dataset") -> str:
    """Send via SMTP. Reads SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD from the environment."""
    host, user, pw = os.environ.get("SMTP_HOST"), os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASSWORD")
    if not (host and user and pw):
        raise RuntimeError("Set SMTP_HOST, SMTP_USER and SMTP_PASSWORD environment variables first.")
    msg = EmailMessage()
    msg["Subject"] = f"Trend Detective: {len(alerts)} alert(s) for {dataset}"
    msg["From"], msg["To"] = user, to
    msg.set_content(format_text(alerts, dataset))
    with smtplib.SMTP_SSL(host, int(os.environ.get("SMTP_PORT", 465)), timeout=20) as s:
        s.login(user, pw)
        s.send_message(msg)
    return "sent"
