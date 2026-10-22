"""Shared data contracts; naive source timestamps are KST per the data README."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TZ = "Asia/Seoul"
VAL_FROM = pd.Timestamp("2024-06-01 01:00", tz=TZ)
VAL_TO = pd.Timestamp("2025-06-01 00:00", tz=TZ)
VAL_END = VAL_TO + pd.Timedelta(hours=1)
SITE_COUNT = 253
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def to_kst(values, *, require_timezone=False):
    result = pd.to_datetime(values, errors="raise")
    if not isinstance(result, pd.Series):
        result = pd.Series(result)
    require(not result.isna().any(), "Null timestamps")
    if result.dt.tz is None:
        require(not require_timezone, "CSV ts must include a timezone, e.g. +09:00")
        result = result.dt.tz_localize(TZ)
    result = result.dt.tz_convert(TZ)
    require(result.eq(result.dt.floor("h")).all(), "Timestamps must be hourly")
    return result


def unique(d, keys, label):
    require(not d[keys].isna().any().any(), f"{label}: null key")
    require(not d.duplicated(keys).any(), f"{label}: duplicate key {keys}")


def read_window(path, columns, start, end):
    """Half-open window [start, end); push down timestamp filtering when possible."""
    field = pq.read_schema(path).field("ts")
    filters = None
    if pa.types.is_timestamp(field.type):
        bounds = [v.tz_convert(field.type.tz) if field.type.tz else v.tz_localize(None)
                  for v in [start, end]]
        filters = [("ts", ">=", bounds[0]), ("ts", "<", bounds[1])]
    d = pd.read_parquet(path, columns=columns, filters=filters)
    d["ts"] = to_kst(d.ts)
    return d[(d.ts >= start) & (d.ts < end)].reset_index(drop=True)


def read_sites(data):
    d = pd.read_csv(Path(data) / "sites.csv", dtype={"site_id": str})
    cols = ["site_id", "capacity_kw", "lat", "lon", "wa_station", "wb_station", "wa_dist", "wb_dist", "current_group"]
    require(set(cols) <= set(d), "sites.csv: missing metadata columns")
    unique(d, ["site_id"], "sites")
    require(len(d) == SITE_COUNT, f"Expected {SITE_COUNT} sites")
    require(not d[cols].isna().any().any(), "sites.csv: missing metadata")
    vals = d[["capacity_kw", "lat", "lon", "wa_dist", "wb_dist"]].to_numpy(float)
    require(np.isfinite(vals).all() and (d.capacity_kw > 0).all(), "Invalid site metadata")
    require(d.current_group.isin(["A", "B"]).all(), "Invalid reference group")
    return d.sort_values("site_id").reset_index(drop=True)


def validate_predictions(pred, sites):
    require(list(pred.columns) == ["site_id", "ts", "kwh"], "Required columns, in order: site_id, ts, kwh")
    pred = pred.copy()
    pred["ts"] = to_kst(pred.ts, require_timezone=True)
    unique(pred, ["site_id", "ts"], "predictions")
    require(set(pred.site_id) == set(sites.site_id), "Unknown or missing site")
    times = pd.date_range(VAL_FROM, VAL_TO, freq="h")
    require(len(pred) == len(sites) * len(times), "Missing/extra prediction rows")
    require(pred.ts.isin(times).all(), "Prediction timestamp outside validation grid")
    require(pred.groupby("site_id").size().eq(len(times)).all(), "Incomplete per-site time grid")
    values = pred.kwh.to_numpy(float)
    require(np.isfinite(values).all() and (values >= 0).all(), "Non-finite or negative kwh")
    return pred
