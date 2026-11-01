"""OIBC 2026: forecast-only training, chronological validation and VPP allocation.

Contract source: supplied data README and competition screenshots.
No external data, pretrained weights, observed weather or generation lags are used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import model_backends as backends

TZ = "Asia/Seoul"
SITE_COUNT = 253
SCORE_HOURS = 8760
MIN_CAPACITY = 20_000.0
WA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press"]
WB = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob",
      "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point"]
REQUIRED = ["sites.csv", "smp.csv", "train/smp.csv", "train/generation.parquet",
            "train/forecast_wa.parquet", "train/forecast_wb.parquet",
            "test/forecast_wa.parquet", "test/forecast_wb.parquet"]


def log(message):
    print(message, flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(value):
    t = pd.Timestamp(value)
    return t.tz_localize(TZ) if t.tzinfo is None else t.tz_convert(TZ)


def timestamps(values, *, aware=False):
    ts = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    require(not ts.hasnans, "Timestamp contains NaT")
    if ts.tz is None:
        require(not aware, "Submission timestamps must include a timezone")
        ts = ts.tz_localize(TZ)  # Source README explicitly defines naive times as KST.
    ts = ts.tz_convert(TZ)
    require((ts == ts.floor("h")).all(), "Expected hourly right-closed timestamps")
    return ts


def operating_days(ts):
    # E.g. June 2 00:00 is the last hour of June 1, NOT June 2.
    return (pd.DatetimeIndex(ts) - pd.Timedelta(hours=1)).normalize()


def forecast_cutoff(first_ts):
    return operating_days([stamp(first_ts)])[0] - pd.Timedelta(days=1) + pd.Timedelta(hours=17)


def k_values(ts):
    hour = pd.DatetimeIndex(ts).hour
    return np.where((hour >= 11) & (hour <= 15), 3.0, 1.0)


def unique(frame, keys, label):
    require(not frame[keys].isna().any().any(), f"{label}: null key")
    require(not frame.duplicated(keys).any(), f"{label}: duplicate {keys}")


def read_sites(root):
    sites = pd.read_csv(Path(root) / "sites.csv").sort_values("site_id").reset_index(drop=True)
    cols = ["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station",
            "wa_dist", "wb_dist"]
    require(set(cols) <= set(sites), "sites.csv: missing columns")
    sites = sites[cols].copy()
    unique(sites, ["site_id"], "sites")
    require(len(sites) == SITE_COUNT, f"Expected {SITE_COUNT} sites")
    require(not sites.isna().any().any(), "sites.csv: missing metadata")
    numeric = sites[["capacity_kw", "lat", "lon", "wa_dist", "wb_dist"]].to_numpy(float)
    require(np.isfinite(numeric).all(), "sites.csv: non-finite metadata")
    require((sites.capacity_kw > 0).all(), "Non-positive capacity")
    require(sites.lat.between(-90, 90).all() and sites.lon.between(-180, 180).all(), "Invalid coordinates")
    require(sites.current_group.isin(["A", "B"]).all(), "Unknown reference group")
    validate_groups(sites, (sites.current_group == "B").to_numpy(np.int8))
    return sites


def read_prices(path, *, evaluation=False):
    d = pd.read_csv(path)
    require({"ts", "smp"} <= set(d), f"{path}: missing price columns")
    d["ts"] = timestamps(d.ts)
    unique(d, ["ts"], str(path))
    d = d.sort_values("ts").reset_index(drop=True)
    require(np.isfinite(d.smp).all() and (d.smp >= 0).all(), "Invalid SMP")
    expected_k = k_values(d.ts)
    if "k" in d:
        require(np.array_equal(d.k.to_numpy(), expected_k), "k differs from the supplied rule")
    d["k"] = expected_k
    if evaluation:
        require(len(d) == SCORE_HOURS, f"Expected {SCORE_HOURS} scoring timestamps")
        expected = pd.date_range(d.ts.iloc[0], periods=len(d), freq="h")
        require(pd.DatetimeIndex(d.ts).equals(expected), "Scoring timestamps must be continuous")
        require(d.ts.iloc[0].hour == 1 and d.ts.iloc[-1].hour == 0, "Incomplete scoring days")
    return d


def validate_groups(sites, groups):
    groups = np.asarray(groups)
    require(groups.shape == (len(sites),) and np.isin(groups, [0, 1]).all(), "Invalid group vector")
    caps = [float(sites.capacity_kw.to_numpy()[groups == g].sum()) for g in [0, 1]]
    require(min(caps) >= MIN_CAPACITY, f"Both groups require >= 20,000 kW; got {caps}")
    return caps


def audit(root):
    root = Path(root)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    result = {"missing_files": missing, "uses_observed_weather": False,
              "uses_external_data_or_pretrained_models": False}
    if (root / "sites.csv").is_file():
        sites = read_sites(root)
        result.update(sites=len(sites), total_capacity_kw=float(sites.capacity_kw.sum()),
                      reference_group_capacity_kw=validate_groups(sites, (sites.current_group == "B").to_numpy()))
    if (root / "smp.csv").is_file():
        prices = read_prices(root / "smp.csv", evaluation=True)
        result.update(scoring_hours=len(prices), first_scoring_ts=str(prices.ts.iloc[0]),
                      last_scoring_ts=str(prices.ts.iloc[-1]),
                      final_training_cutoff=str(forecast_cutoff(prices.ts.iloc[0])))
    return result


def load_forecasts(root, split):
    result = {}
    for provider, cols in [("wa", WA), ("wb", WB)]:
        key = provider + "_station"
        path = Path(root) / split / f"forecast_{provider}.parquet"
        d = pd.read_parquet(path, columns=[key, "round", "ts"] + cols)
        d["ts"] = timestamps(d.ts)
        unique(d, [key, "round", "ts"], str(path))
        require(d["round"].isin([10, 17]).all(), f"{path}: unsupported forecast round")
        # Independent per-row transformations; no interpolation or future-value fills.
        for col in cols:
            d[col] = pd.to_numeric(d[col], errors="raise").astype("float32")
            d[col] = d[col].replace([np.inf, -np.inf], np.nan)
            if col in ["wa_cloud", "wa_humidity", "wb_cloud", "wb_rel_hum", "wb_precip_prob"]:
                d[col] = d[col].clip(0, 100)
            if col in ["wa_rain", "wb_total_liq", "wa_wind_speed", "wb_wind_speed", "wb_vis", "wb_uv_idx"]:
                d.loc[d[col] < 0, col] = np.nan
        if provider == "wb":
            d["wb_wind_speed"] /= 3.6  # km/h -> m/s
        for round_ in [10, 17]:
            part = d[d["round"] == round_].drop(columns="round").copy()
            part[f"{provider}_present_r{round_}"] = np.float32(1)
            part.rename(columns={c: f"{c}_r{round_}" for c in cols}, inplace=True)
            result[(provider, round_)] = part
    return result


def solar(ts, lat, lon):
    """Analytic approximation, adapted from the supplied baseline; no API/data lookup."""
    mid = pd.DatetimeIndex(ts) - pd.Timedelta(minutes=30)
    hour = mid.hour.to_numpy() + mid.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (mid.dayofyear.to_numpy() - 1 + (hour - 12) / 24)
    eq = 229.18 * (.000075 + .001868*np.cos(g) - .032077*np.sin(g)
                   - .014615*np.cos(2*g) - .040849*np.sin(2*g))
    decl = (.006918 - .399912*np.cos(g) + .070257*np.sin(g) - .006758*np.cos(2*g)
            + .000907*np.sin(2*g) - .002697*np.cos(3*g) + .00148*np.sin(3*g))
    ha = np.radians((hour * 60 + eq + 4 * (lon - 135)) / 4 - 180)
    la = np.radians(lat)
    cosz = np.clip(np.sin(la)*np.sin(decl) + np.cos(la)*np.cos(decl)*np.cos(ha), -1, 1)
    return np.degrees(np.arcsin(cosz)), cosz


def features(keys, sites, weather):
    # Never merge energy or SMP into X, even if present in the input keys frame.
    base = keys[["site_id", "ts"]].copy().reset_index(drop=True)
    d = base.merge(sites, on="site_id", how="left", validate="many_to_one", sort=False)
    require(d.capacity_kw.notna().all(), "Unknown site in feature keys")
    for (provider, round_), part in weather.items():
        d = d.merge(part, on=[provider + "_station", "ts"], how="left", validate="many_to_one", sort=False)
    require(d[["site_id", "ts"]].equals(base), "Feature merge changed row ordering")
    x = pd.DataFrame(index=d.index)
    for provider, cols in [("wa", WA), ("wb", WB)]:
        for round_ in [10, 17]:
            for col in cols:
                x[f"{col}_r{round_}"] = d[f"{col}_r{round_}"]
            x[f"{provider}_missing_r{round_}"] = d[f"{provider}_present_r{round_}"].isna().astype("float32")
        for col in [provider + "_cloud", provider + "_temp"]:
            x[col + "_revision"] = x[col + "_r17"] - x[col + "_r10"]
    for round_ in [10, 17]:
        for col in ["cloud", "temp"]:
            x[f"ab_{col}_difference_r{round_}"] = x[f"wa_{col}_r{round_}"] - x[f"wb_{col}_r{round_}"]
    for col in ["capacity_kw", "lat", "lon", "wa_dist", "wb_dist"]:
        x[col] = d[col]
    x["site_code"] = d.site_id.map({s: i for i, s in enumerate(sites.site_id)})
    ts = pd.DatetimeIndex(d.ts)
    x["solar_elevation"], x["cos_zenith"] = solar(ts, d.lat.to_numpy(), d.lon.to_numpy())
    x["positive_cos_zenith"] = x.cos_zenith.clip(lower=0)
    x["hour"] = ts.hour
    x["hour_sin"] = np.sin(2*np.pi*(ts.hour - .5)/24)
    x["hour_cos"] = np.cos(2*np.pi*(ts.hour - .5)/24)
    day = (ts - pd.Timedelta(minutes=30)).dayofyear
    x["day_sin"] = np.sin(2*np.pi*day/365.25)
    x["day_cos"] = np.cos(2*np.pi*day/365.25)
    return x.astype("float32")


@dataclass
class Config:
    train_from: str = "2023-06-01"
    optimize_from: str = "2024-06-01"
    holdout_from: str = "2025-03-01"
    trees: int = 1000
    early_stopping: int = 60
    num_leaves: int = 63
    min_child_samples: int = 100
    threads: int = 4
    seed: int = 2026
    chunk_hours: int = 168
    search_steps: int = 3000
    min_complete_hours: int = 168
    model: str = "lightgbm"
    learning_rate: float = .04
    max_depth: int = 6


def training_matrix(gen, sites, weather, prices, start, end):
    selected = gen[(gen.ts >= start) & (gen.ts <= end)].reset_index(drop=True)
    require(len(selected) > 0, f"No generation rows in {start}..{end}")
    # Remove only physically invalid labels. Missing (site, ts) pairs stay absent.
    good = np.isfinite(selected.energy) & (selected.energy >= 0)
    invalid = int((~good).sum())
    selected = selected[good].reset_index(drop=True)
    require(len(selected) > 0, "No valid labels")
    x = features(selected, sites, weather)
    rate = selected.energy.to_numpy(np.float64) / x.capacity_kw.to_numpy()
    smp = prices.set_index("ts").smp.reindex(pd.DatetimeIndex(selected.ts)).to_numpy(float)
    require(np.isfinite(smp).all(), "Missing training SMP: cannot construct objective weights")
    w = smp * k_values(selected.ts) * x.capacity_kw.to_numpy()
    # Approximate the group penalty with capacity/SMP/k-weighted site absolute errors.
    mask = (x.solar_elevation.to_numpy() > -6) & (w > 0)
    require(mask.any(), "No daylight labels with positive training weights")
    meta = {"requested_start": str(start), "allowed_end": str(end),
            "latest_label": str(selected.loc[mask, "ts"].max()), "rows": int(mask.sum()),
            "invalid_labels_removed": invalid}
    weights = w[mask]
    weights /= weights.mean()
    return x.loc[mask].reset_index(drop=True), rate[mask].astype("float32"), weights.astype("float32"), meta


def fit_model(train, config, *, rounds=None, validation=None):
    return backends.fit_model(train, config, rounds=rounds, validation=validation)


def predict_sites(model, sites, weather, times, chunk_hours):
    require(chunk_hours > 0, "chunk_hours must be positive")
    pieces = []
    names = model.feature_name_ if hasattr(model, "feature_name_") else model.feature_name()
    for start in range(0, len(times), chunk_hours):
        chunk = pd.DatetimeIndex(times[start:start + chunk_hours])
        keys = pd.MultiIndex.from_product([sites.site_id, chunk], names=["site_id", "ts"]).to_frame(index=False)
        # Restrict RHS merges too; avoid rescanning millions of forecasts per chunk.
        local_weather = {key: value[value.ts.between(chunk.min(), chunk.max())]
                         for key, value in weather.items()}
        x = features(keys, sites, local_weather)
        require(list(x.columns) == list(names), "Model/feature schema mismatch")
        pred = np.maximum(model.predict(x), 0) * x.capacity_kw.to_numpy()
        pred[x.solar_elevation.to_numpy() <= -6] = 0
        require(np.isfinite(pred).all(), "Non-finite model output")
        keys["forecast_kwh"] = pred
        pieces.append(keys)
    require(bool(pieces), "Empty prediction time range")
    return pd.concat(pieces, ignore_index=True)


def evaluation_arrays(predictions, gen, sites, prices):
    times = pd.DatetimeIndex(prices.ts)
    ids = sites.site_id.tolist()
    actual = gen.pivot(index="ts", columns="site_id", values="energy").reindex(index=times, columns=ids)
    pred = predictions.pivot(index="ts", columns="site_id", values="forecast_kwh").reindex(index=times, columns=ids)
    require(np.isfinite(pred.to_numpy()).all(), "Incomplete site predictions")
    a = actual.to_numpy(float)
    complete = np.isfinite(a).all(axis=1) & (a >= 0).all(axis=1)
    # Missing true generation is NEVER imputed as zero for optimization/scoring.
    p = pred.to_numpy(float)
    return {"actual": a[complete], "pred": p[complete], "residual": p[complete] - a[complete],
            "weights": (prices.smp * prices.k).to_numpy(float)[complete],
            "smp": prices.smp.to_numpy(float)[complete], "times": times[complete],
            "total_hours": len(times), "complete_hours": int(complete.sum()),
            "coverage": float(complete.mean()) if len(times) else 0.0}


def weighted_error(arrays, groups):
    r = arrays["residual"]
    e0 = r[:, groups == 0].sum(axis=1)
    e1 = r[:, groups == 1].sum(axis=1)
    return float(np.dot(arrays["weights"], np.abs(e0) + np.abs(e1)))


def score(arrays, groups):
    info = {key: arrays[key] for key in ["total_hours", "complete_hours", "coverage"]}
    if arrays["complete_hours"] == 0:
        return dict(info, score_won=None, reason="No timestamps with all site labels; not scored")
    penalty = .10 * weighted_error(arrays, groups)
    sales = float(np.dot(arrays["smp"], arrays["actual"].sum(axis=1)))
    return dict(info, sales_won=sales, penalty_won=penalty, score_won=sales-penalty)


def optimize_groups(sites, arrays, config):
    groups = (sites.current_group == "B").to_numpy(np.int8)
    validate_groups(sites, groups)
    info = {"method": "deterministic capacity-constrained residual local search",
            "complete_hours": arrays["complete_hours"], "accepted_moves": 0}
    if arrays["complete_hours"] < config.min_complete_hours or arrays["weights"].sum() == 0:
        info["fallback"] = "Insufficient complete hours/positive weights; retained reference groups"
        return groups, info
    residual = arrays["residual"]
    total = residual.sum(axis=1)
    left = residual[:, groups == 0].sum(axis=1)
    cap = sites.capacity_kw.to_numpy(float)
    cap0, _ = validate_groups(sites, groups)
    total_cap = cap.sum()
    weights = arrays["weights"]
    value = float(np.dot(weights, np.abs(left) + np.abs(total-left)))
    info["reference_weighted_error"] = value
    rng = np.random.default_rng(config.seed)
    for step in range(config.search_steps):
        # Both single moves and pair swaps preserve every site exactly once.
        if step % 3 == 0:
            candidates = [int(rng.integers(len(groups)))]
        else:
            candidates = [int(rng.choice(np.flatnonzero(groups == g))) for g in [0, 1]]
        delta_cap = sum(cap[i] * (1 if groups[i] == 1 else -1) for i in candidates)
        new_cap = cap0 + delta_cap
        if min(new_cap, total_cap-new_cap) < MIN_CAPACITY:
            continue
        updated = left.copy()
        for i in candidates:
            updated += residual[:, i] * (1 if groups[i] == 1 else -1)
        trial = float(np.dot(weights, np.abs(updated) + np.abs(total-updated)))
        if trial < value - 1e-7:
            for i in candidates:
                groups[i] = 1-groups[i]
            left, cap0, value = updated, new_cap, trial
            info["accepted_moves"] += 1
    info["optimized_weighted_error"] = value
    info["capacities_kw"] = validate_groups(sites, groups)
    return groups, info


def validate_submission(root, output):
    sites = read_sites(root)
    times = pd.DatetimeIndex(read_prices(Path(root) / "smp.csv", evaluation=True).ts)
    composition = pd.read_csv(Path(output) / "composition.csv")
    forecast = pd.read_csv(Path(output) / "forecast.csv")
    require(list(composition.columns) == ["site_id", "group"], "Wrong composition columns")
    require(list(forecast.columns) == ["group", "ts", "forecast_kwh"], "Wrong forecast columns")
    unique(composition, ["site_id"], "composition")
    require(set(composition.site_id) == set(sites.site_id), "Site omission or unknown site")
    require(composition.group.isin(["G1", "G2"]).all(), "Invalid group label")
    aligned = composition.set_index("site_id").reindex(sites.site_id)
    caps = validate_groups(sites, (aligned.group == "G2").to_numpy())
    forecast["ts"] = timestamps(forecast.ts, aware=True)
    unique(forecast, ["group", "ts"], "forecast")
    require(forecast.group.isin(["G1", "G2"]).all(), "Invalid forecast group")
    require(len(forecast) == 2*len(times), "Wrong number of forecast rows")
    for group in ["G1", "G2"]:
        got = pd.DatetimeIndex(forecast.loc[forecast.group == group, "ts"].sort_values())
        require(got.equals(times), f"{group}: timestamp mismatch")
    p = forecast.forecast_kwh.to_numpy(float)
    require(np.isfinite(p).all() and (p >= 0).all(), "Negative/non-finite prediction")
    return {"valid": True, "sites": len(sites), "forecast_rows": len(forecast), "capacity_kw": caps}


def save_submission(root, output, sites, groups, predictions):
    labels = np.where(groups == 0, "G1", "G2")
    composition = pd.DataFrame({"site_id": sites.site_id, "group": labels})
    d = predictions.merge(composition, on="site_id", validate="many_to_one")
    f = d.groupby(["group", "ts"], as_index=False).forecast_kwh.sum().sort_values(["group", "ts"])
    composition.to_csv(Path(output) / "composition.csv", index=False)
    f.to_csv(Path(output) / "forecast.csv", index=False, float_format="%.8f")
    return validate_submission(root, output)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def evaluate_saved(root, output):
    """Re-score saved out-of-sample predictions; no model or test forecasts needed."""
    root, output = Path(root), Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name in ["sites.csv", "train/smp.csv", "train/generation.parquet"]:
        require(fingerprint(root / name) == manifest["input_files"][name],
                f"Input changed since training: {name}")
    warnings = []
    artifacts = manifest.get("holdout_artifacts")
    if artifacts is None:
        warnings.append("Older run: saved prediction/composition fingerprints are unavailable; provenance is not verified.")
    else:
        for name in ["composition.csv", "holdout_site_predictions.parquet"]:
            require(fingerprint(output / name) == artifacts[name], f"Saved holdout artifact changed: {name}")

    sites = read_sites(root)
    first = stamp(manifest["config"]["holdout_from"]) + pd.Timedelta(hours=1)
    last = stamp(manifest["final_cutoff"])
    require(first <= last < stamp(manifest["test_start"]), "Invalid saved holdout interval")
    times = pd.date_range(first, last, freq="h")
    prices = read_prices(root / "train/smp.csv")
    prices = prices[prices.ts.between(first, last)].reset_index(drop=True)
    require(pd.DatetimeIndex(prices.ts).equals(times), "Holdout SMP has gaps")

    composition = pd.read_csv(output / "composition.csv")
    require(list(composition.columns) == ["site_id", "group"], "Wrong composition columns")
    unique(composition, ["site_id"], "composition")
    require(set(composition.site_id) == set(sites.site_id), "Site omission or unknown site")
    require(composition.group.isin(["G1", "G2"]).all(), "Invalid group label")
    groups = (composition.set_index("site_id").reindex(sites.site_id).group == "G2").to_numpy(np.int8)
    capacities = validate_groups(sites, groups)

    pred = pd.read_parquet(output / "holdout_site_predictions.parquet")
    require(set(pred.columns) == {"site_id", "ts", "forecast_kwh"}, "Wrong holdout prediction columns")
    pred["ts"] = timestamps(pred.ts, aware=True)
    unique(pred, ["site_id", "ts"], "holdout predictions")
    require(set(pred.site_id) == set(sites.site_id), "Unknown or missing prediction site")
    require(pred.ts.isin(times).all(), "Prediction timestamp outside the saved holdout")
    require(len(pred) == len(sites) * len(times), "Incomplete holdout prediction grid")
    require(np.isfinite(pred.forecast_kwh).all() and (pred.forecast_kwh >= 0).all(),
            "Negative/non-finite holdout prediction")

    gen = pd.read_parquet(root / "train/generation.parquet", columns=["site_id", "ts", "energy"])
    gen["ts"] = timestamps(gen.ts)
    gen = gen[gen.ts.between(first, last)].copy()
    unique(gen, ["site_id", "ts"], "holdout generation")
    require(gen.site_id.isin(sites.site_id).all(), "Unknown generation site")
    arrays = evaluation_arrays(pred, gen, sites, prices)

    def metrics(assignment):
        result = score(arrays, assignment)
        result.update(group_mae_kwh=None, group_rmse_kwh=None, wwape=None)
        if arrays["complete_hours"]:
            errors = np.column_stack([arrays["residual"][:, assignment == g].sum(axis=1) for g in [0, 1]])
            denominator = float(np.dot(arrays["weights"], arrays["actual"].sum(axis=1)))
            result.update(group_mae_kwh=float(np.abs(errors).mean()),
                          group_rmse_kwh=float(np.sqrt(np.square(errors).mean())),
                          wwape=weighted_error(arrays, assignment) / denominator if denominator > 0 else None)
        return result

    reference = metrics((sites.current_group == "B").to_numpy(np.int8))
    selected = metrics(groups)
    comparison = {"penalty_reduction_won": None, "penalty_reduction_pct": None}
    if arrays["complete_hours"]:
        reduction = reference["penalty_won"] - selected["penalty_won"]
        comparison.update(penalty_reduction_won=reduction,
                          penalty_reduction_pct=100 * reduction / reference["penalty_won"]
                          if reference["penalty_won"] > 0 else None)
    if arrays["coverage"] < 1:
        warnings.append("Scores use only hours with valid actuals for every site; do not treat them as full-period revenue.")
    if arrays["complete_hours"] == 0:
        warnings.append("No complete actual hours: settlement and error metrics are unavailable, not zero.")
    report_path = output / "validation_report.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        fallback = previous.get("group_search", {}).get("fallback")
        if fallback:
            warnings.append("Allocation search retained reference groups: " + fallback)
    return {"evaluation": "local_chronological_holdout", "first_ts": str(first), "last_ts": str(last),
            "capacity_kw": capacities, "holdout_reference": reference, "holdout_selected": selected,
            "comparison": comparison, "warnings": warnings,
            "note": "Earlier holdout model predictions, not final-model performance or an official leaderboard score."}


def print_evaluation(report):
    ref, selected = report["holdout_reference"], report["holdout_selected"]
    print("Local chronological holdout (not official leaderboard)")
    print(f"Period: {report['first_ts']} .. {report['last_ts']}")
    print(f"Complete actual hours: {selected['complete_hours']:,}/{selected['total_hours']:,} "
          f"({selected['coverage']:.2%})")
    print(f"{'Metric':<28} {'Reference':>18} {'Selected':>18}")
    for key, label in [("score_won", "S: settlement (KRW)"), ("penalty_won", "W: penalty (KRW)"),
                       ("group_mae_kwh", "Group MAE (kWh)"), ("group_rmse_kwh", "Group RMSE (kWh)"),
                       ("wwape", "Weighted WAPE")]:
        values = ["N/A" if item.get(key) is None else
                  (f"{item[key]:.2%}" if key == "wwape" else f"{item[key]:,.4f}") for item in [ref, selected]]
        print(f"{label:<28} {values[0]:>18} {values[1]:>18}")
    delta = report["comparison"]["penalty_reduction_won"]
    print("Penalty reduction (positive = better): " + ("N/A" if delta is None else f"{delta:,.4f} KRW"))
    for warning in report["warnings"]:
        print("WARNING: " + warning)
    print(report["note"])


def predict_saved(root, model_dir, output, chunk_hours=168):
    """Reproduce inference without loading labels or retraining."""
    root, model_dir, output = Path(root), Path(model_dir), Path(output)
    require(not output.exists() or not any(output.iterdir()), "Output directory must be empty")
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ["sites.csv", "smp.csv", "test/forecast_wa.parquet", "test/forecast_wb.parquet"]:
        require(fingerprint(root / name) == manifest["input_files"][name], f"Input changed since training: {name}")
    sites = read_sites(root)
    times = pd.DatetimeIndex(read_prices(root / "smp.csv", evaluation=True).ts)
    comp = pd.read_csv(model_dir / "composition.csv")
    unique(comp, ["site_id"], "saved composition")
    require(set(comp.site_id) == set(sites.site_id), "Invalid saved composition site set")
    require(comp.group.isin(["G1", "G2"]).all(), "Invalid saved group")
    groups = (comp.set_index("site_id").reindex(sites.site_id).group == "G2").to_numpy(np.int8)
    validate_groups(sites, groups)
    artifact = manifest.get("model_artifact")
    if artifact is not None:
        cls = backends.backend_class(artifact["backend"])
        require(artifact["file"] == cls.filename, "Unexpected model artifact filename")
        require(fingerprint(model_dir / cls.filename) == artifact["fingerprint"], "Saved model artifact changed")
    model = backends.load_model(model_dir, manifest)
    predictions = predict_sites(model, sites, load_forecasts(root, "test"), times, chunk_hours)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "site_predictions.parquet", index=False)
    return save_submission(root, output, sites, groups, predictions)


def run_pipeline(root, output, config):
    root, output = Path(root), Path(output)
    # Fail before expensive data preparation when an optional backend is absent.
    backends.ensure_available(config.model)
    status = audit(root)
    require(not status["missing_files"], "Missing files:\n" + "\n".join(status["missing_files"]))
    require(not output.exists() or not any(output.iterdir()), "Output directory must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    sites = read_sites(root)
    prices = read_prices(root / "train/smp.csv")
    test_prices = read_prices(root / "smp.csv", evaluation=True)
    start, opt, hold = [stamp(v) for v in [config.train_from, config.optimize_from, config.holdout_from]]
    require(all(v == v.normalize() for v in [start, opt, hold]), "Split dates must start at KST midnight")
    test_day = operating_days([test_prices.ts.iloc[0]])[0]
    require(start < opt < hold < test_day, "Require train_from < optimize_from < holdout_from < evaluation day")
    opt_first, hold_first = opt + pd.Timedelta(hours=1), hold + pd.Timedelta(hours=1)
    cut_opt, cut_hold, cut_final = [forecast_cutoff(t) for t in [opt_first, hold_first, test_prices.ts.iloc[0]]]
    gen = pd.read_parquet(root / "train/generation.parquet", columns=["site_id", "ts", "energy"])
    gen["ts"] = timestamps(gen.ts)
    unique(gen, ["site_id", "ts"], "generation")
    require(gen.site_id.isin(sites.site_id).all(), "Unknown generation site")
    require((gen.ts < test_prices.ts.iloc[0]).all(), "Training labels overlap the evaluation period")
    # Even retrospective scoring avoids labels newer than the first live cutoff.
    gen = gen[(gen.ts >= start + pd.Timedelta(hours=1)) & (gen.ts <= cut_final)].copy()
    days = operating_days(prices.ts)
    p_opt = prices[(days >= opt) & (days < hold)].reset_index(drop=True)
    p_hold = prices[(days >= hold) & (days < test_day)].reset_index(drop=True)
    require(len(p_opt) and len(p_hold), "Empty optimization/holdout prices")
    # Missing SMP must not silently shorten a validation interval.
    for p, first, last in [(p_opt, opt_first, hold), (p_hold, hold_first, test_day)]:
        require(pd.DatetimeIndex(p.ts).equals(pd.date_range(first, last, freq="h")), "Validation SMP has gaps")
    p_hold = p_hold[p_hold.ts <= cut_final].reset_index(drop=True)
    weather = load_forecasts(root, "train")
    log("1/5 Fit initial model; select tree count using only the optimization period")
    initial = training_matrix(gen, sites, weather, prices, start, cut_opt)
    # Tuning labels also respect the cutoff of the later, untouched holdout.
    tune = training_matrix(gen, sites, weather, prices, opt_first, cut_hold)
    initial_model = fit_model(initial, config, validation=tune)
    rounds = int(initial_model.best_iteration_ or config.trees)
    manifest = {"config": asdict(config), "best_iteration": rounds,
                "initial_fit": initial[3], "tree_selection_labels": tune[3],
                "test_start": str(test_prices.ts.iloc[0]), "final_cutoff": str(cut_final)}
    del initial, tune
    gc.collect()
    # Optimization residuals use only labels available before holdout starts.
    p_group = p_opt[p_opt.ts <= cut_hold].reset_index(drop=True)
    opt_predictions = predict_sites(initial_model, sites, weather, pd.DatetimeIndex(p_group.ts), config.chunk_hours)
    a_opt = evaluation_arrays(opt_predictions, gen, sites, p_group)
    groups, search = optimize_groups(sites, a_opt, config)
    reference = (sites.current_group == "B").to_numpy(np.int8)
    report = {"group_search": search,
              "optimization_reference": score(a_opt, reference), "optimization_selected": score(a_opt, groups),
              "note": "Optimization-period results are tuned; only holdout is untouched. Scores use complete hours only."}
    del initial_model, opt_predictions, a_opt
    gc.collect()
    log("2/5 Refit before the holdout cutoff; evaluate the fixed allocation")
    hold_train = training_matrix(gen, sites, weather, prices, start, cut_hold)
    manifest["holdout_fit"] = hold_train[3]
    hold_model = fit_model(hold_train, config, rounds=rounds)
    del hold_train
    hold_predictions = predict_sites(hold_model, sites, weather, pd.DatetimeIndex(p_hold.ts), config.chunk_hours)
    a_hold = evaluation_arrays(hold_predictions, gen, sites, p_hold)
    report["holdout_reference"] = score(a_hold, reference)
    report["holdout_selected"] = score(a_hold, groups)
    # Never choose groups or hyperparameters using the holdout results.
    hold_predictions.to_parquet(output / "holdout_site_predictions.parquet", index=False)
    del hold_model, hold_predictions, a_hold
    gc.collect()
    log("3/5 Fit final model using labels available by the first evaluation forecast cutoff")
    final_train = training_matrix(gen, sites, weather, prices, start, cut_final)
    manifest["final_fit"] = final_train[3]
    final_model = fit_model(final_train, config, rounds=rounds)
    manifest["model_artifact"] = final_model.save(output)
    manifest["model_artifact"]["fingerprint"] = fingerprint(output / final_model.filename)
    importance = final_model.importance()
    if importance is not None:
        importance.to_csv(output / "feature_importance.csv", index=False)
    del final_train, gen, weather
    gc.collect()
    log("4/5 Predict all scoring hours using both forecast rounds; write and validate submission")
    test_weather = load_forecasts(root, "test")
    test_predictions = predict_sites(final_model, sites, test_weather, pd.DatetimeIndex(test_prices.ts), config.chunk_hours)
    test_predictions.to_parquet(output / "site_predictions.parquet", index=False)
    report["submission_validation"] = save_submission(root, output, sites, groups, test_predictions)
    log("5/5 Save audit trail, local validation scores and input hashes")
    manifest["input_files"] = {name: fingerprint(root / name) for name in REQUIRED}
    manifest["code"] = fingerprint(Path(__file__))
    manifest["model_backend_code"] = fingerprint(Path(backends.__file__))
    manifest["python"] = sys.version
    manifest["packages"] = {name: importlib.metadata.version(name) for name in
                            sorted({"numpy", "pandas", "pyarrow", "scikit-learn", *final_model.packages})}
    manifest["features"] = final_model.feature_name_
    manifest["prohibited_inputs_used"] = []
    manifest["holdout_artifacts"] = {name: fingerprint(output / name) for name in
                                     ["composition.csv", "holdout_site_predictions.parquet"]}
    for name, value in [("manifest.json", manifest), ("validation_report.json", report)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    log(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["audit", "run", "validate", "predict", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("data"))
        if name in ["run", "validate", "predict", "evaluate"]:
            p.add_argument("--output", type=Path, default=Path("outputs"))
        if name == "evaluate":
            p.add_argument("--report", type=Path, help="Write evaluation JSON to a new file (never overwrite)")
        if name == "predict":
            p.add_argument("--model-dir", type=Path, required=True)
            p.add_argument("--chunk-hours", type=int, default=168)
        if name == "run":
            for key, value in asdict(Config()).items():
                options = {"choices": backends.MODEL_NAMES} if key == "model" else {}
                p.add_argument("--" + key.replace("_", "-"), type=type(value), default=value, **options)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            result = audit(args.data)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2 if result["missing_files"] else 0
        if args.command == "validate":
            print(json.dumps(validate_submission(args.data, args.output), indent=2))
            return 0
        if args.command == "predict":
            print(json.dumps(predict_saved(args.data, args.model_dir, args.output, args.chunk_hours), indent=2))
            return 0
        if args.command == "evaluate":
            if args.report is not None:
                require(not args.report.exists(), "Report already exists; choose a new filename")
            result = evaluate_saved(args.data, args.output)
            print_evaluation(result)
            if args.report is not None:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                with args.report.open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                print(f"Saved {args.report}")
            return 0
        config = Config(**{key: getattr(args, key) for key in asdict(Config())})
        for key in ["trees", "early_stopping", "num_leaves", "min_child_samples", "threads", "chunk_hours", "min_complete_hours", "max_depth"]:
            require(getattr(config, key) > 0, f"{key} must be positive")
        require(np.isfinite(config.learning_rate) and config.learning_rate > 0, "learning_rate must be finite and positive")
        require(config.search_steps >= 0, "search_steps must be nonnegative")
        run_pipeline(args.data, args.output, config)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
