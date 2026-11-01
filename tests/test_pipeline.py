"""Synthetic tests verify code behavior, not competition accuracy."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oibc2026 as o


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    # Production CLI always checks 253 sites / 8,760 timestamps.
    monkeypatch.setattr(o, "SITE_COUNT", 6)
    monkeypatch.setattr(o, "SCORE_HOURS", 48)
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    sites = pd.DataFrame({
        "site_id": [f"PV{i:03d}" for i in range(1, 7)],
        "current_group": ["A"]*3 + ["B"]*3, "capacity_kw": [7000.]*6,
        "lat": np.linspace(34.5, 36.5, 6), "lon": np.linspace(127., 129., 6),
        "wa_station": ["WA001", "WA001", "WA002", "WA002", "WA003", "WA003"],
        "wb_station": ["WB001", "WB001", "WB002", "WB002", "WB003", "WB003"],
        "wa_dist": [1000.]*6, "wb_dist": [2000.]*6})
    sites.to_csv(root / "sites.csv", index=False)
    train_times = pd.date_range("2025-01-01 01:00", "2025-01-14 00:00", freq="h", tz=o.TZ)
    test_times = pd.date_range("2025-01-14 01:00", periods=48, freq="h", tz=o.TZ)
    for split, times in [("train", train_times), ("test", test_times)]:
        prices = pd.DataFrame({"ts": times, "smp": 100 + times.hour.to_numpy()})
        if split == "test":
            prices["k"] = o.k_values(times)
        prices.to_csv(root / ("smp.csv" if split == "test" else "train/smp.csv"), index=False)
        for provider, cols in [("wa", o.WA), ("wb", o.WB)]:
            grid = pd.MultiIndex.from_product([sites[provider + "_station"].unique(), [10, 17], times],
                                             names=[provider + "_station", "round", "ts"]).to_frame(index=False)
            for i, col in enumerate(cols):
                grid[col] = (i+1)*2 + pd.DatetimeIndex(grid.ts).hour.to_numpy()*.1
            grid.to_parquet(root / split / f"forecast_{provider}.parquet", index=False)
    keys = pd.MultiIndex.from_product([sites.site_id, train_times], names=["site_id", "ts"]).to_frame(index=False)
    merged = keys.merge(sites, on="site_id")
    _, cosz = o.solar(merged.ts, merged.lat.to_numpy(), merged.lon.to_numpy())
    keys["energy"] = np.maximum(cosz, 0) * merged.capacity_kw.to_numpy() * .75
    keys.to_parquet(root / "train/generation.parquet", index=False)
    return root, sites, train_times, test_times


def test_midnight_cutoff():
    assert o.forecast_cutoff("2025-06-02 00:00+09:00") == o.stamp("2025-05-31 17:00")
    assert o.forecast_cutoff("2025-06-01 01:00+09:00") == o.stamp("2025-05-31 17:00")
    assert o.forecast_cutoff("2025-06-02 01:00+09:00") == o.stamp("2025-06-01 17:00")


def test_features_ignore_labels_and_missing_forecasts(tiny_data):
    root, sites, times, _ = tiny_data
    weather = o.load_forecasts(root, "train")
    key = pd.DataFrame({"site_id": [sites.site_id.iloc[0]], "ts": [times[10]], "energy": [1.]})
    x1 = o.features(key, sites, weather)
    key["energy"] = 999999.
    key["smp"] = 888888.
    pd.testing.assert_frame_equal(x1, o.features(key, sites, weather))
    # Missing round-17 stays NaN; round-10 remains available. No future fill.
    weather[("wa", 17)] = weather[("wa", 17)].query("ts != @times[10]").copy()
    x2 = o.features(key, sites, weather)
    assert np.isnan(x2.wa_cloud_r17.iloc[0])
    assert x2.wa_missing_r17.iloc[0] == 1
    assert np.isfinite(x2.wa_cloud_r10.iloc[0])
    assert not any("observed" in c or "energy" in c or "smp" in c for c in x2)


def test_future_forecasts_do_not_change_features(tiny_data):
    root, sites, times, _ = tiny_data
    weather = o.load_forecasts(root, "train")
    key = pd.DataFrame({"site_id": [sites.site_id.iloc[0]], "ts": [times[10]]})
    before = o.features(key, sites, weather)
    for table in weather.values():
        for col in table.select_dtypes(include="number"):
            table.loc[table.ts > times[10], col] = 12345
    pd.testing.assert_frame_equal(before, o.features(key, sites, weather))


def test_training_cutoff_and_missing_labels(tiny_data):
    root, sites, times, _ = tiny_data
    gen = pd.read_parquet(root / "train/generation.parquet")
    weather = o.load_forecasts(root, "train")
    prices = o.read_prices(root / "train/smp.csv")
    cutoff = o.stamp("2025-01-05 17:00")
    a = o.training_matrix(gen, sites, weather, prices, times[0], cutoff)
    gen.loc[gen.ts > cutoff, "energy"] = 1e15
    b = o.training_matrix(gen, sites, weather, prices, times[0], cutoff)
    pd.testing.assert_frame_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert o.stamp(a[3]["latest_label"]) <= cutoff
    partial = gen[(gen.site_id != sites.site_id.iloc[0]) | (gen.ts != times[12])]
    predictions = gen[["site_id", "ts"]].copy()
    predictions["forecast_kwh"] = 0.
    arr = o.evaluation_arrays(predictions, partial, sites, prices)
    assert arr["complete_hours"] == len(times)-1


def test_score_and_capacity_search(tiny_data):
    _, sites, _, _ = tiny_data
    residual = np.array([[10., 10., 10., -10., -10., -10.]] * 24)
    arr = {"residual": residual, "actual": np.full((24, 6), 100.),
           "smp": np.full(24, 100.), "weights": np.full(24, 300.),
           "complete_hours": 24, "total_hours": 24, "coverage": 1.}
    reference = np.array([0, 0, 0, 1, 1, 1])
    result = o.score(arr, reference)
    assert result["sales_won"] == 24*600*100
    assert result["penalty_won"] == pytest.approx(.1*24*60*300)
    groups, info = o.optimize_groups(sites, arr, o.Config(search_steps=200, min_complete_hours=1))
    assert min(o.validate_groups(sites, groups)) >= 20000
    assert o.weighted_error(arr, groups) < o.weighted_error(arr, reference)
    assert info["accepted_moves"] > 0
    with pytest.raises(ValueError, match="20,000"):
        o.validate_groups(sites, np.zeros(6))


def test_duplicate_forecasts_rejected(tiny_data):
    root, _, _, _ = tiny_data
    path = root / "train/forecast_wa.parquet"
    d = pd.read_parquet(path)
    pd.concat([d, d.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        o.load_forecasts(root, "train")


def test_full_pipeline_and_saved_model(tiny_data, tmp_path):
    import lightgbm as lgb
    root, sites, _, test_times = tiny_data
    out = tmp_path / "output"
    config = o.Config(train_from="2025-01-01", optimize_from="2025-01-08", holdout_from="2025-01-11",
                      trees=12, early_stopping=3, num_leaves=7, min_child_samples=5,
                      threads=1, chunk_hours=24, search_steps=20, min_complete_hours=12)
    report = o.run_pipeline(root, out, config)
    assert report["submission_validation"]["valid"]
    assert report["holdout_selected"]["complete_hours"] == 65
    rescored = o.evaluate_saved(root, out)
    for key in ["holdout_reference", "holdout_selected"]:
        assert rescored[key]["score_won"] == pytest.approx(report[key]["score_won"])
        assert rescored[key]["penalty_won"] == pytest.approx(report[key]["penalty_won"])
    assert not rescored["warnings"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert o.stamp(manifest["final_fit"]["latest_label"]) <= o.stamp("2025-01-13 17:00")
    assert o.stamp(manifest["initial_fit"]["latest_label"]) <= o.stamp("2025-01-07 17:00")
    assert o.stamp(manifest["tree_selection_labels"]["latest_label"]) <= o.stamp("2025-01-10 17:00")
    assert o.stamp(manifest["holdout_fit"]["latest_label"]) <= o.stamp("2025-01-10 17:00")
    assert not manifest["prohibited_inputs_used"]
    restored = lgb.Booster(model_file=str(out / "model.txt"))
    p = o.predict_sites(restored, sites, o.load_forecasts(root, "test"), test_times, 24)
    saved = pd.read_parquet(out / "site_predictions.parquet")
    hold_saved = pd.read_parquet(out / "holdout_site_predictions.parquet")
    assert hold_saved.ts.max() <= o.stamp("2025-01-13 17:00")
    np.testing.assert_allclose(p.forecast_kwh, saved.forecast_kwh)
    assert o.validate_submission(root, out)["forecast_rows"] == 96
    reproduced = tmp_path / "reproduced"
    assert o.predict_saved(root, out, reproduced, 24)["valid"]
    assert (out / "forecast.csv").read_bytes() == (reproduced / "forecast.csv").read_bytes()
    # Artifacts from before model selection was introduced still load as LightGBM.
    old_manifest = dict(manifest)
    old_manifest.pop("model_artifact")
    old_manifest["config"] = {k: v for k, v in manifest["config"].items()
                              if k not in ["model", "learning_rate", "max_depth"]}
    (out / "manifest.json").write_text(json.dumps(old_manifest))
    legacy = tmp_path / "legacy_reproduced"
    assert o.predict_saved(root, out, legacy, 24)["valid"]
    assert (out / "forecast.csv").read_bytes() == (legacy / "forecast.csv").read_bytes()
    f = pd.read_csv(out / "forecast.csv")
    f.loc[0, "forecast_kwh"] = -1
    f.to_csv(out / "forecast.csv", index=False)
    with pytest.raises(ValueError, match="Negative"):
        o.validate_submission(root, out)
    f.loc[0, "forecast_kwh"] = 0
    f["ts"] = pd.to_datetime(f.ts).dt.tz_localize(None)
    f.to_csv(out / "forecast.csv", index=False)
    with pytest.raises(ValueError, match="timezone"):
        o.validate_submission(root, out)


def test_submission_missing_site_or_timestamp_rejected(tiny_data, tmp_path):
    root, sites, _, times = tiny_data
    out = tmp_path / "submit"
    out.mkdir()
    keys = pd.MultiIndex.from_product([sites.site_id, times], names=["site_id", "ts"]).to_frame(index=False)
    keys["forecast_kwh"] = 1.
    groups = (sites.current_group == "B").to_numpy(np.int8)
    o.save_submission(root, out, sites, groups, keys)
    comp = pd.read_csv(out / "composition.csv")
    comp.iloc[:-1].to_csv(out / "composition.csv", index=False)
    with pytest.raises(ValueError, match="omission"):
        o.validate_submission(root, out)
    comp.to_csv(out / "composition.csv", index=False)
    forecast = pd.read_csv(out / "forecast.csv")
    forecast.iloc[:-1].to_csv(out / "forecast.csv", index=False)
    with pytest.raises(ValueError, match="number"):
        o.validate_submission(root, out)


@pytest.fixture
def saved_evaluation(tiny_data, tmp_path):
    """Known group errors: reference +/-90 kWh, selected +/-30 kWh each hour."""
    root, sites, _, test_times = tiny_data
    out = tmp_path / "saved_run"
    out.mkdir()
    first, last = o.stamp("2025-01-11 01:00"), o.stamp("2025-01-13 17:00")
    times = pd.date_range(first, last, freq="h")
    gen = pd.read_parquet(root / "train/generation.parquet")
    gen["energy"] = 100.
    gen.to_parquet(root / "train/generation.parquet", index=False)
    pred = pd.MultiIndex.from_product([sites.site_id, times], names=["site_id", "ts"]).to_frame(index=False)
    errors = dict(zip(sites.site_id, [30., 30., 30., -30., -30., -30.]))
    pred["forecast_kwh"] = 100 + pred.site_id.map(errors)
    pred.to_parquet(out / "holdout_site_predictions.parquet", index=False)
    pd.DataFrame({"site_id": sites.site_id, "group": ["G1", "G2", "G1", "G2", "G1", "G2"]}).to_csv(
        out / "composition.csv", index=False)
    manifest = {"config": {"holdout_from": "2025-01-11"}, "final_cutoff": str(last),
                "test_start": str(test_times[0]),
                "input_files": {name: o.fingerprint(root / name) for name in
                                ["sites.csv", "train/smp.csv", "train/generation.parquet"]},
                "holdout_artifacts": {name: o.fingerprint(out / name) for name in
                                      ["composition.csv", "holdout_site_predictions.parquet"]}}
    (out / "manifest.json").write_text(json.dumps(manifest))
    return root, out, manifest, times


def test_saved_evaluation_metrics_and_cli(saved_evaluation, monkeypatch, capsys):
    root, out, _, times = saved_evaluation
    # Re-scoring needs neither model files nor train/test weather nor evaluation SMP.
    for name in ["smp.csv", "train/forecast_wa.parquet", "train/forecast_wb.parquet",
                 "test/forecast_wa.parquet", "test/forecast_wb.parquet"]:
        (root / name).unlink()
    result = o.evaluate_saved(root, out)
    ref, chosen = result["holdout_reference"], result["holdout_selected"]
    weights = sum((100 + t.hour) * (3 if 11 <= t.hour <= 15 else 1) for t in times)
    sales = sum((100 + t.hour) * 600 for t in times)
    assert ref["penalty_won"] == pytest.approx(.1 * weights * 180)
    assert chosen["penalty_won"] == pytest.approx(.1 * weights * 60)
    assert chosen["score_won"] == pytest.approx(sales - .1 * weights * 60)
    assert chosen["group_mae_kwh"] == 30
    assert chosen["group_rmse_kwh"] == 30
    assert chosen["wwape"] == pytest.approx(.1)
    assert result["comparison"]["penalty_reduction_pct"] == pytest.approx(200 / 3)
    target = out / "holdout_evaluation.json"
    monkeypatch.setattr("sys.argv", ["oibc2026.py", "evaluate", "--data", str(root), "--output", str(out),
                                    "--report", str(target)])
    assert o.main() == 0
    assert json.loads(target.read_text()) == result
    assert "Complete actual hours: 65/65" in capsys.readouterr().out
    before = target.read_bytes()
    assert o.main() == 2
    assert "Report already exists" in capsys.readouterr().err
    assert target.read_bytes() == before


@pytest.mark.parametrize("corruption,expected", [
    ("missing_hour", "Incomplete holdout prediction grid"),
    ("duplicate", "duplicate"),
    ("negative", "Negative/non-finite"),
    ("nan", "Negative/non-finite"),
    ("outside", "outside the saved holdout"),
])
def test_saved_evaluation_rejects_invalid_predictions(saved_evaluation, corruption, expected):
    root, out, manifest, times = saved_evaluation
    path = out / "holdout_site_predictions.parquet"
    pred = pd.read_parquet(path)
    if corruption == "missing_hour":
        pred = pred[pred.ts != times[0]]
    elif corruption == "duplicate":
        pred = pd.concat([pred, pred.iloc[:1]])
    elif corruption == "outside":
        pred.loc[0, "ts"] = times[-1] + pd.Timedelta(hours=1)
    else:
        pred.loc[0, "forecast_kwh"] = -1 if corruption == "negative" else np.nan
    pred.to_parquet(path, index=False)
    # Exercise grid/value checks independently of the fingerprint check.
    manifest["holdout_artifacts"][path.name] = o.fingerprint(path)
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=expected):
        o.evaluate_saved(root, out)


@pytest.mark.parametrize("changed", ["sites.csv", "composition.csv", "holdout_site_predictions.parquet"])
def test_saved_evaluation_rejects_changed_files(saved_evaluation, changed):
    root, out, _, _ = saved_evaluation
    path = (root if changed == "sites.csv" else out) / changed
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="changed"):
        o.evaluate_saved(root, out)


def test_saved_evaluation_no_complete_actuals_and_legacy_manifest(saved_evaluation, capsys):
    root, out, manifest, _ = saved_evaluation
    path = root / "train/generation.parquet"
    gen = pd.read_parquet(path)
    gen = gen[gen.site_id != gen.site_id.iloc[0]]
    gen.to_parquet(path, index=False)
    manifest["input_files"]["train/generation.parquet"] = o.fingerprint(path)
    del manifest["holdout_artifacts"]
    (out / "manifest.json").write_text(json.dumps(manifest))
    (out / "validation_report.json").write_text(json.dumps({"group_search": {"fallback": "Insufficient complete hours"}}))
    result = o.evaluate_saved(root, out)
    assert result["holdout_selected"]["complete_hours"] == 0
    assert result["holdout_selected"]["score_won"] is None
    assert result["holdout_selected"]["wwape"] is None
    assert result["comparison"]["penalty_reduction_won"] is None
    assert any("Older run" in warning for warning in result["warnings"])
    assert any("retained reference" in warning for warning in result["warnings"])
    json.dumps(result, allow_nan=False)
    o.print_evaluation(result)
    assert "N/A" in capsys.readouterr().out


def test_saved_evaluation_zero_denominators(saved_evaluation):
    root, out, manifest, _ = saved_evaluation
    for path, column, hashes, name in [
        (root / "train/generation.parquet", "energy", manifest["input_files"], "train/generation.parquet"),
        (out / "holdout_site_predictions.parquet", "forecast_kwh", manifest["holdout_artifacts"],
         "holdout_site_predictions.parquet"),
    ]:
        table = pd.read_parquet(path)
        table[column] = 0.
        table.to_parquet(path, index=False)
        hashes[name] = o.fingerprint(path)
    (out / "manifest.json").write_text(json.dumps(manifest))
    result = o.evaluate_saved(root, out)
    assert result["holdout_selected"]["coverage"] == 1.
    assert result["holdout_selected"]["score_won"] == 0.
    assert result["holdout_selected"]["wwape"] is None
    assert result["comparison"]["penalty_reduction_won"] == 0.
    assert result["comparison"]["penalty_reduction_pct"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("backend,package,filename", [
    ("catboost", "catboost", "model.cbm"),
    ("xgboost", "xgboost", "model.ubj"),
    ("histgb", "sklearn", "model.joblib"),
])
def test_alternative_model_pipeline(tiny_data, tmp_path, backend, package, filename):
    pytest.importorskip(package)
    root, sites, _, times = tiny_data
    # All weather missing at one daytime hour must still produce every site row.
    for provider in ["wa", "wb"]:
        path = root / "test" / f"forecast_{provider}.parquet"
        table = pd.read_parquet(path)
        table[table.ts != times[11]].to_parquet(path, index=False)
    out = tmp_path / backend
    config = o.Config(model=backend, train_from="2025-01-01", optimize_from="2025-01-08",
                      holdout_from="2025-01-11", trees=8, early_stopping=2, num_leaves=7,
                      max_depth=3, min_child_samples=5, threads=1, chunk_hours=24,
                      search_steps=20, min_complete_hours=12)
    report = o.run_pipeline(root, out, config)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model_artifact"]["backend"] == backend
    assert manifest["model_artifact"]["file"] == filename
    assert manifest["best_iteration"] == manifest["model_artifact"]["iterations"]
    assert 1 <= manifest["best_iteration"] <= config.trees
    assert (out / filename).is_file()
    assert (out / "feature_importance.csv").exists() == (backend != "histgb")
    for fit in ["initial_fit", "tree_selection_labels", "holdout_fit", "final_fit"]:
        assert o.stamp(manifest[fit]["latest_label"]) <= o.stamp(manifest[fit]["allowed_end"])
    assert report["submission_validation"]["valid"]
    p = pd.read_parquet(out / "site_predictions.parquet")
    assert len(p) == len(sites) * len(times)
    assert np.isfinite(p.loc[p.ts == times[11], "forecast_kwh"]).all()
    assert o.evaluate_saved(root, out)["holdout_selected"]["score_won"] == pytest.approx(
        report["holdout_selected"]["score_won"])
    reproduced = tmp_path / f"{backend}_reproduced"
    assert o.predict_saved(root, out, reproduced, chunk_hours=7)["valid"]
    assert (out / "forecast.csv").read_bytes() == (reproduced / "forecast.csv").read_bytes()
    # Changed native model files must be rejected before deserialization.
    with (out / filename).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Saved model artifact changed"):
        o.predict_saved(root, out, tmp_path / f"{backend}_corrupted")
