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
