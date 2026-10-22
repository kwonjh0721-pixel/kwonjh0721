import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claude_validation import common, eval_val, make_val, model


@pytest.fixture
def sample(tmp_path, monkeypatch):
    start = pd.Timestamp("2024-06-01 01:00", tz=common.TZ)
    end = pd.Timestamp("2024-06-03 00:00", tz=common.TZ)
    monkeypatch.setattr(common, "SITE_COUNT", 3)
    for module in [common, make_val, eval_val]:
        monkeypatch.setattr(module, "VAL_TO", end)
    for module in [common, make_val]:
        monkeypatch.setattr(module, "VAL_END", end + pd.Timedelta(hours=1))
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    sites = pd.DataFrame({"site_id": ["PV001", "PV002", "PV003"],
                          "current_group": ["A", "B", "B"], "capacity_kw": [100., 150., 200.],
                          "lat": [35., 36., 37.], "lon": [127., 128., 129.],
                          "wa_station": ["WA001"]*3, "wb_station": ["WB001"]*3,
                          "wa_dist": [1000.]*3, "wb_dist": [2000.]*3})
    sites.to_csv(root / "sites.csv", index=False)
    times = pd.date_range(pd.Timestamp("2024-05-28 01:00", tz=common.TZ), end, freq="h")
    for provider, cols in [("wa", model.WA_COLS), ("wb", model.WB_COLS)]:
        f = pd.MultiIndex.from_product([[provider.upper() + "001"], [10, 17], times],
                                      names=[provider + "_station", "round", "ts"]).to_frame(index=False)
        for i, col in enumerate(cols):
            f[col] = i + 1. + pd.DatetimeIndex(f.ts).hour.to_numpy() * 2
        f.to_parquet(root / "train" / f"forecast_{provider}.parquet", index=False)
    gen = pd.MultiIndex.from_product([sites.site_id, times], names=["site_id", "ts"]).to_frame(index=False)
    gen["energy"] = np.maximum(np.sin((pd.DatetimeIndex(gen.ts).hour.to_numpy() - 6) * np.pi/12), 0)*50
    gen.to_parquet(root / "train/generation.parquet", index=False)
    return root, sites, gen, start, end


def test_naive_source_is_kst_and_csv_requires_timezone():
    t = common.to_kst(pd.Series(["2024-06-01 00:00:00"]))
    assert t.iloc[0] == pd.Timestamp("2024-06-01 00:00", tz=common.TZ)
    with pytest.raises(ValueError, match="timezone"):
        common.to_kst(pd.Series(["2024-06-01 00:00:00"]), require_timezone=True)


def test_stats_reject_validation_labels_and_zero_division(sample):
    _, sites, gen, start, _ = sample
    with pytest.raises(ValueError, match="validation"):
        model.site_stats_from(gen, sites)
    before = gen[gen.ts < start].copy()
    before.loc[before.ts.dt.hour.between(14, 17), "energy"] = 0.
    stats = model.site_stats_from(before, sites)
    assert not np.isinf(stats.select_dtypes(include="number").to_numpy()).any()


def test_parquet_boundary_with_and_without_timezone(sample, tmp_path):
    root, _, gen, start, _ = sample
    path = root / "train/generation.parquet"
    tr = common.read_window(path, ["site_id", "ts", "energy"], model.TRAIN_FROM, start)
    assert tr.ts.max() == start - pd.Timedelta(hours=1)
    naive = gen.copy()
    naive["ts"] = naive.ts.dt.tz_localize(None)
    naive.to_parquet(tmp_path / "naive.parquet", index=False)
    other = common.read_window(tmp_path / "naive.parquet", list(naive), model.TRAIN_FROM, start)
    pd.testing.assert_frame_equal(tr, other)


def test_daymean_is_independent_of_label_coverage_and_chunks(sample):
    root, sites, gen, start, end = sample
    wa, wb = make_val.weather_window(root, model.TRAIN_FROM, start)
    stats = model.site_stats_from(gen[gen.ts < start], sites)
    context = model.daily_context(sites, wa, wb)
    base = gen[gen.ts < start][["site_id", "ts"]]
    full = model.build(base, sites, wa, wb, stats, context)
    sparse = model.build(base.iloc[::3], sites, wa, wb, stats, context)
    columns = ["site_id", "ts", "wa_cloud_daymean", "wb_cloud_daymean", "wb_precip_prob_daymean"]
    expected = full[columns].merge(base.iloc[::3], on=["site_id", "ts"])
    pd.testing.assert_frame_equal(sparse[columns].reset_index(drop=True), expected.reset_index(drop=True))
    # Altering validation forecasts cannot change a filtered training feature.
    path = root / "train/forecast_wa.parquet"
    f = pd.read_parquet(path)
    f.loc[f.ts >= start, "wa_cloud"] = 99999
    f.to_parquet(path, index=False)
    next_wa, next_wb = make_val.weather_window(root, model.TRAIN_FROM, start)
    pd.testing.assert_frame_equal(wa, next_wa)
    # Both providers entirely absent at one site remain NaN, without dropping keys.
    missing = sites.copy()
    missing.loc[0, ["wa_station", "wb_station"]] = ["WA999", "WB999"]
    ctx = model.daily_context(missing, wa, wb)
    built = model.build(base, missing, wa, wb, stats, ctx)
    assert built.loc[built.site_id == sites.site_id.iloc[0], "wa_cloud_daymean"].isna().all()
    missing["wa_station"], missing["wb_station"] = "WA999", "WB999"
    ctx = model.daily_context(missing, wa, wb)
    built = model.build(base, missing, wa, wb, stats, ctx)
    assert built.wa_cloud_daymean.isna().all()


def test_duplicate_forecast_is_rejected(sample):
    root, _, _, start, _ = sample
    path = root / "train/forecast_wa.parquet"
    f = pd.read_parquet(path)
    pd.concat([f, f.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        make_val.weather_window(root, model.TRAIN_FROM, start)


def test_original_score_parity_and_missing_price_rejection(sample):
    _, sites, gen, start, _ = sample
    spec = importlib.util.spec_from_file_location("original_team_score", Path(model.__file__).parent / "original/261022/eval_val.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    gen = gen[gen.ts >= start].copy()
    p = gen[["site_id", "ts"]].copy()
    p["kwh"] = gen.energy.to_numpy() * .9
    prices = pd.DataFrame({"ts": sorted(gen.ts.unique()), "smp": 100.})
    comp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    a = original.score(p, comp, gen, prices)
    b = eval_val.score(p, comp, gen, prices)
    for key in a:
        assert a[key] == pytest.approx(b[key])
    with pytest.raises(ValueError, match="SMP"):
        eval_val.score(p, comp, gen, prices.iloc[1:])
    prices["smp"] = 0
    assert eval_val.score(p, comp, gen, prices)["wWAPE"] is None


def test_end_to_end_only_one_csv_and_holdout_target_invariance(sample, tmp_path):
    root, sites, gen, start, end = sample
    options = dict(threads=1, params={"n_estimators": 5, "min_child_samples": 2, "num_leaves": 5})
    out = tmp_path / "first" / "prediction.csv"
    result = make_val.run(root, out, chunk_days=1, **options)
    assert result["rows"] == 3*48
    assert list(out.parent.iterdir()) == [out]
    before = eval_val.load_prediction(out, sites)
    assert before.ts.min() == start and before.ts.max() == end
    # No test/ directory or SMP is required, and future generation has no effect.
    gen.loc[gen.ts >= start, "energy"] = 1e12
    gen.to_parquet(root / "train/generation.parquet", index=False)
    out2 = tmp_path / "second" / "prediction.csv"
    make_val.run(root, out2, chunk_days=2, **options)
    after = eval_val.load_prediction(out2, sites)
    keys = ["site_id", "ts"]
    pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True), after.sort_values(keys).reset_index(drop=True))
    with pytest.raises(ValueError, match="duplicate"):
        common.validate_predictions(pd.concat([before, before.iloc[:1]]), sites)
    with pytest.raises(ValueError, match="rows"):
        common.validate_predictions(before.iloc[1:], sites)
    bad = before.copy()
    bad.loc[0, "kwh"] = -1
    with pytest.raises(ValueError, match="negative"):
        common.validate_predictions(bad, sites)
