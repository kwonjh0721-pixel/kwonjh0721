"""Behavioral tests for optional regressors, weighted fitting and persistence."""
import json

import numpy as np
import pandas as pd
import pytest

import model_backends as b
import oibc2026 as o


@pytest.fixture
def matrices():
    rng = np.random.default_rng(42)
    x = pd.DataFrame({"site_code": np.tile([0., 2., 4.], 40), "cloud": rng.normal(size=120),
                      "missing": np.full(120, np.nan)}).astype("float32")
    x.loc[::6, "cloud"] = np.nan
    y = np.nan_to_num(x.cloud.to_numpy()) * .15 + .5
    w = np.linspace(.1, 3, len(x))
    return (x.iloc[:90].copy(), y[:90], w[:90], {}), (x.iloc[90:].copy(), y[90:], w[90:], {})


def require_backend(name):
    pytest.importorskip(b.backend_class(name).module.split(".")[0])


@pytest.mark.parametrize("name", b.MODEL_NAMES)
def test_roundtrip_missing_values_categories_and_batches(matrices, tmp_path, name):
    require_backend(name)
    train, validation = matrices
    before = train[0].copy(deep=True)
    config = o.Config(model=name, trees=8, early_stopping=2, threads=1, num_leaves=7,
                      min_child_samples=5, max_depth=3)
    fitted = o.fit_model(train, config, validation=validation)
    assert 1 <= fitted.best_iteration_ <= config.trees
    pd.testing.assert_frame_equal(before, train[0])
    x = validation[0].copy()
    x.loc[x.index[:2], "site_code"] = 99.  # Site absent from training labels.
    x.loc[x.index[:3], ["cloud", "missing"]] = np.nan
    pred = fitted.predict(x)
    assert np.isfinite(pred).all()
    pieces = [fitted.predict(x.iloc[i:i+7]) for i in range(0, len(x), 7)]
    np.testing.assert_allclose(pred, np.concatenate(pieces), rtol=0, atol=1e-12)
    if name == "xgboost":
        assert fitted.categories == [0., 2., 4.]
        assert fitted.prepare(x).site_code.iloc[:2].isna().all()
    meta = fitted.save(tmp_path)
    manifest = {"model_artifact": meta, "features": fitted.feature_name_}
    # Metadata must be JSON serializable, including persisted category values.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    restored = b.load_model(tmp_path, manifest)
    np.testing.assert_allclose(pred, restored.predict(x), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="schema mismatch"):
        restored.predict(x[x.columns[::-1]])
    # Subsequent refits use the selected count and do not need validation labels.
    refit = o.fit_model(train, config, rounds=fitted.best_iteration_)
    assert refit.best_iteration_ == fitted.best_iteration_


@pytest.mark.parametrize("name", b.MODEL_NAMES)
def test_training_sample_weights_change_predictions(name):
    require_backend(name)
    x = pd.DataFrame({"site_code": np.repeat(np.arange(3), 40),
                      "cloud": np.repeat(np.arange(60), 2)}).astype("float32")
    y = np.tile([.1, .9], 60)
    high = np.tile([1., 100.], 60)
    low = np.tile([100., 1.], 60)
    high /= high.mean()
    low /= low.mean()
    config = o.Config(model=name, trees=12, learning_rate=.2, threads=1,
                      num_leaves=7, min_child_samples=5, max_depth=3)
    a = o.fit_model((x, y, high, {}), config).predict(x)
    z = o.fit_model((x, y, low, {}), config).predict(x)
    assert float(np.mean(a) - np.mean(z)) > .5


def test_lightgbm_does_not_import_optional_backends(matrices, monkeypatch):
    original = b.importlib.import_module

    def without_optional(module):
        if module in {"catboost", "xgboost"}:
            raise AssertionError("Unselected optional backend was imported")
        return original(module)

    monkeypatch.setattr(b.importlib, "import_module", without_optional)
    model = o.fit_model(matrices[0], o.Config(trees=2, threads=1, min_child_samples=5))
    assert model.name == "lightgbm"


@pytest.mark.parametrize("name", ["catboost", "xgboost"])
def test_missing_optional_dependency_has_install_command(name, monkeypatch):
    original = b.importlib.import_module

    def unavailable(module):
        if module == name:
            raise ModuleNotFoundError(name)
        return original(module)

    monkeypatch.setattr(b.importlib, "import_module", unavailable)
    with pytest.raises(ValueError, match=f"requirements-{name}.txt"):
        b.ensure_available(name)


def test_unknown_model_rejected_by_cli(monkeypatch):
    monkeypatch.setattr("sys.argv", ["oibc2026.py", "run", "--model", "unknown"])
    with pytest.raises(SystemExit) as error:
        o.main()
    assert error.value.code == 2
