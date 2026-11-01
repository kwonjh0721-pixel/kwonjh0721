"""Interchangeable CPU regressors for the common OIBC feature/validation pipeline.

Optional libraries are imported only when their backend is selected. Every backend
uses sample weights and the caller's chronological validation split. Iteration
counts in this interface are one-based counts, never library-specific indices.
"""
from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd


def dependency(module, requirements):
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as error:
        raise ValueError(
            f"Cannot load {module}. Install with: python -m pip install -r {requirements}. "
            f"Also check native runtime libraries (macOS: libomp). Original error: {error}"
        ) from error


class RegressionModel:
    """Backend contract: fit, predict, save/load and optional native importance."""

    name = ""
    filename = ""
    module = ""
    requirements = "requirements.txt"
    packages = ()
    importance_type = None

    def __init__(self, estimator, features, iterations, threads, categories=None):
        self.estimator = estimator
        self.feature_name_ = list(features)
        self.best_iteration_ = int(iterations)
        self.threads = int(threads)
        self.categories = list(categories) if categories is not None else []

    def prepare(self, x):
        if list(x.columns) != self.feature_name_:
            raise ValueError("Model/feature schema mismatch")
        return x

    def predict(self, x):
        return self.estimator.predict(self.prepare(x))

    def metadata(self):
        return {"version": 1, "backend": self.name, "file": self.filename,
                "features": self.feature_name_, "iterations": self.best_iteration_,
                "threads": self.threads, "site_categories": self.categories,
                "importance_type": self.importance_type,
                "package_versions": {name: importlib.metadata.version(name) for name in self.packages}}

    def importance(self):
        return None


class LightGBMModel(RegressionModel):
    name, filename, module = "lightgbm", "model.txt", "lightgbm"
    packages, importance_type = ("lightgbm",), "gain"

    @classmethod
    def fit(cls, train, config, *, rounds=None, validation=None):
        lgb = dependency(cls.module, cls.requirements)
        x, y, w, _ = train
        model = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=rounds or config.trees,
            learning_rate=config.learning_rate, num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples,
            colsample_bytree=.9, subsample=.85, subsample_freq=1,
            reg_lambda=2, max_bin=127, random_state=config.seed,
            n_jobs=config.threads, deterministic=True, force_col_wise=True, verbosity=-1)
        kwargs = {}
        if validation is not None:
            vx, vy, vw, _ = validation
            kwargs.update(eval_set=[(vx, vy)], eval_sample_weight=[vw], eval_metric="l1",
                          callbacks=[lgb.early_stopping(config.early_stopping, verbose=False)])
        model.fit(x, y, sample_weight=w, categorical_feature=["site_code"], **kwargs)
        return cls(model, x.columns, model.best_iteration_ or rounds or config.trees, config.threads)

    def booster(self):
        return self.estimator.booster_ if hasattr(self.estimator, "booster_") else self.estimator

    def save(self, directory):
        self.booster().save_model(str(Path(directory) / self.filename))
        return self.metadata()

    @classmethod
    def load(cls, directory, meta):
        lgb = dependency(cls.module, cls.requirements)
        model = lgb.Booster(model_file=str(Path(directory) / cls.filename))
        return cls(model, model.feature_name(), meta["iterations"], meta["threads"])

    def importance(self):
        # Preserve the existing LightGBM CSV schema and raw gain values.
        return pd.DataFrame({"feature": self.feature_name_,
                             "gain": self.booster().feature_importance(importance_type="gain")}).sort_values(
                                 "gain", ascending=False)


class CatBoostModel(RegressionModel):
    name, filename, module = "catboost", "model.cbm", "catboost"
    requirements = "requirements-catboost.txt"
    packages, importance_type = ("catboost",), "PredictionValuesChange"

    def prepare(self, x):
        x = super().prepare(x).copy()
        # CatBoost categorical columns must contain strings/integers, not float32.
        x["site_code"] = x.site_code.astype("int64").astype(str)
        return x

    @classmethod
    def fit(cls, train, config, *, rounds=None, validation=None):
        cb = dependency(cls.module, cls.requirements)
        x, y, w, _ = train
        if config.max_depth > 16:
            raise ValueError("CatBoost max_depth must be <= 16")
        model = cb.CatBoostRegressor(
            iterations=rounds or config.trees, loss_function="MAE", eval_metric="MAE",
            # Explicit weighted gradient leaf updates; guarded by the weighted
            # regression test (Exact behaved incorrectly on that case in 1.2.8).
            leaf_estimation_method="Gradient",
            learning_rate=config.learning_rate, depth=config.max_depth, l2_leaf_reg=2,
            border_count=127, bootstrap_type="Bernoulli", subsample=.85, rsm=.9,
            random_seed=config.seed, thread_count=config.threads, task_type="CPU",
            allow_writing_files=False, verbose=False)
        fitted = cls(model, x.columns, rounds or config.trees, config.threads)
        pool = cb.Pool(fitted.prepare(x), label=y, weight=w, cat_features=["site_code"])
        kwargs = {"use_best_model": False}
        if validation is not None:
            vx, vy, vw, _ = validation
            kwargs.update(eval_set=cb.Pool(fitted.prepare(vx), label=vy, weight=vw, cat_features=["site_code"]),
                          early_stopping_rounds=config.early_stopping, use_best_model=True)
        model.fit(pool, **kwargs)
        fitted.best_iteration_ = int(model.tree_count_)
        return fitted

    def predict(self, x):
        return self.estimator.predict(self.prepare(x), thread_count=self.threads)

    def save(self, directory):
        self.estimator.save_model(str(Path(directory) / self.filename), format="cbm")
        return dict(self.metadata(), leaf_estimation_method="Gradient")

    @classmethod
    def load(cls, directory, meta):
        cb = dependency(cls.module, cls.requirements)
        model = cb.CatBoostRegressor()
        model.load_model(str(Path(directory) / cls.filename), format="cbm")
        return cls(model, model.feature_names_, meta["iterations"], meta["threads"])

    def importance(self):
        return pd.DataFrame({"feature": self.feature_name_,
                             "importance": self.estimator.get_feature_importance(type=self.importance_type)}).sort_values(
                                 "importance", ascending=False)


class XGBoostModel(RegressionModel):
    name, filename, module = "xgboost", "model.ubj", "xgboost"
    requirements = "requirements-xgboost.txt"
    packages, importance_type = ("xgboost",), "gain"

    def prepare(self, x):
        x = super().prepare(x).copy()
        # Persist the training category dictionary: batches must never re-code IDs.
        # Sites with no training labels become missing categories, not new codes.
        x["site_code"] = pd.Categorical(x.site_code, categories=self.categories)
        return x

    @classmethod
    def fit(cls, train, config, *, rounds=None, validation=None):
        xgb = dependency(cls.module, cls.requirements)
        x, y, w, _ = train
        params = dict(objective="reg:absoluteerror", eval_metric="mae", tree_method="hist", device="cpu",
                      learning_rate=config.learning_rate, max_depth=config.max_depth,
                      colsample_bytree=.9, subsample=.85, reg_lambda=2, max_bin=127,
                      seed=config.seed, nthread=config.threads, verbosity=0)
        categories = sorted(x.site_code.dropna().unique().tolist())
        fitted = cls(None, x.columns, rounds or config.trees, config.threads, categories)
        data = xgb.DMatrix(fitted.prepare(x), label=y, weight=w,
                           enable_categorical=True, nthread=config.threads)
        kwargs = {}
        if validation is not None:
            vx, vy, vw, _ = validation
            valid = xgb.DMatrix(fitted.prepare(vx), label=vy, weight=vw,
                                enable_categorical=True, nthread=config.threads)
            kwargs.update(evals=[(valid, "validation")], early_stopping_rounds=config.early_stopping)
        # Native API avoids sklearn-wrapper compatibility affecting other models.
        model = xgb.train(params, data, num_boost_round=rounds or config.trees, verbose_eval=False, **kwargs)
        fitted.estimator = model
        if validation is not None:
            fitted.best_iteration_ = int(model.best_iteration) + 1
        return fitted

    def predict(self, x):
        xgb = dependency(self.module, self.requirements)
        data = xgb.DMatrix(self.prepare(x), enable_categorical=True, nthread=self.threads)
        return self.estimator.predict(data, iteration_range=(0, self.best_iteration_))

    def save(self, directory):
        self.estimator.save_model(str(Path(directory) / self.filename))
        return self.metadata()

    @classmethod
    def load(cls, directory, meta):
        xgb = dependency(cls.module, cls.requirements)
        model = xgb.Booster(params={"nthread": meta["threads"], "device": "cpu"})
        model.load_model(str(Path(directory) / cls.filename))
        return cls(model, model.feature_names, meta["iterations"], meta["threads"],
                   meta["site_categories"])

    def importance(self):
        values = self.estimator.get_score(importance_type="gain")
        return pd.DataFrame({"feature": self.feature_name_,
                             "importance": [values.get(f, 0.) for f in self.feature_name_]}).sort_values(
                                 "importance", ascending=False)


class HistGBModel(RegressionModel):
    name, filename, module = "histgb", "model.joblib", "sklearn.ensemble"
    packages = ("scikit-learn", "joblib", "threadpoolctl")

    @classmethod
    def fit(cls, train, config, *, rounds=None, validation=None):
        ensemble = dependency(cls.module, cls.requirements)
        from threadpoolctl import threadpool_limits

        x, y, w, _ = train
        params = dict(loss="absolute_error", learning_rate=config.learning_rate,
                      max_iter=rounds or config.trees, max_leaf_nodes=config.num_leaves,
                      min_samples_leaf=config.min_child_samples, l2_regularization=2,
                      categorical_features=["site_code"], max_bins=255,
                      early_stopping=validation is not None, validation_fraction=None,
                      n_iter_no_change=config.early_stopping, scoring="loss", tol=0,
                      random_state=config.seed)
        model = ensemble.HistGradientBoostingRegressor(**params)
        kwargs = {}
        if validation is not None:
            vx, vy, vw, _ = validation
            kwargs.update(X_val=vx, y_val=vy, sample_weight_val=vw)
        with threadpool_limits(limits=config.threads):
            model.fit(x, y, sample_weight=w, **kwargs)
            iterations = int(model.n_iter_)
            if validation is not None:
                # sklearn retains the last ensemble, not the best one. Refit the
                # initial training set at the best positive iteration count.
                iterations = int(np.argmax(model.validation_score_[1:])) + 1
                if iterations != model.n_iter_:
                    params.update(max_iter=iterations, early_stopping=False)
                    model = ensemble.HistGradientBoostingRegressor(**params)
                    model.fit(x, y, sample_weight=w)
        return cls(model, x.columns, iterations, config.threads)

    def predict(self, x):
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=self.threads):
            return self.estimator.predict(self.prepare(x))

    def save(self, directory):
        import joblib

        joblib.dump(self.estimator, Path(directory) / self.filename)
        return self.metadata()

    @classmethod
    def load(cls, directory, meta):
        dependency(cls.module, cls.requirements)
        import joblib

        expected = meta["package_versions"]["scikit-learn"]
        if importlib.metadata.version("scikit-learn") != expected:
            raise ValueError(f"HistGB loading requires the training scikit-learn version: {expected}")
        model = joblib.load(Path(directory) / cls.filename)
        return cls(model, model.feature_names_in_, meta["iterations"], meta["threads"])


BACKENDS = {cls.name: cls for cls in [LightGBMModel, CatBoostModel, XGBoostModel, HistGBModel]}
MODEL_NAMES = tuple(BACKENDS)


def backend_class(name):
    if name not in BACKENDS:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")
    return BACKENDS[name]


def ensure_available(name):
    cls = backend_class(name)
    dependency(cls.module, cls.requirements)


def fit_model(train, config, *, rounds=None, validation=None):
    return backend_class(config.model).fit(train, config, rounds=rounds, validation=validation)


def load_model(directory, manifest):
    meta = manifest.get("model_artifact")
    if meta is None:
        # Existing LightGBM model.txt + manifest.json remain usable unchanged.
        if manifest.get("config", {}).get("model", "lightgbm") != "lightgbm":
            raise ValueError("Missing model_artifact metadata for this backend")
        meta = {"backend": "lightgbm", "file": "model.txt", "version": 1,
                "features": manifest["features"], "iterations": manifest["best_iteration"],
                "threads": manifest.get("config", {}).get("threads", 4)}
    cls = backend_class(meta["backend"])
    if meta["version"] != 1 or meta["file"] != cls.filename:
        raise ValueError("Unsupported model artifact format")
    if meta["iterations"] < 1 or meta["threads"] < 1:
        raise ValueError("Invalid saved model iterations/threads")
    model = cls.load(directory, meta)
    if model.feature_name_ != meta["features"] or model.feature_name_ != manifest["features"]:
        raise ValueError("Saved model/manifest feature schema mismatch")
    return model
