"""수정한 Claude 모델의 학습·특징 함수. 실행 진입점은 make_val.py입니다.

원본의 LightGBM L1, capacity*k 가중치, 800 trees, SCALE=1.00을 유지합니다.
발전소 통계는 검증 시작 전 라벨로만 계산합니다. 일평균 특징은 실적 행과
무관하게 해당 대상일에 발행된 하루전 예보 전체로 계산합니다.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import lightgbm as lgb

if __package__:
    from .common import to_kst, unique, require
else:
    from common import to_kst, unique, require

# ----------------------------------------------------------------------------- 경로 / 설정
TZ = "Asia/Seoul"
SEED = 42
TRAIN_FROM = pd.Timestamp("2021-01-01 01:00", tz=TZ)      # 학습 시작 (전체 사용)
VAL_FROM = pd.Timestamp("2024-06-01 01:00", tz=TZ)        # 자체 검증 구간 시작
VAL_TO = pd.Timestamp("2025-06-01 00:00", tz=TZ)          # 자체 검증 구간 끝
SCALE = 1.00                                              # 최종 예측 배율 (검증 후 조정 예정)

LGB_PARAMS = dict(objective="l1", n_estimators=800, learning_rate=0.05, num_leaves=127,
                  min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=4,
                  deterministic=True, force_col_wise=True)

WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_snow", "wa_wind_speed", "wa_ground_press"]
WB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq",
           "wb_vis", "wb_wind_speed", "wb_dew_point", "wb_ceiling", "wb_snow", "wb_wind_gust_spd"]
R10_COLS = ["wa_cloud", "wb_cloud", "wb_uv_idx", "wb_precip_prob", "wa_rain", "wb_total_liq"]
FEATS = (WA_COLS + WB_COLS + [c + "_r10" for c in R10_COLS]
         + ["elev", "azim", "cosz", "elev_s", "elev_e", "doy_sin", "doy_cos", "hour", "cap", "lat", "lon",
            "wa_dist", "wb_dist", "cs", "cloud_mean", "cloud_diff", "cs_x_clr_a", "cs_x_clr_b", "cs_x_uv",
            "cloud_rev_a", "cloud_rev_b", "wa_cloud_daymean", "wb_cloud_daymean", "wb_precip_prob_daymean",
            "s_cf_q95", "s_cf_q99", "s_cf_mean", "s_am_ratio",
            "wa_missing_r17", "wa_missing_r10", "wb_missing_r17", "wb_missing_r10"])

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def prep_forecast(fa: pd.DataFrame, fb: pd.DataFrame):
    """(관측지점, ts) 단위로 17시 예보를 기본값, 10시 예보 주요 변수는 _r10 열로 추가.
    한쪽 라운드가 결측이면 다른 라운드 값으로 채움 (평가구간 결측 264행 대응)."""
    out = []
    for f, st, cols in ((fa, "wa_station", WA_COLS), (fb, "wb_station", WB_COLS)):
        f = f[[st, "round", "ts"] + cols].copy()
        unique(f, [st, "round", "ts"], st + " forecast")
        require(f["round"].isin([10, 17]).all(), "Unsupported forecast round")
        f[st] = f[st].astype(str)
        for col in cols:
            f[col] = pd.to_numeric(f[col], errors="raise").astype("float32").replace([np.inf, -np.inf], np.nan)
        for col in ["wb_wind_speed", "wb_wind_gust_spd"]:
            if col in f:
                f[col] /= 3.6
        r17 = f[f["round"] == 17].drop(columns="round").set_index([st, "ts"])
        r10 = f[f["round"] == 10].drop(columns="round").set_index([st, "ts"])
        idx = r17.index.union(r10.index)
        r17, r10 = r17.reindex(idx), r10.reindex(idx)
        w = r17.fillna(r10)
        provider = st.split("_")[0]
        w[provider + "_missing_r17"] = r17[cols].isna().all(axis=1).astype("float32")
        w[provider + "_missing_r10"] = r10[cols].isna().all(axis=1).astype("float32")
        k10 = [c for c in cols if c in R10_COLS]
        w = w.join(r10[k10].fillna(r17[k10]).add_suffix("_r10"))
        out.append(w.reset_index())
    return out[0], out[1]


# ----------------------------------------------------------------------------- 특징 생성
def solar(ts: pd.DatetimeIndex, lat, lon):
    """태양 고도·방위 (NOAA 근사). ts 는 우측닫힘이므로 구간 중앙(-30분) 기준."""
    ts = ts - pd.Timedelta(minutes=30)
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * (lon - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cosz = np.clip(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha), -1, 1)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(np.arctan2(-np.sin(ha), np.tan(decl) * np.cos(la) - np.sin(la) * np.cos(ha))) % 360
    return elev.astype("float32"), az.astype("float32"), cosz.astype("float32")


def k_of(ts) -> np.ndarray:
    h = pd.DatetimeIndex(ts).hour
    return np.where((h >= 11) & (h <= 15), 3.0, 1.0)


def site_stats_from(gen: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """발전소별 과거 설비이용률 특성 (학습 구간 데이터로만 계산)."""
    require((gen.ts < VAL_FROM).all(), "Site statistics cannot use validation labels")
    cap = sites.set_index("site_id").capacity_kw
    g = gen[["site_id", "ts", "energy"]].copy()
    g["cf"] = g.energy / g.site_id.map(cap).astype(float)
    h = g.ts.dt.hour
    st = pd.DataFrame({
        "s_cf_q95": g.groupby("site_id").cf.quantile(0.95),
        "s_cf_q99": g.groupby("site_id").cf.quantile(0.99),
        "s_cf_mean": g.groupby("site_id").cf.mean(),
        "s_am_ratio": g[h.between(8, 11)].groupby("site_id").cf.mean()
                      / g[h.between(14, 17)].groupby("site_id").cf.mean(),
    }).astype("float32")
    return st.replace([np.inf, -np.inf], np.nan).reset_index()


def daily_context(sites, wa, wb):
    """Full issued daily forecasts, independent of generation-row availability.

    All hours in one target day belong to its already-issued day-ahead forecast.
    In particular, following midnight belongs to the preceding operating day.
    """
    parts = []
    station_tables = [{key: rows for key, rows in table.groupby(st)}
                      for table, st in [(wa, "wa_station"), (wb, "wb_station")]]
    for site in sites.itertuples(index=False):
        daily = []
        for tables, station, cols in [
            (station_tables[0], site.wa_station, ["wa_cloud"]),
            (station_tables[1], site.wb_station, ["wb_cloud", "wb_precip_prob"]),
        ]:
            table = tables.get(station)
            if table is None or table.empty:
                daily.append(pd.DataFrame(columns=["_day"] + [c + "_daymean" for c in cols]).set_index("_day"))
                continue
            elevation, _, _ = solar(pd.DatetimeIndex(table.ts), site.lat, site.lon)
            v = table[["ts"] + cols].copy()
            v["_day"] = (v.ts - pd.Timedelta(hours=1)).dt.normalize()
            mean = v.loc[elevation > 0].groupby("_day")[cols].mean().add_suffix("_daymean")
            daily.append(mean)
        joined = daily[0].join(daily[1], how="outer").reset_index()
        joined["site_id"] = site.site_id
        if not joined.empty:
            for col in ["wa_cloud_daymean", "wb_cloud_daymean", "wb_precip_prob_daymean"]:
                joined[col] = pd.to_numeric(joined[col]).astype("float32")
            parts.append(joined)
    if not parts:
        return pd.DataFrame({"site_id": pd.Series(dtype="object"),
                             "_day": pd.Series(dtype="datetime64[ns, Asia/Seoul]"),
                             **{c: pd.Series(dtype="float32") for c in
                                ["wa_cloud_daymean", "wb_cloud_daymean", "wb_precip_prob_daymean"]}})
    result = pd.concat(parts, ignore_index=True)
    # Empty-context keys still need a timezone-aware dtype for safe merges.
    result["_day"] = pd.to_datetime(result["_day"], utc=True).dt.tz_convert(TZ)
    unique(result, ["site_id", "_day"], "daily forecast context")
    return result


def build(base: pd.DataFrame, sites: pd.DataFrame, wa: pd.DataFrame, wb: pd.DataFrame,
          stats: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    d = base.merge(sites[["site_id", "capacity_kw", "lat", "lon", "wa_station", "wb_station",
                          "wa_dist", "wb_dist"]], on="site_id", how="left", validate="many_to_one")
    require(d.capacity_kw.notna().all(), "Unknown site")
    d = d.merge(wa, on=["wa_station", "ts"], how="left", validate="many_to_one").merge(
        wb, on=["wb_station", "ts"], how="left", validate="many_to_one")
    for provider in ["wa", "wb"]:
        for round_ in [10, 17]:
            col = f"{provider}_missing_r{round_}"
            d[col] = d[col].fillna(1)
    ts = pd.DatetimeIndex(d.ts)
    lat, lon = d.lat.to_numpy(), d.lon.to_numpy()
    d["elev"], d["azim"], d["cosz"] = solar(ts, lat, lon)
    d["elev_s"], _, _ = solar(ts - pd.Timedelta(minutes=30), lat, lon)
    d["elev_e"], _, _ = solar(ts + pd.Timedelta(minutes=30), lat, lon)
    doy = ts.dayofyear.to_numpy()
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    d["hour"] = ts.hour.to_numpy().astype("int8")
    d["cap"] = d.capacity_kw.astype("float32")
    cs = np.clip(d.cosz.to_numpy(), 0, None)
    d["cs"] = cs.astype("float32")
    d["cloud_mean"] = ((d.wa_cloud + d.wb_cloud) / 2).astype("float32")
    d["cloud_diff"] = (d.wa_cloud - d.wb_cloud).astype("float32")
    d["cs_x_clr_a"] = (cs * (1 - d.wa_cloud / 100)).astype("float32")
    d["cs_x_clr_b"] = (cs * (1 - d.wb_cloud / 100)).astype("float32")
    d["cs_x_uv"] = (cs * d.wb_uv_idx).astype("float32")
    d["cloud_rev_a"] = (d.wa_cloud - d.wa_cloud_r10).astype("float32")
    d["cloud_rev_b"] = (d.wb_cloud - d.wb_cloud_r10).astype("float32")
    # 하루 단위 맥락: 그날 낮 시간 평균 운량·강수확률
    d["_day"] = (ts - pd.Timedelta(hours=1)).normalize()
    d = d.merge(context, on=["site_id", "_day"], how="left", validate="many_to_one")
    d = d.drop(columns="_day")
    return d.merge(stats, on="site_id", how="left", validate="many_to_one")


# ----------------------------------------------------------------------------- 학습 / 예측
def train_model(gen_tr: pd.DataFrame, sites, wa, wb, stats, context, params=None) -> lgb.LGBMRegressor:
    require((gen_tr.ts < VAL_FROM).all(), "Training labels overlap validation")
    require((wa.ts < VAL_FROM).all() and (wb.ts < VAL_FROM).all(), "Training forecasts overlap validation")
    d = build(gen_tr[["site_id", "ts", "energy"]], sites, wa, wb, stats, context)
    d = d[d.elev_e > -1]                                  # 해가 뜨는 시간대만 학습
    y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
    require(len(y) > 0 and np.isfinite(y).all(), "No valid daylight training labels")
    w = (d.cap * k_of(d.ts)).astype("float32")
    w = w / w.mean()
    X = d[FEATS].replace([np.inf, -np.inf], np.nan).astype("float32")
    log(f"  학습 행렬 {X.shape[0]:,}행 x {X.shape[1]}열")
    model = lgb.LGBMRegressor(**(params or LGB_PARAMS)).fit(X, y, sample_weight=w)
    return model


def predict_sites(model, site_ids, ts_index, sites, wa, wb, stats, context) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([list(site_ids), ts_index], names=["site_id", "ts"]).to_frame(index=False)
    v = build(grid, sites, wa, wb, stats, context)
    x = v[FEATS].replace([np.inf, -np.inf], np.nan).astype("float32")
    p = np.where(v.elev_e.to_numpy() > -1, model.predict(x), 0.0)
    out = v[["site_id", "ts"]].copy()
    out["kwh"] = np.clip(p, 0, None) * v.cap.to_numpy() * SCALE
    require(np.isfinite(out.kwh).all(), "Non-finite predictions")
    return out
