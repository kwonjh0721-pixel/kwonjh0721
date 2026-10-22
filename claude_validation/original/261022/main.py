"""
2026 POSTECH OIBC Big Data Challenge - 태양광 집합자원 구성 + 하루전 발전량 예측
=============================================================================
전체 파이프라인 (데이터 전처리 -> 학습 -> 검증 -> 추론 -> 제출 파일 생성 -> 형식 검증)

실행:  python src/main.py              (검증 + 제출 파일 생성)
       python src/main.py --skip-val   (검증 생략, 제출 파일만 생성)

폴더 구조:
    OBIC2026/
    ├─ OIBC2026_dataset/   (주최측 데이터)
    ├─ src/main.py         (이 파일)
    ├─ submissions/        (결과 저장)
    └─ tools/validate_submission.py

채점식 (README):
    S = Σ SMP(t)·실적(t) − 0.10 × Σ_g Σ_t SMP(t)·k(t)·|예측_g(t) − 실적_g(t)|
    k(t) = 3 (ts 의 시가 11~15), 그 외 1

모델 요약:
    - 발전소별 설비이용률(발전량/설비용량)을 LightGBM 으로 예측 후 집합별 합산
    - 절댓값 오차 채점이므로 L1(중앙값) 목적함수 사용
    - 가중치 = 설비용량 × k  (큰 발전소, 벌점 3배 시간대에 집중)
    - 예보: 전일 17시 발행본을 기본, 10시 발행본은 보조 변수(둘 다 예측시점 이전 발행 -> 규칙 준수)
    - 외부 데이터 사용 없음, 평가 구간 실측 사용 없음
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

# ----------------------------------------------------------------------------- 경로 / 설정
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "OIBC2026_dataset")
OUT = os.path.join(BASE, "submissions")
VALIDATOR = os.path.join(BASE, "tools", "validate_submission.py")

TZ = "Asia/Seoul"
SEED = 42
TRAIN_FROM = pd.Timestamp("2021-01-01 01:00", tz=TZ)      # 학습 시작 (전체 사용)
VAL_FROM = pd.Timestamp("2024-06-01 01:00", tz=TZ)        # 자체 검증 구간 시작
VAL_TO = pd.Timestamp("2025-06-01 00:00", tz=TZ)          # 자체 검증 구간 끝
SCALE = 1.00                                              # 최종 예측 배율 (검증 후 조정 예정)

LGB_PARAMS = dict(objective="l1", n_estimators=800, learning_rate=0.05, num_leaves=127,
                  min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)

WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_snow", "wa_wind_speed", "wa_ground_press"]
WB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq",
           "wb_vis", "wb_wind_speed", "wb_dew_point", "wb_ceiling", "wb_snow", "wb_wind_gust_spd"]
R10_COLS = ["wa_cloud", "wb_cloud", "wb_uv_idx", "wb_precip_prob", "wa_rain", "wb_total_liq"]
FEATS = (WA_COLS + WB_COLS + [c + "_r10" for c in R10_COLS]
         + ["elev", "azim", "cosz", "elev_s", "elev_e", "doy_sin", "doy_cos", "hour", "cap", "lat", "lon",
            "wa_dist", "wb_dist", "cs", "cloud_mean", "cloud_diff", "cs_x_clr_a", "cs_x_clr_b", "cs_x_uv",
            "cloud_rev_a", "cloud_rev_b", "wa_cloud_daymean", "wb_cloud_daymean", "wb_precip_prob_daymean",
            "s_cf_q95", "s_cf_q99", "s_cf_mean", "s_am_ratio"])

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- 데이터 읽기
def to_kst(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, utc=True) if not pd.api.types.is_datetime64_any_dtype(s) else s
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(TZ)


def read_parquet(rel: str, columns=None) -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(DATA, *rel.split("/")), columns=columns)
    d["ts"] = to_kst(d["ts"])
    for c in d.columns:
        if d[c].dtype == "float64":
            d[c] = d[c].astype("float32")
    return d


def load_all():
    sites = pd.read_csv(os.path.join(DATA, "sites.csv"))
    smp_test = pd.read_csv(os.path.join(DATA, "smp.csv"))
    smp_test["ts"] = to_kst(smp_test["ts"])
    smp_train = pd.read_csv(os.path.join(DATA, "train", "smp.csv"))
    smp_train["ts"] = to_kst(smp_train["ts"])

    gen = read_parquet("train/generation.parquet")
    gen["site_id"] = gen["site_id"].astype(str)
    gen = gen[gen.energy >= 0]                      # 음수 이상치 40행 제거

    cols_a = ["wa_station", "round", "ts"] + WA_COLS
    cols_b = ["wb_station", "round", "ts"] + WB_COLS
    fa = pd.concat([read_parquet("train/forecast_wa.parquet", cols_a),
                    read_parquet("test/forecast_wa.parquet", cols_a)], ignore_index=True)
    fb = pd.concat([read_parquet("train/forecast_wb.parquet", cols_b),
                    read_parquet("test/forecast_wb.parquet", cols_b)], ignore_index=True)
    wa, wb = prep_forecast(fa, fb)
    return sites, smp_train, smp_test, gen, wa, wb


def prep_forecast(fa: pd.DataFrame, fb: pd.DataFrame):
    """(관측지점, ts) 단위로 17시 예보를 기본값, 10시 예보 주요 변수는 _r10 열로 추가.
    한쪽 라운드가 결측이면 다른 라운드 값으로 채움 (평가구간 결측 264행 대응)."""
    out = []
    for f, st, cols in ((fa, "wa_station", WA_COLS), (fb, "wb_station", WB_COLS)):
        f = f[[st, "round", "ts"] + cols].copy()
        f[st] = f[st].astype(str)
        r17 = f[f["round"] == 17].drop(columns="round").set_index([st, "ts"])
        r10 = f[f["round"] == 10].drop(columns="round").set_index([st, "ts"])
        idx = r17.index.union(r10.index)
        r17, r10 = r17.reindex(idx), r10.reindex(idx)
        w = r17.fillna(r10)
        k10 = [c for c in cols if c in R10_COLS]
        w = w.join(r10[k10].fillna(r17[k10]).add_suffix("_r10"))
        out.append(w.reset_index())
    return out[0], out[1]


# ----------------------------------------------------------------------------- 특징 생성
def solar(ts: pd.DatetimeIndex, lat, lon):
    """태양 고도·방위 (NOAA 근사). ts 는 우측닫힘이므로 구간 중앙(-30분) 기준."""
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = (hour - 0.5) * 60 + eqtime + 4 * (lon - 135.0)
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
    return st.reset_index()


def build(base: pd.DataFrame, sites: pd.DataFrame, wa: pd.DataFrame, wb: pd.DataFrame,
          stats: pd.DataFrame) -> pd.DataFrame:
    d = base.merge(sites[["site_id", "capacity_kw", "lat", "lon", "wa_station", "wb_station",
                          "wa_dist", "wb_dist"]], on="site_id", how="left")
    d = d.merge(wa, on=["wa_station", "ts"], how="left").merge(wb, on=["wb_station", "ts"], how="left")
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
    dl = d.elev > 0
    for c in ("wa_cloud", "wb_cloud", "wb_precip_prob"):
        m = d[dl].groupby(["site_id", "_day"])[c].mean().rename(c + "_daymean")
        d = d.merge(m.reset_index(), on=["site_id", "_day"], how="left")
    d = d.drop(columns="_day")
    return d.merge(stats, on="site_id", how="left")


# ----------------------------------------------------------------------------- 학습 / 예측
def train_model(gen_tr: pd.DataFrame, sites, wa, wb, stats) -> lgb.LGBMRegressor:
    d = build(gen_tr[["site_id", "ts", "energy"]], sites, wa, wb, stats)
    d = d[d.elev_e > -1]                                  # 해가 뜨는 시간대만 학습
    y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
    w = (d.cap * k_of(d.ts)).astype("float32")
    w = w / w.mean()
    X = d[FEATS].astype("float32")
    log(f"  학습 행렬 {X.shape[0]:,}행 x {X.shape[1]}열")
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y, sample_weight=w)
    return model


def predict_sites(model, site_ids, ts_index, sites, wa, wb, stats) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([list(site_ids), ts_index], names=["site_id", "ts"]).to_frame(index=False)
    v = build(grid, sites, wa, wb, stats)
    p = np.where(v.elev_e.to_numpy() > -1, model.predict(v[FEATS].astype("float32")), 0.0)
    out = v[["site_id", "ts"]].copy()
    out["kwh"] = np.clip(p, 0, None) * v.cap.to_numpy() * SCALE
    return out


# ----------------------------------------------------------------------------- 자체 채점
def score(pred, comp, gen, smp) -> dict:
    """README 채점식. 실적이 있는 (발전소, 시각)만 집합별로 합산."""
    a = gen[["site_id", "ts", "energy"]].merge(pred, on=["site_id", "ts"], how="inner")
    a["group"] = a.site_id.map(comp)
    g = a.groupby(["group", "ts"], as_index=False)[["energy", "kwh"]].sum()
    g = g.merge(smp[["ts", "smp"]], on="ts", how="inner")
    g["k"] = k_of(g.ts)
    pen = g.smp * g.k * (g.kwh - g.energy).abs()
    rev = float((g.smp * g.energy).sum())
    W = float(0.10 * pen.sum())
    return {"S": rev - W, "revenue": rev, "W": W,
            "W_G1": float(0.10 * pen[g.group == "G1"].sum()),
            "W_G2": float(0.10 * pen[g.group == "G2"].sum())}


# ----------------------------------------------------------------------------- 메인
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-val", action="store_true", help="자체 검증 생략")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    log("1) 데이터 읽기")
    sites, smp_train, smp_test, gen, wa, wb = load_all()
    gen = gen[gen.ts >= TRAIN_FROM]
    comp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})   # 참조 구성 사용
    log(f"  발전소 {len(sites)}개, 발전량 {len(gen):,}행, 예보 A {len(wa):,} / B {len(wb):,}")

    if not args.skip_val:
        log("2) 자체 검증: 2024-06 이전으로 학습 -> 2024-06~2025-05 예측")
        gtr = gen[gen.ts < VAL_FROM]
        stats_v = site_stats_from(gtr, sites)
        m_v = train_model(gtr, sites, wa, wb, stats_v)
        pv = predict_sites(m_v, sites.site_id, pd.date_range(VAL_FROM, VAL_TO, freq="h"), sites, wa, wb, stats_v)
        first = gen.groupby("site_id").ts.min()
        warm = first[first < VAL_FROM - pd.Timedelta(days=60)].index     # 학습 기록이 있던 발전소
        r = score(pv, comp, gen[gen.site_id.isin(warm)], smp_train)
        log(f"  [검증] 발전소 {len(warm)}개 기준  벌점 W = {r['W']/1e6:,.3f}백만원 "
            f"(G1 {r['W_G1']/1e6:,.3f} / G2 {r['W_G2']/1e6:,.3f})")
        vp_path = os.path.join(OUT, "val_pred_claude_v1.csv")
        pv.to_csv(vp_path, index=False)                     # 팀 공통 채점기(eval_val.py)용
        log(f"  검증 예측 저장: {vp_path}")
        del m_v, pv
    else:
        log("2) 자체 검증 생략")

    log("3) 최종 학습: 학습 구간 전체 사용")
    stats = site_stats_from(gen, sites)
    model = train_model(gen, sites, wa, wb, stats)

    log("4) 평가 구간 예측")
    index = pd.DatetimeIndex(smp_test.ts)                 # 채점 시간축 8,760시각
    ps = predict_sites(model, sites.site_id, index, sites, wa, wb, stats)
    ps["group"] = ps.site_id.map(comp)
    fc = (ps.groupby(["group", "ts"], as_index=False).kwh.sum()
            .rename(columns={"kwh": "forecast_kwh"}).sort_values(["group", "ts"]))
    fc["forecast_kwh"] = fc.forecast_kwh.clip(lower=0).round(3)

    comp_path = os.path.join(OUT, "composition.csv")
    fc_path = os.path.join(OUT, "forecast.csv")
    comp.rename("group").rename_axis("site_id").reset_index().to_csv(comp_path, index=False)
    fc.to_csv(fc_path, index=False)
    for g in ("G1", "G2"):
        cap_g = sites.capacity_kw[sites.site_id.map(comp) == g].sum()
        log(f"  {g}: 설비용량 {cap_g:,.1f} kW, 연간 예측 합계 {fc[fc.group == g].forecast_kwh.sum()/1e6:,.2f} GWh")
    log(f"  저장: {comp_path}")
    log(f"  저장: {fc_path} ({len(fc):,}행)")

    log("5) 주최측 형식 검증")
    if os.path.isfile(VALIDATOR):
        res = subprocess.run([sys.executable, VALIDATOR, "--composition", comp_path,
                              "--forecast", fc_path, "--data", DATA],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(res.stdout.strip() or res.stderr.strip())
    else:
        log("  tools/validate_submission.py 가 없어 형식 검증을 건너뜁니다.")

    print("=" * 70)
    print(f"실행 시간: 총 {time.time() - T0:.0f}초  |  결과 폴더: {OUT}")
    print("완료되었습니다. 터미널을 복붙해서 클로드에게 보내주세요.")   # ※ 코드 제출 전 이 줄 삭제
    print("=" * 70)


if __name__ == "__main__":
    main()
