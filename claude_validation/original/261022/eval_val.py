"""
OIBC 2026 - 팀 내부 공통 채점기 (자체 검증용, 제출용 아님)
==============================================================
어떤 모델이든 아래 규칙으로 만든 '검증 예측 파일'을 넣으면 같은 기준으로 점수를 매긴다.

[검증 예측 파일 만드는 규칙]
  1. 학습에는 2024-06-01 01:00 KST 이전 데이터만 사용 (generation, 예보, 실측 모두)
  2. 2024-06-01 01:00 ~ 2025-06-01 00:00 KST 를 예측 (예보는 이 기간 train 예보 사용)
  3. CSV 컬럼:  site_id, ts, kwh    (발전소별·시각별 예측 발전량 kWh)
     - ts 는 '2024-06-01 01:00:00+09:00' 처럼 타임존 포함
     - 253개 발전소 전부 넣는 것을 권장 (없는 발전소·시각은 0으로 간주)

실행:
  python src/eval_val.py submissions/val_pred_내모델.csv
  python src/eval_val.py 파일1.csv 파일2.csv ...   (여러 개 비교)

채점식 (README):
  W = 0.10 × Σ_g Σ_t SMP(t)·k(t)·|예측_g(t) − 실적_g(t)|,  k=3 (시 11~15), 그 외 1
  - 집합 G1/G2 = sites.csv 의 current_group (A->G1, B->G2) 고정
  - 실적이 있는 (발전소, 시각)만 집합별로 합산
  - 주 지표: 2024-04-01 이전부터 기록이 있는 발전소 174개 기준 W (실제 평가와 조건이 비슷)
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "OIBC2026_dataset")
TZ = "Asia/Seoul"
VAL_FROM = pd.Timestamp("2024-06-01 01:00", tz=TZ)
VAL_TO = pd.Timestamp("2025-06-01 00:00", tz=TZ)


def to_kst(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, utc=True) if not pd.api.types.is_datetime64_any_dtype(s) else s
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(TZ)


def k_of(ts) -> np.ndarray:
    h = pd.DatetimeIndex(ts).hour
    return np.where((h >= 11) & (h <= 15), 3.0, 1.0)


def score(pred, comp, gen, smp) -> dict:
    a = gen[["site_id", "ts", "energy"]].merge(pred, on=["site_id", "ts"], how="left")
    a["kwh"] = a.kwh.fillna(0.0)
    a["group"] = a.site_id.map(comp)
    g = a.groupby(["group", "ts"], as_index=False)[["energy", "kwh"]].sum()
    g = g.merge(smp[["ts", "smp"]], on="ts", how="inner")
    g["k"] = k_of(g.ts)
    pen = g.smp * g.k * (g.kwh - g.energy).abs()
    return {"W": float(0.10 * pen.sum()),
            "W_G1": float(0.10 * pen[g.group == "G1"].sum()),
            "W_G2": float(0.10 * pen[g.group == "G2"].sum()),
            "wWAPE": float(pen.sum() / (g.smp * g.k * g.energy).sum())}


def main() -> None:
    files = sys.argv[1:]
    if not files:
        print("사용법: python src/eval_val.py 검증예측.csv [검증예측2.csv ...]")
        return

    sites = pd.read_csv(os.path.join(DATA, "sites.csv"))
    comp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    smp = pd.read_csv(os.path.join(DATA, "train", "smp.csv"))
    smp["ts"] = to_kst(smp.ts)
    gen = pd.read_parquet(os.path.join(DATA, "train", "generation.parquet"))
    gen["site_id"] = gen.site_id.astype(str)
    gen["ts"] = to_kst(gen.ts)
    gen = gen[gen.energy >= 0]
    first = gen.groupby("site_id").ts.min()
    gen = gen[(gen.ts >= VAL_FROM) & (gen.ts <= VAL_TO)]
    warm = set(first[first < VAL_FROM - pd.Timedelta(days=60)].index)

    print("=" * 78)
    print(f"검증 구간 {VAL_FROM:%Y-%m-%d %H:%M} ~ {VAL_TO:%Y-%m-%d %H:%M} KST  |  주 지표: 발전소 {len(warm)}개 기준")
    print("=" * 78)
    for f in files:
        p = pd.read_csv(f)
        miss = {"site_id", "ts", "kwh"} - set(p.columns)
        if miss:
            print(f"[오류] {f}: 컬럼 누락 {sorted(miss)} (site_id, ts, kwh 필요)")
            continue
        p["site_id"] = p.site_id.astype(str)
        p["ts"] = to_kst(p.ts)
        p = p[(p.ts >= VAL_FROM) & (p.ts <= VAL_TO)].groupby(["site_id", "ts"], as_index=False).kwh.sum()
        n_sites = p.site_id.nunique()
        r = score(p, comp, gen[gen.site_id.isin(warm)], smp)
        ra = score(p, comp, gen, smp)
        print(f"{os.path.basename(f)}")
        print(f"  [주 지표 174개] W = {r['W']/1e6:9.3f}백만원  (G1 {r['W_G1']/1e6:8.3f} / G2 {r['W_G2']/1e6:8.3f})"
              f"  가중오차율 {r['wWAPE']*100:5.2f}%")
        print(f"  [참고 253개]   W = {ra['W']/1e6:9.3f}백만원  (예측 파일의 발전소 수: {n_sites})")
    print("=" * 78)
    print("참고: 주최측 Baseline = 147.0백만원, Claude v1 = 136.7백만원 (주 지표 기준, 작을수록 좋음)")
    print("완료되었습니다. 터미널을 복붙해서 클로드에게 보내주세요.")


if __name__ == "__main__":
    main()
