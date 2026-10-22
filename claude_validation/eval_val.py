"""Team validation scorer, not the official competition scorer.

Training: generation/forecast/observed data must have ts < 2024-06-01 01:00 KST.
Predict 2024-06-01 01:00 .. 2025-06-01 00:00 KST from train/ forecasts.
CSV: site_id,ts,kwh with explicit timezone; 253 sites x 8760 hours.

Reference groups A->G1, B->G2. Score only site-hour pairs with valid actuals.
Primary cohort follows the ORIGINAL EXECUTABLE rule: first valid record before
VAL_FROM - 60 days = 2024-04-02 01:00 KST, not April 1 as its prose stated.
The cohort size is calculated, never assumed to be 174.

python claude_validation/eval_val.py val_pred_claude_fixed.csv --data OIBC2026_dataset
Missing predictions are errors by default; --allow-incomplete restores the
original missing-as-zero behavior and reports the missing row count.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import DEFAULT_DATA, VAL_FROM, VAL_TO, to_kst, read_sites, require, unique, validate_predictions
else:
    from common import DEFAULT_DATA, VAL_FROM, VAL_TO, to_kst, read_sites, require, unique, validate_predictions


def k_of(ts):
    h = pd.DatetimeIndex(ts).hour
    return np.where((h >= 11) & (h <= 15), 3., 1.)


def score(pred, comp, gen, smp):
    unique(pred, ["site_id", "ts"], "predictions")
    unique(gen, ["site_id", "ts"], "generation")
    unique(smp, ["ts"], "SMP")
    a = gen[["site_id", "ts", "energy"]].merge(pred, on=["site_id", "ts"], how="left", validate="one_to_one")
    missing = int(a.kwh.isna().sum())
    a["kwh"] = a.kwh.fillna(0.)
    a["group"] = a.site_id.map(comp)
    require(a.group.notna().all(), "Generation site has no reference group")
    g = a.groupby(["group", "ts"], as_index=False)[["energy", "kwh"]].sum()
    g = g.merge(smp[["ts", "smp"]], on="ts", how="left", validate="many_to_one")
    require(np.isfinite(g.smp).all() and (g.smp >= 0).all(), "Missing/invalid SMP for scored hours")
    pen = g.smp * k_of(g.ts) * (g.kwh - g.energy).abs()
    denominator = float((g.smp * k_of(g.ts) * g.energy).sum())
    return {"W": float(.10 * pen.sum()),
            "W_G1": float(.10 * pen[g.group == "G1"].sum()),
            "W_G2": float(.10 * pen[g.group == "G2"].sum()),
            "wWAPE": float(pen.sum() / denominator) if denominator > 0 else None,
            "observed_site_hours": len(a), "missing_predictions_on_actuals": missing}


def load_prediction(path, sites, allow_incomplete=False):
    p = pd.read_csv(path, dtype={"site_id": str})
    require(list(p.columns) == ["site_id", "ts", "kwh"], "Required columns: site_id,ts,kwh")
    p["ts"] = to_kst(p.ts, require_timezone=True)
    unique(p, ["site_id", "ts"], "predictions")
    require(p.site_id.isin(sites.site_id).all(), "Unknown site")
    require(p.ts.between(VAL_FROM, VAL_TO).all(), "Out-of-window prediction")
    require(np.isfinite(p.kwh).all() and (p.kwh >= 0).all(), "Invalid kwh")
    return p if allow_incomplete else validate_predictions(p, sites)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    try:
        sites = read_sites(args.data)
        comp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
        smp = pd.read_csv(args.data / "train/smp.csv")
        smp["ts"] = to_kst(smp.ts)
        gen = pd.read_parquet(args.data / "train/generation.parquet", columns=["site_id", "ts", "energy"])
        gen["ts"] = to_kst(gen.ts)
        unique(gen, ["site_id", "ts"], "generation")
        gen = gen[np.isfinite(gen.energy) & (gen.energy >= 0)]
        first = gen.groupby("site_id").ts.min()
        warm_cutoff = VAL_FROM - pd.Timedelta(days=60)
        warm = first[first < warm_cutoff].index
        gen = gen[gen.ts.between(VAL_FROM, VAL_TO)]
        require(not gen.empty, "No validation actuals to score")
        print(f"Primary cohort: {len(warm)} sites; first valid record < {warm_cutoff}")
        failed = False
        for path in args.files:
            try:
                p = load_prediction(path, sites, args.allow_incomplete)
                result = {"file": str(path), "primary_site_count": len(warm),
                          "primary": score(p, comp, gen[gen.site_id.isin(warm)], smp),
                          "all_sites": score(p, comp, gen, smp)}
                if result["primary"]["observed_site_hours"] == 0:
                    result["primary"] = None
                print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            except (ValueError, FileNotFoundError) as error:
                failed = True
                print(f"ERROR {path}: {error}")
        return 2 if failed else 0
    except (ValueError, FileNotFoundError) as error:
        ap.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
