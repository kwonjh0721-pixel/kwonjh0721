"""Create ONE validation CSV using only the supplied train/ data.

python claude_validation/make_val.py --data OIBC2026_dataset --output val_pred_claude_fixed.csv

Train: ts < 2024-06-01 01:00 KST (generation and forecasts).
Predict: 2024-06-01 01:00 through 2025-06-01 00:00 KST, inclusive.
Output: site_id,ts,kwh; all 253 sites x 8760 hours = 2,216,280 rows.
No validation targets, early stopping on the validation year, or test/ files.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

if __package__:
    from . import model
    from .common import DEFAULT_DATA, VAL_FROM, VAL_TO, VAL_END, read_sites, read_window, require, unique, validate_predictions
else:
    import model
    from common import DEFAULT_DATA, VAL_FROM, VAL_TO, VAL_END, read_sites, read_window, require, unique, validate_predictions


def weather_window(data, start, end):
    tables = []
    for provider, columns in [("wa", model.WA_COLS), ("wb", model.WB_COLS)]:
        path = Path(data) / "train" / f"forecast_{provider}.parquet"
        table = read_window(path, [provider + "_station", "round", "ts"] + columns, start, end)
        tables.append(table)
    return model.prep_forecast(*tables)


def run(data, output, *, threads=4, chunk_days=7, params=None):
    data, output = Path(data), Path(output)
    required = ["sites.csv", "train/generation.parquet", "train/forecast_wa.parquet", "train/forecast_wb.parquet"]
    missing = [p for p in required if not (data / p).is_file()]
    require(not missing, "Missing input files:\n" + "\n".join(missing))
    require(not output.exists(), "Output already exists; choose a new filename")
    require(threads > 0 and chunk_days > 0, "threads and chunk-days must be positive")
    sites = read_sites(data)
    model.log("Read training generation strictly before the validation boundary")
    gen = read_window(data / "train/generation.parquet", ["site_id", "ts", "energy"], model.TRAIN_FROM, VAL_FROM)
    unique(gen, ["site_id", "ts"], "training generation")
    require(gen.site_id.isin(sites.site_id).all(), "Unknown generation site")
    valid = np.isfinite(gen.energy) & (gen.energy >= 0)
    model.log(f"Invalid generation rows removed: {int((~valid).sum())}")
    gen = gen[valid].copy()
    require(not gen.empty, "No valid training generation")
    require(gen.ts.max() < VAL_FROM, "Training cutoff violated")
    model.log(f"Training labels: {gen.ts.min()} .. {gen.ts.max()} ({len(gen):,} rows)")
    wa, wb = weather_window(data, model.TRAIN_FROM, VAL_FROM)
    require(not wa.empty and not wb.empty, "Missing training forecasts for a provider")
    stats = model.site_stats_from(gen, sites)
    context = model.daily_context(sites, wa, wb)
    fit_params = dict(model.LGB_PARAMS)
    if params:
        fit_params.update(params)
    fit_params["n_jobs"] = threads
    trained = model.train_model(gen, sites, wa, wb, stats, context, fit_params)
    del gen, wa, wb, context
    gc.collect()
    # Validation forecasts are loaded only after fitting is complete.
    model.log("Predict validation year using train/ forecasts only")
    wa, wb = weather_window(data, VAL_FROM, VAL_END)
    require(not wa.empty and not wb.empty, "Missing validation forecasts for a provider")
    context = model.daily_context(sites, wa, wb)
    times = pd.date_range(VAL_FROM, VAL_TO, freq="h")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            for offset in range(0, len(times), chunk_days * 24):
                chunk = times[offset:offset + chunk_days * 24]
                a = wa[wa.ts.between(chunk.min(), chunk.max())]
                b = wb[wb.ts.between(chunk.min(), chunk.max())]
                pred = model.predict_sites(trained, sites.site_id, chunk, sites, a, b, stats, context)
                pred.to_csv(stream, index=False, header=(offset == 0), float_format="%.8f")
                if offset % (chunk_days * 24 * 8) == 0:
                    model.log(f"Predicted {min(offset + len(chunk), len(times)):,}/{len(times):,} hours per site")
        checked = validate_predictions(pd.read_csv(temporary, dtype={"site_id": str}), sites)
        # Reserve the final path exclusively; never overwrite an existing user file.
        with output.open("x"):
            pass
        os.replace(temporary, output)
        model.log(f"Saved {output}: {len(checked):,} rows; columns=site_id,ts,kwh; KST timezone checked")
        return {"rows": len(checked), "sites": len(sites), "hours_per_site": len(times)}
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("val_pred_claude_fixed.csv"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-days", type=int, default=7)
    args = parser.parse_args()
    try:
        run(args.data, args.output, threads=args.threads, chunk_days=args.chunk_days)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
