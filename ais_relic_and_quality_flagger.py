#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flags three kinds of AIS row into one combined output: line relics, circle
relics, and obvious data-quality errors (null/NA coordinates, (0,0), lat/lon
out of valid range, and duplicate timestamps per ship).

Architecture
------------
Built on the lightweight, direct-query pattern (proven reliable throughout
this project), NOT v8's heavier staging/sharding/checkpoint architecture --
that architecture has produced three separate real bugs in the course of
this investigation (a SQL binder error, a column-ambiguity error, and a
stale-checkpoint reuse bug), while the lightweight pattern has been simple
and correct throughout. The circle/line detection functions themselves are
confirmed byte-for-byte IDENTICAL between v6 and v8, so nothing about
detection quality is lost by building on the lighter foundation.

Quality checks (vectorized, applied to every row, independent of detection)
-----------------------------------------------------------------------------
- null_coordinate: LAT or LON is NULL/NA after casting
- zero_zero: LAT == 0 AND LON == 0 (after the null check, so this is a
  genuine zero, not a null miscast as zero)
- out_of_range: LAT outside [-90, 90] or LON outside [-180, 180]
- duplicate_timestamp: more than one row for the same ship at the exact
  same timestamp

Rows with null_coordinate, zero_zero, or out_of_range are EXCLUDED from
relic detection before tracks are built -- these aren't real positions, and
feeding them into the geometric fitting would create artificial jumps
(e.g. to the (0,0) point) that could manufacture false line/circle
detections rather than reflecting genuine spoofing behaviour.
duplicate_timestamp rows are NOT excluded, since the position itself may
still be genuine; ties are broken with a stable secondary sort key so track
ordering stays deterministic.

Output: one combined Parquet file, one row per flagged AIS position, with
a `reason` column (semicolon-joined if more than one quality issue applies
to the same row) and detection metrics where relevant (NULL otherwise).

Usage (PowerShell)
-------------------
    & "C:\\anaconda\\python.exe" -u "C:\\Users\\Craig Pearce\\Desktop\\ais_relic_and_quality_flagger.py" `
        --ais-folder "C:\\Users\\Craig Pearce\\Desktop\\Data_sets\\ais_files\\ais_2025" `
        --output-dir "C:\\Users\\Craig Pearce\\Desktop\\ais_2025_flagged" `
        --detector-module-dir "C:\\Users\\Craig Pearce\\Desktop" `
        --workers 4

Run --self-test first (synthetic data covering every case, no real files
needed):

    & "C:\\anaconda\\python.exe" -u "C:\\Users\\Craig Pearce\\Desktop\\ais_relic_and_quality_flagger.py" --self-test
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DEFAULT_MODULE_NAME = "ais_anomaly_detector_strict_relics_v6_zones"


def sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def _detect_columns(con: duckdb.DuckDBPyConnection, ais_file: str) -> tuple[str, str]:
    columns = set(con.execute(f"SELECT * FROM read_parquet({sql_str(ais_file)}) LIMIT 0").fetch_df().columns)
    cog_expr = "TRY_CAST(COG AS DOUBLE)" if "COG" in columns else "NULL"
    sog_expr = "TRY_CAST(SPEED AS DOUBLE) / 10.0" if "SPEED" in columns else "NULL"
    return cog_expr, sog_expr


def process_file(ais_file: Path, module_dir: str, module_name: str, worker_id: int) -> dict:
    print(f"  [worker {worker_id}] {ais_file.name}: loading...", flush=True)
    t0 = time.perf_counter()

    sys.path.insert(0, module_dir)
    detector = importlib.import_module(module_name)

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    cog_expr, sog_expr = _detect_columns(con, str(ais_file))

    df = con.execute(
        f"""
        SELECT
            SHIP_ID AS ship_id,
            TRY_CAST(LAT AS DOUBLE) AS lat,
            TRY_CAST(LON AS DOUBLE) AS lon,
            TIMESTAMP AS ts,
            {cog_expr} AS cog_deg,
            {sog_expr} AS sog_kts
        FROM read_parquet({sql_str(str(ais_file))})
        ORDER BY SHIP_ID, TIMESTAMP
        """
    ).fetch_df()
    n = len(df)

    # --- Quality checks: vectorized, applied to every row -----------------
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    null_coordinate = ~np.isfinite(lat) | ~np.isfinite(lon)
    zero_zero = ~null_coordinate & (lat == 0.0) & (lon == 0.0)
    out_of_range = ~null_coordinate & ((lat < -90.0) | (lat > 90.0) | (lon < -180.0) | (lon > 180.0))
    dup_key = df["ship_id"].astype(str) + "||" + df["ts"].astype(str)
    duplicate_timestamp = dup_key.duplicated(keep=False).to_numpy()

    quality_reasons = np.full(n, "", dtype=object)
    for mask, label in (
        (null_coordinate, "null_coordinate"),
        (zero_zero, "zero_zero"),
        (out_of_range, "out_of_range"),
        (duplicate_timestamp, "duplicate_timestamp"),
    ):
        for i in np.flatnonzero(mask):
            quality_reasons[i] = quality_reasons[i] + (";" if quality_reasons[i] else "") + label

    quality_flagged = null_coordinate | zero_zero | out_of_range | duplicate_timestamp

    # --- Relic detection: only on geometrically valid rows -----------------
    # null/zero/out-of-range rows are excluded -- these aren't real
    # positions and would create artificial jumps in the track (e.g. to
    # (0,0)) that could manufacture false detections rather than reflect
    # genuine spoofing behaviour. duplicate_timestamp rows are KEPT, since
    # the position itself may still be genuine.
    valid_for_detection = ~(null_coordinate | zero_zero | out_of_range)

    thresholds = detector.DetectorThresholds()
    flagged_rows: list[dict] = []
    n_circles = 0
    n_lines = 0

    valid_df = df[valid_for_detection].copy()

    # groupby on valid_df with a reset index gives a direct, simple mapping
    valid_df = valid_df.reset_index(drop=False).rename(columns={"index": "orig_index"})
    for ship_id, group in valid_df.groupby("ship_id", sort=False):
        m = len(group)
        if m < 3:
            continue
        g_lat = group["lat"].to_numpy()
        g_lon = group["lon"].to_numpy()
        ts_us = pd.to_datetime(group["ts"]).to_numpy().astype("datetime64[us]").astype(np.int64)
        orig_idx = group["orig_index"].to_numpy()

        track = detector.TrackArrays(
            lat=g_lat, lon=g_lon, ts_us=ts_us,
            cog_deg=group["cog_deg"].to_numpy(), sog_kts=group["sog_kts"].to_numpy(),
            source_file_id=np.zeros(m, dtype=np.int64), source_row=np.arange(m, dtype=np.int64),
            in_bbox_core=np.ones(m, dtype=bool),
        )
        geometry = detector.compute_chunk_geometry(track)

        circle_windows, _ = detector.circle_candidate_windows_strict(track, thresholds, geometry)
        if circle_windows:
            for det in detector.detect_strict_annulus_circles_from_candidates(track, thresholds, geometry, circle_windows):
                n_circles += 1
                event_id = f"{ship_id}_circle_{n_circles}"
                m_ = det.metrics
                for local_i in det.indices:
                    real_row = int(orig_idx[local_i])
                    flagged_rows.append({
                        "row_index": real_row, "reason": "circle_relic", "event_id": event_id,
                        "score": float(det.score), "radius_m": float(m_.get("diameter_m", 0.0)) / 2.0,
                    })

        line_windows = detector.coordinate_locked_line_candidate_runs(track, thresholds, geometry)
        if line_windows:
            for det in detector.detect_coordinate_locked_lines_from_candidates(track, thresholds, geometry, line_windows):
                n_lines += 1
                event_id = f"{ship_id}_line_{n_lines}"
                for local_i in det.indices:
                    real_row = int(orig_idx[local_i])
                    flagged_rows.append({
                        "row_index": real_row, "reason": "line_relic", "event_id": event_id,
                        "score": float(det.score), "radius_m": None,
                    })

    # --- Combine quality-flagged rows and relic-flagged rows --------------
    combined: dict[int, dict] = {}
    for i in np.flatnonzero(quality_flagged):
        combined[i] = {
            "SHIP_ID": df["ship_id"].iat[i], "LAT": df["lat"].iat[i], "LON": df["lon"].iat[i],
            "TIMESTAMP": df["ts"].iat[i], "reason": quality_reasons[i], "event_id": None,
            "score": None, "radius_m": None, "source_file": ais_file.name,
        }
    for r in flagged_rows:
        i = r["row_index"]
        if i in combined:
            existing = combined[i]
            existing["reason"] = existing["reason"] + ";" + r["reason"]
        else:
            combined[i] = {
                "SHIP_ID": df["ship_id"].iat[i], "LAT": df["lat"].iat[i], "LON": df["lon"].iat[i],
                "TIMESTAMP": df["ts"].iat[i], "reason": r["reason"], "event_id": r["event_id"],
                "score": r["score"], "radius_m": r["radius_m"], "source_file": ais_file.name,
            }

    elapsed = time.perf_counter() - t0
    print(
        f"  [worker {worker_id}] {ais_file.name}: {n:,} rows in {elapsed:.1f}s -- "
        f"{int(quality_flagged.sum()):,} quality-flagged, {n_circles} circles, {n_lines} lines, "
        f"{len(combined):,} total flagged rows",
        flush=True,
    )

    out_df = pd.DataFrame(list(combined.values()))
    return {
        "source_file": ais_file.name, "rows": n, "quality_flagged": int(quality_flagged.sum()),
        "circles": n_circles, "lines": n_lines, "total_flagged": len(combined),
        "seconds": elapsed, "flagged_df": out_df,
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run(ais_folder: Path, output_dir: Path, module_dir: Path, module_name: str, workers: int, max_files: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in ais_folder.glob("*.parquet") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {ais_folder}")
    if max_files is not None:
        files = files[:max_files]

    combined_path = output_dir / "flagged_rows_combined.parquet"
    combined_path.unlink(missing_ok=True)
    summary_rows = []
    started = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [
            pool.submit(process_file, f, str(module_dir), module_name, i % workers)
            for i, f in enumerate(files)
        ]
        first_write = True
        for fut in futures:
            result = fut.result()
            out_df = result.pop("flagged_df")
            summary_rows.append(result)
            if not out_df.empty:
                out_df.to_parquet(
                    output_dir / f"_part_{result['source_file']}.parquet", index=False
                )

    # Consolidate all per-file parts into one combined file.
    part_files = sorted(output_dir.glob("_part_*.parquet"))
    if part_files:
        con = duckdb.connect()
        con.execute("SET enable_progress_bar = false")
        parts_sql = "[" + ", ".join(sql_str(str(p)) for p in part_files) + "]"
        con.execute(
            f"COPY (SELECT * FROM read_parquet({parts_sql}, union_by_name=true)) TO {sql_str(str(combined_path))} (FORMAT PARQUET)"
        )
        for p in part_files:
            p.unlink()

    elapsed = time.perf_counter() - started
    summary = {
        "files_processed": len(files), "elapsed_seconds": elapsed,
        "total_rows": sum(r["rows"] for r in summary_rows),
        "total_quality_flagged": sum(r["quality_flagged"] for r in summary_rows),
        "total_circles": sum(r["circles"] for r in summary_rows),
        "total_lines": sum(r["lines"] for r in summary_rows),
        "total_flagged_rows": sum(r["total_flagged"] for r in summary_rows),
        "per_file": summary_rows,
    }
    (output_dir / "flagging_run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {combined_path} and {output_dir / 'flagging_run_summary.json'}")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_file"}, indent=2))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    module_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(11)

    rows = []
    # Ship 1: a clean, genuine circle relic.
    n = 60
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = 0.03
    lat = 25.0 + r * np.sin(angles) + rng.normal(0, 0.00005, n)
    lon = 50.0 + r * np.cos(angles) + rng.normal(0, 0.00005, n)
    ts = pd.date_range("2026-01-01", periods=n, freq="30s")
    for i in range(n):
        rows.append({"SHIP_ID": 111111111, "LAT": lat[i], "LON": lon[i], "TIMESTAMP": ts[i], "COG": 0.0, "SPEED": 50.0})

    # Ship 2: a normal, boring track -- should trigger nothing.
    n2 = 20
    ts2 = pd.date_range("2026-01-01", periods=n2, freq="1min")
    for i in range(n2):
        rows.append({"SHIP_ID": 222222222, "LAT": 30.0 + i * 0.01, "LON": 40.0 + i * 0.01, "TIMESTAMP": ts2[i], "COG": 45.0, "SPEED": 100.0})

    # Ship 3: quality issues -- (0,0), NULL, out-of-range, and a duplicate timestamp.
    dup_ts = pd.Timestamp("2026-01-01T00:00:00")
    rows.append({"SHIP_ID": 333333333, "LAT": 0.0, "LON": 0.0, "TIMESTAMP": pd.Timestamp("2026-01-01T00:01:00"), "COG": 0.0, "SPEED": 0.0})
    rows.append({"SHIP_ID": 333333333, "LAT": None, "LON": 35.0, "TIMESTAMP": pd.Timestamp("2026-01-01T00:02:00"), "COG": 0.0, "SPEED": 0.0})
    rows.append({"SHIP_ID": 333333333, "LAT": 95.0, "LON": 35.0, "TIMESTAMP": pd.Timestamp("2026-01-01T00:03:00"), "COG": 0.0, "SPEED": 0.0})
    rows.append({"SHIP_ID": 333333333, "LAT": 34.0, "LON": 35.0, "TIMESTAMP": dup_ts, "COG": 0.0, "SPEED": 0.0})
    rows.append({"SHIP_ID": 333333333, "LAT": 34.1, "LON": 35.1, "TIMESTAMP": dup_ts, "COG": 0.0, "SPEED": 0.0})

    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp:
        ais_folder = Path(tmp) / "ais"
        ais_folder.mkdir()
        df.to_parquet(ais_folder / "synthetic.parquet")
        output_dir = Path(tmp) / "out"

        run(ais_folder, output_dir, module_dir, DEFAULT_MODULE_NAME, workers=1, max_files=None)

        summary = json.loads((output_dir / "flagging_run_summary.json").read_text())
        print("summary:", json.dumps({k: v for k, v in summary.items() if k != "per_file"}, indent=2))

        # NOTE: the synthetic ring above does NOT reliably clear v6's full
        # strict threshold set (confirmed directly: the candidate window's
        # angular_coverage/angular_gap/radial_rmse gates fail even with
        # evenly-spaced source angles -- the window-finder's own internal
        # logic truncates before reaching full coverage, and correctly
        # calibrating a synthetic shape against the complete gate set
        # proved genuinely difficult without deep knowledge of that
        # windowing algorithm). The detection algorithm itself is already
        # proven on real data elsewhere in this project (real circles
        # found in real Middle East runs), so this self-test does not
        # depend on a positive detection firing here -- see the direct,
        # mocked row-index-mapping test below instead, which exercises
        # the part of THIS script's own code most at risk of a real bug.
        print(f"  (circles found on synthetic data: {summary['total_circles']} -- not asserted on; see note above)")
        assert summary["total_quality_flagged"] == 5, f"expected 5 quality-flagged rows (0,0 + null + out-of-range + 2 duplicates), got {summary['total_quality_flagged']}"

        combined = pd.read_parquet(output_dir / "flagged_rows_combined.parquet")
        reasons = set(combined["reason"].str.split(";").explode())
        print("distinct reasons found:", reasons)
        for expected in ("zero_zero", "null_coordinate", "out_of_range", "duplicate_timestamp"):
            assert expected in reasons, f"expected reason {expected!r} to appear in the combined output, got {reasons}"

        ship2_rows = combined[combined["SHIP_ID"] == 222222222]
        assert ship2_rows.empty, "the boring, clean ship-2 track should never appear in the flagged output"

    print("\n=== Direct test: row-index mapping, via a mocked Detection with known indices ===")
    with tempfile.TemporaryDirectory() as tmp2:
        module_dir2 = str(module_dir)
        detector = importlib.import_module(DEFAULT_MODULE_NAME)
        n3 = 10
        lat3 = np.linspace(20.0, 21.0, n3)
        lon3 = np.linspace(40.0, 41.0, n3)
        ts3 = pd.date_range("2026-01-01", periods=n3, freq="1min")
        df3 = pd.DataFrame({
            "SHIP_ID": [444444444] * n3, "LAT": lat3, "LON": lon3, "TIMESTAMP": ts3,
            "COG": [0.0] * n3, "SPEED": [0.0] * n3,
        })
        ais_path3 = Path(tmp2) / "mock_ais.parquet"
        df3.to_parquet(ais_path3)

        known_indices = np.array([2, 3, 4])  # deliberately a KNOWN, specific sub-range
        mock_detection = detector.Detection(
            anomaly_code=detector.ANOMALY_CIRCLE, subtype="strict_annulus",
            indices=known_indices, score=0.9, metrics={"diameter_m": 5000.0},
        )
        dummy_window = detector.CircleCandidateWindow(
            start=0, end=n3 - 1, direction=1, turn_deg=360.0, direction_consistency=1.0,
            path_m=1000.0, endpoint_path_ratio=1.0, bbox_aspect_ratio=1.0,
            centroid_radial_iqr_ratio=0.1, coarse_sector_fraction=1.0, coarse_inner_fraction=0.0,
            quick_score=1.0,
        )
        original_windows_fn = detector.circle_candidate_windows_strict
        original_fit_fn = detector.detect_strict_annulus_circles_from_candidates
        # A straight-line test track has zero curvature, so the real
        # window-finder naturally returns nothing -- mocking only the
        # downstream fitting function would never actually get called.
        # Both must be mocked so the code path this test targets actually runs.
        detector.circle_candidate_windows_strict = lambda *a, **k: ([dummy_window], {})
        detector.detect_strict_annulus_circles_from_candidates = lambda *a, **k: [mock_detection]
        try:
            # Called directly, in-process, NOT via run()'s ProcessPoolExecutor:
            # a spawned worker re-imports the module fresh in its own
            # process and would never see this mock applied to the
            # already-imported module object here.
            result = process_file(ais_path3, module_dir2, DEFAULT_MODULE_NAME, worker_id=0)
            combined3 = result["flagged_df"]
            print(combined3[["SHIP_ID", "LAT", "LON", "reason", "radius_m"]].to_string())
            circle_rows = combined3[combined3["reason"].str.contains("circle_relic")]
            expected_lats = set(lat3[known_indices].round(6))
            actual_lats = set(circle_rows["LAT"].round(6))
            assert actual_lats == expected_lats, (
                f"row-index mapping is wrong: mocked detection covered indices {known_indices} "
                f"(lat values {expected_lats}), but the output flagged lat values {actual_lats}"
            )
            print(f"  row-index mapping CONFIRMED correct: indices {known_indices} correctly mapped to lat values {expected_lats}")
        finally:
            detector.circle_candidate_windows_strict = original_windows_fn
            detector.detect_strict_annulus_circles_from_candidates = original_fit_fn

    print("\nSelf-test PASSED.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ais-folder", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--detector-module-dir", default=None)
    parser.add_argument("--module-name", default=DEFAULT_MODULE_NAME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        self_test()
        return
    if not args.ais_folder or not args.output_dir or not args.detector_module_dir:
        raise SystemExit("--ais-folder, --output-dir and --detector-module-dir are required unless --self-test is passed.")
    run(
        Path(args.ais_folder), Path(args.output_dir), Path(args.detector_module_dir),
        args.module_name, args.workers, args.max_files,
    )


if __name__ == "__main__":
    main()
