#!/usr/bin/env python
"""
Batch visualization for MuProMAC event logs + stability inspection.

For every CSV in FOLDER, this script:
- Prints basic info
- Plots WIP (cases in system) over time
- Plots total queue length over time
- Prints per-station queue statistics (early/late & slopes)
- Computes simple stability indicators and writes a summary CSV.

Plots are saved as PNGs next to each CSV file:
- <logname>_wip.png
- <logname>_queue.png

Summary CSV:
- <FOLDER>/FIFO_EXP_stability_summary.csv
"""

import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 🔧 CHANGE THIS TO YOUR LOG FOLDER 🔧
FOLDER = r"out/251110/results"


# -----------------------------
# Helpers
# -----------------------------

def load_log(path: str) -> pd.DataFrame:
    # Only load the columns we actually use in this script
    needed = [
        "status",
        "timestamp",
        "case_id",
        "activity",
        "resource",
        "end_time",
        "cycle_time",
        "queue_start",
    ]

    # usecols with a callable: ignore all other columns
    df = pd.read_csv(path, usecols=lambda c: c in needed)

    # ensure expected columns exist (handle missing gracefully)
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    return df


def build_step_trajectory(events):
    """
    Generic helper:
    - events: list of (time, delta)
    - returns times, values for a step-like trajectory
    """
    if not events:
        return np.array([]), np.array([])

    events_sorted = sorted(events, key=lambda x: x[0])
    times = []
    values = []
    current = 0.0

    for t, delta in events_sorted:
        current += delta
        times.append(float(t))
        values.append(current)

    return np.array(times), np.array(values)


def build_wip_trajectory(df: pd.DataFrame):
    """
    WIP(t) = number of cases that have started but not yet completed.
    Uses START and COMPLETE events.
    """
    starts = df[df["status"] == "START"]["timestamp"].values
    comps = df[df["status"] == "COMPLETE"]["timestamp"].values

    events = []
    for t in starts:
        events.append((t, +1))
    for t in comps:
        events.append((t, -1))

    return build_step_trajectory(events)


def build_global_queue_trajectory(df: pd.DataFrame):
    """
    Total queue length = number of jobs with status 'queued' not yet
    started ('running') anywhere, aggregated over all activities.
    """
    qs = df[df["status"] == "queued"]["timestamp"].values
    rs = df[df["status"] == "running"]["timestamp"].values

    events = []
    for t in qs:
        events.append((t, +1))
    for t in rs:
        events.append((t, -1))

    return build_step_trajectory(events)


def summarize_step_trajectory(times: np.ndarray,
                              values: np.ndarray,
                              label: str,
                              verbose: bool = True):
    """
    Compute summary stats for a step trajectory and (optionally) print them.

    Returns a dict with:
      - label, T, max, early_mean, late_mean, ratio, slope
    """
    if times.size == 0:
        if verbose:
            print(f"{label}: no data.")
        return {
            "label": label,
            "T": 0.0,
            "max": 0.0,
            "early_mean": 0.0,
            "late_mean": 0.0,
            "ratio": 1.0,
            "slope": 0.0,
        }

    T = times[-1]
    v_max = float(values.max())
    cut_early = 0.2 * T
    cut_late = 0.8 * T

    early_vals = [v for t, v in zip(times, values) if t <= cut_early]
    late_vals = [v for t, v in zip(times, values) if t >= cut_late]

    early_mean = float(np.mean(early_vals)) if early_vals else 0.0
    late_mean = float(np.mean(late_vals)) if late_vals else 0.0
    eps = 1e-6
    ratio = (late_mean + eps) / (early_mean + eps)

    # rough slope (trend over time)
    x = times - times.mean()
    y = values.astype(float)
    if len(x) > 1:
        slope, _ = np.polyfit(x, y, 1)
    else:
        slope = 0.0

    if verbose:
        print(f"{label}:")
        print(f"  Horizon T          : {T:.2f}")
        print(f"  Max value          : {v_max:.2f}")
        print(f"  Mean early (0–20%) : {early_mean:.2f}")
        print(f"  Mean late (80–100%): {late_mean:.2f}")
        print(f"  Late / early ratio : {ratio:.2f}")
        print(f"  Slope (approx)     : {slope:.4f}")
        print()

    return {
        "label": label,
        "T": T,
        "max": v_max,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "ratio": ratio,
        "slope": slope,
    }


# -----------------------------
# Queue behaviour per station
# -----------------------------

def queue_stability_stats(df: pd.DataFrame, activity_label: str):
    """
    Reconstruct queue length over time for one activity using queued/running events.
    Returns a dict including a 'suspect_overload' flag and slope.
    """
    sub = df[(df["activity"] == activity_label) &
             (df["status"].isin(["queued", "running"]))].copy()
    if sub.empty:
        return {
            "activity": activity_label,
            "has_data": False,
            "max_q": 0,
            "early_mean": 0.0,
            "late_mean": 0.0,
            "ratio": 0.0,
            "slope": 0.0,
            "suspect_overload": False,
        }

    sub = sub.sort_values("timestamp")
    q = 0
    times = []
    sizes = []
    for _, row in sub.iterrows():
        if row["status"] == "queued":
            q += 1
        elif row["status"] == "running":
            q = max(0, q - 1)
        times.append(row["timestamp"])
        sizes.append(q)

    times = np.array(times, dtype=float)
    sizes = np.array(sizes, dtype=float)

    T = times[-1]
    cut_early = T * 0.2
    cut_late = T * 0.8
    early = [s for t, s in zip(times, sizes) if t <= cut_early]
    late = [s for t, s in zip(times, sizes) if t >= cut_late]

    early_mean = float(np.mean(early)) if early else 0.0
    late_mean = float(np.mean(late)) if late else 0.0
    max_q = int(max(sizes)) if sizes.size > 0 else 0

    # linear trend (slope) of queue vs time
    x = times - times.mean()
    y = sizes
    if len(x) > 1:
        slope, _ = np.polyfit(x, y, 1)
    else:
        slope = 0.0

    eps = 1e-6
    ratio = (late_mean + eps) / (early_mean + eps)

    # same heuristic as before
    suspect_overload = (
        slope > 0.01 and
        late_mean > early_mean + 1.0 and
        ratio > 1.5 and
        late_mean > 2.0
    )

    return {
        "activity": activity_label,
        "has_data": True,
        "max_q": max_q,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "ratio": ratio,
        "slope": slope,
        "suspect_overload": suspect_overload,
    }


def all_station_queue_summary(df: pd.DataFrame, verbose: bool = True):
    """
    Compute queue stability stats for all activities that ever have 'queued' events.

    Returns:
      - summary_df (per-activity stats) or None
      - agg (dict with n_stations, n_suspect, max_ratio, max_slope)
    """
    queued_acts = df[df["status"] == "queued"]["activity"].dropna().unique()
    results = []

    for act in queued_acts:
        res = queue_stability_stats(df, act)
        if res["has_data"]:
            results.append(res)

    if not results:
        if verbose:
            print("No queue data per station.")
            print()
        agg = {
            "n_stations": 0,
            "n_suspect": 0,
            "max_station_ratio": 0.0,
            "max_station_slope": 0.0,
        }
        return None, agg

    summary = pd.DataFrame(results)
    summary = summary.sort_values("ratio", ascending=False)

    if verbose:
        print("=== ALL-STATION QUEUE SUMMARY (sorted by late/early ratio) ===")
        cols = ["activity", "max_q", "early_mean", "late_mean",
                "ratio", "slope", "suspect_overload"]
        print(summary[cols].to_string(index=False))
        print()

    agg = {
        "n_stations": int(len(summary)),
        "n_suspect": int(summary["suspect_overload"].sum()),
        "max_station_ratio": float(summary["ratio"].max()),
        "max_station_slope": float(summary["slope"].max()),
    }
    return summary, agg


# -----------------------------
# Main visualization for one log
# -----------------------------

def visualize_log(path: str):
    """
    Visualize + compute stability stats for one log.

    Returns a dict with stability indicators for later aggregation.
    """
    print("\n" + "=" * 80)
    print(f"LOG: {path}")
    print("=" * 80)

    df = load_log(path)

    n_events = len(df)
    n_cases = df["case_id"].nunique()
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    horizon_span = t_max - t_min

    print(f"Events  : {n_events}")
    print(f"Cases   : {n_cases}")
    print(f"Horizon : [{t_min:.2f}, {t_max:.2f}] span={horizon_span:.2f}")

    # WIP trajectory
    wip_t, wip_v = build_wip_trajectory(df)
    wip_stats = summarize_step_trajectory(wip_t, wip_v, "WIP (cases in system)")

    # Global queue trajectory
    q_t, q_v = build_global_queue_trajectory(df)
    q_stats = summarize_step_trajectory(q_t, q_v, "Total queue length")

    # Per-station queue summary
    station_df, station_agg = all_station_queue_summary(df)

    # --- Heuristic classification: is this log unstable?  # NEW
    is_unstable = (
        (wip_stats["ratio"] > 2.0) or
        (wip_stats["slope"] > 0.01) or
        (q_stats["ratio"] > 2.0) or
        (q_stats["slope"] > 0.01) or
        (station_agg["n_suspect"] >= 1)
    )

    if is_unstable:
        print("🚨 Marked as UNSTABLE (suspect overload).")
    else:
        print("✅ Marked as STABLE (no obvious overload).")
    print()

    # Save plots next to the CSV
    base_dir = os.path.dirname(path)
    base_name = os.path.splitext(os.path.basename(path))[0]

    # WIP plot
    if wip_t.size > 0:
        plt.figure(figsize=(8, 4))
        plt.step(wip_t, wip_v, where="post")
        plt.xlabel("time")
        plt.ylabel("WIP (cases in system)")
        plt.title(f"WIP over time – {base_name}")
        plt.tight_layout()
        out_path = os.path.join(base_dir, base_name + "_wip.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"WIP plot saved to: {out_path}")
    else:
        print("No WIP data to plot.")

    # Queue plot
    if q_t.size > 0:
        plt.figure(figsize=(8, 4))
        plt.step(q_t, q_v, where="post")
        plt.xlabel("time")
        plt.ylabel("Total queue length")
        plt.title(f"Total queue over time – {base_name}")
        plt.tight_layout()
        out_path = os.path.join(base_dir, base_name + "_queue.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Queue plot saved to: {out_path}")
    else:
        print("No queue data to plot.")

    print("=== VISUALIZATION DONE ===\n")

    # Row for stability summary CSV  # NEW
    row = {
        "log_name": base_name,
        "path": path,
        "n_events": n_events,
        "n_cases": n_cases,
        "t_min": float(t_min),
        "t_max": float(t_max),
        "horizon_span": float(horizon_span),

        # WIP stats
        "wip_T": wip_stats["T"],
        "wip_max": wip_stats["max"],
        "wip_early_mean": wip_stats["early_mean"],
        "wip_late_mean": wip_stats["late_mean"],
        "wip_ratio_late_early": wip_stats["ratio"],
        "wip_slope": wip_stats["slope"],

        # Global queue stats
        "q_T": q_stats["T"],
        "q_max": q_stats["max"],
        "q_early_mean": q_stats["early_mean"],
        "q_late_mean": q_stats["late_mean"],
        "q_ratio_late_early": q_stats["ratio"],
        "q_slope": q_stats["slope"],

        # Per-station aggregate
        "n_stations_with_queue": station_agg["n_stations"],
        "n_suspect_stations": station_agg["n_suspect"],
        "max_station_ratio": station_agg["max_station_ratio"],
        "max_station_slope": station_agg["max_station_slope"],

        # Final flag
        "is_unstable": bool(is_unstable),
    }
    return row


# -----------------------------
# Batch loop over all CSV logs
# -----------------------------

def main():
    if not os.path.isdir(FOLDER):
        print(f"Directory not found: {FOLDER}")
        return

    files = sorted(glob.glob(os.path.join(FOLDER, "FIFO_EXP_*.csv")))
    if not files:
        print(f"No CSV files found in directory: {FOLDER}")
        return

    print(f"Found {len(files)} CSV log(s) in {FOLDER}.\n")

    summary_rows = []  # NEW: collect stability info

    for path in files:
        row = visualize_log(path)
        summary_rows.append(row)

    # Save stability summary CSV  # NEW
    summary_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(FOLDER, "FIFO_EXP_stability_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"Stability summary saved to: {out_csv}")


if __name__ == "__main__":
    main()
