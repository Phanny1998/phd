#!/usr/bin/env python

"""
Build Verenich-style logdata CSVs from MuProMAC event-log splits
(train / val / test) for use with time-prediction-benchmark.

Input:
    out/251110/event_logs_splits/
        <scenario>_warm10_train.csv
        <scenario>_warm10_val.csv
        <scenario>_warm10_test.csv
    (possibly multiple scenarios)

For each <scenario>, we:

  1) Load train + val + test for that scenario.
  2) Concatenate them -> effectively reconstruct the warm-up-trimmed full log.
  3) Keep only activity-completion events (status == "running").
  4) Compute Verenich-style features:

        Case ID
        Activity
        Resource
        Complete Timestamp      (we use end_time, in simulation time units)
        activity_duration       = end_time - timestamp
        timesincelastevent      = gap since previous completion in the case
        timesincecasestart      = completion time - first completion time
        event_nr                = 1,2,3,... per case
        open_cases              = #cases active at this completion time
        remtime                 = case_complete_time - completion time

Output:
    experiments/logdata/mupromac_<scenario>.csv
"""

import os
import glob

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIG – change paths if your layout differs
# ------------------------------------------------------------------
SPLIT_FOLDER   = r"out/251110/event_logs_splits"
LOGDATA_FOLDER = r"verenichbenchmark\experiments\logdata"
OUTPUT_PREFIX  = "mupromac_"   # dataset name will be e.g. mupromac_actuator8
# ------------------------------------------------------------------


def load_scenario_splits(split_folder: str):
    """
    Find all scenarios by looking for *_warm10_train.csv files.
    For each scenario base name, load its train/val/test (if present)
    and return a dict: {scenario_name: df_all_splits}.
    """
    pattern_train = os.path.join(split_folder, "*_warm10_train.csv")
    train_files = sorted(glob.glob(pattern_train))

    if not train_files:
        raise SystemExit(f"No '*_warm10_train.csv' files found in {split_folder}")

    scenario_to_df = {}

    for train_path in train_files:
        base = os.path.basename(train_path)
        # everything before "_warm10_train.csv" is the scenario name
        scenario = base.replace("_warm10_train.csv", "")
        print(f"\n=== Scenario: {scenario} ===")

        val_path  = os.path.join(
            split_folder, f"{scenario}_warm10_val.csv"
        )
        test_path = os.path.join(
            split_folder, f"{scenario}_warm10_test.csv"
        )

        dfs = []

        # load train
        print(f"  Loading TRAIN: {train_path}")
        dfs.append(pd.read_csv(train_path))

        # load val if exists
        if os.path.exists(val_path):
            print(f"  Loading VAL   : {val_path}")
            dfs.append(pd.read_csv(val_path))
        else:
            print(f"  WARNING: no val file found for {scenario}")

        # load test if exists
        if os.path.exists(test_path):
            print(f"  Loading TEST  : {test_path}")
            dfs.append(pd.read_csv(test_path))
        else:
            print(f"  WARNING: no test file found for {scenario}")

        df_all = pd.concat(dfs, axis=0, ignore_index=True)
        print(f"  -> concatenated {len(df_all)} rows, "
              f"{df_all['case_id'].nunique()} cases.")
        scenario_to_df[scenario] = df_all

    return scenario_to_df


def build_logdata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MuProMAC event log into Verenich-style logdata.

    Assumptions:
    - df has columns: case_id, activity, resource, timestamp, end_time, status
    - timestamp and end_time are numeric "simulation time" (seconds, minutes, etc.)
    - status == "running" means an activity execution:
        * timestamp  = start_of_service (simulation time)
        * end_time   = completion_time  (simulation time)
    """

    required = ["case_id", "activity", "resource", "timestamp", "end_time", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    # 1) Keep only completed-activity rows (like Production_Data, which has only completions)
    events = df[df["status"] == "running"].copy()
    if events.empty:
        raise SystemExit("No status == 'running' rows; nothing to build.")

    # Treat end_time as the completion time (simulation clock units)
    events["Complete Timestamp"] = events["end_time"].astype(float)

    # Sort by case + completion time
    events = events.sort_values(["case_id", "Complete Timestamp"]).reset_index(drop=True)

    # 2) Per-case start/end based on completion times (Verenich-style choice)
    per_case = (
        events.groupby("case_id")["Complete Timestamp"]
        .agg(["min", "max"])
        .rename(columns={"min": "case_start_time", "max": "case_complete_time"})
        .reset_index()
    )
    events = events.merge(per_case, on="case_id", how="left")

    # 3) Activity duration: end_time - timestamp (in same simulation units)
    events["activity_duration"] = (
        events["end_time"].astype(float) - events["timestamp"].astype(float)
    )

    # 4) Intra-case timing:
    #    timesincelastevent: gap between consecutive COMPLETIONS in a case
    events["timesincelastevent"] = (
        events.groupby("case_id")["Complete Timestamp"]
        .diff()
        .fillna(0.0)
    )

    #    timesincecasestart: completion - first completion in the case
    events["timesincecasestart"] = (
        events["Complete Timestamp"] - events["case_start_time"]
    )

    #    remtime: last completion - current completion
    events["remtime"] = (
        events["case_complete_time"] - events["Complete Timestamp"]
    )

    #    event_nr: 1,2,3,... within each case (completion order)
    events["event_nr"] = events.groupby("case_id").cumcount() + 1

    # 5) Global WIP: open_cases based on [case_start_time, case_complete_time)
    intervals = per_case.rename(
        columns={
            "case_start_time": "start_time",
            "case_complete_time": "end_time",
        }
    )

    if len(intervals) == 0:
        events["open_cases"] = 0
    else:
        starts = (
            intervals[["start_time"]]
            .sort_values("start_time")
            .reset_index(drop=True)
        )
        starts["n_starts"] = np.arange(1, len(starts) + 1, dtype=np.int64)

        ends = (
            intervals[["end_time"]]
            .sort_values("end_time")
            .reset_index(drop=True)
        )
        ends["n_ends"] = np.arange(1, len(ends) + 1, dtype=np.int64)

        # Use merge_asof on numeric simulation time
        ev_sorted = events.sort_values("Complete Timestamp").reset_index(drop=True)

        ev_sorted = pd.merge_asof(
            ev_sorted,
            starts,
            left_on="Complete Timestamp",
            right_on="start_time",
            direction="backward",
        )
        ev_sorted = pd.merge_asof(
            ev_sorted,
            ends,
            left_on="Complete Timestamp",
            right_on="end_time",
            direction="backward",
        )

        ev_sorted["n_starts"] = ev_sorted["n_starts"].fillna(0)
        ev_sorted["n_ends"]   = ev_sorted["n_ends"].fillna(0)

        ev_sorted["open_cases"] = (
            ev_sorted["n_starts"] - ev_sorted["n_ends"]
        ).astype(int)

        # Drop helper cols and restore
        ev_sorted = ev_sorted.drop(
            columns=[c for c in ["start_time", "end_time", "n_starts", "n_ends"]
                     if c in ev_sorted.columns]
        )
        events = ev_sorted

    # 6) Rename to Verenich-style columns and select
    events["Case ID"] = events["case_id"]
    events["Activity"] = events["activity"]
    events["Resource"] = events["resource"]

    keep_cols = [
        "Case ID",
        "Activity",
        "Resource",
        "Complete Timestamp",
        "activity_duration",
        "timesincelastevent",
        "timesincecasestart",
        "event_nr",
        "open_cases",
        "remtime",
    ]

    # Keep scenario/meta info if present (won't be used by their code but useful for debugging)
    for extra in ["scenario", "simulation_run", "process", "method"]:
        if extra in events.columns:
            keep_cols.append(extra)

    logdata = events[keep_cols].copy()

    print(f"  Logdata rows: {len(logdata)}, "
          f"cases: {logdata['Case ID'].nunique()}")
    return logdata


def main():
    scenario_to_df = load_scenario_splits(SPLIT_FOLDER)

    os.makedirs(LOGDATA_FOLDER, exist_ok=True)

    for scenario, df_all in scenario_to_df.items():
        print(f"\n--- Building logdata for scenario: {scenario} ---")
        logdata = build_logdata(df_all)

        out_name = f"{OUTPUT_PREFIX}{scenario}.csv"
        out_path = os.path.join(LOGDATA_FOLDER, out_name)
        logdata.to_csv(out_path, sep=";", index=False)
        print(f"  Saved Verenich-style logdata -> {out_path}")


if __name__ == "__main__":
    main()
