#!/usr/bin/env python

import os
import shutil
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
RUN_TAG = "251110"
OUT_ROOT = f"out/{RUN_TAG}"
RESULTS_DIR = os.path.join(OUT_ROOT, "results")

# Point to the descriptors recalculated in Step 2
DESC_FILE = os.path.join(RESULTS_DIR, "FIFO_EXP_path_descriptors_UNIFORM.csv")

# Input: The raw MuProMAC logs that were trimmed to N_min in Step 1
RAW_SOURCE_DIR = os.path.join(OUT_ROOT, "stable_logs_uniform_trimmed")

# Output: New directory to avoid overwriting old processed logs
EVENT_OUT_DIR = os.path.join(OUT_ROOT, "event_logs_processed_UNIFORM")

# MUST BE 0.0 because Step 1 already applied the warm-up trim
WARMUP_FRAC = 0.0

REQUIRE_COMPLETE = True
REQUIRED_RAW_COLS = ["case_id", "timestamp", "status", "end_time", "activity", "resource"]

RUNNING_STATUS = "running"
COMPLETE_STATUS = "COMPLETE"

# -----------------------------
# Core Logic Functions
# -----------------------------

def extract_activity_stage(events: pd.DataFrame) -> pd.DataFrame:
    """Removes machine indices from activity labels (e.g., MOULDING_1 -> MOULDING)"""
    def parse_activity_stage(act_label):
        parts = str(act_label).split('_')
        if len(parts) == 1: return act_label
        if parts[0] in ['MOULDING', 'SORTING', 'PACKAGING']: return parts[0]
        if parts[0] in ['ASSEMBLY', 'INSPECTION'] and len(parts) >= 2: return f"{parts[0]}_{parts[1]}"
        return act_label
    events['activity'] = events['activity'].apply(parse_activity_stage)
    return events

def build_completion_stream(df_raw: pd.DataFrame) -> pd.DataFrame:
    events = df_raw[df_raw["status"] == RUNNING_STATUS].copy()
    if events.empty: return events
    events["Complete Timestamp"] = events["end_time"]
    events["activity_duration"] = (events["end_time"] - events["timestamp"]).fillna(0.0)
    return events.sort_values(["case_id", "Complete Timestamp"]).reset_index(drop=True)

def compute_case_times(events: pd.DataFrame) -> pd.DataFrame:
    case_times = events.groupby("case_id")["Complete Timestamp"].agg(["min", "max"]).rename(
        columns={"min": "case_start_time", "max": "case_complete_time"}).reset_index()
    return events.merge(case_times, on="case_id", how="left")

def compute_event_features(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["case_id", "Complete Timestamp"]).reset_index(drop=True)
    events["event_nr"] = events.groupby("case_id").cumcount() + 1
    events["timesincelastevent"] = events.groupby("case_id")["Complete Timestamp"].diff().fillna(0.0)
    events["timesincecasestart"] = events["Complete Timestamp"] - events["case_start_time"]
    return events

def compute_open_cases(events: pd.DataFrame) -> pd.DataFrame:
    case_times = events[["case_id", "case_start_time", "case_complete_time"]].drop_duplicates()
    if case_times.empty: return events
    starts = case_times[["case_start_time"]].copy().rename(columns={"case_start_time": "st"}).sort_values("st")
    starts["n_starts"] = np.arange(1, len(starts) + 1)
    ends = case_times[["case_complete_time"]].copy().rename(columns={"case_complete_time": "et"}).sort_values("et")
    ends["n_ends"] = np.arange(1, len(ends) + 1)
    ev = events.sort_values("Complete Timestamp")
    ev = pd.merge_asof(ev, starts, left_on="Complete Timestamp", right_on="st", direction="backward")
    ev = pd.merge_asof(ev, ends, left_on="Complete Timestamp", right_on="et", direction="backward")
    ev["open_cases"] = (ev["n_starts"].fillna(0) - ev["n_ends"].fillna(0)).astype(int)
    return ev.drop(columns=["st", "et", "n_starts", "n_ends"], errors='ignore')

def add_remaining_time_label(events: pd.DataFrame) -> pd.DataFrame:
    events["remtime"] = events["case_complete_time"] - events["Complete Timestamp"]
    return events

def prepare_output_format(events: pd.DataFrame) -> pd.DataFrame:
    events = extract_activity_stage(events)
    output_cols = ["case_id", "activity", "resource", "Complete Timestamp", "activity_duration", 
                   "timesincelastevent", "timesincecasestart", "event_nr", "open_cases", "remtime"]
    for extra in ["scenario", "method", "l", "simulation_run", "process", "queue_style", "hetero"]:
        if extra in events.columns: output_cols.append(extra)
    return events[output_cols].copy()

# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.exists(EVENT_OUT_DIR): os.makedirs(EVENT_OUT_DIR)
    
    # Read the uniform descriptor file to get the list of logs to process
    df_desc = pd.read_csv(DESC_FILE)
    log_names = df_desc["log_name"].unique().tolist()
    
    print(f"Preprocessing {len(log_names)} UNIFORM logs...")

    for log_name in log_names:
        input_path = os.path.join(RAW_SOURCE_DIR, log_name)
        if not os.path.exists(input_path):
            print(f" [SKIP] {log_name} not found in uniform trimmed folder.")
            continue
            
        print(f" Processing: {log_name}")
        df_raw = pd.read_csv(input_path)
        
        # We don't need to filter for 'COMPLETE' here because Step 1 already did it,
        # but build_completion_stream is still needed to extract 'running' events.
        events = build_completion_stream(df_raw)
        events = compute_case_times(events)
        events = compute_event_features(events)
        
        # Warmup trim (WARMUP_FRAC is 0.0, so this keeps everything)
        # Included for structural consistency with your original pipeline
        case_times = events[["case_id", "case_start_time"]].drop_duplicates().sort_values("case_start_time")
        k_drop = int(np.floor(WARMUP_FRAC * len(case_times)))
        keep_ids = case_times["case_id"].iloc[k_drop:].tolist()
        events = events[events["case_id"].isin(keep_ids)].copy()
        
        events = compute_open_cases(events)
        events = add_remaining_time_label(events)
        events = prepare_output_format(events)
        
        # Save as semicolon separated for the Verenich pipeline
        out_file = os.path.join(EVENT_OUT_DIR, log_name)
        events.to_csv(out_file, index=False, sep=";")
        print(f"  [OK] Wrote {len(events)} events.")

if __name__ == "__main__":
    main()