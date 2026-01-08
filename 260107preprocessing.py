#!/usr/bin/env python

import os
import shutil
import numpy as np
import pandas as pd

"""
Preprocessing for MuProMAC logs (final version):

For each STABLE log:
  1) Load raw MuProMAC event log
  2) (Optional) Keep only cases that have a COMPLETE event
  3) Convert to "completion event stream" (running rows only)
  4) Compute per-event features:
        - activity_duration
        - timesincelastevent  
        - timesincecasestart (elapsed from first event)
        - event_nr (position in case)
        - open_cases (WIP at this timestamp)
  5) Compute case-level remaining time (label)
  6) Warm-up trim: drop first WARMUP_FRAC of cases
  7) Write single event-level CSV per log

The pipeline will then:
  - split_data() for train/test split (80/20)
  - generate_prefix_data() to create prefixes
"""

# -----------------------------
# CONFIG
# -----------------------------
RUN_TAG = "251110"
OUT_ROOT = f"out/{RUN_TAG}"
RESULTS_DIR = os.path.join(OUT_ROOT, "results")

DESC_FILE = os.path.join(RESULTS_DIR, "FIFO_EXP_path_descriptors_machine_and_activity_level_STABLE.csv")

COPY_RAW_LOGS = True
RAW_DEST_DIR = os.path.join(OUT_ROOT, "stable_logs_raw")

# Output event logs (single file per log, NO prefixes, NO splits)
EVENT_OUT_DIR = os.path.join(OUT_ROOT, "event_logs_processed_v2")

# Warm-up trim: drop first X fraction of cases
WARMUP_FRAC = 0.17

# If True, keep only cases that have a COMPLETE event
REQUIRE_COMPLETE = True

# Required columns in raw MuProMAC logs
REQUIRED_RAW_COLS = ["case_id", "timestamp", "status", "end_time", "activity", "resource"]

RUNNING_STATUS = "running"
COMPLETE_STATUS = "COMPLETE"


# -----------------------------
# Helpers
# -----------------------------
def normalize_log_name_to_csv_candidates(log_name: str) -> list[str]:
    log_name = str(log_name).strip()
    cands = [log_name]
    if not log_name.lower().endswith(".csv"):
        cands.append(log_name + ".csv")
    return cands


def find_log_file(log_name: str) -> str | None:
    candidates = normalize_log_name_to_csv_candidates(log_name)
    
    # 1) direct in results dir
    for cand in candidates:
        direct = os.path.join(RESULTS_DIR, cand)
        if os.path.isfile(direct):
            return direct
    
    # 2) recursive in OUT_ROOT
    cand_set = set(candidates)
    for root, _, files in os.walk(OUT_ROOT):
        for f in files:
            if f in cand_set:
                return os.path.join(root, f)
    
    return None


def strip_csv_suffix(path_or_name: str) -> str:
    fn = os.path.basename(path_or_name)
    return fn[:-4] if fn.lower().endswith(".csv") else fn


# -----------------------------
# Core logic
# -----------------------------
def keep_complete_cases_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only cases that have at least one COMPLETE row."""
    if not REQUIRE_COMPLETE:
        return df_raw
    complete_cases = set(df_raw.loc[df_raw["status"] == COMPLETE_STATUS, "case_id"].unique())
    return df_raw[df_raw["case_id"].isin(complete_cases)].copy()


def build_completion_stream(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 'running' rows (activity executions).
    Use end_time as the completion timestamp.
    """
    events = df_raw[df_raw["status"] == RUNNING_STATUS].copy()
    if events.empty:
        return events
    
    # Completion timestamp
    events["Complete Timestamp"] = events["end_time"]
    
    # Activity duration
    events["activity_duration"] = (events["end_time"] - events["timestamp"]).fillna(0.0)
    
    # Sort for stable behavior
    events = events.sort_values(["case_id", "Complete Timestamp"]).reset_index(drop=True)
    return events


def compute_case_times(events: pd.DataFrame) -> pd.DataFrame:
    """Add case start/end times based on completion timestamps."""
    case_times = (
        events.groupby("case_id")["Complete Timestamp"]
        .agg(["min", "max"])
        .rename(columns={"min": "case_start_time", "max": "case_complete_time"})
        .reset_index()
    )
    events = events.merge(case_times, on="case_id", how="left")
    return events


def compute_event_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-event features matching Verenich pipeline expectations:
    - event_nr: position in case (1, 2, 3, ...)
    - timesincelastevent: time since previous event in case
    - timesincecasestart: elapsed time from case start
    """
    events = events.sort_values(["case_id", "Complete Timestamp"]).reset_index(drop=True)
    
    # Event number within case
    events["event_nr"] = events.groupby("case_id").cumcount() + 1
    
    # Time since last event
    events["timesincelastevent"] = events.groupby("case_id")["Complete Timestamp"].diff().fillna(0.0)
    
    # Time since case start
    events["timesincecasestart"] = events["Complete Timestamp"] - events["case_start_time"]
    
    return events


def compute_open_cases(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute open_cases (WIP) at each event timestamp.
    This is computed on the FULL set of cases (after warmup trim, before train/test split).
    Matches original Verenich methodology.
    """
    case_times = events[["case_id", "case_start_time", "case_complete_time"]].drop_duplicates()
    
    if case_times.empty:
        events["open_cases"] = 0
        return events
    
    # Rename to avoid collision with merge_asof keys
    starts = case_times[["case_start_time"]].copy()
    starts = starts.rename(columns={"case_start_time": "start_time"})
    starts = starts.sort_values("start_time").reset_index(drop=True)
    starts["n_starts"] = np.arange(1, len(starts) + 1, dtype=np.int64)
    
    ends = case_times[["case_complete_time"]].copy()
    ends = ends.rename(columns={"case_complete_time": "end_time"})
    ends = ends.sort_values("end_time").reset_index(drop=True)
    ends["n_ends"] = np.arange(1, len(ends) + 1, dtype=np.int64)
    
    ev = events.sort_values("Complete Timestamp").reset_index(drop=True)
    
    # Merge with RENAMED columns so we don't overwrite case_start_time/case_complete_time
    ev = pd.merge_asof(ev, starts, left_on="Complete Timestamp", right_on="start_time", direction="backward")
    ev = pd.merge_asof(ev, ends, left_on="Complete Timestamp", right_on="end_time", direction="backward")
    
    ev["n_starts"] = ev["n_starts"].fillna(0)
    ev["n_ends"] = ev["n_ends"].fillna(0)
    ev["open_cases"] = (ev["n_starts"] - ev["n_ends"]).astype(int)
    
    # Drop only the merge artifacts and temporary columns (ignore if they don't exist)
    ev = ev.drop(columns=["start_time", "end_time", "n_starts", "n_ends"], errors='ignore')
    
    return ev


def add_remaining_time_label(events: pd.DataFrame) -> pd.DataFrame:
    """
    Add remtime label: time from current event completion to case completion.
    This will be used as the regression target.
    """
    events["remtime"] = events["case_complete_time"] - events["Complete Timestamp"]
    return events


def warmup_trim_cases(events: pd.DataFrame) -> pd.DataFrame:
    """Drop first WARMUP_FRAC of cases ordered by case_start_time."""
    case_times = events[["case_id", "case_start_time"]].drop_duplicates()
    case_times_sorted = case_times.sort_values("case_start_time").reset_index(drop=True)
    
    n = len(case_times_sorted)
    k_drop = int(np.floor(WARMUP_FRAC * n))
    kept_cases = case_times_sorted["case_id"].iloc[k_drop:].tolist()
    
    events_trimmed = events[events["case_id"].isin(kept_cases)].copy()
    
    print(f"     Warm-up trim: dropped {k_drop}/{n} cases, kept {len(kept_cases)}")
    return events_trimmed
def extract_activity_stage(events: pd.DataFrame) -> pd.DataFrame:
    """
    Clean activity column by removing machine identifiers.
    
    Converts:
    - MOULDING_5 → MOULDING
    - ASSEMBLY_1_3 → ASSEMBLY_1
    - ASSEMBLY_2_2 → ASSEMBLY_2
    - SORTING → SORTING (no change)
    - PACKAGING_2 → PACKAGING
    - INSPECTION_1_2 → INSPECTION_1
    - INSPECTION_2_3 → INSPECTION_2
    
    Resource column already contains machine-specific information.
    """
    def parse_activity_stage(act_label):
        parts = act_label.split('_')
        
        # Single part (e.g., SORTING with no machine number)
        if len(parts) == 1:
            return act_label
        
        # Core activities: MOULDING_5, PACKAGING_2, etc. → ACTIVITY
        if parts[0] in ['MOULDING', 'SORTING', 'PACKAGING']:
            return parts[0]
        
        # ASSEMBLY: ASSEMBLY_1_3 → ASSEMBLY_1
        if parts[0] == 'ASSEMBLY' and len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        
        # INSPECTION: INSPECTION_1_2 → INSPECTION_1
        if parts[0] == 'INSPECTION' and len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        
        # Fallback
        return act_label
    
    # Modify activity column in-place
    events['activity'] = events['activity'].apply(parse_activity_stage)
    
    return events
def prepare_output_format(events: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order columns to match pipeline expectations.
    Activity column is cleaned to contain activity names only.
    Resource column contains machine-specific information.
    """
    # Clean activity column (remove machine identifiers)
    events = extract_activity_stage(events)
    
    output_cols = [
        "case_id",
        "activity",
        "resource",
        "Complete Timestamp",
        "activity_duration",
        "timesincelastevent",
        "timesincecasestart",
        "event_nr",
        "open_cases",
        "remtime",
    ]
    
    # Add metadata columns if present
    for extra in ["scenario", "method", "l", "simulation_run", "process"]:
        if extra in events.columns and extra not in output_cols:
            output_cols.append(extra)
    
    return events[output_cols].copy()


def write_output(events: pd.DataFrame, base: str, out_dir: str):
    """Write single processed event log."""
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, f"{base}.csv")
    
    # Use semicolon separator to match Verenich pipeline's read_csv
    events.to_csv(out_file, index=False, sep=";")
    
    return out_file


# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.isfile(DESC_FILE):
        raise FileNotFoundError(f"Descriptor file not found: {DESC_FILE}")
    
    df_desc = pd.read_csv(DESC_FILE).replace([np.inf, -np.inf], np.nan)
    if "log_name" not in df_desc.columns:
        raise ValueError("Descriptor file missing required column: log_name")
    
    log_names = sorted(df_desc["log_name"].dropna().unique().tolist())
    
    if COPY_RAW_LOGS:
        os.makedirs(RAW_DEST_DIR, exist_ok=True)
    os.makedirs(EVENT_OUT_DIR, exist_ok=True)
    
    print(f"\nProcessing {len(log_names)} STABLE logs from:\n  {DESC_FILE}\n")
    print(f"Warm-up fraction: {WARMUP_FRAC}")
    print(f"NOTE: No train/test split - pipeline handles this")
    print(f"NOTE: No prefix generation - pipeline handles this\n")
    
    missing_files = []
    processed = 0
    skipped = 0
    
    for log_name in log_names:
        print(f"\n{'='*70}")
        print(f"Processing: {log_name}")
        
        src = find_log_file(log_name)
        if src is None:
            missing_files.append(log_name)
            print(f"[SKIP] File not found")
            continue
        
        # Copy raw log
        if COPY_RAW_LOGS:
            dst_raw = os.path.join(RAW_DEST_DIR, os.path.basename(src))
            shutil.copy2(src, dst_raw)
            read_path = dst_raw
        else:
            read_path = src
        
        base = strip_csv_suffix(read_path)
        
        # Load raw log
        df_raw = pd.read_csv(read_path)
        missing_cols = [c for c in REQUIRED_RAW_COLS if c not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"{read_path} missing required columns: {missing_cols}")
        
        print(f"     Raw events: {len(df_raw)}, cases: {df_raw['case_id'].nunique()}")
        
        # 1) Filter complete cases
        df_raw = keep_complete_cases_raw(df_raw)
        if df_raw.empty:
            print(f"[SKIP] Empty after COMPLETE filtering")
            skipped += 1
            continue
        
        # 2) Build completion stream
        events = build_completion_stream(df_raw)
        if events.empty:
            print(f"[SKIP] No '{RUNNING_STATUS}' rows")
            skipped += 1
            continue
        
        print(f"     Completion events: {len(events)}, cases: {events['case_id'].nunique()}")
        
        # 3) Compute case times
        events = compute_case_times(events)
        
        # 4) Compute event features
        events = compute_event_features(events)
        
        # 5) Warm-up trim BEFORE computing open_cases
        events = warmup_trim_cases(events)
        if events.empty:
            print(f"[SKIP] Empty after warm-up trim")
            skipped += 1
            continue
        
        # 6) Compute open_cases on trimmed set
        events = compute_open_cases(events)
        
        # 7) Add remaining time label
        events = add_remaining_time_label(events)
        
        # 8) Format output
        events = prepare_output_format(events)
        
        # 9) Write single file
        out_file = write_output(events, base, EVENT_OUT_DIR)
        
        print(f"[OK] Wrote: {out_file}")
        print(f"     Cases: {events['case_id'].nunique()}, Events: {len(events)}")
        
        processed += 1
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Processed: {processed}/{len(log_names)}")
    print(f"  Skipped: {skipped}")
    print(f"  Output directory: {EVENT_OUT_DIR}")
    if COPY_RAW_LOGS:
        print(f"  Raw logs copied to: {RAW_DEST_DIR}")
    
    if missing_files:
        print(f"\nWARNING: Could not find {len(missing_files)} log files:")
        for m in missing_files[:10]:
            print(f"  - {m}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files)-10} more")


if __name__ == "__main__":
    main()