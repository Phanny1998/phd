import os
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
RUN_TAG = "251110"
OUT_ROOT = f"out/{RUN_TAG}"
RAW_DIR = os.path.join(OUT_ROOT, "stable_logs_raw") 
TRIMMED_DIR = os.path.join(OUT_ROOT, "stable_logs_uniform_trimmed")

WARMUP_FRAC = 0.17
RANDOM_SEED = 42
COMPLETE_STATUS = "COMPLETE"
START_STATUS = "START"

os.makedirs(TRIMMED_DIR, exist_ok=True)

def compute_true_wip(df):
    """Calculates WIP based on ALL cases in the dataframe before any are deleted."""
    # Get the global start and end time for every case in the raw file
    case_times = df.groupby("case_id")["timestamp"].agg(["min", "max"]).reset_index()
    
    # We define a case as "open" from its START event until its COMPLETE event
    starts = case_times["min"].sort_values().values
    ends = case_times["max"].sort_values().values
    
    # Sort df by timestamp to use searchsorted for speed
    df = df.sort_values("timestamp")
    current_times = df["timestamp"].values
    
    # How many started at or before this time minus how many ended strictly before this time
    n_started = np.searchsorted(starts, current_times, side='right')
    n_ended = np.searchsorted(ends, current_times, side='left')
    
    df["open_cases_true"] = n_started - n_ended
    return df

# 1. Pass One: Identify valid cases and find global N_min
log_data_info = {}
log_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

print(f"Scanning {len(log_files)} logs for valid steady-state completed cases...")

for f in log_files:
    df = pd.read_csv(os.path.join(RAW_DIR, f))
    
    t_max = df["timestamp"].max()
    warmup_threshold = WARMUP_FRAC * t_max
    
    # Filter IDs: Steady-state AND Completed
    steady_ids = set(df[(df["status"] == START_STATUS) & (df["timestamp"] >= warmup_threshold)]["case_id"].unique())
    complete_ids = set(df[df["status"] == COMPLETE_STATUS]["case_id"].unique())
    valid_ids = list(steady_ids.intersection(complete_ids))
    
    if len(valid_ids) > 0:
        log_data_info[f] = valid_ids
    else:
        print(f"  Warning: {f} has no valid cases.")

N_MIN = min(len(v) for v in log_data_info.values())
print(f"\nGlobal Minimum Cases: {N_MIN}. Calculating True WIP and Sampling...")
print("-" * 50)

# 2. Pass Two: Calculate True WIP on full data, then sample N_MIN
for f, valid_ids in log_data_info.items():
    df = pd.read_csv(os.path.join(RAW_DIR, f))
    
    # --- CRITICAL: Calculate WIP on the FULL log before sampling ---
    df = compute_true_wip(df)
    
    # Randomly sample exactly N_MIN from the valid candidates
    np.random.seed(RANDOM_SEED)
    keep_ids = np.random.choice(valid_ids, size=N_MIN, replace=False)
    
    # Filter to only the rows for these sampled cases
    # These rows now carry the 'open_cases_true' value from the original context
    df_trimmed = df[df["case_id"].isin(keep_ids)].copy()
    
    output_path = os.path.join(TRIMMED_DIR, f)
    df_trimmed.to_csv(output_path, index=False)
    print(f"Saved: {f} (N={N_MIN}, WIP Context Preserved)")

print(f"\nStep 1 Complete. Files in {TRIMMED_DIR} are raw format with 'open_cases_true' column.")