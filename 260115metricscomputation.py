#!/usr/bin/env python

import os
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
RUN_TAG  = "251110"
OUT_ROOT = f"out/{RUN_TAG}"
RESULTS_DIR = os.path.join(OUT_ROOT, "results")

### CHANGE 1: Point to the UNIFORM logs created in Step 1 ###
INPUT_LOG_DIR = os.path.join(OUT_ROOT, "stable_logs_uniform_trimmed")

### CHANGE 2: Set Warm-up to 0 (Step 1 already applied it) ###
# If we keep it at 0.17, we will lose 17% of our N_min cases.
WARMUP_FRAC = 0.0 

# Keep math functions exactly the same to ensure no "logic drift"
def pop_var(x: pd.Series) -> float:
    x = x.to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    mu = x.mean()
    return ((x - mu) ** 2).mean()

def _normalized_shannon(pi: np.ndarray) -> tuple[float, float, int]:
    pi = pi[pi > 0]
    K = len(pi)
    if K == 0: return 0.0, 0.0, 0
    H = -np.sum(pi * np.log(pi))
    H_norm = H / np.log(K) if K > 1 else 0.0
    return float(H), float(H_norm), K

def _normalized_renyi2(pi: np.ndarray) -> tuple[float, float]:
    pi = pi[pi > 0]
    K = len(pi)
    if K == 0: return 0.0, 0.0
    coll = np.sum(pi ** 2)
    H2 = -np.log(coll)
    H2_norm = H2 / np.log(K) if K > 1 else 0.0
    return float(H2), float(H2_norm)

def compute_descriptors_for_file(csv_path: str, warmup_frac: float = 0.0) -> dict:
    print(f"--- Processing {os.path.basename(csv_path)} ---")
    df = pd.read_csv(csv_path)

    # Note: df_steady_complete will be the same as df because of Step 1 logic
    df_start = df[df["status"] == "START"][["case_id", "timestamp"]].copy()
    df_start = df_start.rename(columns={"timestamp": "start_time"})
    df_complete = df[(df["status"] == "COMPLETE")].copy()
    df_complete = df_complete.merge(df_start, on="case_id", how="left")

    t_max = df["timestamp"].max()
    warmup_threshold = warmup_frac * t_max
    df_steady_complete = df_complete[df_complete["start_time"] >= warmup_threshold].copy()
    
    steady_case_ids = df_steady_complete["case_id"].unique()
    df_steady = df[df["case_id"].isin(steady_case_ids)].copy()

    cycle_times = df_steady_complete["cycle_time"].to_numpy(dtype=float)
    mu_total = cycle_times.mean()
    var_total = ((cycle_times - mu_total) ** 2).mean()

    df_run = df_steady[df_steady["status"] == "running"].copy()
    df_run = df_run.sort_values(["case_id", "timestamp", "activity", "resource"])

    # MACHINE-LEVEL PATHS
    def build_machine_path(group: pd.DataFrame) -> str:
        nodes = group["resource"].dropna().astype(str).tolist()
        return ">".join(nodes) if nodes else None

    machine_paths = df_run.groupby("case_id").apply(build_machine_path).reset_index(name="path")
    machine_paths = machine_paths.dropna(subset=["path"])

    # ACTIVITY-LEVEL PATHS
    def normalize_activity_name(activity_str: str) -> str:
        parts = activity_str.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit(): return parts[0]
        return activity_str

    def build_activity_path(group: pd.DataFrame) -> str:
        return ">".join(group["activity"].astype(str).apply(normalize_activity_name).tolist())

    activity_paths = df_run.groupby("case_id").apply(build_activity_path).reset_index(name="act_path")

    # VARIANCE DECOMPOSITION
    case_level = df_steady_complete[["case_id", "cycle_time"]].merge(machine_paths, on="case_id", how="inner")
    N = len(case_level)
    
    path_stats = case_level.groupby("path")["cycle_time"].agg(n="count", mean="mean", var=pop_var).reset_index()
    weights = path_stats["n"] / N
    mu_from_paths = (path_stats["mean"] * path_stats["n"]).sum() / N
    Var_between = ((path_stats["mean"] - mu_from_paths) ** 2 * weights).sum()
    Var_within  = (path_stats["var"] * weights).sum()
    D_paths = Var_between / var_total if var_total > 0 else np.nan

    pi_paths = (path_stats["n"] / N).to_numpy()
    H_paths, H_norm_paths, K_paths = _normalized_shannon(pi_paths)
    H2_paths, H2_norm_paths = _normalized_renyi2(pi_paths)

    # ACTIVITY ENTROPIES
    case_level_act = df_steady_complete[["case_id"]].merge(activity_paths, on="case_id", how="inner")
    act_stats = case_level_act.groupby("act_path")["case_id"].agg(n="count").reset_index()
    pi_act = (act_stats["n"] / len(case_level_act)).to_numpy()
    H_act, H_norm_act, K_paths_act = _normalized_shannon(pi_act)

    return {
        "n_steady_cases": N,
        "mu_total": mu_total,
        "var_total": var_total,
        "D_paths": D_paths,
        "H_norm": H_norm_paths,
        "K_paths": K_paths,
        "H_norm_act": H_norm_act,
        "K_paths_act": K_paths_act
    }

def parse_log_name(log_name: str) -> dict:
    # Logic remains the same to parse file parameters
    base = os.path.basename(log_name).replace(".csv", "").replace("FIFO_EXP_", "")
    parts = base.split("_")
    params = {"log_name": log_name}
    try:
        params["l"] = float(parts[0][1:])
        params["style"] = parts[1]
        params["count_preset"] = parts[2]
        params["variant_count"] = int(parts[3][1:])
        params["activity_total"] = int(parts[4][1:])
        params["qc_level"] = int(parts[5][2:]) / 100.0
        params["hetero"] = parts[6] if len(parts) > 6 else None
    except: pass
    return params

def main():
    # Loop through the files in the Step 1 output directory
    log_files = [f for f in os.listdir(INPUT_LOG_DIR) if f.endswith(".csv")]
    
    rows = []
    for f in log_files:
        full_path = os.path.join(INPUT_LOG_DIR, f)
        stats = compute_descriptors_for_file(full_path, warmup_frac=WARMUP_FRAC)
        params = parse_log_name(f)
        stats.update(params)
        rows.append(stats)

    summary_df = pd.DataFrame(rows)

    ### CHANGE 3: Save to a NEW filename to preserve original results ###
    out_path = os.path.join(
        RESULTS_DIR,
        "FIFO_EXP_path_descriptors_UNIFORM.csv"
    )
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved UNIFORM summary to {out_path}")

if __name__ == "__main__":
    main()