import os
import pandas as pd
import numpy as np

RUN_TAG  = "251110"
OUT_ROOT = f"out/{RUN_TAG}"
RESULTS_DIR = os.path.join(OUT_ROOT, "results")

# === match fifo.py experiment menu ===
l_values        = [0.28]                        # ARRIVAL_RATES in config
styles          = ["pooled", "hybrid30", "dedicated"]
count_keys      = ["C1"]                        # same as counts in fifo
variant_counts  = [6]                           # variants
activity_totals = [5, 8, 12]                    # acts
qc_levels       = [0.97, 0.99, 1.0, 0.5, 0.7]   # qc_lvls
hets            = ["identical", "mild_all", "strong_all"]  # hets

# Same warm-up fraction
WARMUP_FRAC = 0.17


def pop_var(x: pd.Series) -> float:
    x = x.to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    mu = x.mean()
    return ((x - mu) ** 2).mean()


def _normalized_shannon(pi: np.ndarray) -> tuple[float, float, int]:
    """
    Return (H, H_norm, K) where:
      - H = Shannon entropy
      - H_norm = H / log(K)   (0 if K <= 1)
      - K = number of non-zero-probability states
    """
    # filter out zeros just in case
    pi = pi[pi > 0]
    K = len(pi)
    if K == 0:
        return 0.0, 0.0, 0
    H = -np.sum(pi * np.log(pi))
    if K <= 1:
        H_norm = 0.0
    else:
        H_norm = H / np.log(K)
    return float(H), float(H_norm), K


def _normalized_renyi2(pi: np.ndarray) -> tuple[float, float]:
    """
    Return (H2, H2_norm) where:
      - H2 = Rényi-2 entropy = -log(sum pi^2)
      - H2_norm = H2 / log(K)
    Uses K = number of non-zero-probability states.
    """
    pi = pi[pi > 0]
    K = len(pi)
    if K == 0:
        return 0.0, 0.0
    coll = np.sum(pi ** 2)        # collision probability
    H2 = -np.log(coll)            # Rényi-2
    if K <= 1:
        H2_norm = 0.0
    else:
        H2_norm = H2 / np.log(K)
    return float(H2), float(H2_norm)


def compute_descriptors_for_file(csv_path: str, warmup_frac: float = 0.17) -> dict:
    print(f"\n=== Processing {os.path.basename(csv_path)} ===")
    df = pd.read_csv(csv_path)

    # -------- warm-up selection on cases (KEEP THIS) --------
    df_start = df[df["status"] == "START"][["case_id", "timestamp"]].copy()
    df_start = df_start.rename(columns={"timestamp": "start_time"})

    df_complete = df[(df["status"] == "COMPLETE") & (df["activity"] == "END")].copy()
    if df_complete.empty:
        raise ValueError(f"No COMPLETE END events in {csv_path}")

    df_complete = df_complete.merge(df_start, on="case_id", how="left")

    t_max = df["timestamp"].max()
    warmup_threshold = warmup_frac * t_max
    df_steady_complete = df_complete[df_complete["start_time"] >= warmup_threshold].copy()
    if df_steady_complete.empty:
        raise ValueError(f"No cases left after warm-up in {csv_path}")

    steady_case_ids = df_steady_complete["case_id"].unique()
    df_steady = df[df["case_id"].isin(steady_case_ids)].copy()

    # -------- overall cycle-time stats (KEEP THIS) --------
    cycle_times = df_steady_complete["cycle_time"].to_numpy(dtype=float)
    mu_total = cycle_times.mean()
    var_total = ((cycle_times - mu_total) ** 2).mean()

    # -------- build MACHINE-LEVEL + ACTIVITY-LEVEL paths from running events --------
    df_run = df_steady[df_steady["status"] == "running"].copy()
    if df_run.empty:
        raise ValueError(f"No 'running' events for steady cases in {csv_path}")

    df_run = df_run.sort_values(["case_id", "timestamp", "activity", "resource"])

    # machine-level path: resource if available, else activity
    def build_machine_path(group: pd.DataFrame) -> str:
        nodes = []
        for _, row in group.iterrows():
            res = row.get("resource", None)
            node = res if isinstance(res, str) and res != "" else str(row["activity"])
            nodes.append(node)
        return ">".join(nodes)

    # activity-level path: ignore machine, use activity label only
    def build_activity_path(group: pd.DataFrame) -> str:
        return ">".join(group["activity"].astype(str).tolist())

    machine_paths = (
        df_run.groupby("case_id", group_keys=False)
              .apply(build_machine_path)
              .reset_index(name="path")
    )

    activity_paths = (
        df_run.groupby("case_id", group_keys=False)["activity"]
              .apply(lambda col: ">".join(col.astype(str)))
              .reset_index(name="act_path")
    )

    # -------- case-level table for machine paths (as before) --------
    case_level = df_steady_complete[["case_id", "cycle_time", "start_time"]].merge(
        machine_paths, on="case_id", how="left"
    )
    case_level = case_level.dropna(subset=["path"]).copy()
    N = len(case_level)
    if N == 0:
        raise ValueError(f"No cases with machine-level paths in {csv_path}")

    # -------- variance decomposition by machine-level path (unchanged) --------
    path_stats = (
        case_level.groupby("path")["cycle_time"]
                  .agg(n="count", mean="mean", var=pop_var)
                  .reset_index()
    )

    N_check = path_stats["n"].sum()
    assert N_check == N

    weights = path_stats["n"] / N_check
    mu_from_paths = (path_stats["mean"] * path_stats["n"]).sum() / N_check

    Var_between = ((path_stats["mean"] - mu_from_paths) ** 2 * weights).sum()
    Var_within  = (path_stats["var"] * weights).sum()
    Var_total_check = Var_between + Var_within

    if var_total > 0:
        D_paths = Var_between / var_total
    else:
        D_paths = np.nan

    # -------- machine-level path entropies (Shannon + Rényi-2) --------
    pi_paths = (path_stats["n"] / N_check).to_numpy(dtype=float)

    H_paths, H_norm_paths, K_paths = _normalized_shannon(pi_paths)
    H2_paths, H2_norm_paths = _normalized_renyi2(pi_paths)

    # -------- activity-level path distribution + entropies --------
    case_level_act = (
        df_steady_complete[["case_id"]]
        .merge(activity_paths, on="case_id", how="left")
        .dropna(subset=["act_path"])
        .copy()
    )
    N_act = len(case_level_act)
    if N_act == 0:
        # degenerate, but handle gracefully
        H_act = H_norm_act = H2_act = H2_norm_act = 0.0
        K_paths_act = 0
    else:
        act_stats = (
            case_level_act.groupby("act_path")["case_id"]
                          .agg(n="count")
                          .reset_index()
        )
        N_act_check = act_stats["n"].sum()
        assert N_act_check == N_act

        pi_act = (act_stats["n"] / N_act_check).to_numpy(dtype=float)

        H_act, H_norm_act, K_paths_act = _normalized_shannon(pi_act)
        H2_act, H2_norm_act = _normalized_renyi2(pi_act)

    print(f"  N_steady (with paths): {N}")
    print(f"  mu_total: {mu_total:.4f}, Var_total: {var_total:.4f}")
    print(f"  Var_between: {Var_between:.4f}, Var_within: {Var_within:.4f}")
    print(f"  Var_between + Var_within: {Var_total_check:.4f}")
    print(f"  K_paths (machine): {K_paths}, D_paths: {D_paths:.4f}, "
          f"H_norm (machine): {H_norm_paths:.4f}, H2_norm (machine): {H2_norm_paths:.4f}")
    print(f"  K_paths_act (activity): {K_paths_act}, "
          f"H_norm_act: {H_norm_act:.4f}, H2_norm_act: {H2_norm_act:.4f}")

    return {
        # existing metrics
        "n_steady_cases": N,
        "mu_total": mu_total,
        "var_total": var_total,
        "var_between": Var_between,
        "var_within": Var_within,
        "var_bw_plus_within": Var_total_check,
        "D_paths": D_paths,
        "H": H_paths,
        "H_norm": H_norm_paths,
        "K_paths": K_paths,
        "warmup_frac": warmup_frac,
        "warmup_threshold": warmup_threshold,
        "t_max": t_max,

        # NEW: machine-level Rényi-2
        "H2_paths": H2_paths,
        "H2_norm_paths": H2_norm_paths,

        # NEW: activity-level entropies
        "N_act_cases": N_act,
        "K_paths_act": K_paths_act,
        "H_act": H_act,
        "H_norm_act": H_norm_act,
        "H2_act": H2_act,
        "H2_norm_act": H2_norm_act,
    }


# ---------- outer loop over all experiment files ----------
rows = []
for l_value in l_values:
    for style in styles:
        for count_key in count_keys:
            for variant_count in variant_counts:
                for activity_total in activity_totals:
                    for qc_level in qc_levels:
                        for hk in hets:
                            fname = (
                                f"FIFO_EXP_l{l_value}_{style}_{count_key}_"
                                f"V{variant_count}_A{activity_total}_QC{int(qc_level*100)}_{hk}.csv"
                            )
                            csv_path = os.path.join(RESULTS_DIR, fname)
                            if not os.path.exists(csv_path):
                                print(f"WARNING: missing file, skipping: {csv_path}")
                                continue

                            stats = compute_descriptors_for_file(csv_path, warmup_frac=WARMUP_FRAC)
                            stats.update({
                                "l": l_value,
                                "style": style,
                                "count_preset": count_key,
                                "variant_count": variant_count,
                                "activity_total": activity_total,
                                "qc_level": qc_level,
                                "hetero": hk,
                            })
                            rows.append(stats)

summary_df = pd.DataFrame(rows)

print("\n=== Path-based descriptor summary (machine-level + activity-level entropies) ===")
print(summary_df[[
    "l", "style", "hetero", "variant_count", "activity_total",
    "n_steady_cases", "mu_total", "var_total",
    "var_between", "var_within", "D_paths",
    "H_norm", "H2_norm_paths",
    "H_norm_act", "H2_norm_act",
    "K_paths", "K_paths_act"
]])

out_path = os.path.join(RESULTS_DIR, "FIFO_EXP_path_descriptors_machine_and_activity_level.csv")
summary_df.to_csv(out_path, index=False)
print(f"\nSaved summary to {out_path}")
