import os
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
RUN_TAG = "251110"
RESULTS_DIR = f"out/{RUN_TAG}/results"

### CHANGE 1: Point to the UNIFORM descriptors from Step 2 ###
DESC_FILE = os.path.join(RESULTS_DIR, "FIFO_EXP_path_descriptors_UNIFORM.csv")

# ----- choose entropy axis here -----
ENTROPY_COL = "H_norm"          
D_COL = "D_paths"

# Updated Proxies (Columns that exist in the UNIFORM descriptor file)
PROXIES = ["var_total", "K_paths", "K_paths_act", "mu_total"]

# Quantile cutpoints
Q_LOW, Q_HIGH = (1/3, 2/3)

# Optional: benchmark scheme
PROXY_OPT_D_EDGES = (0.15, 0.45)
PROXY_OPT_H_EDGES = (0.85, 0.95)

### CHANGE 2: Updated Output paths to include "UNIFORM" ###
OUT_QUANT_ALLOC = os.path.join(RESULTS_DIR, f"FIFO_EXP_alloc_3x3_QUANT_UNIFORM_{ENTROPY_COL}_vs_{D_COL}.csv")
OUT_PROXY_ALLOC = os.path.join(RESULTS_DIR, f"FIFO_EXP_alloc_3x3_PROXYOPT_UNIFORM_{ENTROPY_COL}_vs_{D_COL}.csv")

# -----------------------------
# Helper Functions (Stay the same)
# -----------------------------
def cut3_LMH(x: pd.Series, a: float, b: float) -> pd.Categorical:
    return pd.cut(x.astype(float), bins=[-np.inf, a, b, np.inf], labels=["L", "M", "H"], include_lowest=True)

def cell_id(e_bin: str, d_bin: str) -> int:
    e_map, d_map = {"L": 0, "M": 1, "H": 2}, {"L": 0, "M": 1, "H": 2}
    return 3 * e_map[e_bin] + d_map[d_bin] + 1

def compute_proxy_diagnostics(tmp: pd.DataFrame) -> dict:
    mat = pd.crosstab(tmp["E_bin"], tmp["D_bin"], dropna=False).reindex(index=["L","M","H"], columns=["L","M","H"], fill_value=0)
    grp = tmp.groupby(["E_bin","D_bin"], observed=True)[PROXIES].mean().reindex(pd.MultiIndex.from_product([["L","M","H"], ["L","M","H"]]))
    spreads_norm = {}
    for p in PROXIES:
        if grp[p].isna().any(): spreads_norm[p] = np.nan; continue
        sd = float(tmp[p].std(ddof=0))
        spreads_norm[p] = ((grp[p].max() - grp[p].min()) / sd) if sd > 1e-12 else 0.0
    return {"counts": mat, "spreads_norm": spreads_norm, "min_cell": int(mat.min().min())}

def allocate(df: pd.DataFrame, d_edges: tuple[float,float], e_edges: tuple[float,float], scheme_name: str) -> pd.DataFrame:
    tmp = df.copy()
    d1, d2 = d_edges
    e1, e2 = e_edges
    tmp["D_cut1"], tmp["D_cut2"], tmp["E_cut1"], tmp["E_cut2"] = float(d1), float(d2), float(e1), float(e2)
    tmp["D_bin"] = cut3_LMH(tmp[D_COL], d1, d2)
    tmp["E_bin"] = cut3_LMH(tmp[ENTROPY_COL], e1, e2)
    tmp["cell_id"] = [cell_id(e, d) for e, d in zip(tmp["E_bin"].astype(str), tmp["D_bin"].astype(str))]
    tmp["cell_label"] = "E_" + tmp["E_bin"].astype(str) + "__D_" + tmp["D_bin"].astype(str)
    tmp["scheme"], tmp["entropy_col"], tmp["d_col"] = scheme_name, ENTROPY_COL, D_COL
    return tmp

def main():
    if not os.path.isfile(DESC_FILE):
        raise FileNotFoundError(f"Uniform descriptor file not found: {DESC_FILE}")

    df = pd.read_csv(DESC_FILE).replace([np.inf, -np.inf], np.nan)
    need = ["log_name", D_COL, ENTROPY_COL] + [p for p in PROXIES if p in df.columns]
    df = df.dropna(subset=["log_name", D_COL, ENTROPY_COL]).copy()

    # SCHEME A: Quantile cutpoints
    d1, d2 = float(df[D_COL].quantile(Q_LOW)), float(df[D_COL].quantile(Q_HIGH))
    e1, e2 = float(df[ENTROPY_COL].quantile(Q_LOW)), float(df[ENTROPY_COL].quantile(Q_HIGH))

    quant = allocate(df, (d1, d2), (e1, e2), scheme_name="QUANTILES_Q33_Q66_UNIFORM")
    
    # Report
    diag = compute_proxy_diagnostics(quant)
    print(f"\n=== UNIFORM Binning Report ({ENTROPY_COL} vs {D_COL}) ===")
    print(f"Cutpoints: D({d1:.4f}, {d2:.4f}) | E({e1:.4f}, {e2:.4f})")
    print("\n3x3 Counts:")
    print(diag["counts"])

    # Save
    quant.to_csv(OUT_QUANT_ALLOC, index=False)
    print(f"\nSaved UNIFORM quantile allocations to: {OUT_QUANT_ALLOC}")

if __name__ == "__main__":
    main()