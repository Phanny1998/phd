import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import t
import numpy as np


import numpy as np
import pandas as pd
from typing import Literal, Optional, Dict, Any
from scipy import stats

def compare_methods_ttest(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    alpha: float = 0.05,
    agg: Literal["run", "case"] = "run",
    scenario: Optional[str] = None,
    l: Optional[float] = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> Dict[str, Any]:
    """
    Student's t-test (two-sample, equal variances) on cycle_time.

    Parameters
    ----------
    df : DataFrame with columns at least: ['method','cycle_time'].
         For agg='run' also needs 'simulation_run'. If present, rows with
         status!='COMPLETE' are dropped.
    method_a, method_b : method names in df['method'].
    alpha : significance level for CI and decision.
    agg : 'run' -> test uses per-run mean cycle_time (recommended),
          'case' -> uses all case-level cycle_time values.
    scenario : optional filter df['scenario']==scenario (if column exists).
    l : optional filter df['l']==l (if column exists).
    alternative : 'two-sided', 'less' (mean_a < mean_b), or 'greater' (mean_a > mean_b).

    Returns
    -------
    dict with keys:
      {'method_a','method_b','n_a','n_b','mean_a','mean_b','diff',
       'stat','pvalue','ci','alpha','significant','agg','test',
       'cohen_d','hedges_g'}
    """
    d = df.copy()

    if "status" in d.columns:
        d = d[d["status"] == "COMPLETE"]
    if scenario is not None and "scenario" in d.columns:
        d = d[d["scenario"] == scenario]
    if l is not None and "l" in d.columns:
        d = d[d["l"] == l]

    def values_for(m: str) -> np.ndarray:
        dm = d[d["method"] == m]
        if dm.empty:
            return np.array([])
        if agg == "run":
            if "simulation_run" not in dm.columns:
                raise ValueError("agg='run' requires a 'simulation_run' column.")
            return dm.groupby("simulation_run")["cycle_time"].mean().dropna().to_numpy()
        return dm["cycle_time"].dropna().to_numpy()

    a = values_for(method_a)
    b = values_for(method_b)
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"Not enough samples: {method_a} n={len(a)}, {method_b} n={len(b)}")

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    diff = mean_a - mean_b
    na, nb = len(a), len(b)

    # Student’s t-test (equal variances)
    t_stat, pval = stats.ttest_ind(a, b, equal_var=True, alternative=alternative)

    # CI for difference in means under equal variances
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    dfree = na + nb - 2
    sp2 = ((na - 1) * va + (nb - 1) * vb) / dfree          
    se_diff = np.sqrt(sp2 * (1/na + 1/nb))

    if alternative == "two-sided":
        tcrit = stats.t.ppf(1 - alpha/2, dfree)
        ci = (diff - tcrit * se_diff, diff + tcrit * se_diff)
    elif alternative == "greater":
        # H1: mean_a > mean_b → lower one-sided bound
        tcrit = stats.t.ppf(alpha, dfree)
        ci = (diff - tcrit * se_diff, np.inf)
    else:  # 'less'
        # H1: mean_a < mean_b → upper one-sided bound
        tcrit = stats.t.ppf(1 - alpha, dfree)
        ci = (-np.inf, diff + tcrit * se_diff)

    sp = np.sqrt(sp2)
    cohen_d = diff / sp if sp > 0 else np.nan
    J = 1 - 3/(4*(na+nb) - 9)  # Hedges correction
    hedges_g = J * cohen_d

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_a": int(na),
        "n_b": int(nb),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": diff,
        "stat": float(t_stat),
        "pvalue": float(pval),
        "ci": (float(ci[0]), float(ci[1])),
        "alpha": float(alpha),
        "significant": bool(pval < alpha),
        "agg": agg,
        "test": "students_equal_var",
        "cohen_d": float(cohen_d),
        "hedges_g": float(hedges_g),
    }


from scipy.stats import t
import numpy as np
import pandas as pd

def paper_tables(df: pd.DataFrame, show_process: bool = False, confidence: float = 0.95,
                 include_unfinished: bool = False):
    """
    If include_unfinished=False (default): use ONLY completed cases (no imputation).
    """

    if include_unfinished:
        # ------- original pre-processing (impute unfinished) -------
        case_start = (df.groupby(['method','l','simulation_run','case_id'])['timestamp']
                        .min().rename('case_start').reset_index())
        run_end = (df.groupby(['method','l','simulation_run'])['timestamp']
                     .max().rename('run_end').reset_index())
        df = df.merge(case_start, on=['method','l','simulation_run','case_id'], how='left')
        df = df.merge(run_end,   on=['method','l','simulation_run'],              how='left')

        completed = (df[df['status']=='COMPLETE']
                       .groupby(['method','l','simulation_run','case_id'])
                       .size().reset_index(name='complete_count'))
        idx_completed = (completed[completed['complete_count']>0]
                         .set_index(['method','l','simulation_run','case_id']).index)
        df['has_complete'] = df.set_index(['method','l','simulation_run','case_id']).index.isin(idx_completed)

        case_last = (df.groupby(['method','l','simulation_run','case_id'])['timestamp']
                       .max().rename('case_last').reset_index())
        df = df.merge(case_last, on=['method','l','simulation_run','case_id'], how='left')

        cond = df['cycle_time'].isna() & (~df['has_complete']) & (df['timestamp'] == df['case_last'])
        df.loc[cond, 'cycle_time'] = df.loc[cond, 'run_end'] - df.loc[cond, 'case_start']

        df.drop(columns=['case_start','run_end','has_complete','case_last'], inplace=True)
        df_filtered = df[df['cycle_time'].notna()].copy()

    else:
        df_filtered = df[df['status'] == 'COMPLETE'].copy()

    # ---------- aggregation ----------
    if not show_process:
        per_proc = (df_filtered
                    .groupby(['method','l','simulation_run','process'])['cycle_time']
                    .mean().reset_index(name='proc_mean'))
        gact_per_run = (per_proc
                        .groupby(['method','l','simulation_run'])['proc_mean']
                        .mean().reset_index(name='gact'))
        summary = (gact_per_run
                   .groupby(['method','l'])
                   .agg(mean_gact=('gact','mean'),
                        std_gact =('gact','std'),
                        n_runs   =('gact','count'))
                   .reset_index())
        summary['CI'] = summary.apply(
            lambda r: t.ppf((1+confidence)/2, r['n_runs']-1) * (r['std_gact']/np.sqrt(max(r['n_runs'],1)))
                      if r['n_runs'] > 1 else 0.0,
            axis=1
        )
        result = summary[['method','l','mean_gact','CI']].round(2)
        return result.pivot(index='method', columns='l', values=['mean_gact','CI']).transpose()
    else:
        act_per_run = (df_filtered
                       .groupby(['method','l','simulation_run','process'])['cycle_time']
                       .mean().reset_index(name='act'))
        summary_p = (act_per_run
                     .groupby(['method','l','process'])
                     .agg(mean_act=('act','mean'),
                          std_act =('act','std'),
                          n_runs  =('act','count'))
                     .reset_index())
        summary_p['CI'] = summary_p.apply(
            lambda r: t.ppf((1+confidence)/2, r['n_runs']-1) * (r['std_act']/np.sqrt(max(r['n_runs'],1)))
                      if r['n_runs'] > 1 else 0.0,
            axis=1
        )
        result = summary_p[['method','process','l','mean_act','CI']].round(2)
        return result.pivot(index=['method','process'], columns='l', values=['mean_act','CI']).transpose()



from scipy.stats import t
import numpy as np
import pandas as pd

def weighted_avg_cycle_time(df, l, confidence=0.95, show_process=False):
    """
    Computes the average of the weighted average cycle times per simulation run,
    and the confidence interval across runs. Optionally broken down per process.
    """
    # Filter: completed events for selected lambda
    df_filtered = df[(df['l'] == l) & (df['status'] == 'COMPLETE') & (df['cycle_time'].notnull())]

    # Group columns
    group_cols = ['method', 'simulation_run'] if not show_process else ['method', 'process', 'simulation_run']

    # Compute weighted avg per run (total duration / number of executions)
    run_avgs = (
        df_filtered
        .groupby(group_cols)
        .agg(
            total_duration=('cycle_time', 'sum'),
            executions=('cycle_time', 'count')
        )
        .reset_index()
    )
    run_avgs['weighted_avg'] = run_avgs['total_duration'] / run_avgs['executions']

    # Now compute mean and CI across runs
    if show_process:
        summary = run_avgs.groupby(['method', 'process']).agg(
            mean_weighted_avg=('weighted_avg', 'mean'),
            std=('weighted_avg', 'std'),
            n=('weighted_avg', 'count')
        ).reset_index()

        summary['CI'] = summary.apply(
            lambda row: t.ppf((1 + confidence) / 2, row['n'] - 1) * (row['std'] / np.sqrt(row['n']))
            if row['n'] > 1 else 0, axis=1
        )

        # Pivot for table format
        result = summary.pivot(index='method', columns='process', values=['mean_weighted_avg', 'CI']).round(2)
        return result.transpose(), run_avgs

    else:
        summary = run_avgs.groupby('method').agg(
            mean_weighted_avg=('weighted_avg', 'mean'),
            std=('weighted_avg', 'std'),
            n=('weighted_avg', 'count')
        ).reset_index()

        summary['CI'] = summary.apply(
            lambda row: t.ppf((1 + confidence) / 2, row['n'] - 1) * (row['std'] / np.sqrt(row['n']))
            if row['n'] > 1 else 0, axis=1
        )

        result = summary[['method', 'mean_weighted_avg', 'CI']].round(2)
        return result.set_index('method').transpose(), run_avgs




def throughput_per_method(df: pd.DataFrame, confidence: float = 0.95, show_process: bool = True) -> pd.DataFrame:
    """
    Compute throughput (completed cases per run) with 95% CI.

    Expects `df` to have columns:
      ['method','l','simulation_run','count']  (+ 'process' if show_process=True)

    If show_process == False:
        groups by ['method','l'] across runs.
        Returns a pivot with columns=λ and rows=(throughput, CI) transposed.

    If show_process == True:
        groups by ['method','l','process'] across runs.
        Returns a pivot with columns=(λ, process) and rows=(throughput, CI) transposed.
    """

    needed = {'method', 'l', 'simulation_run', 'count'}
    if show_process:
        needed.add('process')
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    group_fields = ['method', 'l'] + (['process'] if show_process else [])

    summary = (
        df.groupby(group_fields)
          .agg(
              throughput=('count', 'mean'),          # mean count per run
              std_count=('count', 'std'),
              n_runs=('simulation_run', 'nunique')
          )
          .reset_index()
    )

    def ci_fn(row):
        n = int(row['n_runs'])
        if n <= 1 or pd.isna(row['std_count']):
            return 0.0
        return t.ppf((1 + confidence) / 2.0, n - 1) * (row['std_count'] / np.sqrt(n))

    summary['CI'] = summary.apply(ci_fn, axis=1)

    summary['throughput'] = summary['throughput'].round(2)
    summary['CI']         = summary['CI'].round(2)

    if show_process:
        pivot = summary.pivot(index='method', columns=['l', 'process'], values=['throughput', 'CI'])
    else:
        pivot = summary.pivot(index='method', columns='l', values=['throughput', 'CI'])

    return pivot.transpose()

from scipy.stats import ttest_ind


FIGURE_SIZE = (5, 4)
GACT = 'GACT'
LINEWITH = .5
SETLINEWITH = 1
SETMARKERSIZE = 2

def plot_results_broken_y_axis(folder_path, scenario, top=6.4, bottom=18, legend=False, FIGURE_SIZE=(5,4)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import glob
    

    dataframes = []
    csv_files = glob.glob(folder_path)
    for file in csv_files:
        df = pd.read_csv(file)#, index_col="Unnamed: 0"
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    #df = df[df['status'] == 'COMPLETE']
    
    # Set the style for a clean and professional look
    sns.set_theme(style="whitegrid", context="talk")
    #sns.set_theme(context="talk")
    #sns.set_style("darkgrid")

    f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2,
                                          figsize=FIGURE_SIZE, 
                                          sharex=True, 
                                          gridspec_kw={'height_ratios': [1, 6], 'hspace': 0.1})
    
    

    for ax in [ax_top, ax_bottom]:
        sns.lineplot(
            x="l", 
            y="cycle_time",
            hue="method", 
            style="method", 
            data=df,
            hue_order=["Random","SPT","FIFO", "RLRAM","DRL",  "MuProMAC"],
            palette={"FIFO": sns.color_palette()[2], "RLRAM": sns.color_palette()[1], "SPT": sns.color_palette()[4],"DRL": sns.color_palette()[2], "MuProMAC": sns.color_palette()[3], "Random": sns.color_palette()[0]},#{"RLRAM": "tab:blue", "DRL": "tab:green", "MuProMAC": "tab:orange", "Random": "#FFB482"},
            markers={"FIFO": "P","RLRAM": "o", "DRL": "P", "MuProMAC": "s", "Random": "X", "SPT":"s"},  
            dashes={"FIFO": (1,3), "Random": (1,3), "RLRAM": (3, 1, 1, 1), "DRL": (5, 2), "MuProMAC": (2, 2), "SPT":(1,3)},
            legend=False,
            ax=ax
        )
        ax.grid(True, linestyle=":", linewidth=LINEWITH, color=".6")
    
    for line in ax.lines:
        line.set_linewidth(SETLINEWITH)    
        line.set_markersize(SETMARKERSIZE) 
    
    font_dict = {"fontweight": "bold"}
    ax_bottom.set_xlabel("Lambda (λ)", fontsize=16) # ,**font_dict
    ax_bottom.set_ylabel("GACT", fontsize=16)
    ax_top.set_ylabel("", fontsize=14)
    
    ax_bottom.tick_params(axis='both', labelsize=13, width=2)
    ax_top.tick_params(axis='both', labelsize=13, width=2)
    for label in ax_bottom.get_xticklabels() + ax_bottom.get_yticklabels():
        label.set_fontweight("bold")
    for label in ax_top.get_xticklabels() + ax_top.get_yticklabels():
        label.set_fontweight("bold")
    
    ax_top.set_ylim(bottom=bottom)
    ax_top.set_xlim(0.18,1.01)
    ax_bottom.set_ylim(bottom=0, top=top)
    ax_bottom.set_xlim(0.18,1.01)
    sns.despine(ax=ax_bottom)
    sns.despine(ax=ax_top, bottom=True)
    
    d = .03  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs) 
    kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
    ax_bottom.plot((-d, +d), (1 - d + 0.02, 1 + d - 0.02), **kwargs)  # bottom-left diagonal

    f.tight_layout()
    f.savefig(f"results/plots/{scenario}_cycle_time_vs_lambda.png", dpi=600, bbox_inches="tight")
    
    f.show()

def plot_shared_resources(folder_path, scenario,l=.6, legend=True, fig_name="shared_resources"):
    legend_config = 'auto' if legend else False
    dataframes = []
    csv_files = glob.glob(folder_path)

    for file in csv_files:
        if "papershared2" in file:
            col_val=2
            df = pd.read_csv(file)#, index_col="Unnamed: 0"
            df['num_shared_r'] = col_val
            dataframes.append(df)
        elif "papershared3" in file:
            col_val=3
            df = pd.read_csv(file)#, index_col="Unnamed: 0"
            df['num_shared_r'] = col_val
            dataframes.append(df)
        elif "papershared4" in file:
            col_val=4
            df = pd.read_csv(file)#, index_col="Unnamed: 0"
            df['num_shared_r'] = col_val
            dataframes.append(df)
        elif "papershared5" in file:
            col_val=5
            df = pd.read_csv(file)#, index_col="Unnamed: 0"
            df['num_shared_r'] = col_val
            dataframes.append(df)
        

    df = pd.concat(dataframes, ignore_index=True)
    #df = df[df['status'] == 'COMPLETE']
    df = df[df['method'] != 'Random']
    df_sub = df[df['l']==l]
    #df['process']= df['process'].apply(lambda x: encode_process_names[x])
    #df.rename(columns={'method':'Method', 'process':'Process'}, inplace=True)
    import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(5,4))
    sns.lineplot(
        x="num_shared_r", 
        y="cycle_time",
        hue="method", 
        style="method", 
        data=df_sub,
        hue_order=["RLRAM","DRL", "MuProMAC", "FIFO", "SPT"],
        palette={"FIFO": sns.color_palette()[2],  "SPT": sns.color_palette()[4],"RLRAM": sns.color_palette()[1], "DRL": sns.color_palette()[2], "MuProMAC": sns.color_palette()[3], "Random": sns.color_palette()[0]},#{"RLRAM": "tab:blue", "DRL": "tab:green", "MuProMAC": "tab:orange", "Random": "#FFB482"},
        markers={"FIFO": "P", "SPT":"s","RLRAM": "o", "DRL": "P", "MuProMAC": "s", "Random": "X"},  
        dashes={"FIFO": (1,3), "SPT":(1,3),"Random": (1,3), "RLRAM": (3, 1, 1, 1), "DRL": (5, 2), "MuProMAC": (2, 2)},
        legend=legend_config
    )
    ax = plt.gca()
    for line in ax.lines:
        line.set_linewidth(SETLINEWITH)    
        line.set_markersize(SETMARKERSIZE) 

    if legend:
        plt.legend(
            title="Method",
            loc='best',#"best",
            fontsize=11,
            title_fontsize=14,
            #bbox_to_anchor=(1.8, -0.2),
            ncol = 1
        )
    plt.xlabel("# Shared Resources", fontsize=16)
    plt.ylabel('GACT', fontsize=16)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.grid(visible=True, which="major", linestyle="--", linewidth=LINEWITH, alpha=0.7)
    sns.despine()

    plt.tight_layout()

    plt.savefig(f"results/plots/{scenario}_{fig_name}.png", dpi=600, bbox_inches="tight")

    plt.show()



from scipy.stats import t
import numpy as np
import pandas as pd
import glob

def table_shared_resources(folder_path, scenario, l=0.6, confidence=0.95):
    dataframes = []
    csv_files = glob.glob(folder_path)

    for file in csv_files:
        if "papershared2" in file:
            col_val = 2
        elif "papershared3" in file:
            col_val = 3
        elif "papershared4" in file:
            col_val = 4
        elif "papershared5" in file:
            col_val = 5
        else:
            continue

        df = pd.read_csv(file)
        df['num_shared_r'] = col_val
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df[df['status'] == 'COMPLETE']
    df = df[df['method'] != 'Random']
    df = df[df['l'] == l]

    def compute_gact(group):
        return group.groupby('process')['cycle_time'].mean().mean()

    gact_runs = (
        df.groupby(['method', 'num_shared_r', 'simulation_run'])
        .apply(compute_gact)
        .reset_index(name='gact')
    )

    summary = (
        gact_runs.groupby(['method', 'num_shared_r'])
        .agg(
            mean_gact=('gact', 'mean'),
            std_gact=('gact', 'std'),
            n_runs=('gact', 'count')
        )
        .reset_index()
    )

    summary['CI'] = summary.apply(
        lambda row: t.ppf((1 + confidence) / 2, row['n_runs'] - 1) * (row['std_gact'] / np.sqrt(row['n_runs']))
        if row['n_runs'] > 1 else 0,
        axis=1
    )

    summary = summary[['method', 'num_shared_r', 'mean_gact', 'CI']].round(2)
    pivoted = summary.pivot(index='method', columns='num_shared_r', values=['mean_gact', 'CI'])

    return pivoted


def plot_process_details_cycle_time(folder_path, scenario, legend=False, fig_name="cycle_time_vs_lambda_deep_view",encode_process_names={'process_a':'P1', 'process_b':'P2'}):

    legend_config = 'auto' if legend else False
    csv_files = glob.glob(folder_path)
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)#, index_col="Unnamed: 0"
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    #df = df[df['status'] == 'COMPLETE']
    df['process']= df['process'].apply(lambda x: encode_process_names[x])
    df.rename(columns={'method':'Method', 'process':'Process'}, inplace=True)
    import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(5,4))
    sns.lineplot(
        x="l", 
        y="cycle_time",
        hue="Method", 
        style="Process", 
        data=df,
        palette={"FIFO": sns.color_palette()[2],"RLRAM": sns.color_palette()[1],"SPT": sns.color_palette()[4], "DRL": sns.color_palette()[2], "MuProMAC": sns.color_palette()[3], "Random": sns.color_palette()[0]},#{"RLRAM": "tab:blue", "DRL": "tab:green", "MuProMAC": "tab:orange", "Random": "#FFB482"},
        markers={"P1":'o', "P2":'X'}, 
        #dashes={"Random": (1,3), "RLRAM": (3, 1, 1, 1), "DRL": (5, 2), "MuProMAC": (2, 2)}, 
        hue_order=["Random", "SPT","FIFO","RLRAM", "MuProMAC","DRL"],
        legend=legend_config
    )
    ax = plt.gca()
    for line in ax.lines:
        line.set_linewidth(SETLINEWITH)    
        line.set_markersize(SETMARKERSIZE) 
    if legend:
        plt.legend(
            title="Method & Process",
            loc='lower center',#"best"
            fontsize=11,
            title_fontsize=14,
            bbox_to_anchor=(1.8, -0.2),
            ncol = 1
        )
    plt.xlabel("Lambda (λ)", fontsize=16)
    plt.ylabel('ACT', fontsize=16)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.grid(visible=True, which="major", linestyle="--", linewidth=LINEWITH, alpha=0.7)
    sns.despine()

    plt.tight_layout()

    plt.savefig(f"results/plots/{scenario}_{fig_name}.png", dpi=600, bbox_inches="tight")

    plt.show()

def plot_process_details_throughput(folder_path, scenario, legend=False, fig_name="throughput_vs_lambda_deep_view",encode_process_names={'process_a':'P1', 'process_b':'P2'}):

    legend_config = 'auto' if legend else False
    csv_files = glob.glob(folder_path)
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)#, index_col="Unnamed: 0"
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    #df = df[df['status'] == 'COMPLETE']
    df['process']= df['process'].apply(lambda x: encode_process_names[x])
    df.rename(columns={'method':'Method', 'process':'Process'}, inplace=True)
    count_data = df.groupby(['l', 'Method', 'Process','simulation_run']).size().reset_index(name='count')
    import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(5,4))
    sns.lineplot(
        x="l", 
        y="count",
        hue="Method", 
        style="Process", 
        data=count_data,
        palette={"FIFO": sns.color_palette()[2],"RLRAM": sns.color_palette()[1],"SPT": sns.color_palette()[4], "DRL": sns.color_palette()[2], "MuProMAC": sns.color_palette()[3], "Random": sns.color_palette()[0]},#{"RLRAM": "tab:blue", "DRL": "tab:green", "MuProMAC": "tab:orange", "Random": "#FFB482"},
        markers={"P1":'o', "P2":'X'}, 
        #dashes={"Random": (1,3), "RLRAM": (3, 1, 1, 1), "DRL": (5, 2), "MuProMAC": (2, 2)}, 
        hue_order=["Random", "SPT", "FIFO","RLRAM",  "MuProMAC","DRL"],
        legend=legend_config
    )
    ax = plt.gca()
    for line in ax.lines:
        line.set_linewidth(SETLINEWITH)    
        line.set_markersize(SETMARKERSIZE) 
    if legend:
        plt.legend(
            title="Method & Process",
            loc='lower center',#"best"
            fontsize=11,
            title_fontsize=14,
            bbox_to_anchor=(1.8, -0.2),
            ncol = 1
        )
    plt.xlabel("Lambda (λ)", fontsize=16)
    plt.ylabel('Throughput (TP)', fontsize=16)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.grid(visible=True, which="major", linestyle="--", linewidth=LINEWITH, alpha=0.7)
    sns.despine()

    plt.tight_layout()

    plt.savefig(f"results/plots/{scenario}_{fig_name}.png", dpi=600, bbox_inches="tight")

    plt.show()



def process_data(source, from_memory=False):
   
    import glob
    import pandas as pd

    if from_memory:
        df = source.copy()
    else:
        import os
        csv_files = glob.glob(source)
        dataframes = []
        for file in csv_files:
            df_file = pd.read_csv(file)
            dataframes.append(df_file)
        df = pd.concat(dataframes, ignore_index=True)

    df_copy = df.copy()
    df_copy = df_copy[df_copy['status'] == 'COMPLETE']
    count_data = df_copy.groupby(['l', 'method', 'process', 'simulation_run']).size().reset_index(name='count')
    return df, count_data





def plot_results(folder_path, scenario, legend=False):
    csv_files = glob.glob(folder_path)

    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    #df = df[df['status'] == 'COMPLETE']
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=FIGURE_SIZE)
    
    sns.lineplot(
        x="l", 
        y="cycle_time",
        hue="method", 
        style="method", 
        data=df,
        markers=True,  # Add markers to emphasize points
        #dashes=False,  # Use solid lines for better readability
        hue_order=["RLRAM","DRL","MuProMAC"],
        legend=legend,
        #alpha=1,
        #linewidth=.3
    )
    ax = plt.gca()
    for line in ax.lines:
        line.set_linewidth(SETLINEWITH)    
        line.set_markersize(SETMARKERSIZE) 
    
    if legend:
        plt.legend(
            title="Method",
            loc='upper center',#"best"
            fontsize=11,
            title_fontsize=14,
            bbox_to_anchor=(2, 1)
        )

    #plt.title("Average Cycle Time vs. Lambda (λ)", fontsize=16, fontweight='bold')
    plt.xlabel("Lambda (λ)", fontsize=14)
    plt.ylabel(GACT, fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(visible=True, which="major", linestyle="--", linewidth=LINEWITH, alpha=0.7)
    #plt.grid(visible=True)
    sns.despine()

    plt.tight_layout()

    plt.savefig(f"results/plots/{scenario}_cycle_time_vs_lambda.png", dpi=600, bbox_inches="tight")

    plt.show()

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=FIGURE_SIZE)
    sns.lineplot(
        x="l", 
        y="cycle_time",
        hue="method", 
        style="process", 
        data=df,
        markers=True,  
        dashes=False,  
        hue_order=["RLRAM","DRL","MuProMAC"]
    )
    ax = plt.gca()
    for line in ax.lines:
        line.set_linewidth(SETLINEWITH)    
        line.set_markersize(SETMARKERSIZE) 
    
    plt.legend(
        title="Method & Process",
        loc="best", 
        fontsize=11,
        title_fontsize=14,
    )

    #plt.title("Average Cycle Time vs. Lambda (λ)", fontsize=16, fontweight='bold')
    plt.xlabel("Lambda (λ)", fontsize=14)
    plt.ylabel(GACT, fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(visible=True, which="major", linestyle="--", linewidth=LINEWITH, alpha=0.7)
    sns.despine()

    plt.tight_layout()

    plt.savefig(f"results/plots/{scenario}_cycle_time_vs_lambda.png", dpi=600, bbox_inches="tight")

    plt.show()
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid", context="talk")

    count_data = df.groupby(['l', 'method', 'process','simulation_run']).size().reset_index(name='count')

    plt.figure(figsize=FIGURE_SIZE)
    sns.lineplot(
        x="l", 
        y="count", 
        hue="method", 
        style="process", 
        data=count_data,
        markers=True,  
        dashes=False,  
        hue_order=["RLRAM","DRL","MuProMAC"]#["FIFO","RLRAM","DRL","MuProMAC", "Random"]
    )
    ax = plt.gca()
    for line in ax.lines:
        line.set_linewidth(SETLINEWITH)    
        line.set_markersize(SETMARKERSIZE) 

    plt.legend(
        title="Method & Process",
        loc="best",
        fontsize=11,
        title_fontsize=14,
    )

    #plt.title("Count of Cases vs. Lambda (l)", fontsize=16, fontweight='bold')
    plt.xlabel("Lambda (λ)", fontsize=14)
    plt.ylabel("Throughput (TP)", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(visible=True, which="major", linestyle="--", linewidth=LINEWITH, alpha=0.7)
    sns.despine()

    plt.tight_layout()

    plt.savefig(f"results/plots/{scenario}_throughput_vs_lambda.png", dpi=600, bbox_inches="tight")

    plt.show()
    return df, count_data




import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def plot_gact_lambda(df, scenario_label, out = None):
        
    def tidy_from_pivot(df_wide, scenario_label="(i)"):
     
        df = df_wide.copy()
        if df.index.nlevels == 2:
            df.index.set_names(["stat", "lambda"], inplace=True)

        means = df.loc["mean_gact"].stack().reset_index()
        means.columns = ["lambda", "method", "gact_mean"]

        cis = df.loc["CI"].stack().reset_index()
        cis.columns = ["lambda", "method", "gact_ci"]

        out = means.merge(cis, on=["lambda", "method"])
        out["scenario"] = scenario_label
        out = out.sort_values(["lambda", "method"]).reset_index(drop=True)
        return out

    df_plot = tidy_from_pivot(df, scenario_label=scenario_label)

    name_map = {
        "DRL_SingleAgent": "DRL (SA)",
        "DRL_MultiAgent_SingleLike": "DRL (MA)",
        "DoubleDQNExact": "DDQN (SA)",
        "MuProMAC": "MuProMAC",
        "RLRAM": "RLRAM",
        "FIFO": "FIFO",
        "SPT": "SPT",
        "Random": "Random",
    }
    df_plot["method"] = df_plot["method"].map(name_map).fillna(df_plot["method"])


    BASE = 12  
    mpl.rcParams.update({
        "font.size": BASE,             
        "axes.labelsize": BASE + 1,     # x/y labels
        "axes.titlesize": BASE + 1,     # subplot titles
        "xtick.labelsize": BASE,        # tick labels
        "ytick.labelsize": BASE,
        "legend.fontsize": BASE - 1,    # legend text
        "legend.title_fontsize": BASE - 1,
    })


    # ---- config ----
    methods = ["DRL (MA)", "DRL (SA)", "SPT", "RLRAM", "MuProMAC"]
    palette = plt.get_cmap("tab10")
    offsets = np.linspace(-0.015, 0.035, len(methods))  # tiny x-dodges
    PRINT_BW = False  

    styles = {
        "DRL (MA)": dict(ls="-",  marker="o"),
        "DRL (SA)": dict(ls="--", marker="s"),
        "MuProMAC": dict(ls="-.", marker="^"),
        "RLRAM":    dict(ls=":",  marker="D"),
        "SPT":      dict(ls=(0, (1, 1)), marker="P"),  # densely dotted
    }
    hatches = ["///", "\\\\", "xx", "++", ".."] 

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    lams = sorted(df_plot["lambda"].unique())

    colors = {m: (palette(i) if not PRINT_BW else "black") for i, m in enumerate(methods)}

    for i, m in enumerate(methods):
        sub = df_plot[df_plot["method"] == m].sort_values("lambda")
        x  = sub["lambda"].to_numpy()
        xd = x + offsets[i]
        y  = sub["gact_mean"].to_numpy()
        ci = sub["gact_ci"].to_numpy()

        poly = ax.fill_between(
            xd, y - ci, y + ci,
            facecolor=(colors[m] if not PRINT_BW else "0.85"),
            alpha=(0.15 if not PRINT_BW else 1.0),
            linewidth=0, zorder=1
        )
        if PRINT_BW:
            poly.set_hatch(hatches[i % len(hatches)])
            poly.set_edgecolor("0.3")
            poly.set_linewidth(0.5)

        ax.plot(
            xd, y,
            lw=2.0,
            color=colors[m],
            label=m,
            **styles[m],
            markersize=5.5,
            markerfacecolor="white",       
            markeredgewidth=1.1,
            zorder=3+i
        )

    # axes & labels
    ax.set_xlabel("λ", fontweight="bold")
    ax.set_ylabel("GACT",  fontweight="bold")
    ax.grid(True, axis="y", alpha=.9, linewidth=.8)

    # show only these ticks
    ticks = [0.2, 0.4, 0.6, 0.8]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])

    ax.set_xlim(ticks[0] - 0.03, ticks[-1] + 0.08)

    # despine top/right
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # bold tick labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    #legend (compact, bold)
    # leg = ax.legend(title="Method", ncol=5, frameon=False, fontsize=9, title_fontsize=9, bbox_to_anchor=(3, 1.05))
    # for txt in leg.get_texts():
    #     txt.set_fontweight("bold")
    # if leg.get_title():
    #     leg.get_title().set_fontweight("bold")

    plt.tight_layout()
    if out:
        os.makedirs("figures", exist_ok=True)
        out = out
        #fig.savefig(out + ".pdf", bbox_inches="tight")              # vector for LaTeX
        fig.savefig(out + ".png", dpi=300, bbox_inches="tight")     # high-res raster

    plt.show()




import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.pyplot as plt

def break_plot_gact_lambda(df, scenario_label, low_ylim, high_ylim, out = None):

    def tidy_from_pivot(df_wide, scenario_label="(i)"):
       
        df = df_wide.copy()
        if df.index.nlevels == 2:
            df.index.set_names(["stat", "lambda"], inplace=True)

        means = df.loc["mean_gact"].stack().reset_index()
        means.columns = ["lambda", "method", "gact_mean"]

        cis = df.loc["CI"].stack().reset_index()
        cis.columns = ["lambda", "method", "gact_ci"]

        out = means.merge(cis, on=["lambda", "method"])
        out["scenario"] = scenario_label
        
        out = out.sort_values(["lambda", "method"]).reset_index(drop=True)
        return out

    df_plot = tidy_from_pivot(df, scenario_label=scenario_label)

    name_map = {
        "DRL_SingleAgent": "DRL (SA)",
        "DRL_MultiAgent_SingleLike": "DRL (MA)",
        "DoubleDQNExact": "DDQN (SA)",
        "MuProMAC": "MuProMAC",
        "RLRAM": "RLRAM",
        "FIFO": "FIFO",
        "SPT": "SPT",
        "Random": "Random",
    }
    df_plot["method"] = df_plot["method"].map(name_map).fillna(df_plot["method"])


    BASE = 12  
    mpl.rcParams.update({
        "font.size": BASE,              # default text
        "axes.labelsize": BASE + 1,     # x/y labels
        "axes.titlesize": BASE + 1,     # subplot titles (if any)
        "xtick.labelsize": BASE,        # tick labels
        "ytick.labelsize": BASE,
        "legend.fontsize": BASE - 1,    # legend text
        "legend.title_fontsize": BASE - 1,
    })


    methods = ["DRL (MA)", "DRL (SA)", "SPT", "RLRAM", "MuProMAC"]
    palette = plt.get_cmap("tab10")
    offsets = np.linspace(-0.015, 0.035, len(methods))  # tiny x-dodges
    PRINT_BW = False  

    styles = {
        "DRL (MA)": dict(ls="-",  marker="o"),
        "DRL (SA)": dict(ls="--", marker="s"),
        "MuProMAC": dict(ls="-.", marker="^"),
        "RLRAM":    dict(ls=":",  marker="D"),
        "SPT":      dict(ls=(0, (1, 1)), marker="P"),  # densely dotted
    }
    hatches = ["///", "\\\\", "xx", "++", ".."]  # for CI ribbons in B/W mode
   

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(5.6, 4.2),
        gridspec_kw={"height_ratios":[1, 2], "hspace": 0.05}
    )

    lams    = sorted(df_plot["lambda"].unique())
    colors  = {m: (palette(i) if not PRINT_BW else "black") for i, m in enumerate(methods)}
    ticks   = [0.2, 0.4, 0.6, 0.8]

    def plot_on(ax):
        for i, m in enumerate(methods):
            sub = df_plot[df_plot["method"] == m].sort_values("lambda")
            x  = sub["lambda"].to_numpy()
            xd = x + offsets[i]
            y  = sub["gact_mean"].to_numpy()
            ci = sub["gact_ci"].to_numpy()

            poly = ax.fill_between(
                xd, y-ci, y+ci,
                facecolor=(colors[m] if not PRINT_BW else "0.85"),
                alpha=(0.15 if not PRINT_BW else 1.0),
                linewidth=0, zorder=1
            )
            if PRINT_BW:
                poly.set_hatch(hatches[i % len(hatches)])
                poly.set_edgecolor("0.3")
                poly.set_linewidth(0.5)

            ax.plot(
                xd, y, lw=2.0, color=colors[m], label=m, **styles[m],
                markersize=5.5, markerfacecolor="white", markeredgewidth=1.1, zorder=3+i
            )

    plot_on(ax_top)
    plot_on(ax_bot)

    ax_bot.set_ylim(*low_ylim)
    ax_top.set_ylim(*high_ylim)

    ax_bot.set_xlabel("λ", fontweight="bold")
    ax_bot.set_ylabel("GACT", fontweight="bold")

    ax_bot.set_xticks(ticks)
    ax_bot.set_xticklabels([f"{t:.1f}" for t in ticks])
    ax_bot.set_xlim(ticks[0] - 0.03, ticks[-1] + 0.08)

    for ax in (ax_top, ax_bot):
        ax.grid(True, axis="y", alpha=.25)
        ax.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)   # no bottom ticks on top
    ax_bot.tick_params(top=False)                      # no top ticks on bottom


    for lbl in ax_bot.get_xticklabels() + ax_bot.get_yticklabels() + ax_top.get_yticklabels():
        lbl.set_fontweight("bold")

    d = .015
    kw = dict(color="k", clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)              # left diag top
    ax_top.plot((1-d, 1+d), (-d, +d), transform=ax_top.transAxes, **kw)            # right diag top
    ax_bot.plot((-d, +d), (1-d, 1+d), transform=ax_bot.transAxes, **kw)            # left diag bottom
    ax_bot.plot((1-d, 1+d), (1-d, 1+d), transform=ax_bot.transAxes, **kw)          # right diag bottom

    plt.tight_layout()
    if out:
        os.makedirs("figures", exist_ok=True)
        out = out
        #fig.savefig(out + ".pdf", bbox_inches="tight")              
        fig.savefig(out + ".png", dpi=300, bbox_inches="tight")    
    plt.show()



def plot_process_level_ACT(df, out = None):
    df = paper_tables(df,show_process=True)

    def tidy_from_pivot_process(df_wide, scenario_label=None):
        
        df = df_wide.copy()
        if df.index.nlevels == 2:
            df.index.set_names(["stat", "lambda"], inplace=True)

        means = df.loc["mean_act"].stack(level=[0,1]).reset_index()
        means.columns = ["lambda", "method", "process", "act_mean"]

        cis = df.loc["CI"].stack(level=[0,1]).reset_index()
        cis.columns = ["lambda", "method", "process", "act_ci"]

        out = means.merge(cis, on=["lambda", "method", "process"])
        if scenario_label is not None:
            out["scenario"] = scenario_label
        
        return out.sort_values(["method","process","lambda"]).reset_index(drop=True)

    name_map = {
        "DRL_SingleAgent": "DRL (SA)",
        "DRL_MultiAgent_SingleLike": "DRL (MA)",
        #"DoubleDQNExact": "DDQN (SA)",
        "MuProMAC": "MuProMAC",
    }

    df_proc_plot = tidy_from_pivot_process(df, scenario_label=None)
    df_proc_plot["method"] = df_proc_plot["method"].map(name_map).fillna(df_proc_plot["method"])

    keep_methods = ["DRL (MA)", "DRL (SA)",  "MuProMAC"]#"DDQN (SA)",
    df_proc_plot = df_proc_plot[df_proc_plot["method"].isin(keep_methods)]

    BASE = 12
    mpl.rcParams.update({
        "font.size": BASE,
        "axes.labelsize": BASE + 1,
        "axes.titlesize": BASE + 1,
        "xtick.labelsize": BASE,
        "ytick.labelsize": BASE,
        "legend.fontsize": BASE - 1,
        "legend.title_fontsize": BASE - 1,
    })

    palette = plt.get_cmap("tab10")
    PRINT_BW = False   
    SHOW_CI = True     

    methods_sorted = keep_methods
    colors = {m: (palette(i) if not PRINT_BW else "black") for i, m in enumerate(methods_sorted)}

    processes_sorted = sorted(df_proc_plot["process"].unique())   
    proc_styles = {}
    proc_markers = {}
    line_styles = ["-", "--", "-.", ":"]
    markers     = ["o", "s", "^", "D"]
    for i, p in enumerate(processes_sorted):
        proc_styles[p]  = line_styles[i % len(line_styles)]
        proc_markers[p] = markers[i % len(markers)]

    hatches = ["///", "\\\\", "xx", "++", ".."]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    ticks = [0.2, 0.4, 0.6, 0.8]
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0] - 0.03, ticks[-1] + 0.08)

    offsets = np.linspace(-0.018, 0.018, len(processes_sorted))

    for m_i, m in enumerate(methods_sorted):
        df_m = df_proc_plot[df_proc_plot["method"] == m]
        for p_i, p in enumerate(processes_sorted):
            sub = df_m[df_m["process"] == p].sort_values("lambda")
            if sub.empty:
                continue

            x  = sub["lambda"].to_numpy()
            xd = x + offsets[p_i]              
            y  = sub["act_mean"].to_numpy()
            ci = sub["act_ci"].to_numpy()

            if SHOW_CI:
                poly = ax.fill_between(
                    xd, y - ci, y + ci,
                    facecolor=(colors[m] if not PRINT_BW else "0.85"),
                    alpha=(0.15 if not PRINT_BW else 1.0),
                    linewidth=0, zorder=1
                )
                if PRINT_BW:
                    poly.set_hatch(hatches[p_i % len(hatches)])
                    poly.set_edgecolor("0.3")
                    poly.set_linewidth(0.5)

            ax.plot(
                xd, y,
                lw=2.0,
                color=colors[m],
                label=f"{m} – {p}",             
                ls=proc_styles[p],
                marker=proc_markers[p],
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
                zorder=3 + m_i
            )

    ax.set_xlabel(r"λ", fontweight="bold")
    ax.set_ylabel("ACT (per process)", fontweight="bold")
    ax.grid(True, axis="y", alpha=.25)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    # Legend (compact)
    # leg = ax.legend(title="Method – Process", ncol=2, frameon=False, loc="best")
    # for txt in leg.get_texts():
    #     txt.set_fontweight("bold")
    # if leg.get_title():
    #     leg.get_title().set_fontweight("bold")
    from matplotlib.lines import Line2D


    proc_label_map = {"process_a": "P1", "process_b": "P2"}  
    proc_labels = [proc_label_map.get(p, p) for p in processes_sorted]

    method_handles = [
        Line2D([0], [0], color=colors[m], lw=2, ls='-', marker=None)
        for m in methods_sorted
    ]
    method_labels = methods_sorted  

    process_handles = [
        Line2D([0], [0], color='black', lw=2, ls=proc_styles[p],
            marker=proc_markers[p], markersize=6,
            markerfacecolor='white', markeredgewidth=1.1)
        for p in processes_sorted
    ]

    header = Line2D([], [], color='none', marker=None, linestyle='none')

    handles = (
        [header] + method_handles +
        [header] + process_handles
    )
    labels  = (
        ["Method"] + method_labels +
        ["Process"] + proc_labels
    )

    leg = ax.legend(
        handles, labels,
        title="Method & Process",
        ncol=1, frameon=False, fancybox=False, framealpha=0.9,
        borderpad=0.6, handlelength=2.2, handletextpad=0.8,
        loc="best"
    )

    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    for txt in leg.get_texts():
        if txt.get_text() in ("Method", "Process"):
            txt.set_fontweight("heavy")   

    if leg.get_title():
        leg.get_title().set_fontweight("bold")

    plt.tight_layout()
    if out:
        os.makedirs("figures", exist_ok=True)
        #fig.savefig(out + ".pdf", bbox_inches="tight")              
        fig.savefig(out + ".png", dpi=300, bbox_inches="tight")     
    plt.show()



import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def tidy_from_pivot_throughput_generic(df_wide: pd.DataFrame, scenario_label=None) -> pd.DataFrame:
    
    df = df_wide.copy()

    stat_level = None
    for lvl in range(df.index.nlevels):
        vals = set(map(str, df.index.get_level_values(lvl)))
        if {'throughput', 'CI'}.intersection(vals):
            stat_level = lvl
            break
    if stat_level is None:
        raise ValueError("Could not find 'throughput'/'CI' in the index levels of the pivot.")

    means = df.xs('throughput', level=stat_level).stack().reset_index()
    means.columns = ['lambda', 'process', 'method', 'tp_mean']

    cis = df.xs('CI', level=stat_level).stack().reset_index()
    cis.columns = ['lambda', 'process', 'method', 'tp_ci']

    out = means.merge(cis, on=['lambda', 'process', 'method'])
    if scenario_label is not None:
        out['scenario'] = scenario_label

    out['lambda'] = out['lambda'].astype(float)
    return out.sort_values(['method', 'process', 'lambda']).reset_index(drop=True)


def plot_process_throughput_from_pivot(
    df_thr_pivot: pd.DataFrame,
    included_methods=("DRL_MultiAgent_SingleLike", "DRL_SingleAgent", "DoubleDQNExact", "MuProMAC"),
    process_label_map=None,
    out=None,
    print_bw=False,
    show_ci=True
):
    name_map = {
        "DRL_SingleAgent": "DRL (SA)",
        "DRL_MultiAgent_SingleLike": "DRL (MA)",
        "DoubleDQNExact": "DDQN (SA)",
        "MuProMAC": "MuProMAC",
        "RLRAM": "RLRAM", "FIFO": "FIFO", "SPT": "SPT", "Random": "Random",
    }

    df_plot = tidy_from_pivot_throughput_generic(df_thr_pivot)
    if process_label_map:
        df_plot["process"] = df_plot["process"].map(process_label_map).fillna(df_plot["process"])
    df_plot = df_plot[df_plot["method"].isin(included_methods)].copy()
    df_plot["method"] = df_plot["method"].map(name_map).fillna(df_plot["method"])

    BASE = 12
    mpl.rcParams.update({
        "font.size": BASE, "axes.labelsize": BASE+1, "axes.titlesize": BASE+1,
        "xtick.labelsize": BASE, "ytick.labelsize": BASE,
        "legend.fontsize": BASE-1, "legend.title_fontsize": BASE-1,
    })

    palette = plt.get_cmap("tab10")
    methods_sorted   = list(dict.fromkeys(df_plot["method"]))
    colors = {m: (palette(i) if not print_bw else "black") for i, m in enumerate(methods_sorted)}
    processes_sorted = sorted(df_plot["process"].unique())

    line_styles = ["-", "--", "-.", ":"]
    markers     = ["o", "s", "^", "D"]
    proc_styles = {p: line_styles[i % len(line_styles)] for i, p in enumerate(processes_sorted)}
    proc_marks  = {p: markers[i % len(markers)]       for i, p in enumerate(processes_sorted)}
    hatches     = ["///", "\\\\", "xx", "++", ".."]

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    lams = sorted(df_plot["lambda"].unique())
    ticks = [0.2, 0.4, 0.6, 0.8] if set([0.2,0.4,0.6,0.8]).issubset(set(lams)) else lams
    ax.set_xticks(ticks); ax.set_xlim(min(ticks)-0.03, max(ticks)+0.08)
    #offsets = np.linspace(-0.018, 0.018, len(processes_sorted))
    offsets = np.linspace(-0.0, 0.0, len(processes_sorted))

    for m_i, m in enumerate(methods_sorted):
        df_m = df_plot[df_plot["method"] == m]
        for p_i, p in enumerate(processes_sorted):
            sub = df_m[df_m["process"] == p].sort_values("lambda")
            if sub.empty: continue
            x  = sub["lambda"].to_numpy()
            xd = x + offsets[p_i]
            y  = sub["tp_mean"].to_numpy()
            ci = sub["tp_ci"].to_numpy()

            if show_ci:
                poly = ax.fill_between(
                    xd, y-ci, y+ci,
                    facecolor=(colors[m] if not print_bw else "0.85"),
                    alpha=(0.15 if not print_bw else 1.0),
                    linewidth=0, zorder=1
                )
                if print_bw:
                    poly.set_hatch(hatches[p_i % len(hatches)])
                    poly.set_edgecolor("0.3"); poly.set_linewidth(0.5)

            ax.plot(
                xd, y, lw=2.0, color=colors[m],
                label=f"{m} – {p}", ls=proc_styles[p], marker=proc_marks[p],
                markersize=5.5, markerfacecolor="white", markeredgewidth=1.1, zorder=3+m_i
            )

    ax.set_xlabel(r"λ", fontweight="bold")
    ax.set_ylabel("ATP (per process)", fontweight="bold")
    ax.grid(True, axis="y", alpha=.25)
    for spine in ("top","right"): ax.spines[spine].set_visible(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels(): lbl.set_fontweight("bold")

    method_handles  = [Line2D([0],[0], color=colors[m], lw=2, ls='-') for m in methods_sorted]
    process_handles = [Line2D([0],[0], color='black', lw=2, ls=line_styles[i%4],
                              marker=markers[i%4], markersize=6,
                              markerfacecolor='white', markeredgewidth=1.1)
                       for i, _ in enumerate(processes_sorted)]
    header = Line2D([], [], color='none', marker=None, linestyle='none')
    handles = [header] + method_handles + [header] + process_handles
    labels  = ["Method"] + methods_sorted + ["Process"] + processes_sorted
    leg = ax.legend(handles, labels, title="Method & Process",
                    ncol=1, frameon=False, fancybox=False, framealpha=0.9,
                    borderpad=0.6, handlelength=2.2, handletextpad=0.8, loc="best")
    for txt in leg.get_texts(): txt.set_fontweight("bold")
    if leg.get_title(): leg.get_title().set_fontweight("bold")

    plt.tight_layout()
    if out:
        os.makedirs("figures", exist_ok=True)
        fig.savefig(out + ".png", dpi=300, bbox_inches="tight") 
    plt.show()



import os
import re
import matplotlib.pyplot as plt
import pandas as pd

def _pretty_name_from_path(path):
    p = path.lower()
    if "mupromac" in p: return "MuProMAC"
    if "drl_multi" in p or "drlmultiagent" in p: return "DRL (MA)"
    if "drl_single" in p or "drlsingleagent" in p: return "DRL (SA)"
    if "rlram" in p: return "RLRAM"
    if "spt" in p: return "SPT"
    if "random" in p: return "Random"
    if "doubledqn" in p or "ddqn" in p: return "DDQN (SA)"
    return os.path.basename(path)

def _read_episode_avgct_minimal(path):
    """Read only episode and avg_cycle_time by splitting each line at most 3 commas."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header_seen = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_seen and line.lower().startswith("episode,"):
                header_seen = True
                continue
            parts = line.split(",", 3)
            if len(parts) < 3:
                continue
            ep_str, act_str, _elapsed_and_rest = parts[0], parts[1], parts[2]
            
            ep_str  = ep_str.strip().strip('"').strip("'")
            act_str = act_str.strip().strip('"').strip("'")
            try:
                ep  = int(float(ep_str))   # handle "0.0"
                act = float(act_str)
                rows.append((ep, act))
            except ValueError:
                
                continue
    return pd.DataFrame(rows, columns=["episode", "avg_cycle_time"])

def plot_avg_ct_over_episodes(
    csv_paths,
    method_name_overrides=None,   # dict {path: "Pretty Name"}
    methods_filter=None,          # list of names to include
    smooth=5,                     # rolling window (None or int)
    out=None,
    figsize=(6.4, 4.2),
    dpi=300
):
    all_rows = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"[skip] not found: {p}")
            continue
        df = _read_episode_avgct_minimal(p)
        if df.empty:
            print(f"[warn] no data parsed from: {p}")
            continue
        name = (method_name_overrides or {}).get(p, _pretty_name_from_path(p))
        df["method"] = name
        all_rows.append(df)

    if not all_rows:
        print("[ERROR] nothing to plot.")
        return None

    tidy = pd.concat(all_rows, ignore_index=True)

    if methods_filter:
        tidy = tidy[tidy["method"].isin(methods_filter)]
        if tidy.empty:
            print("[warn] filter removed all data.")
            return None

    ycol = "avg_cycle_time"
    if isinstance(smooth, int) and smooth > 1:
        tidy = (tidy.sort_values(["method", "episode"])
                    .groupby("method", group_keys=False)
                    .apply(lambda g: g.assign(avg_cycle_time_smooth=g["avg_cycle_time"]
                                              .rolling(smooth, min_periods=1).mean())))
        ycol = "avg_cycle_time_smooth"

    plt.figure(figsize=figsize)
    for i, (method, g) in enumerate(tidy.groupby("method")):
        g = g.sort_values("episode")
        ls = ["-", "-", "-.", ":", (0, (1,1)), (0,(3,1,1,1))][i % 6]
        mk = ["o","o", "s", "^", "D", "P", "X", "v", "*"][i % 8]
        plt.plot(
            g["episode"], g[ycol],
            ls=ls,  lw=2.0,#marker=mk, markersize=4.5,
            #markerfacecolor="white", markeredgewidth=2.1,
            label=method
        )

    plt.xlabel("Episode", fontweight="bold")
    plt.ylabel("GACT", fontweight="bold")
    plt.grid(True, axis="y", alpha=.25)
    ax = plt.gca()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    plt.legend(title="Method", frameon=False, ncol=2)

    plt.tight_layout()
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out}")
        return None
    else:
        plt.show()
        return plt.gcf()
