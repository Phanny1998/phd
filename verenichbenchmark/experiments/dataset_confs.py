# dataset_confs.py
# Configuration for datasets used by time-prediction-benchmark.
# Here we define ONE ENTRY PER SCENARIO, all with the same column layout.

case_id_col = {}
activity_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

# ----------------------------------------------------------------------
# SCENARIO 1
# ----------------------------------------------------------------------

dataset = "mup_scen1"   # <- name you will pass to experiments_final.py

# Path relative to this experiments/ folder:
# i.e. the file is at experiments/logdata/mup_scen1.csv
filename[dataset] = "logdata/mupromac_log_FIFO_run0_EXP_dedicated_C1.csv"

# Core columns (these MUST match the headers in your CSV)
case_id_col[dataset]   = "Case ID"
activity_col[dataset]  = "Activity"
timestamp_col[dataset] = "Complete Timestamp"

# Remaining-time label (in seconds), produced by your logdata builder
label_col[dataset] = "remtime"

# For regression these labels are not really used; keep dummy values
pos_label[dataset] = "regular"
neg_label[dataset] = "deviant"

# Dynamic categorical features (change over the case)
dynamic_cat_cols[dataset] = [
    "Activity",
    "Resource",
]

# Static categorical features (per-case, known from the start)
# If your logdata includes e.g. "scenario" or "process" and you want to use them,
# you can add here later.
static_cat_cols[dataset] = []

# Dynamic numeric features (change over time)
dynamic_num_cols[dataset] = [
    "activity_duration",    # seconds, end_time - start_time for this activity
    "timesincelastevent",   # seconds since previous completion in this case
    "timesincecasestart",   # seconds since first completion in this case
    "event_nr",             # 1,2,3,... within case
    "open_cases",           # WIP at this completion time
]

# Static numeric features (fixed per case)
static_num_cols[dataset] = []

