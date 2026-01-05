'''# dataset_confs.py
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
static_num_cols[dataset] = []'''

# dataset_confs.py
# Configuration for datasets used by time-prediction-benchmark.

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
# ORIGINAL DATASETS (keep these if you still want to use them)
# ----------------------------------------------------------------------

# ... your original bpic2011, bpic2015, etc. configurations ...

# ----------------------------------------------------------------------
# AUTO-CONFIGURE MuProMAC SCENARIOS
# ----------------------------------------------------------------------

import os
import glob

# CONTROL HOW MANY LOGS TO CONFIGURE
# Set to None to configure ALL logs, or set a number (e.g., 10) to limit
MAX_LOGS = 10  # Change this to None to process all 84 logs

# Find all processed MuProMAC logs
mupromac_logs = glob.glob("../../out/251110/event_logs_processed/*.csv")

if not mupromac_logs:
    # Try relative path from experiments folder
    mupromac_logs = glob.glob("../out/251110/event_logs_processed/*.csv")

# Sort for consistent ordering
mupromac_logs = sorted(mupromac_logs)

# Limit to first N logs if MAX_LOGS is set
if MAX_LOGS is not None:
    mupromac_logs = mupromac_logs[:MAX_LOGS]

for log_path in mupromac_logs:
    base_name = os.path.basename(log_path).replace(".csv", "")
    dataset = f"mup_{base_name}"
    
    filename[dataset] = log_path
    case_id_col[dataset] = "case_id"
    activity_col[dataset] = "activity"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "remtime"
    pos_label[dataset] = "regular"
    neg_label[dataset] = "deviant"
    dynamic_cat_cols[dataset] = ["activity", "resource"]
    static_cat_cols[dataset] = []
    dynamic_num_cols[dataset] = ["activity_duration", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = []

# print(f"Auto-configured {len(mupromac_logs)} MuProMAC datasets")
# if MAX_LOGS is not None:
#     print(f"  (Limited to first {MAX_LOGS} logs - set MAX_LOGS=None to process all)")
# if mupromac_logs:
#     print(f"First dataset: {list(filename.keys())[0] if filename else 'None'}")

