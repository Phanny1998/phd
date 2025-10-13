# Multi-Process Multi-Agent Coordination (MuProMAC)

MuProMAC is a research framework for **allocating shared resources across multiple business processes**. It provides a discrete-event simulator, classic baselines, and several RL planners (single-agent, multi-agent, and our cooperative MuProMAC variant). The environment follows a multi-process setting with **shared and dedicated resources**, gateways, and stochastic arrivals/processing times.

> Core idea: centralized training with a **global critic** + decentralized execution by **per-process agents** that act on **local** information.

---


## Installation

**OS:** Windows  
**Python:** 3.11.10

1) **Clone**
```bat
git clone https://anonymous.4open.science/r/MuProMAC-2025.git
cd MuProMAC
```

2) **Create venv (recommended)**
```bat
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

3) **Install dependencies**
```bat
pip install -r requirements.txt
```

> **GPU:** PyTorch with CUDA is recommended for RL training.

---

## Repository structure

```
MUPROMAC_REVISION/
├─ framework/
│  ├─ __init__.py
│  ├─ config.py                # Scenario definitions (processes, tasks, resources, arrival rates, runtime)
│  ├─ config_training.py       # Training defaults (episodes, convergence, etc.)
│  ├─ rw_config_training.py    # Alternative training config (revision/ablation settings)
│  ├─ processes.py             # Process, Task, Resource, Gateway, simulator-facing API
│  ├─ result_helper.py         # Helpers
│  └─ simulator.py             # Discrete-event simulator 
├─ .gitignore
├─ fifo.py                     # Baseline FIFO
├─ randomAllocation.py         # Baseline random assignment
├─ run_ddqn.py                 # DDQN runner
├─ run_drl_multiagent_DRL.py   # DRL multi-agent
├─ run_drl_singleagent_DRL.py  # DRL single-agent
├─ run_mupromac.py             # MuProMAC
├─ rw_train_ddqn.py            
├─ rw_train_drl_multiagent_DRL.py   
├─ rw_train_drl_singleagent_DRL.py  
├─ spt.py                      # SPT (shortest processing time)
├─ train_ddqn.py               
├─ train_drl_multiagent_DRL.py 
├─ train_drl_singleagent_DRL.py
├─ train_mupromac.py           # Train MuProMAC 
├─ paper_results.ipynb         # Paper results
├─ README.md                   # (this file)
└─ requirements.txt
```
---

## Configure scenarios

Open **`framework/config.py`** and set:

- `ARRIVAL_RATES` – list of λ values (e.g., `[.3]`)
- `SCENARIO_NAMES` – list of scenario IDs (e.g., `['bpi2020_2processes_massive_share']`)
- `SIMULATION_RUN_TIME` – simulated time horizon per run
- `SIMULATION_RUNS` – number of replications (independent runs)

Training-specific in **`framework/config_training.py`** (or `rw_config_training.py`), e.g.:

- `SIMULATION_UNIT` – simulated time per episode
- `MAX_EPISODES`, `CONVERGENCE_WINDOW`, `IMPROVEMENT_THRESHOLD`

---

## Run a quick  test

If you just want to verify the environment and get a small CSV:

1) In **`framework/config.py`**, set:
```python
ARRIVAL_RATES=[.2]
SCENARIO_NAMES=['bpi2020_2processes_massive_share']
SIMULATION_RUN_TIME = 500
SIMULATION_RUNS = 1
```

2) **Run a simple baseline** (FIFO):
```bat
python fifo.py
```

This will simulate and write a CSV under `event_logs/test/…csv`.  
Use `paper_results.ipynb` to summarize results.

---

## Train & run 

### MuProMAC 

**Train**
```bat
python train_mupromac.py
```

**Evaluate / run with saved models**
```bat
python run_mupromac.py
```

Models are saved under:
```
trained_models/MuProMAC/<SCENARIO>_lambda<LAMBDA>/
  actor_<process>.pt
  critic.pt
  gru_<process>.pt
```

## Results, logs, and analysis

- **Logs**  
  Every simulation writes a CSV to:
  - `event_logs/train/log_<method>_run<id>_<scenario>.csv`
  - `event_logs/test/log_<method>_run<id>_<scenario>.csv`



---


## Contact

Questions or issues?  
Open a GitHub issue or email **kiran.busch@klu.org**.
