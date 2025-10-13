import os
import time
import json
import gc
import warnings
from multiprocessing import Pool, set_start_method

import torch
import pandas as pd
from tqdm import trange

from framework.config import process_function
from framework.simulator import Simulator
from framework.result_helper import paper_tables, process_data
from framework.config_training import (
    SCENARIOS,
    ARRIVAL_RATES,
    SIMULATION_UNIT,
    MAX_EPISODES,
    CONVERGENCE_WINDOW,
    IMPROVEMENT_THRESHOLD,
)

from run_ddqn import DoubleDQNProcess

warnings.filterwarnings("ignore")

CHECKPOINT_INTERVAL = 2     
WARMUP_EPISODES = 50          
EPSILON_DECREASING_FACTOR = 0.09 



def get_ep(fname: str) -> int:
    try:
        return int(fname.split("_ep")[-1].split(".pt")[0])
    except Exception:
        return -1


def safe_json_load(path: str, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def score_from_proc_ct(avg_ct_by_proc: dict, throughput_by_proc: dict | None) -> float:
   
    if not avg_ct_by_proc:
        return float("inf")

    vals = {k: float(v) for k, v in avg_ct_by_proc.items() if pd.notna(v)}
    if not vals:
        return float("inf")

    if throughput_by_proc:
        weights = {k: float(throughput_by_proc.get(k, 0.0)) for k in vals}
        total_w = sum(weights.values())
        if total_w > 0.0:
            return sum(vals[k] * weights[k] for k in vals) / total_w

    # equal-weight fallback
    return sum(vals.values()) / len(vals)


# ---------------------- Training ----------------------

def train_double_dqn_exact(scenario_name: str, lam: float):
    
    SAVE_DIR = f"trained_models/DoubleDQN/{scenario_name}_lambda{lam}"
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_meta_path = os.path.join(SAVE_DIR, "best_meta.json")
    log_path = os.path.join(SAVE_DIR, "training_log.csv")

    if os.path.exists(os.path.join(SAVE_DIR, "online.pt")) and os.path.exists(os.path.join(SAVE_DIR, "target.pt")):
        print(f"[INFO] Training already completed for {scenario_name} λ={lam}. Skipping.")
        return

    available_eps = [
        get_ep(f) for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".pt") and "_ep" in f
    ]
    resume_episode = max(available_eps) if available_eps else 0

    best_meta = safe_json_load(best_meta_path, default={}) or {}
    best_score = float(best_meta.get("score", float("inf")))

    # Create log header if new
    if resume_episode == 0 and not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("episode,avg_cycle_time,elapsed_time,avg_cycle_time_by_process,throughput_by_process\n")

    print(f"\n=== Training DoubleDQN(EXACT) on '{scenario_name}' with λ={lam} ===")
    processes = process_function(lam, scenario_name)
    agent_proc = DoubleDQNProcess(processes, scenario_name, save_dir=SAVE_DIR)

    if resume_episode > 0:
        print(f"[INFO] Resuming training from episode {resume_episode}")
        online_path = os.path.join(CHECKPOINT_DIR, f"online_ep{resume_episode}.pt")
        target_path = os.path.join(CHECKPOINT_DIR, f"target_ep{resume_episode}.pt")
        agent_proc.agent.online.load_state_dict(torch.load(online_path, map_location="cpu"))
        agent_proc.agent.target.load_state_dict(torch.load(target_path, map_location="cpu"))

    if os.path.exists(log_path):
        try:
            df_log = pd.read_csv(log_path, comment='#')  
            cycle_time_history = pd.to_numeric(df_log['avg_cycle_time'], errors='coerce') \
                                .dropna().astype(float).tolist()
        except Exception:
            cycle_time_history = []
    else:
        cycle_time_history = []


    eps_anneal_episodes = max(1, int(MAX_EPISODES * EPSILON_DECREASING_FACTOR))
    converged = False

    for episode in trange(resume_episode, MAX_EPISODES, desc=f"Training {scenario_name} λ={lam}"):
        start_time = time.time()

        epsilon = max(1.0 - episode / float(eps_anneal_episodes), 0.1)
        agent_proc.current_epsilon = epsilon
        agent_proc.training_enabled = (episode > WARMUP_EPISODES)

        simulator = Simulator(
            simulation_run=episode,
            process=agent_proc,
            scenario_name='DoubleDQN',
            log_only_case_completion=True,
            is_training=True
        )
        simulator.run(SIMULATION_UNIT)
        elapsed = time.time() - start_time

        df = simulator.event_log.get_dataframe()
        df["method"] = "TrainEval"
        df["l"] = lam
        df["simulation_run"] = episode

        df_proc, _ = process_data(df, from_memory=True)
        gact_table = paper_tables(df_proc, show_process=False)
        avg_cycle_time = float(gact_table.at[("mean_gact", lam), "TrainEval"])
        cycle_time_history.append(avg_cycle_time)

        by_proc = df.groupby("process", dropna=True)
        avg_ct_by_proc = by_proc["cycle_time"].mean().to_dict()
        throughput_by_proc = (by_proc.size()).to_dict()  

        print(f"[Episode {episode}] Avg Cycle Time: {avg_cycle_time:.2f} | Time: {elapsed:.2f}s")
        with open(log_path, "a") as f:
            f.write(
                f'{episode},{avg_cycle_time:.4f},{elapsed:.2f},'
                f'"{json.dumps({k: float(v) for k, v in avg_ct_by_proc.items()})}",'
                f'"{json.dumps({k: float(v) for k, v in throughput_by_proc.items()})}"\n'
            )

        this_score = score_from_proc_ct(avg_ct_by_proc, throughput_by_proc)
        if this_score < best_score - 1e-9:
            torch.save(agent_proc.agent.online.state_dict(), os.path.join(SAVE_DIR, "online.pt"))
            torch.save(agent_proc.agent.target.state_dict(), os.path.join(SAVE_DIR, "target.pt"))
            best_score = this_score
            best_meta = {
                "episode": int(episode),
                "score": float(this_score),
                "avg_cycle_time_global": float(avg_cycle_time),
                "avg_ct_by_proc": {k: float(v) for k, v in avg_ct_by_proc.items()},
                "throughput_by_proc": {k: float(v) for k, v in throughput_by_proc.items()},
                "elapsed_sec": float(elapsed),
            }
            with open(best_meta_path, "w") as f:
                json.dump(best_meta, f, indent=2)
            print(f"[BEST] ↓ improved best per-process score to {this_score:.4f} at episode {episode}. Saved best networks.")

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(
                agent_proc.agent.online.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"online_ep{episode + 1}.pt")
            )
            torch.save(
                agent_proc.agent.target.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"target_ep{episode + 1}.pt")
            )
            print(f"[Checkpoint] Saved models at episode {episode + 1}")

        del simulator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(agent_proc, "reset_after_episode"):
            agent_proc.reset_after_episode()

        if len(cycle_time_history) >= CONVERGENCE_WINDOW:
            recent = pd.to_numeric(pd.Series(cycle_time_history[-CONVERGENCE_WINDOW:]),
                                errors='coerce').dropna()
            if len(recent) >= 2 and (recent.max() - recent.min()) < IMPROVEMENT_THRESHOLD:
                print(f"[INFO] Training converged after {episode + 1} episodes "
                    f"(Δ cycle_time < {IMPROVEMENT_THRESHOLD})")
                with open(log_path, "a") as f:
                    f.write(f"# Converged at episode {episode + 1}\n")
                converged = True
                break


    print(f"\n[INFO] Saving final (last) models to {SAVE_DIR}")
    torch.save(agent_proc.agent.online.state_dict(), os.path.join(SAVE_DIR, "online_last.pt"))
    torch.save(agent_proc.agent.target.state_dict(), os.path.join(SAVE_DIR, "target_last.pt"))
    if not converged:
        print("[INFO] Max episodes reached before convergence.")



if __name__ == "__main__":
    set_start_method("spawn", force=True)
    args_list = [(s, l) for s in SCENARIOS for l in ARRIVAL_RATES]
    with Pool(processes=min(4, len(args_list))) as pool:
        pool.starmap(train_double_dqn_exact, args_list)
