
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
from run_drl_singleagent_DRL import SingleAgentPPO
from framework.result_helper import paper_tables, process_data
from framework.rw_config_training import (
    SCENARIOS,
    ARRIVAL_RATES,
    SIMULATION_UNIT,
    MAX_EPISODES,
    CONVERGENCE_WINDOW,
    IMPROVEMENT_THRESHOLD,
)

warnings.filterwarnings("ignore")

CHECKPOINT_INTERVAL = 5  


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
    return sum(vals.values()) / len(vals)


def compute_episode_metrics(df: pd.DataFrame, lam: float, method_name: str):
    
    df_local = df.copy()
    df_local["method"] = method_name
    df_local["l"] = lam

    df_proc, _ = process_data(df_local, from_memory=True)
    gact_table = paper_tables(df_proc, show_process=False)

    row_key = ("mean_gact", lam)
    if method_name in gact_table.columns:
        method_col = method_name
    elif "TrainEval" in gact_table.columns:
        method_col = "TrainEval"
    else:
        method_col = next(iter(gact_table.columns))
    try:
        avg_cycle_time = float(gact_table.loc[row_key, method_col])
    except Exception:
        avg_cycle_time = float(df_local[df_local["status"] == "COMPLETE"]["cycle_time"].mean())

    df_done = df_local[df_local["status"] == "COMPLETE"]
    throughput_by_process = df_done.groupby("process").size().to_dict()
    total_throughput = int(df_done.shape[0])

    avg_cycle_time_by_process = df_done.groupby("process")["cycle_time"].mean().to_dict()

    return avg_cycle_time, avg_cycle_time_by_process, throughput_by_process, total_throughput


def train_single_agent(scenario_name: str, lam: float):
    SAVE_DIR = f"trained_models/DRL_SingleAgent/{scenario_name}_lambda{lam}"
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_meta_path = os.path.join(SAVE_DIR, "best_meta.json")
    log_path = os.path.join(SAVE_DIR, "training_log.csv")

    if os.path.exists(os.path.join(SAVE_DIR, "critic.pt")) and os.path.exists(os.path.join(SAVE_DIR, "actor.pt")):
        print(f"[INFO] Training already completed for {scenario_name} λ={lam}. Skipping.")
        return

    available_eps = [get_ep(f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt") and "_ep" in f]
    resume_episode = max(available_eps) if available_eps else 0

    best_meta = safe_json_load(best_meta_path, default={}) or {}
    best_score = float(best_meta.get("score", float("inf")))

    if resume_episode == 0 and not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("episode,avg_cycle_time,total_throughput,elapsed_time,avg_cycle_time_by_process,throughput_by_process\n")

    if os.path.exists(log_path):
        try:
            df_log = pd.read_csv(log_path)
            cycle_time_history = df_log['avg_cycle_time'].tolist()
        except Exception:
            cycle_time_history = []
    else:
        cycle_time_history = []

    print(f"\n=== Training DRL_SingleAgent on '{scenario_name}' with λ={lam} ===")
    processes = process_function(lam, scenario_name)
    agent_proc = SingleAgentPPO(processes, scenario_name, resume_from_checkpoint=(resume_episode > 0))

    if resume_episode > 0:
        print(f"[INFO] Resuming training from episode {resume_episode}")
        critic_path = os.path.join(CHECKPOINT_DIR, f"critic_ep{resume_episode}.pt")
        actor_path = os.path.join(CHECKPOINT_DIR, f"actor_ep{resume_episode}.pt")
        agent_proc.agent.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
        agent_proc.agent.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))

    converged = False

    for episode in trange(resume_episode, MAX_EPISODES, desc=f"Training {scenario_name} λ={lam}"):
        start_time = time.time()
        simulator = Simulator(
            simulation_run=episode,
            process=agent_proc,
            scenario_name='DRL_SingleAgent',
            log_only_case_completion=True,
            is_training=True
        )
        simulator.run(SIMULATION_UNIT)
        elapsed = time.time() - start_time

        df = simulator.event_log.get_dataframe()
        
        avg_ct, avg_ct_by_proc, thr_by_proc, total_thr = compute_episode_metrics(df, lam, "TrainEval")
        cycle_time_history.append(avg_ct)

        print(f"[Episode {episode}] Avg CT: {avg_ct:.3f} | Total throughput: {total_thr} | Time: {elapsed:.2f}s")
        with open(log_path, "a") as f:
            f.write(
                f'{episode},{avg_ct:.6f},{total_thr},{elapsed:.3f},'
                f'"{json.dumps({k: float(v) for k, v in avg_ct_by_proc.items()})}",'
                f'"{json.dumps({k: int(v) for k, v in thr_by_proc.items()})}"\n'
            )

        this_score = score_from_proc_ct(avg_ct_by_proc, thr_by_proc)
        if this_score < best_score - 1e-9:
            torch.save(agent_proc.agent.critic.state_dict(), os.path.join(SAVE_DIR, "critic.pt"))
            torch.save(agent_proc.agent.actor.state_dict(),  os.path.join(SAVE_DIR, "actor.pt"))
            best_score = this_score
            best_meta = {
                "episode": int(episode),
                "score": float(this_score),
                "avg_cycle_time_global": float(avg_ct),
                "avg_ct_by_proc": {k: float(v) for k, v in avg_ct_by_proc.items()},
                "throughput_by_proc": {k: int(v) for k, v in thr_by_proc.items()},
                "total_throughput": int(total_thr),
                "elapsed_sec": float(elapsed),
            }
            with open(best_meta_path, "w") as f:
                json.dump(best_meta, f, indent=2)
            print(f"[BEST] ↓ improved score to {this_score:.4f} at episode {episode}.")

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(agent_proc.agent.critic.state_dict(), os.path.join(CHECKPOINT_DIR, f"critic_ep{episode + 1}.pt"))
            torch.save(agent_proc.agent.actor.state_dict(),  os.path.join(CHECKPOINT_DIR, f"actor_ep{episode + 1}.pt"))
            print(f"[Checkpoint] Saved models at episode {episode + 1}")

        del simulator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(agent_proc, "reset_after_episode"):
            agent_proc.reset_after_episode()

        if len(cycle_time_history) >= CONVERGENCE_WINDOW:
            recent = cycle_time_history[-CONVERGENCE_WINDOW:]
            if max(recent) - min(recent) < IMPROVEMENT_THRESHOLD:
                print(f"[INFO] Training converged after {episode + 1} episodes (Δ CT < {IMPROVEMENT_THRESHOLD}).")
                with open(log_path, "a") as f:
                    f.write(f"# Converged at episode {episode + 1}\n")
                converged = True
                break

    print(f"\n[INFO] Saving last weights to {SAVE_DIR}")
    torch.save(agent_proc.agent.critic.state_dict(), os.path.join(SAVE_DIR, "critic_last.pt"))
    torch.save(agent_proc.agent.actor.state_dict(),  os.path.join(SAVE_DIR, "actor_last.pt"))
    if not converged:
        print("[INFO] Max episodes reached before convergence.")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    args_list = [(s, l) for s in SCENARIOS for l in ARRIVAL_RATES]
    with Pool(processes=min(10, len(args_list))) as pool:
        pool.starmap(train_single_agent, args_list)
