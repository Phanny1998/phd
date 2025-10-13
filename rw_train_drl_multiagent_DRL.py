
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
from run_drl_multiagent_DRL import MultiAgentPPO
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


def get_ep(fname):
    try:
        return int(fname.split("_ep")[-1].split(".pt")[0])
    except:
        return -1


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


def train_multi_agent(s, l):
    processes = process_function(l, s)
    SAVE_DIR = f"trained_models/DRLMultiAgent/{s}_lambda{l}"
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    expected = [f"critic_{p.name}.pt" for p in processes] + [f"actor_{p.name}.pt" for p in processes]
    if all(os.path.exists(os.path.join(SAVE_DIR, f)) for f in expected):
        print(f"[INFO] Training already completed for {s} λ={l}. Skipping.")
        return

    available_eps = [get_ep(f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt") and "_ep" in f]
    resume_episode = max(available_eps) if available_eps else 0

    agent_proc = MultiAgentPPO(processes, s, resume_from_checkpoint=(resume_episode > 0))

    if resume_episode > 0:
        print(f"[INFO] Resuming training from episode {resume_episode}")
        for pname, agent in agent_proc.agents.items():
            cp = os.path.join(CHECKPOINT_DIR, f"critic_{pname}_ep{resume_episode}.pt")
            ap = os.path.join(CHECKPOINT_DIR, f"actor_{pname}_ep{resume_episode}.pt")
            if os.path.exists(cp):
                agent.critic.load_state_dict(torch.load(cp, map_location="cpu"))
            if os.path.exists(ap):
                agent.actor.load_state_dict(torch.load(ap, map_location="cpu"))

    log_path = os.path.join(SAVE_DIR, "training_log.csv")
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

    converged = False

    for episode in trange(resume_episode, MAX_EPISODES, desc=f"Training {s} λ={l}"):
        start_time = time.time()
        simulator = Simulator(
            simulation_run=episode,
            process=agent_proc,
            scenario_name='DRL_MultiAgent',
            log_only_case_completion=True,
            is_training=True
        )
        simulator.run(SIMULATION_UNIT)
        elapsed = time.time() - start_time

        df = simulator.event_log.get_dataframe()
        method_name = agent_proc.allocation_method_name  # keep method label
        avg_ct, avg_ct_by_proc, thr_by_proc, total_thr = compute_episode_metrics(df, l, method_name)
        cycle_time_history.append(avg_ct)

        print(f"[Episode {episode}] Avg CT: {avg_ct:.3f} | Total throughput: {total_thr} | Time: {elapsed:.2f}s")
        with open(log_path, "a") as f:
            f.write(
                f'{episode},{avg_ct:.6f},{total_thr},{elapsed:.3f},'
                f'"{json.dumps({k: float(v) for k, v in avg_ct_by_proc.items()})}",'
                f'"{json.dumps({k: int(v) for k, v in thr_by_proc.items()})}"\n'
            )

        del simulator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            for pname, agent in agent_proc.agents.items():
                torch.save(agent.critic.state_dict(), os.path.join(CHECKPOINT_DIR, f"critic_{pname}_ep{episode+1}.pt"))
                torch.save(agent.actor.state_dict(),  os.path.join(CHECKPOINT_DIR, f"actor_{pname}_ep{episode+1}.pt"))
            print(f"[Checkpoint] Saved models at episode {episode + 1}")

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

    print(f"\n[INFO] Saving final models to {SAVE_DIR}")
    for pname, agent in agent_proc.agents.items():
        torch.save(agent.critic.state_dict(), os.path.join(SAVE_DIR, f"critic_{pname}.pt"))
        torch.save(agent.actor.state_dict(),  os.path.join(SAVE_DIR, f"actor_{pname}.pt"))
    if not converged:
        print("[INFO] Max episodes reached before convergence.")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    args_list = [(s, l) for s in SCENARIOS for l in ARRIVAL_RATES]
    with Pool(processes=min(10, len(args_list))) as pool:
        pool.starmap(train_multi_agent, args_list)
