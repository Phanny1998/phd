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
from run_mupromac import MuProMAC
from framework.result_helper import paper_tables, process_data
from framework.config_training import (
    SCENARIOS,
    ARRIVAL_RATES,
    SIMULATION_UNIT,
    MAX_EPISODES,
    CONVERGENCE_WINDOW,
    IMPROVEMENT_THRESHOLD,
)

warnings.filterwarnings("ignore")

CHECKPOINT_INTERVAL = 5



def get_checkpoint_episode(filename: str) -> int:
    try:
        return int(filename.split("_ep")[-1].split(".pt")[0])
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



def train_mupromac(s: str, l: float):
    processes_for_names = process_function(l, s)

    SAVE_DIR = f"trained_models/MuProMAC/{s}_lambda{l}"
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_meta_path = os.path.join(SAVE_DIR, "best_meta.json")
    log_file_path = os.path.join(SAVE_DIR, "training_log.csv")

    expected_files = (
        ["critic.pt"]
        + [f"actor_{p.name}.pt" for p in processes_for_names]
        + [f"gru_{p.name}.pt" for p in processes_for_names]
    )
    if all(os.path.exists(os.path.join(SAVE_DIR, fname)) for fname in expected_files):
        print(f"[INFO] Training already completed for {s} λ={l}. Skipping.")
        return

    available_episodes = [
        get_checkpoint_episode(f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".pt") and "_ep" in f
    ]
    resume_episode = max(available_episodes) if available_episodes else 0

    best_meta = safe_json_load(best_meta_path, default={}) or {}
    best_score = float(best_meta.get("score", float("inf")))

    if resume_episode == 0 and not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("episode,avg_cycle_time,elapsed_time,avg_cycle_time_by_process,throughput_by_process\n")

    print(f"[INFO] Initializing MuProMAC for {s} λ={l}")
    processes = process_function(l, s)
    mupromac_process = MuProMAC(processes, s, resume_from_checkpoint=(resume_episode > 0))

    if resume_episode > 0:
        print(f"[INFO] Resuming training from episode {resume_episode}")
        critic_path = os.path.join(CHECKPOINT_DIR, f"critic_ep{resume_episode}.pt")
        if os.path.exists(critic_path):
            mupromac_process.agent.critic.load_state_dict(
                torch.load(critic_path, map_location=mupromac_process.device)
            )

        for pname in mupromac_process.agent.actors:
            actor_path = os.path.join(CHECKPOINT_DIR, f"actor_{pname}_ep{resume_episode}.pt")
            if os.path.exists(actor_path):
                mupromac_process.agent.actors[pname].load_state_dict(
                    torch.load(actor_path, map_location=mupromac_process.device)
                )

        for pname in mupromac_process.gru_models:
            gru_path = os.path.join(CHECKPOINT_DIR, f"gru_{pname}_ep{resume_episode}.pt")
            if os.path.exists(gru_path):
                mupromac_process.gru_models[pname].load_state_dict(
                    torch.load(gru_path, map_location=mupromac_process.device)
                )

    if os.path.exists(log_file_path):
        try:
            df_log = pd.read_csv(log_file_path)
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
            process=mupromac_process,
            scenario_name='MuProMAC',
            log_only_case_completion=True,
            is_training=True
        )
        simulator.run(SIMULATION_UNIT)
        elapsed = time.time() - start_time

        df = simulator.event_log.get_dataframe()
        df["method"] = "TrainEval"
        df["l"] = l
        df["simulation_run"] = episode

        df_proc, _ = process_data(df, from_memory=True)
        gact_table = paper_tables(df_proc, show_process=False)
        avg_cycle_time = gact_table.at[("mean_gact", l), "TrainEval"]
        cycle_time_history.append(avg_cycle_time)

        by_proc = df.groupby("process", dropna=True)
        avg_ct_by_proc = by_proc["cycle_time"].mean().to_dict()
        throughput_by_proc = (by_proc.size()).to_dict()  

        print(f"[Episode {episode}] Avg Cycle Time: {avg_cycle_time:.2f} | Time: {elapsed:.2f}s")
        with open(log_file_path, "a") as f:
            f.write(
                f'{episode},{avg_cycle_time:.4f},{elapsed:.2f},'
                f'"{json.dumps({k: float(v) for k, v in avg_ct_by_proc.items()})}",'
                f'"{json.dumps({k: float(v) for k, v in throughput_by_proc.items()})}"\n'
            )

        this_score = score_from_proc_ct(avg_ct_by_proc, throughput_by_proc)
        if this_score < best_score - 1e-9:
            torch.save(mupromac_process.agent.critic.state_dict(), os.path.join(SAVE_DIR, "critic.pt"))
            for pname, actor in mupromac_process.agent.actors.items():
                torch.save(actor.state_dict(), os.path.join(SAVE_DIR, f"actor_{pname}.pt"))
            for pname, gru in mupromac_process.gru_models.items():
                torch.save(gru.state_dict(), os.path.join(SAVE_DIR, f"gru_{pname}.pt"))

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
            print(f"[BEST] ↓ improved best per-process score to {this_score:.4f} at episode {episode}. Saved best critic/actors/grus.")

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(
                mupromac_process.agent.critic.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"critic_ep{episode + 1}.pt")
            )
            for pname, actor in mupromac_process.agent.actors.items():
                torch.save(actor.state_dict(), os.path.join(CHECKPOINT_DIR, f"actor_{pname}_ep{episode + 1}.pt"))
            for pname, gru in mupromac_process.gru_models.items():
                torch.save(gru.state_dict(), os.path.join(CHECKPOINT_DIR, f"gru_{pname}_ep{episode + 1}.pt"))
            print(f"[Checkpoint] Saved models at episode {episode + 1}")

        del simulator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(mupromac_process, "reset_after_episode"):
            mupromac_process.reset_after_episode()

        if len(cycle_time_history) >= CONVERGENCE_WINDOW:
            recent = cycle_time_history[-CONVERGENCE_WINDOW:]
            if max(recent) - min(recent) < IMPROVEMENT_THRESHOLD:
                print(f"[INFO] Training converged after {episode + 1} episodes "
                      f"(Δ cycle_time < {IMPROVEMENT_THRESHOLD})")
                with open(log_file_path, "a") as f:
                    f.write(f"# Converged at episode {episode + 1}\n")
                converged = True
                break

    print(f"\n[INFO] Saving final (last) models to {SAVE_DIR}")
    torch.save(mupromac_process.agent.critic.state_dict(), os.path.join(SAVE_DIR, "critic_last.pt"))
    for pname, actor in mupromac_process.agent.actors.items():
        torch.save(actor.state_dict(), os.path.join(SAVE_DIR, f"actor_{pname}_last.pt"))
    for pname, gru in mupromac_process.gru_models.items():
        torch.save(gru.state_dict(), os.path.join(SAVE_DIR, f"gru_{pname}_last.pt"))

    if not all(os.path.exists(os.path.join(SAVE_DIR, f)) for f in expected_files):
        print("[INFO] Note: best models were not created during this run (no improvement). "
              "You still have last weights saved as *_last.pt.")
    print("[INFO] All models saved.")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    args_list = [(s, l) for s in SCENARIOS for l in ARRIVAL_RATES]
    with Pool(processes=min(16, len(args_list))) as pool:
        pool.starmap(train_mupromac, args_list)
