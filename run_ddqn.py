import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single = True
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x.squeeze(0) if single else x


@dataclass
class DDQNConfig:
    gamma: float = 1.0
    lr: float = 1e-3
    batch_size: int = 32
    replay_capacity: int = 24_000
    min_replay_to_train: int = 0
    target_update_steps: int = 10_000


class ReplayBuffer:
    def __init__(self, capacity: int):
        from collections import deque
        self.buf = deque(maxlen=capacity)
    def __len__(self): return len(self.buf)
    def push(self, s, a, r, sp): self.buf.append((s, a, r, sp))
    def sample(self, batch_size: int):
        import random, numpy as np
        batch = random.sample(self.buf, batch_size)
        s, a, r, sp = map(np.array, zip(*batch))
        return s.astype(np.float32), a.astype(np.int64), r.astype(np.float32), sp.astype(np.float32)


class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device=None, cfg: DDQNConfig = DDQNConfig()):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg

        self.online = QNet(state_dim, action_dim).to(self.device)
        self.target = QNet(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()          
        self.online.train()         

        self.optim = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(self.cfg.replay_capacity)
        self.steps = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float) -> int:
        
        self.steps += 1
        # explore
        if random.random() < epsilon:
            return random.randint(0, self.online.out.out_features - 1)

        # exploit
        was_training = self.online.training
        self.online.eval()  
        try:
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            q = self.online(s_t)             
            a = int(torch.argmax(q).item())
        finally:
            if was_training:
                self.online.train()          
        return a

    def push(self, s, a, r, sp):
        self.replay.push(s, a, r, sp)

    def train_step(self) -> float | None:
        if len(self.replay) < max(self.cfg.min_replay_to_train, self.cfg.batch_size):
            return None

        self.online.train()

        s, a, r, sp = self.replay.sample(self.cfg.batch_size)
        s_t  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a_t  = torch.tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r_t  = torch.tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        sp_t = torch.tensor(sp, dtype=torch.float32, device=self.device)

        q_sa = self.online(s_t).gather(1, a_t)

        with torch.no_grad():
            a_star = self.online(sp_t).argmax(dim=1, keepdim=True)
            q_sp = self.target(sp_t).gather(1, a_star)
            y = r_t + self.cfg.gamma * q_sp

        loss = self.loss_fn(q_sa, y)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()

        if self.steps % self.cfg.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()

        return float(loss.item())
    
    def save(self, online_path: str, target_path: str | None = None):
        torch.save(self.online.state_dict(), online_path)
        if target_path:
            torch.save(self.target.state_dict(), target_path)

    def load(self, online_path: str, target_path: str | None = None, map_location="cpu"):
        self.online.load_state_dict(torch.load(online_path, map_location=map_location))
        if target_path and os.path.exists(target_path):
            self.target.load_state_dict(torch.load(target_path, map_location=map_location))
        else:
            self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.online.train()




import os
import numpy as np
from collections import defaultdict

from framework.processes import SimpleProcess, Task

class DoubleDQNProcess(SimpleProcess):
   
    def __init__(self, processes, scenario_name: str, save_dir: str | None = None, resume_from_checkpoint: bool = False):
        super().__init__(processes)
        self.allocation_method_name = "DoubleDQN"
        self.scenario_name = scenario_name
        self.save_dir = save_dir
        self.resume_from_checkpoint = resume_from_checkpoint

        self.activities = list({
            t.name: t
            for p in processes
            for t in p.tasks.values()
            if isinstance(t, Task) and t.resources
        }.values())
        self.activity_names = [t.name for t in self.activities]
        self.act_index = {name: i for i, name in enumerate(self.activity_names)}

        self.resources = list({
            r.name: r
            for p in processes
            for t in p.tasks.values()
            if isinstance(t, Task)
            for r in t.resources
        }.values())

        self.nR = len(self.resources)
        self.nT = len(self.activities)
        self.state_dim = self.nR + self.nT
        self.action_dim = self.nR * self.nT + 1  # +1 for NOOP

        self.agent = DDQNAgent(self.state_dim, self.action_dim)

        self.current_epsilon = 0.1
        self.training_enabled = False

        self.open_tr = {}

        self._loss_sum = 0.0
        self._loss_n = 0


    def load_models_if_available(self, lam: float | None = None, prefer: str = "best") -> bool:
        
        model_dir = self.save_dir
        if model_dir is None:
            if lam is None:
                try:
                    lam = next(iter(self.process_definitions.values())).arrival_distribution
                except Exception:
                    pass
            if lam is None:
                print("[DoubleDQN] Could not infer lambda — please pass lam to load_models_if_available().")
                return False
            model_dir = f"trained_models/DoubleDQN/{self.scenario_name}_lambda{lam}"

        ckpt_dir = os.path.join(model_dir, "checkpoints")

        candidates = []
        if prefer == "best":
            candidates.extend([
                (os.path.join(model_dir, "online.pt"),      os.path.join(model_dir, "target.pt")),
                (os.path.join(model_dir, "online_last.pt"), os.path.join(model_dir, "target_last.pt")),
            ])
        else:
            candidates.extend([
                (os.path.join(model_dir, "online_last.pt"), os.path.join(model_dir, "target_last.pt")),
                (os.path.join(model_dir, "online.pt"),      os.path.join(model_dir, "target.pt")),
            ])

        if os.path.isdir(ckpt_dir):
            try:
                def get_ep(fname: str) -> int:
                    try:
                        return int(fname.split("_ep")[-1].split(".pt")[0])
                    except Exception:
                        return -1
                eps = [get_ep(f) for f in os.listdir(ckpt_dir) if f.startswith("online_ep") and f.endswith(".pt")]
                if eps:
                    latest = max(eps)
                    candidates.append(
                        (os.path.join(ckpt_dir, f"online_ep{latest}.pt"),
                         os.path.join(ckpt_dir, f"target_ep{latest}.pt"))
                    )
            except Exception:
                pass

        # try to load the first existing pair
        for online_path, target_path in candidates:
            if os.path.exists(online_path) and os.path.exists(target_path):
                try:
                    self.agent.online.load_state_dict(
                        __import__("torch").load(online_path, map_location="cpu")
                    )
                    self.agent.target.load_state_dict(
                        __import__("torch").load(target_path, map_location="cpu")
                    )
                    print(f"[DoubleDQN] Loaded pretrained weights:\n  {online_path}\n  {target_path}")
                    return True
                except Exception as e:
                    print(f"[DoubleDQN] Failed to load from:\n  {online_path}\n  {target_path}\n  Error: {e}")
                    # try next candidate
        print(f"[DoubleDQN] No pretrained weights found in {model_dir}.")
        return False

    def _decode_action(self, a_idx: int):
        if a_idx == 0:
            return None, None, True  # NOOP
        a = a_idx - 1
        r_idx = a // self.nT
        t_idx = a %  self.nT
        return r_idx, t_idx, False

    def _state_from_shadow(self, temp_available, temp_unassigned_ids, unassigned_tasks):
        avail_bits = [1.0 if r in temp_available else 0.0 for r in self.resources]
        counts = [0] * self.nT
        for tid in temp_unassigned_ids:
            nm = unassigned_tasks[tid].label
            if nm in self.act_index:
                counts[self.act_index[nm]] += 1
        total = sum(counts) or 1
        mix = [c / total for c in counts]
        return np.array(avail_bits + mix, dtype=np.float32)

    def _close_all_open_with_next(self, next_state):
        for cid, tr in list(self.open_tr.items()):
            self.agent.push(tr["s"], tr["a"], 0.0, next_state)
            del self.open_tr[cid]
            if self.training_enabled:
                loss = self.agent.train_step()
                if loss is not None:
                    self._loss_sum += loss
                    self._loss_n += 1

    def _resource_can_do_task(self, res_name: str, task_name: str) -> bool:
        t = next((t for t in self.activities if t.name == task_name), None)
        return any(r.name == res_name for r in t.resources) if t else False

    def assign_resources(self, unassigned_tasks, available_resources, resource_available_times):
        assignments = []
        temp_available = set(available_resources)
        temp_unassigned_ids = set(unassigned_tasks.keys())

        while True:
            if not temp_available or not temp_unassigned_ids:
                break

            s = self._state_from_shadow(temp_available, temp_unassigned_ids, unassigned_tasks)
            a_idx = self.agent.act(s, self.current_epsilon)

            r_idx, t_idx, is_noop = self._decode_action(a_idx)
            if is_noop:
                self._close_all_open_with_next(s)
                break

            res_obj = self.resources[r_idx]
            t_name = self.activities[t_idx].name

            eligible = self._resource_can_do_task(res_obj.name, t_name)

            chosen_tid = next((tid for tid in temp_unassigned_ids if unassigned_tasks[tid].label == t_name), None)
            if (chosen_tid is None) or (res_obj not in temp_available) or (not eligible):
                self._close_all_open_with_next(s)
                break

            task_obj = unassigned_tasks[chosen_tid]
            cid = task_obj.case_id
            self.open_tr[cid] = dict(s=s, a=a_idx)

            assignments.append((task_obj, res_obj))
            temp_available.remove(res_obj)
            temp_unassigned_ids.remove(chosen_tid)

        if self.training_enabled:
            loss = self.agent.train_step()
            if loss is not None:
                self._loss_sum += loss
                self._loss_n += 1

        return assignments

    def handle_case_completion_reward(self, process_element):
        cid = process_element.case_id
        temp_available = set(self.simulator.available_resources)
        temp_unassigned_ids = set(self.simulator.unassigned_tasks.keys())
        s_next = self._state_from_shadow(temp_available, temp_unassigned_ids, self.simulator.unassigned_tasks)

        if cid in self.open_tr:
            s, a = self.open_tr[cid]["s"], self.open_tr[cid]["a"]
            self.agent.push(s, a, 1.0, s_next)
            del self.open_tr[cid]
        else:
            self.agent.push(s_next, 0, 1.0, s_next)

        if self.training_enabled:
            loss = self.agent.train_step()
            if loss is not None:
                self._loss_sum += loss
                self._loss_n += 1

    def reset_after_episode(self):
        self.open_tr.clear()
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self.agent.save(
                online_path=os.path.join(self.save_dir, "online_last.pt"),
                target_path=os.path.join(self.save_dir, "target_last.pt"),
            )
        self._loss_sum = 0.0
        self._loss_n = 0

    def get_episode_loss_mean(self) -> float | None:
        if self._loss_n == 0:
            return None
        return self._loss_sum / self._loss_n



import os
import time
import multiprocessing as mp

import torch
import pandas as pd
from tqdm import tqdm

from framework.config import (
    ARRIVAL_RATES,
    SCENARIO_NAMES,
    SIMULATION_RUN_TIME,
    SIMULATION_RUNS,
    process_function,
)
from framework.simulator import Simulator
mp.set_start_method("spawn", force=True)

os.makedirs("results", exist_ok=True)
os.makedirs("runtime", exist_ok=True)

RUNTIME_CSV_PATH = "runtime/DoubleDQN_runtimes.csv"
if not os.path.exists(RUNTIME_CSV_PATH):
    with open(RUNTIME_CSV_PATH, "w") as f:
        f.write("method,scenario,l,run,time,loaded,which\n")


def run_double_dqn_simulation(args):
    l, scenario_name, run_id = args

    import torch
    from framework.config import process_function
    from framework.simulator import Simulator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Run {run_id}] Using device: {device} | scenario={scenario_name} λ={l}")

    processes = process_function(l, scenario_name)
    agent = DoubleDQNProcess(processes, scenario_name)

    loaded = agent.load_models_if_available(lam=l, prefer="best")
    which = "best"
    if not loaded:
        # fallback to last
        loaded = agent.load_models_if_available(lam=l, prefer="last")
        which = "last" if loaded else "none"

    start_time = time.time()
    simulator = Simulator(simulation_run=run_id, process=agent, scenario_name="DoubleDQN_EXACT", log_only_case_completion=True)
    simulator.run(SIMULATION_RUN_TIME)
    runtime_seconds = time.time() - start_time

    df = simulator.event_log.get_dataframe()
    df["run"] = run_id
    df["method"] = "DoubleDQNExact"
    df["l"] = l
    df["scenario"] = scenario_name

    return df, runtime_seconds, run_id, loaded, which


if __name__ == "__main__":
    for l in ARRIVAL_RATES:
        for scenario_name in SCENARIO_NAMES:
            print(f"\n=== Running scenario: {scenario_name}, arrival rate: {l} (DoubleDQNExact) ===")

            args_list = [(l, scenario_name, i) for i in range(SIMULATION_RUNS)]

            with mp.Pool(processes=min(10, SIMULATION_RUNS)) as pool:
                results = list(tqdm(pool.imap(run_double_dqn_simulation, args_list), total=len(args_list)))

            combined_logs = []
            for df, runtime_seconds, run_id, loaded, which in results:
                combined_logs.append(df)
                with open(RUNTIME_CSV_PATH, "a") as f:
                    f.write(f"DoubleDQNExact,{scenario_name},{l},{run_id},{runtime_seconds:.2f},{int(loaded)},{which}\n")
                print(f"[Run {run_id}] Completed in {runtime_seconds:.2f}s | loaded={loaded}({which})")

            combined_df = pd.concat(combined_logs, ignore_index=True)
            result_filename = f"results/DoubleDQNExact_l{l}_{scenario_name}.csv"
            combined_df.to_csv(result_filename, index=False)
            print(f"Saved results to {result_filename}")
