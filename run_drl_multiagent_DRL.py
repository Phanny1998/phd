

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import itertools

from framework.processes import SimpleProcess, Task

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)



class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.999, epsilon=0.2,
                 lr=3e-5, entropy_coef=0.00, gae_lambda=0.95):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but required for this run.")
        self.device = torch.device("cuda")
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        self.actor = Actor(state_size, action_size).to(self.device)
        self.critic = Critic(state_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.initial_lr = lr

    def set_lr(self, new_lr: float):
        for opt in (self.actor_optimizer, self.critic_optimizer):
            for g in opt.param_groups:
                g["lr"] = new_lr

    @staticmethod
    def _masked_dist(probs, masks):
        masked = probs * masks
        masked = masked / (masked.sum(dim=1, keepdim=True) + 1e-10)
        return masked

    @torch.no_grad()
    def _masked_dist_nograd(self, probs, masks):
        return self._masked_dist(probs, masks)

    def select_action(self, state, mask):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(s)
        masked = self._masked_dist_nograd(probs, m).squeeze(0)
        if masked.sum().item() == 0:
            return None, None
        dist = torch.distributions.Categorical(masked)
        a = dist.sample()
        return a.item(), dist.log_prob(a).item()

    def train(self, rollout, epochs=5, minibatch_size=256):
        
        states, actions, rewards, next_states, dones, old_log_probs, masks = zip(*rollout)
        states      = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        masks_t     = torch.tensor(np.array(masks), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            v      = self.critic(states).squeeze(-1)
            v_next = self.critic(next_states).squeeze(-1)

        T = len(rewards)
        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * (1.0 - dones[t]) * v_next[t] - v[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            adv[t] = gae
        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        if minibatch_size is None or minibatch_size >= T:
            idx_iter = [torch.arange(T, device=self.device)]
        else:
            perm = torch.randperm(T, device=self.device)
            idx_iter = [perm[i:i+minibatch_size] for i in range(0, T, minibatch_size)]

        for _ in range(epochs):
            for idx in idx_iter:
                raw = self.actor(states[idx])
                masked = (raw * masks_t[idx])
                masked = masked / (masked.sum(dim=1, keepdim=True) + 1e-10)

                sel = masked.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                logp = torch.log(sel + 1e-10)
                ratios = torch.exp(logp - old_log_probs[idx])

                clipped = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                pg_loss = -torch.min(ratios * adv[idx], clipped * adv[idx]).mean()

                dist = torch.distributions.Categorical(masked)
                entropy = dist.entropy().mean()
                actor_loss = pg_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                v_pred = self.critic(states[idx]).squeeze(-1)
                critic_loss = F.mse_loss(v_pred, ret[idx])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


class MultiAgentPPO(SimpleProcess):
    POSTPONE_KEY = -1

    def __init__(self, processes, scenario_name, resume_from_checkpoint=False, postpone_penalty: float = -0.1):
        super().__init__(processes)
        self.global_res_by_name = {r.name: r for r in self.resources}

        self.allocation_method_name = "DRL_MultiAgent"
        self.scenario_name = scenario_name
        self.postpone_penalty = float(postpone_penalty)

        self.train_every = 500
        self.target_env_steps = 50_000_000 

        self.process_names = [p.name for p in processes]
        self.process_activities = {
            p.name: [t for t in p.tasks.values() if isinstance(t, Task) and t.resources]
            for p in processes
        }
        # self.process_resources = {
        #     p.name: list({r.name: r for t in p.tasks.values() if isinstance(t, Task) for r in t.resources}.values())
        #     for p in processes
        # }
        self.process_resources = {
            p.name: list({
                r.name: self.global_res_by_name[r.name]
                for t in p.tasks.values() if isinstance(t, Task)
                for r in t.resources
            }.values())
            for p in processes
        }
        # self.res_by_proc_and_name = {
        #     pname: {r.name: r for r in rlist}
        #     for pname, rlist in self.process_resources.items()
        # }

        self.res_by_proc_and_name = {
            pname: {r.name: self.global_res_by_name[r.name] for r in rlist}
            for pname, rlist in self.process_resources.items()
        }
        self.state_sizes = {
            pname: 2 * len(self.process_resources[pname]) + len(self.process_activities[pname])
            for pname in self.process_names
        }
        self.action_spaces = {
            pname: ([(res.name, task.name)
                     for task in self.process_activities[pname]
                     for res in task.resources] + ["POSTPONE"])
            for pname in self.process_names
        }

        self.agents = {pname: PPOAgent(self.state_sizes[pname], len(self.action_spaces[pname]))
                       for pname in self.process_names}
        self.buffers = {pname: deque(maxlen=25700) for pname in self.process_names}
        self.assignment_counters = {pname: 0 for pname in self.process_names}
        self.total_env_steps = {pname: 0 for pname in self.process_names}

        self.open_tr = defaultdict(dict)
        self.case_transitions = defaultdict(list)

        # optional pretrained load
        self.models_loaded = False
        if not resume_from_checkpoint:
            self._load_models_if_available()

    def _load_models_if_available(self):
        try:
            lambda_val = self.process_definitions[self.process_names[0]].arrival_distribution
        except Exception:
            lambda_val = self.processes[0].arrival_distribution  # fallback
        model_dir = f"trained_models/DRLMultiAgent/{self.scenario_name}_lambda{lambda_val}"
        ok = True
        for pname, agent in self.agents.items():
            ap = os.path.join(model_dir, f"actor_{pname}.pt")
            cp = os.path.join(model_dir, f"critic_{pname}.pt")
            if not (os.path.exists(ap) and os.path.exists(cp)):
                ok = False
                break
            agent.actor.load_state_dict(torch.load(ap, map_location=agent.device))
            agent.critic.load_state_dict(torch.load(cp, map_location=agent.device))
        self.models_loaded = ok
        if ok:
            print(f"[Model Load] Loaded all agents from {model_dir}")
        else:
            print(f"[Model Load] No complete pretrained set at {model_dir}")

    def _local_activity_index(self, pname, task_name):
        for i, t in enumerate(self.process_activities[pname]):
            if t.name == task_name:
                return i
        return -1

    def _shadow_state_for_process(self, pname, temp_available, temp_unassigned_ids, unassigned_tasks, shadow_busy):
        
        local_res = self.process_resources[pname]
        local_acts = self.process_activities[pname]

        availability_bits = [1 if r in temp_available else 0 for r in local_res]

        assign_ids = []
        L = max(1, len(local_acts))
        for r in local_res:
            t = shadow_busy.get(r)
            if t is not None:
                idx = self._local_activity_index(pname, t.label)
                assign_ids.append(idx / L if idx != -1 else 0.0)
            else:
                assign_ids.append(0.0)

        name_to_idx = {t.name: i for i, t in enumerate(local_acts)}
        queues = [0.0] * len(local_acts)
        for tid in temp_unassigned_ids:
            task = unassigned_tasks[tid]
            if task.case_type == pname:
                j = name_to_idx.get(task.label, None)
                if j is not None:
                    queues[j] = min(1.0, queues[j] + 1 / 100.0)

        return np.array(availability_bits + assign_ids + queues, dtype=np.float32)

    # def _build_mask_for_process(self, pname, temp_available, temp_unassigned_ids, unassigned_tasks):
    #     counts = defaultdict(int)
    #     for tid in temp_unassigned_ids:
    #         task = unassigned_tasks[tid]
    #         if task.case_type == pname:
    #             counts[task.label] += 1

    #     mask = []
    #     space = self.action_spaces[pname]
    #     res_map = self.res_by_proc_and_name[pname]
    #     for pair in space[:-1]:
    #         res_name, task_name = pair
    #         res = res_map[res_name]
    #         feasible = (res in temp_available) and (counts[task_name] > 0)
    #         mask.append(1 if feasible else 0)
    #     mask.append(1)  # POSTPONE always allowed
    #     return mask

    def _build_mask_for_process(self, pname, temp_available, temp_unassigned_ids, unassigned_tasks):
        counts = defaultdict(int)
        for tid in temp_unassigned_ids:
            task = unassigned_tasks[tid]
            if task.case_type == pname:
                counts[task.label] += 1

        mask = []
        space = self.action_spaces[pname]
        for res_name, task_name in space[:-1]:
            res = self.global_res_by_name[res_name]       
            feasible = (res in temp_available) and (counts[task_name] > 0)
            mask.append(1 if feasible else 0)
        mask.append(1)  # pOSTPONE
        return mask

    def _close_open_for_process(self, pname, next_state):
        if self.models_loaded:
            self.open_tr[pname].clear()
            return
        for key, tr in list(self.open_tr[pname].items()):
            s, a, r, lp, mk = tr['s'], tr['a'], tr['r'], tr['logp'], tr['mask']
            if key == self.POSTPONE_KEY:
                self.buffers[pname].append((s, a, r, next_state, False, lp, mk))
            else:
                self.case_transitions[(pname, key)].append((s, a, r, next_state, False, lp, mk))
            del self.open_tr[pname][key]

    def _first_task_process(self, temp_unassigned_ids, unassigned_tasks):
        for tid in temp_unassigned_ids:
            return unassigned_tasks[tid].case_type
        return None


    def assign_resources(self, unassigned_tasks, available_resources, resource_available_times):
        assignments = []

        temp_available = set(available_resources)
        temp_unassigned_ids = set(unassigned_tasks.keys())
        shadow_busy = dict(self.simulator.busy_resources)

        while True:
            any_non_postpone = False
            for pname in self.process_names:
                mask = self._build_mask_for_process(pname, temp_available, temp_unassigned_ids, unassigned_tasks)
                if sum(mask[:-1]) > 0:
                    any_non_postpone = True
                    break
            if not any_non_postpone:
                break

            first_p = self._first_task_process(temp_unassigned_ids, unassigned_tasks)
            if first_p in self.process_names:
                order = [first_p] + [p for p in self.process_names if p != first_p]
            else:
                order = list(self.process_names)

            did_any = False
            for pname in order:
                mask = self._build_mask_for_process(pname, temp_available, temp_unassigned_ids, unassigned_tasks)
                if sum(mask[:-1]) == 0:
                    continue

                state = self._shadow_state_for_process(pname, temp_available, temp_unassigned_ids, unassigned_tasks, shadow_busy)

                self._close_open_for_process(pname, state)

                agent = self.agents[pname]
                action, log_prob = agent.select_action(state, mask)
                if action is None:
                    continue

                space = self.action_spaces[pname]
                POSTPONE_IDX = len(space) - 1

                if action == POSTPONE_IDX:
                    if not self.models_loaded:
                        self.open_tr[pname][self.POSTPONE_KEY] = {
                            's': state, 'a': action, 'r': self.postpone_penalty,
                            'logp': log_prob, 'mask': mask
                        }
                        self.assignment_counters[pname] += 1
                    continue  

                res_name, task_name = space[action]
                #res_obj = self.res_by_proc_and_name[pname][res_name]
                res_obj = self.global_res_by_name[res_name]
                if res_obj not in temp_available:
                    continue
                selected_task, selected_tid = None, None
                for tid in temp_unassigned_ids:
                    t = unassigned_tasks[tid]
                    if t.case_type == pname and t.label == task_name:
                        selected_task, selected_tid = t, tid
                        break
                if selected_task is None:
                    continue

                if not self.models_loaded:
                    cid = selected_task.case_id
                    self.open_tr[pname][cid] = {
                        's': state, 'a': action, 'r': 0.0, 'logp': log_prob, 'mask': mask
                    }
                    self.assignment_counters[pname] += 1

                assignments.append((selected_task, res_obj))
                temp_available.remove(res_obj)
                temp_unassigned_ids.remove(selected_tid)
                shadow_busy[res_obj] = selected_task
                did_any = True

            if not did_any:
                break

        self._train_agents()
        return assignments


    def on_element_completed(self, process_element):
        return

    def handle_case_completion_reward(self, process_element):
        pname = process_element.case_type
        cid = process_element.case_id
        start_time = self.simulator.case_start_times[cid]
        cycle_time = self.simulator.now - start_time
        reward = 1.0 / (cycle_time + 1)

        if self.models_loaded:
            return

        if cid in self.open_tr[pname]:
            tr = self.open_tr[pname].pop(cid)
            s, a, r0, lp, mk = tr['s'], tr['a'], tr['r'], tr['logp'], tr['mask']
            temp_avail = set(self.simulator.available_resources)
            temp_uids = set(self.simulator.unassigned_tasks.keys())
            shadow_now = dict(self.simulator.busy_resources)
            next_state = self._shadow_state_for_process(pname, temp_avail, temp_uids, self.simulator.unassigned_tasks, shadow_now)
            self.buffers[pname].append((s, a, reward, next_state, True, lp, mk))
        else:
            key = (pname, cid)
            traj = self.case_transitions.get(key, [])
            if traj:
                s, a, _, sp, _, lp, mk = traj[-1]
                traj[-1] = (s, a, reward, sp, True, lp, mk)
                for t in traj:
                    self.buffers[pname].append(t)
                del self.case_transitions[key]


    def _train_agents(self, rollout_length=25600, epochs=5, minibatch_size=256):
        if self.models_loaded:
            return
        for pname, agent in self.agents.items():
            if self.assignment_counters[pname] >= self.train_every and len(self.buffers[pname]) >= rollout_length:
                start = len(self.buffers[pname]) - rollout_length
                rollout = list(itertools.islice(self.buffers[pname], start, len(self.buffers[pname])))

                progress_remaining = max(0.0, 1.0 - (self.total_env_steps[pname] / self.target_env_steps))
                agent.set_lr(agent.initial_lr * progress_remaining)

                agent.train(rollout, epochs=epochs, minibatch_size=minibatch_size)
                self.total_env_steps[pname] += len(rollout)

                self.buffers[pname].clear()
                self.assignment_counters[pname] = 0


    def reset_after_episode(self):
        self.case_transitions.clear()
        self.open_tr.clear()
        # for pname in self.buffers:
        #     self.buffers[pname].clear()
        # (do not reset total_env_steps to keep LR schedule across episodes)



import os
import torch
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from framework.config import ARRIVAL_RATES, SCENARIO_NAMES, SIMULATION_RUN_TIME, process_function, SIMULATION_RUNS
from framework.simulator import Simulator

mp.set_start_method("spawn", force=True)

os.makedirs("results", exist_ok=True)
os.makedirs("runtime", exist_ok=True)

runtime_csv_path = "runtime/DRLMultiAgent_runtimes.csv"

if not os.path.exists(runtime_csv_path):
    with open(runtime_csv_path, "w") as f:
        f.write("method,l,run,time\n")


def run_simulation(args):
    import torch
    import time

    l, scenario_name, run_id = args

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"[Run {run_id}] Using device: {device}")

    # Initialize processes and MuProMAC
    processes = process_function(l, scenario_name)
    maac_process = MultiAgentPPO(processes, scenario_name)  

    start_time = time.time()
    simulator = Simulator(simulation_run=run_id, process=maac_process)
    simulator.run(SIMULATION_RUN_TIME)
    end_time = time.time()
    runtime_seconds = end_time - start_time

    df = simulator.event_log.get_dataframe()
    df["run"] = run_id

    return df, runtime_seconds, run_id

if __name__ == "__main__":
    for l in ARRIVAL_RATES:
        for scenario_name in SCENARIO_NAMES:
            print(f"\n=== Running scenario: {scenario_name}, arrival rate: {l} ===")

            args_list = [(l, scenario_name, i) for i in range(SIMULATION_RUNS)]

            with mp.Pool(processes=min(20, SIMULATION_RUNS)) as pool:#with mp.Pool(processes=min(mp.cpu_count(), SIMULATION_RUNS)) as pool:
                results = list(tqdm(pool.imap(run_simulation, args_list), total=len(args_list)))

            combined_logs = []
            for df, runtime_seconds, run_id in results:
                combined_logs.append(df)
                with open(runtime_csv_path, "a") as f:
                    f.write(f"DRLMultiAgent,{l},{run_id},{runtime_seconds:.2f}\n")
                print(f"[Run {run_id}] Completed in {runtime_seconds:.2f}s")

            combined_df = pd.concat(combined_logs, ignore_index=True)
            result_filename = f"results/DRLMultiAgent_l{l}_{scenario_name}.csv"
            combined_df.to_csv(result_filename, index=False)
            print(f"Saved results to {result_filename}")
