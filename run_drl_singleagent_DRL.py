import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from framework.processes import SimpleProcess, Task
import itertools


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
    def __init__(self, state_size, action_size, gamma=0.999, epsilon=0.2, lr=3e-5, entropy_coef=0.00, gae_lambda=0.95, total_decay_steps: int | None = None):
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

    def _masked_dist(self, probs, masks):
        
        masked = probs * masks
        denom = masked.sum(dim=1, keepdim=True)
        masked = masked / (denom + 1e-10)
        return masked

    @torch.no_grad()
    def _masked_dist_nograd(self, probs, masks):
        masked = probs * masks
        denom = masked.sum(dim=1, keepdim=True)
        masked = masked / (denom + 1e-10)
        return masked

    def select_action(self, state, mask):
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, S]
        probs = self.actor(state_tensor)                                      # [1, A]
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)    # [1, A]
        masked_probs = self._masked_dist_nograd(probs, mask_tensor).squeeze(0)
        if masked_probs.sum().item() == 0:
            return None, None
        dist = torch.distributions.Categorical(masked_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
            
    def train(self, rollout, epochs=5, minibatch_size=None):
        
        states, actions, rewards, next_states, dones, old_log_probs, masks = zip(*rollout)
        states      = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        masks_t     = torch.tensor(np.array(masks), dtype=torch.float32, device=self.device)  # [T, A]

        with torch.no_grad():
            values      = self.critic(states).squeeze(-1)             # V(s_t)
            next_values = self.critic(next_states).squeeze(-1)        # V(s_{t+1})

        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * (1.0 - dones[t]) * next_values[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values  # targets for critic

        # normalize advantages
        adv_mean, adv_std = advantages.mean(), advantages.std(unbiased=False)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        if minibatch_size is None or minibatch_size >= T:
            idx_iter = [torch.arange(T, device=self.device)]
        else:
            perm = torch.randperm(T, device=self.device)
            idx_iter = [perm[i:i+minibatch_size] for i in range(0, T, minibatch_size)]

        for _ in range(epochs):
            for idx in idx_iter:
                raw_probs = self.actor(states[idx])                     # [B, A]
                masked_probs = (raw_probs * masks_t[idx])
                masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-10)

                selected_probs = masked_probs.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                log_probs = torch.log(selected_probs + 1e-10)
                ratios = torch.exp(log_probs - old_log_probs[idx])

                adv = advantages[idx]
                clipped = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                policy_loss = -torch.min(ratios * adv, clipped * adv).mean()

                dist = torch.distributions.Categorical(masked_probs)
                entropy = dist.entropy().mean()
                actor_loss = policy_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                value_pred = self.critic(states[idx]).squeeze(-1)
                critic_loss = F.mse_loss(value_pred, returns[idx])

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


    


class SingleAgentPPO(SimpleProcess):
    
    POSTPONE_KEY = -1  # key for open POSTPONE transition

    def __init__(self, processes, scenario_name, resume_from_checkpoint=False, postpone_penalty: float = -0.1):
        super().__init__(processes)
        self.allocation_method_name = "DRL_SingleAgent"
        self.train_every = 500
        self.processes = processes
        self.scenario_name = scenario_name
        self.postpone_penalty = float(postpone_penalty) 

        self.total_env_steps = 0                     
        self.target_env_steps = 50_000_000 

        self.activities = list({
            t.name: t
            for p in processes
            for t in p.tasks.values()
            if isinstance(t, Task) and t.resources
        }.values())

        self.resources = list({
            r.name: r
            for p in processes
            for t in p.tasks.values()
            if isinstance(t, Task)
            for r in t.resources
        }.values())
        self.resource_by_name = {r.name: r for r in self.resources}

        self.state_size = 2 * len(self.resources) + len(self.activities)

        self.action_space = [
            (res.name, task.name)
            for task in self.activities
            for res in task.resources
        ] + ["POSTPONE"]
        print('------ACTION SPACE---------')
        print(self.action_space)
        print('--------------')

        self.agent = PPOAgent(self.state_size, len(self.action_space))

        self.buffer = deque(maxlen=25700)
        self.assignment_counter = 0
        self.models_loaded = False

        self.case_transitions = defaultdict(list)

        self.open_tr = {}

        if not resume_from_checkpoint:
            self.load_models_if_available()

    def load_models_if_available(self):
        lambda_val = self.processes[0].arrival_distribution
        model_dir = f"trained_models/DRL_SingleAgent/{self.scenario_name}_lambda{lambda_val}"

        actor_path = os.path.join(model_dir, "actor.pt")
        critic_path = os.path.join(model_dir, "critic.pt")

        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            print(f"[Model Load] No pretrained models found at {model_dir}")
            return

        try:
            self.agent.actor.load_state_dict(torch.load(actor_path, map_location=self.agent.device))
            self.agent.critic.load_state_dict(torch.load(critic_path, map_location=self.agent.device))
            self.models_loaded = True
            print(f"[Model Load] Loaded actor & critic for '{self.scenario_name}' λ={lambda_val}")
        except Exception as e:
            print(f"[Model Load] Error loading models: {e}")
            self.models_loaded = False

    def activity_index(self, task_name):
        for i, task in enumerate(self.activities):
            if task.name == task_name:
                return i
        return -1

    def _shadow_state(self, temp_available, temp_unassigned_ids, unassigned_tasks, shadow_busy):
       
        availability_bits = [1 if r in temp_available else 0 for r in self.resources]

        assignment_ids = []
        for r in self.resources:
            t = shadow_busy.get(r)
            if t:
                idx = self.activity_index(t.label)
                assignment_ids.append(idx / len(self.activities) if idx != -1 else 0.0)
            else:
                assignment_ids.append(0.0)

        queue_lengths = [0.0] * len(self.activities)
        name_to_idx = {act.name: i for i, act in enumerate(self.activities)}
        for tid in temp_unassigned_ids:
            lbl = unassigned_tasks[tid].label
            i = name_to_idx.get(lbl, None)
            if i is not None:
                queue_lengths[i] = min(queue_lengths[i] + 1/100.0, 1.0)

        return np.array(availability_bits + assignment_ids + queue_lengths, dtype=np.float32)

    def _close_all_open_with(self, next_state):
        if self.models_loaded:
            self.open_tr.clear()
            return
        for cid, tr in list(self.open_tr.items()):
            s, a, r, lp, mk = tr['s'], tr['a'], tr['r'], tr['logp'], tr['mask']
            if cid == self.POSTPONE_KEY:
                self.buffer.append((s, a, r, next_state, False, lp, mk))
            else:
                self.case_transitions[cid].append((s, a, r, next_state, False, lp, mk))
            del self.open_tr[cid]

    def assign_resources(self, unassigned_tasks, available_resources, resource_available_times):
        
        assignments = []

        temp_available = set(available_resources)
        temp_unassigned_ids = set(unassigned_tasks.keys())
        shadow_busy = dict(self.simulator.busy_resources)  

        POSTPONE_IDX = len(self.action_space) - 1

        while True:
            unassigned_by_label = {}
            for tid in temp_unassigned_ids:
                lbl = unassigned_tasks[tid].label
                unassigned_by_label[lbl] = unassigned_by_label.get(lbl, 0) + 1

            mask = []
            for res_name, task_name in self.action_space[:-1]:
                res = self.resource_by_name[res_name]
                feasible = (res in temp_available) and (unassigned_by_label.get(task_name, 0) > 0)
                mask.append(1 if feasible else 0)
            mask.append(1)  # POSTPONE

            if sum(mask[:-1]) == 0:
                break

            state = self._shadow_state(temp_available, temp_unassigned_ids, unassigned_tasks, shadow_busy)

            self._close_all_open_with(state)

            action, log_prob = self.agent.select_action(state, mask)
            if action is None:
                break

            if action == POSTPONE_IDX:
                if not self.models_loaded:
                    self.open_tr[self.POSTPONE_KEY] = {
                        's': state, 'a': action, 'r': self.postpone_penalty, 'logp': log_prob, 'mask': mask
                    }
                    self.assignment_counter += 1
                break 

            res_name, task_name = self.action_space[action]
            res_obj = self.resource_by_name[res_name]
            if res_obj not in temp_available or unassigned_by_label.get(task_name, 0) <= 0:
                
                break

            selected_task = None
            for tid in temp_unassigned_ids:
                if unassigned_tasks[tid].label == task_name:
                    selected_task = unassigned_tasks[tid]
                    selected_tid = tid
                    break
            if selected_task is None:
                break

            if not self.models_loaded:
                cid = selected_task.case_id
                self.open_tr[cid] = {
                    's': state, 'a': action, 'r': 0.0, 'logp': log_prob, 'mask': mask
                }
                self.assignment_counter += 1

            assignments.append((selected_task, res_obj))

            temp_available.remove(res_obj)
            temp_unassigned_ids.remove(selected_tid)
            shadow_busy[res_obj] = selected_task  

        self.train_agent()
        return assignments

    def on_element_completed(self, process_element):
        return

    def handle_case_completion_reward(self, process_element):
        
        case_id = process_element.case_id
        start_time = self.simulator.case_start_times[case_id]
        cycle_time = self.simulator.now - start_time
        reward = 1.0 / (cycle_time + 1)

        if self.models_loaded:
            return

        if case_id in self.open_tr:
            tr = self.open_tr.pop(case_id)
            s, a, r0, lp, mk = tr['s'], tr['a'], tr['r'], tr['logp'], tr['mask']
            
            temp_available = set(self.simulator.available_resources)
            temp_unassigned_ids = set(self.simulator.unassigned_tasks.keys())
            shadow_busy_now = dict(self.simulator.busy_resources)
            next_state = self._shadow_state(
                temp_available, temp_unassigned_ids, self.simulator.unassigned_tasks, shadow_busy_now
            )
            self.case_transitions[case_id].append((s, a, reward, next_state, True, lp, mk))
        else:
            transitions = self.case_transitions.get(case_id, [])
            if transitions:
                s, a, _, sp, _, lp, mk = transitions[-1]
                transitions[-1] = (s, a, reward, sp, True, lp, mk)
            else:
                pass

        if case_id in self.case_transitions and len(self.case_transitions[case_id]) > 0:
            for t in self.case_transitions[case_id]:
                self.buffer.append(t)
            del self.case_transitions[case_id]

    def train_agent(self, rollout_length=25600, epochs=5, minibatch_size=256):
        if self.models_loaded:
            return
        if self.assignment_counter >= self.train_every and len(self.buffer) >= rollout_length:
            
            start = len(self.buffer) - rollout_length
            rollout = list(itertools.islice(self.buffer, start, len(self.buffer)))
            
            progress_remaining = max(0.0, 1.0 - (self.total_env_steps / self.target_env_steps))
            new_lr = self.agent.initial_lr * progress_remaining  
            self.agent.set_lr(new_lr)
            
            self.agent.train(rollout, epochs=epochs, minibatch_size=minibatch_size)

            self.total_env_steps += len(rollout)
            
            self.buffer.clear()
            self.assignment_counter = 0


    def reset_after_episode(self):
        self.case_transitions.clear()
        self.open_tr.clear()
        #self.buffer.clear()




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

runtime_csv_path = "runtime/SingleActor_runtimes.csv"

if not os.path.exists(runtime_csv_path):
    with open(runtime_csv_path, "w") as f:
        f.write("method,l,run,time\n")

def run_single_actor_simulation(args):
    import torch
    import time

    l, scenario_name, run_id = args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Run {run_id}] Using device: {device}")

    processes = process_function(l, scenario_name)
    agent = SingleAgentPPO(processes, scenario_name)

    start_time = time.time()
    simulator = Simulator(simulation_run=run_id, process=agent)
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

            with mp.Pool(processes=min(10, SIMULATION_RUNS)) as pool:
                results = list(tqdm(pool.imap(run_single_actor_simulation, args_list), total=len(args_list)))

            combined_logs = []
            for df, runtime_seconds, run_id in results:
                combined_logs.append(df)
                with open(runtime_csv_path, "a") as f:
                    f.write(f"SingleActor,{l},{run_id},{runtime_seconds:.2f}\n")
                print(f"[Run {run_id}] Completed in {runtime_seconds:.2f}s")

            combined_df = pd.concat(combined_logs, ignore_index=True)
            result_filename = f"results/SingleActor_l{l}_{scenario_name}.csv"
            combined_df.to_csv(result_filename, index=False)
            print(f"Saved results to {result_filename}")
