
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from cachetools import LRUCache

from framework.processes import SimpleProcess, ProcessElement, Task


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.out(x), dim=-1)
        return x


class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)



class MAACAgent:
    def __init__(
        self, processes, state_sizes, critic_state_size, action_sizes,
        gamma=0.99, epsilon=0.2, lr=3e-5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.processes = processes
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.device = device

        self.actors = {
            p: ActorNetwork(state_sizes[p], action_sizes[p]).to(self.device)
            for p in processes
        }
        self.actor_optimizers = {
            p: optim.Adam(self.actors[p].parameters(), lr=self.lr)
            for p in processes
        }
        self.critic = CriticNetwork(critic_state_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # per-process replay buffers of trajectory dicts
        self.buffers = {p: deque(maxlen=10000) for p in processes}

        # each entry is a dict with keys: 'RO','s','global_s','a','logp','task_id', optionally 's_prime','global_s_prime','r'
        self.case_buffers = defaultdict(lambda: defaultdict(list))

    def store_selection(self, process_name, task_id, case_id, request_object,
                        actor_state, critic_state, action, log_prob, is_postpone=False):
        trajectory = {
            'RO': request_object,
            's': actor_state,
            'global_s': critic_state,
            'a': action,
            'logp': log_prob,
            'task_id': task_id,
            'is_postpone': is_postpone
        }
        self.case_buffers[process_name][case_id].append(trajectory)

    def finalize_case(self, process_name, case_id, reward,
                      postpone_counts=None,
                      terminal_actor_state=None,
                      terminal_global_state=None):
        
        if case_id not in self.case_buffers[process_name]:
            return

        case_trajectories = self.case_buffers[process_name][case_id]

        for traj in case_trajectories:
            if 's_prime' not in traj:
                traj['s_prime'] = terminal_actor_state if terminal_actor_state is not None else traj['s']
            if 'global_s_prime' not in traj:
                traj['global_s_prime'] = terminal_global_state if terminal_global_state is not None else traj['global_s']

        for traj in case_trajectories:
            adjusted_reward = reward
            if traj.get('logp', 0.0) == 0.0:
                adjusted_reward -= 2  # fallback penalty
            if traj.get('is_postpone', False):
                adjusted_reward -= 0.5
            traj['r'] = adjusted_reward
            self.buffers[process_name].append(traj)

        del self.case_buffers[process_name][case_id]

    def all_buffers_ready(self, batch_size: int) -> bool:
        return all(len(self.buffers[p]) >= batch_size for p in self.processes)

    def train_single_actor(self, pname, batch, actor_net, actor_opt, critic, gamma, epsilon, device, epochs):
        local_states = torch.tensor([b['s'] for b in batch], dtype=torch.float32, device=device)
        actions = torch.tensor([b['a'] for b in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([b['r'] for b in batch], dtype=torch.float32, device=device)
        global_states = torch.tensor([b['global_s'] for b in batch], dtype=torch.float32, device=device)
        global_nexts = torch.tensor([b['global_s_prime'] for b in batch], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor([b['logp'] for b in batch], dtype=torch.float32, device=device)

        with torch.no_grad():
            target_values = rewards + gamma * critic(global_nexts).squeeze()

        for _ in range(epochs):
            probs = actor_net(local_states)
            selected_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            log_probs = torch.log(selected_probs + 1e-10)

            values = critic(global_states).squeeze().detach()
            advantages = target_values - values

            ratios = torch.exp(log_probs - old_log_probs)
            clipped = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
            actor_loss = -torch.min(ratios * advantages, clipped * advantages).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

    def train(self, batch_size=256, epochs=3):
        all_critic_batches = []
        actor_jobs = []

        with ThreadPoolExecutor(max_workers=len(self.processes)) as executor:
            for pname in self.processes:
                buffer = self.buffers[pname]
                if len(buffer) < batch_size:
                    continue

                batch = random.sample(buffer, batch_size)

                all_critic_batches.extend(list(zip(
                    [torch.tensor(b['global_s'], dtype=torch.float32, device=self.device) for b in batch],
                    [b['r'] for b in batch],
                    [torch.tensor(b['global_s_prime'], dtype=torch.float32, device=self.device) for b in batch]
                )))

                actor_jobs.append(executor.submit(
                    self.train_single_actor,
                    pname, batch,
                    self.actors[pname],
                    self.actor_optimizers[pname],
                    self.critic,
                    self.gamma,
                    self.epsilon,
                    self.device,
                    epochs
                ))

            for job in actor_jobs:
                job.result()

        if not all_critic_batches:
            return

        critic_states, rewards, critic_nexts = zip(*all_critic_batches)
        critic_states = torch.stack(critic_states)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        critic_nexts = torch.stack(critic_nexts)

        with torch.no_grad():
            targets = rewards + self.gamma * self.critic(critic_nexts).squeeze()

        for _ in range(epochs):
            values = self.critic(critic_states).squeeze()
            critic_loss = F.mse_loss(values, targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


class GRUAutoencoder(nn.Module):
    def __init__(self, embedding_size):
        super(GRUAutoencoder, self).__init__()
        self.embedding_size = embedding_size
        self.encoder_gru = nn.GRU(input_size=3, hidden_size=embedding_size, batch_first=True)
        self.decoder_gru = nn.GRU(input_size=3, hidden_size=embedding_size, batch_first=True)
        self.output_layer = nn.Linear(embedding_size, 3)

    def forward(self, x):
        _, hidden = self.encoder_gru(x)
        decoded_output, _ = self.decoder_gru(x, hidden)
        reconstructed = self.output_layer(decoded_output)
        return reconstructed, hidden.squeeze(0)

    def generate_embedding(self, sequence_list, device='cuda' if torch.cuda.is_available() else 'cpu'):
        input_tensor = torch.tensor([sequence_list], dtype=torch.float32, device=device)
        self.eval()
        with torch.no_grad():
            _, embedding = self.forward(input_tensor)
        return embedding.cpu().numpy().flatten()

class MuProMAC(SimpleProcess):
    def __init__(self, processes, scenario_name, resume_from_checkpoint=False):
        super().__init__(processes)
        self.allocation_method_name = "MuProMAC"
        num_processes = len(processes)
        self.embedding_size = 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processes = processes
        self.scenario_name = scenario_name
        self.all_unique_resources = self.get_all_unique_resources()

        state_sizes = {p.name: self.get_actor_state_size(p.name) for p in processes}
        action_sizes = {p.name: len(self.get_unique_resources(p.name)) + 1 for p in processes}

        self.arrival_lambda = processes[0].arrival_distribution
        self.model_dir = f"trained_models/MuProMAC/{self.scenario_name}_lambda{self.arrival_lambda}"

        critic_state_size = self.get_critic_state_size()
        self.task_ids = self.assign_task_ids()
        self.actions = {p.name: sorted(self.get_unique_resources(p.name)) + ["POSTPONE"] for p in processes}
        self.shared_resources = {p.name: self.get_shared_resources(p.name) for p in processes}

        self.task_to_resources = self.task_to_resources_mapping()
        self.resource_to_tasks = self.resource_to_tasks_mapping()
        self.all_tasks_with_resources = self.get_tasks_by_resource()

        self.embedding_cache = LRUCache(maxsize=50)

        self.last_task_completion_time = dict()

        self.agent = MAACAgent(
            processes=[p.name for p in processes],
            state_sizes=state_sizes,
            critic_state_size=critic_state_size,
            action_sizes=action_sizes,
            device=self.device
        )

        self.task_first_ready_time = dict()
        self.task_postpone_count = defaultdict(int)
        self.max_postpone = 3

        self.total_decision_steps = 0
        self.train_every_n_steps = 500

        self.gru_batch_size = 256
        self.sequence_buffers = {p.name: deque(maxlen=self.gru_batch_size * 10) for p in processes}
        self.gru_train_counts = {p.name: 0 for p in processes}
        self.validation_buffers = {p.name: deque(maxlen=1000) for p in processes}
        self.validation_loss_history = {p.name: deque(maxlen=50) for p in processes}
        self.stop_gru_training = {p.name: False for p in processes}
        self.gru_train_interval = 200
        self.decision_step_counts_for_gru_training = {p.name: 0 for p in self.processes}

        self.gru_models = {p.name: GRUAutoencoder(self.embedding_size).to(self.device) for p in processes}
        self.models_loaded = False

        if not resume_from_checkpoint:
            self.load_models_if_available()

    def reset_embedding_cache(self):
        self.embedding_cache = {}

    def load_models_if_available(self):
        lambda_val = self.processes[0].arrival_distribution
        model_dir = f"trained_models/MuProMAC/{self.scenario_name}_lambda{lambda_val}"
        self.models_loaded = False

        if not os.path.exists(model_dir):
            print(f"[Model Load] Directory not found: {model_dir}")
            return

        try:
            critic_path = os.path.join(model_dir, "critic.pt")
            if not os.path.exists(critic_path):
                print(f"[Model Load] Missing: {critic_path}")
                return
            self.agent.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

            for pname in self.agent.actors:
                actor_path = os.path.join(model_dir, f"actor_{pname}.pt")
                if not os.path.exists(actor_path):
                    print(f"[Model Load] Missing: {actor_path}")
                    return
                self.agent.actors[pname].load_state_dict(torch.load(actor_path, map_location=self.device))

            for pname in self.gru_models:
                gru_path = os.path.join(model_dir, f"gru_{pname}.pt")
                if not os.path.exists(gru_path):
                    print(f"[Model Load] Missing: {gru_path}")
                    return
                self.gru_models[pname].load_state_dict(torch.load(gru_path, map_location=self.device))

            self.models_loaded = True
            self.stop_gru_training = {p.name: True for p in self.processes}
            print(f"[Model Load] Successfully loaded all models for '{self.scenario_name}' λ={lambda_val}")

        except Exception as e:
            print(f"[Model Load] Error loading models: {e}")
            self.models_loaded = False

    def train_gru_model(self, model, input_tensor, epochs=5):
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = model(input_tensor)
            loss = loss_fn(reconstructed, input_tensor)
            loss.backward()
            optimizer.step()

    def update_open_tasks(self, open_tasks, selected_task_label):
        related_tasks = set()
        if selected_task_label in self.task_to_resources:
            for resource in self.task_to_resources[selected_task_label]:
                related_tasks.update(self.resource_to_tasks[resource])
        updated_open_tasks = {
            key: value for key, value in open_tasks.items()
            if value.label in related_tasks
        }
        return updated_open_tasks

    def generate_embedding(self, sequence_list, process_name, is_actor, train=True):
        max_seq_len = 20
        sequence = sequence_list[:max_seq_len]

        if len(sequence) < max_seq_len:
            padding = [[0, 0, 0]] * (max_seq_len - len(sequence))
            sequence += padding

        key = (process_name, is_actor, tuple(map(tuple, sequence)))
        if key in self.embedding_cache:
            return self.embedding_cache[key]

        if train and is_actor and not self.models_loaded and not self.stop_gru_training[process_name]:
            self.decision_step_counts_for_gru_training[process_name] += 1

            if random.random() < 0.05:
                self.validation_buffers[process_name].append(sequence)
            else:
                self.sequence_buffers[process_name].append(sequence)

            if (
                self.decision_step_counts_for_gru_training[process_name] % self.gru_train_interval == 0
                and len(self.sequence_buffers[process_name]) >= self.gru_batch_size
            ):
                sampled_batch = random.sample(self.sequence_buffers[process_name], self.gru_batch_size)
                batch_tensor = torch.tensor(sampled_batch, dtype=torch.float32, device=self.device)
                self.train_gru_model(self.gru_models[process_name], batch_tensor)
                self.gru_train_counts[process_name] += 1
                self.decision_step_counts_for_gru_training[process_name] = 0

                if len(self.validation_buffers[process_name]) >= 100:
                    sampled_val = random.sample(self.validation_buffers[process_name], k=100)
                    val_tensor = torch.tensor(sampled_val, dtype=torch.float32, device=self.device)
                    val_loss = self.evaluate_gru_loss(self.gru_models[process_name], val_tensor)
                    self.validation_loss_history[process_name].append(val_loss)

                    if len(self.validation_loss_history[process_name]) >= 5:
                        recent = list(self.validation_loss_history[process_name])[-5:]
                        if all(recent[i] > recent[i - 1] for i in range(1, 5)):
                            self.stop_gru_training[process_name] = True
                            print(f"[GRU Training Stopped for {process_name}]")

        embedding = self.gru_models[process_name].generate_embedding(sequence, device=self.device)
        embedding = embedding.tolist()
        self.embedding_cache[key] = embedding
        return embedding

    def evaluate_gru_loss(self, model, val_tensor):
        model.eval()
        with torch.no_grad():
            reconstructed, _ = model(val_tensor)
            loss = F.mse_loss(reconstructed, val_tensor)
        return loss.item()


    def get_actor_state_size(self, process_name):
        current_item_size = 3
        available_resources = len(self.get_unique_resources(process_name)) + 1  
        queue_embedding_size = self.embedding_size * len(self.tasks_with_resources(process_name))
        return available_resources + queue_embedding_size + current_item_size

    @lru_cache(maxsize=None)
    def get_unique_resources(self, process_name):
        unique_resources = set()
        for process in self.processes:
            if process.name == process_name:
                for task in process.tasks.values():
                    if not isinstance(task, Task):
                        continue
                    for resource in task.resources:
                        unique_resources.add(resource.name)
        return sorted(unique_resources)

    def get_shared_resources(self, process_name):
        process_resources = self.get_unique_resources(process_name)
        shared_resources = set()
        for process in self.processes:
            if process.name != process_name:
                for task in process.tasks.values():
                    if not isinstance(task, Task):
                        continue
                    for resource in task.resources:
                        if resource.name in process_resources:
                            shared_resources.add(resource.name)
        return list(shared_resources)

    def tasks_with_resources(self, process_name=None):
        tasks_with_resources = []
        for process in self.processes:
            if process_name and process.name != process_name:
                continue
            for task in process.tasks.values():
                if not isinstance(task, Task):
                    continue
                if task.resources:
                    tasks_with_resources.append(task.name)
        return list(set(tasks_with_resources))

    def get_critic_state_size(self):
        available_resources = len(self.all_unique_resources)
        queue_embedding_size = self.embedding_size * self.count_all_tasks_with_resources()
        return available_resources + queue_embedding_size

    def get_all_unique_resources(self):
        all_resources = set()
        for process in self.processes:
            for task in process.tasks.values():
                if not isinstance(task, Task):
                    continue
                for resource in task.resources:
                    all_resources.add(resource.name)
        return list(all_resources)

    def count_all_tasks_with_resources(self):
        return sum(
            1 for process in self.processes
            for task in process.tasks.values()
            if isinstance(task, Task) and task.resources
        )

    def task_to_resources_mapping(self):
        mapping = {}
        for process in self.processes:
            for task_name, task in process.tasks.items():
                if not isinstance(task, Task):
                    continue
                mapping[task_name] = {res.name for res in task.resources}
        return mapping

    def resource_to_tasks_mapping(self):
        mapping = defaultdict(set)
        for process in self.processes:
            for task_name, task in process.tasks.items():
                if not isinstance(task, Task):
                    continue
                for res in task.resources:
                    mapping[res.name].add(task_name)
        return mapping

    def assign_task_ids(self):
        task_id_mapping = {}
        task_id = 1
        for process in self.processes:
            for task_label in process.tasks.keys():
                if task_label not in task_id_mapping:
                    task_id_mapping[task_label] = task_id
                    task_id += 1
        return task_id_mapping


    def get_item_information(self, task):
        task_label = task.label
        current_task_id = self.task_ids[task_label]
        next_task = self.get_next_task(task_label)
        next_task_id = self.task_ids[next_task[0]] if next_task else 999
        time_since_arrival = self.simulator.now - self.simulator.case_start_times[task.case_id]
        return [current_task_id, next_task_id, time_since_arrival]

    def get_available_resources(self, available_resources, process_name=None):
        all_available_names = [r.name for r in available_resources]
        resource_pool = self.actions[process_name] if process_name else self.all_unique_resources
        return [1 if res in all_available_names else 0 for res in resource_pool]

    def get_tasks_by_resource(self, process_name=None):
        task_map = defaultdict(set)
        for process in self.processes:
            if process_name and process.name != process_name:
                continue
            for task_name, task in process.tasks.items():
                if not isinstance(task, Task):
                    continue
                for res in task.resources:
                    task_map[res.name].add(task_name)
        return task_map

    def get_next_task(self, task_label):
        for process in self.processes:
            if task_label in process.tasks:
                return process.tasks[task_label].next_tasks
        return []

    def get_actor_state(self, task, available_resources, unassigned_tasks, process_name):
        item_info = self.get_item_information(task)
        resources = self.get_available_resources(available_resources, process_name)
        embeddings = []
        for activity in self.tasks_with_resources(process_name):
            task_queue = self.update_open_tasks(unassigned_tasks, activity)
            sequence = [self.get_item_information(t) for t in task_queue.values()] if task_queue else []
            embedding = self.generate_embedding(sequence, process_name, is_actor=True) if sequence else [0] * self.embedding_size
            embeddings.extend(embedding)
        return item_info + resources + embeddings

    def get_critic_state(self, task, available_resources, unassigned_tasks, process_name):
        resources = self.get_available_resources(available_resources)
        embeddings = []
        for activity in self.tasks_with_resources():
            task_queue = self.update_open_tasks(unassigned_tasks, activity)
            sequence = [self.get_item_information(t) for t in task_queue.values()] if task_queue else []
            embedding = self.generate_embedding(sequence, process_name, is_actor=False) if sequence else [0] * self.embedding_size
            embeddings.extend(embedding)
        return resources + embeddings


    def on_element_completed(self, process_element: ProcessElement):
        
        return

    def handle_case_completion_reward(self, process_element: ProcessElement):
        case_id = process_element.case_id
        process_name = process_element.case_type

        start_time = self.simulator.case_start_times.get(case_id, self.simulator.now)
        end_time = self.simulator.now
        case_duration = end_time - start_time
        reward = -case_duration

        if self.models_loaded:
            return

        terminal_actor = self.get_actor_state(process_element, self.simulator.available_resources, self.simulator.unassigned_tasks, process_name)
        terminal_critic = self.get_critic_state(process_element, self.simulator.available_resources, self.simulator.unassigned_tasks, process_name)

        postpone_counts = {
            t.id: self.task_postpone_count.get(t.id, 0)
            for t in self.simulator.busy_cases.get(case_id, [])
        }

        self.agent.finalize_case(
            process_name=process_name,
            case_id=case_id,
            reward=reward,
            postpone_counts=postpone_counts,
            terminal_actor_state=terminal_actor,
            terminal_global_state=terminal_critic
        )

        for task in self.simulator.busy_cases.get(case_id, []):
            self.task_first_ready_time.pop(task.id, None)
            self.task_postpone_count.pop(task.id, None)

    def assign_resources(self, unassigned_tasks, available_resources, resource_available_times):
        self.reset_embedding_cache()

        temp_available = available_resources.copy()
        temp_unassigned_task = unassigned_tasks.copy()

        if not unassigned_tasks or not available_resources:
            return []

        assignments = []
        for task in list(unassigned_tasks.values()):
            if len(temp_available) == 0:
                break

            process_name = task.case_type
            valid_resources = self.task_to_resources[task.label]
            resource_map = {r.name: r for r in temp_available}
            mask = np.array([
                1 if res == "POSTPONE" or (res in valid_resources and res in resource_map) else 0
                for res in self.actions[process_name]
            ])
            if not np.any(mask):
                continue

            item_info = self.get_item_information(task)
            actor_state = self.get_actor_state(task, temp_available, temp_unassigned_task, process_name)
            critic_state = self.get_critic_state(task, temp_available, temp_unassigned_task, process_name)

            input_tensor = torch.tensor([actor_state], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                probs = self.agent.actors[process_name](input_tensor).squeeze(0).cpu().numpy()

            probs = probs * mask
            if np.sum(probs) == 0:
                continue
            probs /= np.sum(probs)
            if self.models_loaded:
                action_index = np.argmax(probs)
            else:
                action_index = np.random.choice(len(probs), p=probs)
            selected_name = self.actions[process_name][action_index]

            if task.id not in self.task_first_ready_time:
                self.task_first_ready_time[task.id] = self.simulator.now

            if selected_name == "POSTPONE":
                self.task_postpone_count[task.id] += 1

                if self.task_postpone_count[task.id] >= self.max_postpone:
                    fallback_options = [res for res in valid_resources if res in resource_map]
                    resource_indices = [
                        idx for idx, res_name in enumerate(self.actions[process_name])
                        if res_name in fallback_options
                    ]
                    if resource_indices:
                        best_idx = max(resource_indices, key=lambda i: probs[i])
                        best_resource_name = self.actions[process_name][best_idx]
                        fallback_resource = resource_map[best_resource_name]

                        assignments.append((task, fallback_resource))
                        temp_available.remove(fallback_resource)
                        del temp_unassigned_task[task.id]

                        if not self.models_loaded:
                            prev_transitions = self.agent.case_buffers[process_name][task.case_id]
                            for traj in reversed(prev_transitions):
                                if traj['task_id'] == task.id and 's_prime' not in traj:
                                    traj['s_prime'] = actor_state
                                    traj['global_s_prime'] = critic_state
                                    break

                            self.agent.store_selection(
                                process_name=task.case_type,
                                task_id=task.id,
                                case_id=task.case_id,
                                request_object=item_info,
                                actor_state=actor_state,
                                critic_state=critic_state,
                                action=best_idx,
                                log_prob=0.0
                            )
                            self.total_decision_steps += 1

                        continue  
                    else:
                        continue  

                if not self.models_loaded:
                    prev_transitions = self.agent.case_buffers[process_name][task.case_id]
                    for traj in reversed(prev_transitions):
                        if traj['task_id'] == task.id and 's_prime' not in traj:
                            traj['s_prime'] = actor_state
                            traj['global_s_prime'] = critic_state
                            break

                    self.agent.store_selection(
                        process_name=task.case_type,
                        task_id=task.id,
                        case_id=task.case_id,
                        request_object=item_info,
                        actor_state=actor_state,
                        critic_state=critic_state,
                        action=action_index,
                        log_prob=float(np.log(probs[action_index] + 1e-10)),
                        is_postpone=True
                    )
                    self.total_decision_steps += 1

                continue 

            if selected_name in resource_map:
                selected_res = resource_map[selected_name]
                assignments.append((task, selected_res))
                temp_available.remove(selected_res)
                del temp_unassigned_task[task.id]

                if not self.models_loaded:
                    prev_transitions = self.agent.case_buffers[process_name][task.case_id]
                    for traj in reversed(prev_transitions):
                        if traj['task_id'] == task.id and 's_prime' not in traj:
                            traj['s_prime'] = actor_state
                            traj['global_s_prime'] = critic_state
                            break

                    self.agent.store_selection(
                        process_name=task.case_type,
                        task_id=task.id,
                        case_id=task.case_id,
                        request_object=item_info,
                        actor_state=actor_state,
                        critic_state=critic_state,
                        action=action_index,
                        log_prob=float(np.log(probs[action_index] + 1e-10))
                    )
                    self.total_decision_steps += 1

        if (
            not self.models_loaded
            and self.total_decision_steps > self.train_every_n_steps
            and self.agent.all_buffers_ready(256)
        ):
            self.agent.train()
            self.total_decision_steps = 0

        return assignments


    def reset_after_episode(self):
        self.agent.case_buffers = defaultdict(lambda: defaultdict(list))

        self.task_postpone_count.clear()
        self.task_first_ready_time.clear()

        self.sequence_buffers = {p.name: deque(maxlen=self.gru_batch_size * 10) for p in self.processes}
        self.validation_buffers = {p.name: deque(maxlen=1000) for p in self.processes}
        self.validation_loss_history = {p.name: deque(maxlen=50) for p in self.processes}
        self.decision_step_counts_for_gru_training = {p.name: 0 for p in self.processes}

        self.reset_embedding_cache()


import os
import time
import torch
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from framework.config import ARRIVAL_RATES, SCENARIO_NAMES, SIMULATION_RUN_TIME, process_function, SIMULATION_RUNS
from framework.simulator import Simulator

mp.set_start_method("spawn", force=True)

os.makedirs("results", exist_ok=True)
os.makedirs("runtime", exist_ok=True)

runtime_csv_path = "runtime/MuProMAC.csv"

if not os.path.exists(runtime_csv_path):
    with open(runtime_csv_path, "w") as f:
        f.write("method,l,run,time\n")


def run_simulation(args):
    import torch
    import time
    import gc

    l, scenario_name, run_id = args

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"[Run {run_id}] Using device: {device}")

    processes = process_function(l, scenario_name)
    maac_process = MuProMAC(processes, scenario_name) 

    start_time = time.time()
    simulator = Simulator(simulation_run=run_id, process=maac_process, scenario_name=scenario_name, log_only_case_completion=True)
    simulator.run(SIMULATION_RUN_TIME)
    end_time = time.time()
    runtime_seconds = end_time - start_time

    df = simulator.event_log.get_dataframe()
    df["run"] = run_id

    del simulator
    del maac_process
    del processes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df, runtime_seconds, run_id

if __name__ == "__main__":
    for l in ARRIVAL_RATES:
        for scenario_name in SCENARIO_NAMES:
            print(f"\n=== Running scenario: {scenario_name}, arrival rate: {l} ===")

            args_list = [(l, scenario_name, i) for i in range(SIMULATION_RUNS)]

            with mp.Pool(processes=min(22, SIMULATION_RUNS)) as pool:
                results = list(tqdm(pool.imap(run_simulation, args_list), total=len(args_list)))

            combined_logs = []
            for df, runtime_seconds, run_id in results:
                combined_logs.append(df)
                with open(runtime_csv_path, "a") as f:
                    f.write(f"MuProMAC,{l},{run_id},{runtime_seconds:.2f}\n")
                print(f"[Run {run_id}] Completed in {runtime_seconds:.2f}s")

            combined_df = pd.concat(combined_logs, ignore_index=True)
            result_filename = f"results/MuProMAC_l{l}_{scenario_name}.csv"
            combined_df.to_csv(result_filename, index=False)
            print(f"Saved results to {result_filename}")


