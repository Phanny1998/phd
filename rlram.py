from framework.processes import SimpleProcess, ProcessElement
from framework.simulator import Simulator
import random

import numpy as np

class RLRAM(SimpleProcess):
    def __init__(self, processes):
        super().__init__(processes)
        self.allocation_method_name = "RLRAM"
        self.agents = {process.name: {
            'q_table': {},  # Q-table specific to this process
            'alpha': 0.1,  # Learning rate
            'gamma': 0.9,  # Discount factor
            'epsilon': 0.1  # Exploration rate
        } for process in processes}  # separate agent for each process

    # def initialize_q_table(self, q_table, states, actions):
    #     for state in states:
    #         if state not in q_table:
    #             q_table[state] = {}
    #             for action in actions:
    #                 q_table[state][action] = 0.0
    def initialize_q_table(self, q_table, states, actions):
        for state in states:
            if state not in q_table:
                q_table[state] = {}
            for action in actions:
                if action not in q_table[state]:
                    q_table[state][action] = 0.0


    def select_action(self, q_table, state, available_resources):
        if np.random.random() < self.agents[state[0]]['epsilon']:  # explore
            return random.choice(available_resources) if available_resources else None
        else:  # exploit
            if state in q_table and q_table[state]:
                return max(q_table[state], key=q_table[state].get)
            else:
                return random.choice(available_resources) if available_resources else None

    # def update_q_table(self, q_table, state, action, reward, next_state, available_resources):
    #     max_next_q = max(q_table[next_state][r] for r in available_resources if next_state in q_table and r in q_table[next_state]) if next_state in q_table else 0
    #     q_table[state][action] = q_table[state][action] + self.agents[state[0]]['alpha'] * (
    #         reward + self.agents[state[0]]['gamma'] * max_next_q - q_table[state][action]
    #     )
    def update_q_table(self, q_table, state, action, reward, next_state, available_resources):
       
        if state not in q_table:
            q_table[state] = {}
        
        # initialize the action for the state if not present
        if action not in q_table[state]:
            q_table[state][action] = 0.0
        
        # compute max Q-value for the next state
        max_next_q = max(
            q_table[next_state][r] for r in available_resources
            if next_state in q_table and r in q_table[next_state]
        ) if next_state in q_table else 0
        
        # update the Q-value for the current state-action pair
        q_table[state][action] += self.agents[state[0]]['alpha'] * (
            reward + self.agents[state[0]]['gamma'] * max_next_q - q_table[state][action]
        )


    def assign_resources(self, unassigned_tasks, available_resources, resource_available_times):
        assignments = []
        resources_queue = list(available_resources)  # maintain a queue of available resources
        tasks_queue = list(unassigned_tasks.values())
        tasks_queue.sort(key=lambda task: task.id)  # assuming task.id represents arrival order

        for task in tasks_queue:
            process_name = task.case_type
            agent = self.agents[process_name]
            state = self.get_task_state(task)

            # filter resources based on task constraints
            task_definition = self.process_definitions[process_name].tasks[task.label]
            compatible_resources = {res.name for res in task_definition.resources}
            valid_resources = [r for r in resources_queue if r.name in compatible_resources]

            self.initialize_q_table(agent['q_table'], [state], valid_resources)
            selected_resource = self.select_action(agent['q_table'], state, valid_resources)

            if selected_resource and selected_resource in resources_queue:
                assignments.append((task, selected_resource))
                resources_queue.remove(selected_resource)

        return assignments

    def get_task_state(self, task):
        workload = len(self.simulator.unassigned_tasks)  
        return (task.case_type, task.label, workload)

    # def update_q_values_on_completion(self, task, resource, completion_time):
    #     state = self.get_task_state(task)
    #     process_name = task.case_type
    #     agent = self.agents[process_name]
    #     reward = -(completion_time - self.simulator.now)  # Negative flow time as cost
    #     next_state = "terminal"  # No next state after task completion

    #     self.update_q_table(agent['q_table'], state, resource.name, reward, next_state, self.resources)

    def on_element_completed(self, process_element: ProcessElement):
        
        if process_element.is_task():
            #print(process_element.label)
            #print(self.simulator.busy_resources)
            # find the resource assigned to this task
            resource = None
            for r, task in self.simulator.busy_resources.items():
                if task.id == process_element.id:
                    resource = r
                    break
            
            if resource:
                #print(resource)
                state = self.get_task_state(process_element)

                reward = -(self.simulator.now - self.simulator.case_start_times[process_element.case_id])

                next_state = "terminal"
                #print(reward)
                #print('------------------------------------------------')
                # Update Q-table
                agent = self.agents[process_element.case_type]
                
                self.update_q_table(
                    q_table=agent['q_table'],
                    state=state,
                    action=resource.name,
                    reward=reward,
                    next_state=next_state,
                    available_resources=self.simulator.available_resources
                )
            else:
                print(f"[Warning] No resource found for task ID {process_element.id}. Q-value not updated.")




import os
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from framework.config import (
    ARRIVAL_RATES, SCENARIO_NAMES, SIMULATION_RUN_TIME,
    process_function, SIMULATION_RUNS
)
from framework.simulator import Simulator


def run_rlram_simulation(args):
    import time as _time

    l, scenario_name, run_id = args
    processes = process_function(l, scenario_name)
    allocator = RLRAM(processes)   

    start = _time.time()
    simulator = Simulator(simulation_run=run_id, process=allocator)
    simulator.run(SIMULATION_RUN_TIME)
    end = _time.time()
    runtime_seconds = end - start

    df = simulator.event_log.get_dataframe()
    df["run"] = run_id
    return df, runtime_seconds, run_id


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    os.makedirs("results", exist_ok=True)
    os.makedirs("runtime", exist_ok=True)

    runtime_csv_path = "runtime/RLRAM_runtimes.csv"
    if not os.path.exists(runtime_csv_path):
        with open(runtime_csv_path, "w") as f:
            f.write("method,l,run,time\n")

    for l in ARRIVAL_RATES:
        for scenario_name in SCENARIO_NAMES:
            print(f"\n=== Running RLRAM: scenario={scenario_name}, λ={l} ===")
            args_list = [(l, scenario_name, i) for i in range(SIMULATION_RUNS)]

            with mp.Pool(processes=min(10, SIMULATION_RUNS)) as pool:
                results = list(tqdm(
                    pool.imap(run_rlram_simulation, args_list, chunksize=1),
                    total=len(args_list)
                ))

            combined_logs = []
            for df, runtime_seconds, run_id in results:
                combined_logs.append(df)
                with open(runtime_csv_path, "a") as f:
                    f.write(f"RLRAM,{l},{run_id},{runtime_seconds:.2f}\n")
                print(f"[Run {run_id}] {runtime_seconds:.2f}s")

            final_df = pd.concat(combined_logs, ignore_index=True)
            out_path = f"results/RLRAM_l{l}_{scenario_name}.csv"
            final_df.to_csv(out_path, index=False)
            print(f"Saved combined results to: {out_path}")
