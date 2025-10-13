import random
from framework.processes import SimpleProcess
from framework.simulator import Simulator


import random

class SPTAllocation(SimpleProcess):
    def __init__(self, processes):
        super().__init__(processes)
        self.allocation_method_name = "SPT"

    def assign_resources(self, unassigned_tasks, available_resources, resource_available_times):
        """
        Greedy Shortest Processing Time (SPT) allocator.

        - Picks the (task, resource) pair with the smallest expected processing time.
        - Breaks ties uniformly at random across all equal-best candidates.
        - Repeats until either tasks or resource are empty.
        """
        assignments = []
        resources_list = list(available_resources)
        tasks_list = list(unassigned_tasks.values())


        while tasks_list and resources_list:
            shortest_time = float("inf")
            best_candidates = []  

            for task in tasks_list:
                task_def = self.process_definitions[task.case_type].tasks[task.label]

                allowed_names = {res.name for res in task_def.resources}

                compatible_resources = [r for r in resources_list if r.name in allowed_names]

                for resource in compatible_resources:
                    for resource in compatible_resources:
                        mean_time = next(
                            r2.execution_distribution[0]
                            for r2 in task_def.resources
                            if r2.name == resource.name
                        )
                        if mean_time < shortest_time:
                            shortest_time = mean_time
                            best_candidates = [(task, resource)]
                        elif mean_time == shortest_time:
                            best_candidates.append((task, resource))

                    if mean_time < shortest_time:
                        shortest_time = mean_time
                        best_candidates = [(task, resource)]
                    elif mean_time == shortest_time:
                        best_candidates.append((task, resource))

            if not best_candidates:
                break

            task, resource = random.choice(best_candidates)

            assignments.append((task, resource))
            tasks_list.remove(task)
            resources_list.remove(resource)

        return assignments


import os
import time
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from framework.config import ARRIVAL_RATES, SCENARIO_NAMES, SIMULATION_RUN_TIME, process_function, SIMULATION_RUNS
from framework.simulator import Simulator

mp.set_start_method("spawn", force=True)

os.makedirs("results", exist_ok=True)
os.makedirs("runtime", exist_ok=True)

runtime_csv_path = "runtime/SPTAllocation_runtimes.csv"

if not os.path.exists(runtime_csv_path):
    with open(runtime_csv_path, "w") as f:
        f.write("method,l,run,time\n")


def run_spt_simulation(args):
    
    import time  
    l, scenario_name, run_id = args

    processes = process_function(l, scenario_name)
    allocator = SPTAllocation(processes)

    start_time = time.time()
    simulator = Simulator(simulation_run=run_id, process=allocator)
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
                results = list(tqdm(pool.imap(run_spt_simulation, args_list), total=len(args_list)))

            combined_logs = []
            for df, runtime_seconds, run_id in results:
                combined_logs.append(df)
                with open(runtime_csv_path, "a") as f:
                    f.write(f"SPTAllocation,{l},{run_id},{runtime_seconds:.2f}\n")
                print(f"[Run {run_id}] Completed in {runtime_seconds:.2f}s")

            combined_df = pd.concat(combined_logs, ignore_index=True)
            result_filename = f"results/SPTAllocation_l{l}_{scenario_name}.csv"
            combined_df.to_csv(result_filename, index=False)
            print(f"Saved results to {result_filename}")

