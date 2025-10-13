import random
from framework.processes import SimpleProcess
from framework.simulator import Simulator

class RandomAllocation(SimpleProcess):
    def __init__(self, processes):
        super().__init__(processes)
        self.allocation_method_name = "Random"

    def assign_resources(self, unassigned_tasks, available_resources, resource_available_times):
        assignments = []
        resources_list = list(available_resources)  

        for task  in unassigned_tasks.values():
            assigned = False

            task_definition = self.process_definitions[task.case_type].tasks[task.label]
            compatible_resources = {res.name for res in task_definition.resources}

            compatible_available_resources = [
                resource for resource in resources_list if resource.name in compatible_resources
            ]

            if compatible_available_resources:
                selected_resource = random.choice(compatible_available_resources)

                assignments.append((task, selected_resource))

                resources_list.remove(selected_resource)
                assigned = True

            if not assigned:
                break

        #print(assignments)
        return assignments


import os
import time
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from framework.config import (
    ARRIVAL_RATES, SCENARIO_NAMES, SIMULATION_RUN_TIME,
    process_function, SIMULATION_RUNS
)
from framework.simulator import Simulator



def run_random_simulation(args):
 
    import time as _time

    l, scenario_name, run_id = args
    processes = process_function(l, scenario_name)
    allocator = RandomAllocation(processes)

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

    runtime_csv_path = "runtime/Random_runtimes.csv"
    if not os.path.exists(runtime_csv_path):
        with open(runtime_csv_path, "w") as f:
            f.write("method,l,run,time\n")

    for l in ARRIVAL_RATES:
        for scenario_name in SCENARIO_NAMES:
            print(f"\n=== Running scenario: {scenario_name}, arrival rate: {l} ===")

            args_list = [(l, scenario_name, i) for i in range(SIMULATION_RUNS)]

            pool_size = min(10, SIMULATION_RUNS)
            with mp.Pool(processes=pool_size) as pool:
                results = list(tqdm(
                    pool.imap(run_random_simulation, args_list, chunksize=1),
                    total=len(args_list)
                ))

            combined_logs = []
            for df, runtime_seconds, run_id in results:
                combined_logs.append(df)
                with open(runtime_csv_path, "a") as f:
                    f.write(f"Random,{l},{run_id},{runtime_seconds:.2f}\n")
                print(f"[Run {run_id}] Completed in {runtime_seconds:.2f}s")

            combined_df = pd.concat(combined_logs, ignore_index=True)
            out_path = f"results/Random_l{l}_{scenario_name}.csv"
            combined_df.to_csv(out_path, index=False)
            print(f"Saved combined results to: {out_path}")

