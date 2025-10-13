from __future__ import annotations
from typing import Dict, List, Set, Tuple
from framework.processes import SimpleProcess, Task, Resource, ProcessElement

class FIFO(SimpleProcess):
    """
    - Order cases by earliest case start time (ties by case_id).
    - For each case (in that order), try its waiting tasks in arrival order (task.id).
    - Assign at most one task per case per pass, then move to the next case.
    - For a chosen task, pick any compatible available resource, tie-break by earliest availability time
    """

    def __init__(self, processes):
        super().__init__(processes)
        self.allocation_method_name = "FIFO"

    def assign_resources(
        self,
        unassigned_tasks: Dict[int, ProcessElement],            
        available_resources: Set[Resource],
        resource_available_times: Dict[Resource, float],
    ) -> List[Tuple[ProcessElement, Resource]]:
        assignments: List[Tuple[ProcessElement, Resource]] = []
        if not unassigned_tasks or not available_resources:
            return assignments

        temp_available: Set[Resource] = set(available_resources)
        used_task_ids: Set[int] = set()

        tasks_list: List[ProcessElement] = list(unassigned_tasks.values())
        unique_case_ids = {t.case_id for t in tasks_list}
        case_order = sorted(
            unique_case_ids,
            key=lambda cid: (
                self.simulator.case_start_times.get(cid, float("inf")),
                cid,
            ),
        )

        while temp_available:
            made_progress = False

            for cid in case_order:
                if not temp_available:
                    break

                case_tasks = [
                    t for t in unassigned_tasks.values()
                    if t.case_id == cid and t.id not in used_task_ids
                ]
                if not case_tasks:
                    continue

                case_tasks.sort(key=lambda t: t.id)

                for t in case_tasks:
                    task_def: Task = self.process_definitions[t.case_type].tasks[t.label]
                    
                    compatible_available = [r for r in task_def.resources if r in temp_available]
                    if not compatible_available:
                        continue

                    chosen_res = min(
                        compatible_available,
                        key=lambda r: (resource_available_times.get(r, 0.0), r.name),
                    )

                    assignments.append((t, chosen_res))
                    temp_available.remove(chosen_res)
                    used_task_ids.add(t.id)
                    made_progress = True
                    break  

            if not made_progress:
                break

        return assignments



import os
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from framework.config import (
    ARRIVAL_RATES,
    SCENARIO_NAMES,
    SIMULATION_RUN_TIME,
    process_function,
    SIMULATION_RUNS,
)
from framework.simulator import Simulator


mp.set_start_method("spawn", force=True)

os.makedirs("results", exist_ok=True)
os.makedirs("runtime", exist_ok=True)

runtime_csv_path = "runtime/FIFO_runtimes.csv"
if not os.path.exists(runtime_csv_path):
    with open(runtime_csv_path, "w") as f:
        f.write("method,l,run,time\n")

def run_fifo_simulation(args):
    
    import time  
    l, scenario_name, run_id = args

    processes = process_function(l, scenario_name)
    allocator = FIFO(processes)

    start_time = time.time()
    simulator = Simulator(simulation_run=run_id, process=allocator)
    simulator.run(SIMULATION_RUN_TIME)
    runtime_seconds = time.time() - start_time

    df = simulator.event_log.get_dataframe()
    df["run"] = run_id
    df["l"] = l
    df["scenario"] = scenario_name
    df["method"] = "FIFO"

    return df, runtime_seconds, run_id

if __name__ == "__main__":
    for l in ARRIVAL_RATES:
        for scenario_name in SCENARIO_NAMES:
            print(f"\n=== Running FIFO: scenario={scenario_name}, λ={l} ===")

            args_list = [(l, scenario_name, i) for i in range(SIMULATION_RUNS)]
            pool_size = min(10, SIMULATION_RUNS)

            with mp.Pool(processes=pool_size) as pool:
                results = list(tqdm(pool.imap(run_fifo_simulation, args_list), total=len(args_list)))

            combined_logs = []
            for df, runtime_seconds, run_id in results:
                combined_logs.append(df)
                with open(runtime_csv_path, "a") as f:
                    f.write(f"FIFO,{l},{run_id},{runtime_seconds:.2f}\n")

            combined_df = pd.concat(combined_logs, ignore_index=True)
            result_filename = f"results/FIFO_l{l}_{scenario_name}.csv"
            combined_df.to_csv(result_filename, index=False)
            print(f"Saved results to {result_filename}")
