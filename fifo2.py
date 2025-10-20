from __future__ import annotations
from typing import Dict, List, Set, Tuple
import os
import time
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

from myproject.processes import SimpleProcess, Task, Resource, ProcessElement
from myproject.config import (
    ARRIVAL_RATES,
    SCENARIO_NAMES,
    SIMULATION_RUN_TIME,
    process_function,
    SIMULATION_RUNS,
)
from myproject.simulator import Simulator


class FIFO(SimpleProcess):
    """
    Fair, work-conserving FIFO allocator:

    - Order cases by earliest case start time (ties by case_id).
    - Within a case, try waiting tasks in their arrival order (task.id).
    - Assign at most one task per case per pass (prevents one case from hogging all resources).
    - For the chosen task, pick a compatible resource with the earliest availability time
      (tie-break by resource name to keep deterministic behavior).
    """

    def __init__(self, processes):
        super().__init__(processes)
        self.allocation_method_name = "FIFO"

    '''def assign_resources(
        self,
        unassigned_tasks: Dict[int, ProcessElement],
        available_resources: Set[Resource],
        resource_available_times: Dict[Resource, float],
    ) -> List[Tuple[ProcessElement, Resource]]:
        assignments: List[Tuple[ProcessElement, Resource]] = []
        if not unassigned_tasks or not available_resources:
            return assignments

        # Work on a temp set so we don't mutate simulator state mid-iteration
        temp_available: Set[Resource] = set(available_resources)
        used_task_ids: Set[int] = set()

        tasks_list: List[ProcessElement] = list(unassigned_tasks.values())
        unique_case_ids = {t.case_id for t in tasks_list}

        # Fairness across cases: earliest started case first
        case_order = sorted(
            unique_case_ids,
            key=lambda cid: (self.simulator.case_start_times.get(cid, float("inf")), cid),
        )

        while temp_available:
            made_progress = False

            for cid in case_order:
                if not temp_available:
                    break

                # Tasks for this case that are still unassigned this round
                case_tasks = [
                    t for t in unassigned_tasks.values()
                    if t.case_id == cid and t.id not in used_task_ids
                ]
                if not case_tasks:
                    continue

                # FIFO within the case (earliest task id == earliest task activation)
                case_tasks.sort(key=lambda t: t.id)

                for t in case_tasks:
                    task_def: Task = self.process_definitions[t.case_type].tasks[t.label]  # type: ignore

                    # Which of the task's resources are currently available?
                    compatible_available = [r for r in task_def.resources if r in temp_available]
                    if not compatible_available:
                        continue

                    # Choose the resource that has been idle for the longest (earliest available time)
                    chosen_res = min(
                        compatible_available,
                        key=lambda r: (resource_available_times.get(r, 0.0), r.name),
                    )

                    assignments.append((t, chosen_res))
                    temp_available.remove(chosen_res)
                    used_task_ids.add(t.id)
                    made_progress = True
                    break  # assign at most one task for this case in this pass

            if not made_progress:
                break

        return assignments'''
    
    def assign_resources(
        self,
        unassigned_tasks: Dict[int, ProcessElement],
        available_resources: Set[Resource],
        resource_available_times: Dict[Resource, float],
    ) -> List[Tuple[ProcessElement, Resource]]:
        """
        Strict FIFO per station (activity label):
        - Build a queue per station label ordered by task activation (task.id).
        - For each station, assign available machines to the head of that station's queue.
        - Machine tie-break: earliest available time (longest idle), then name.
        """
        assignments: List[Tuple[ProcessElement, Resource]] = []
        if not unassigned_tasks or not available_resources:
            return assignments

        # 1) Build per-station FIFO queues (only TASKs get here)
        station_queues: Dict[str, List[ProcessElement]] = {}
        for pe in unassigned_tasks.values():
            station_queues.setdefault(pe.label, []).append(pe)
        '''for q in station_queues.values():
            # activation order proxy: smaller task.id == earlier activation
            q.sort(key=lambda t: t.id)'''
        for q in station_queues.values():
            q.sort(key=lambda t: (self.task_first_ready_time.get(t.id, float("inf")), t.id))


        # 2) For each station label present, get the compatible resources
        #    (use a representative task's case_type to fetch that station's Task definition)
        station_resources: Dict[str, List[Resource]] = {}
        for label, queue in station_queues.items():
            pe0 = queue[0]
            task_def: Task = self.process_definitions[pe0.case_type].tasks[label]  # type: ignore
            station_resources[label] = list(task_def.resources) if getattr(task_def, "resources", None) else []

        temp_available: Set[Resource] = set(available_resources)

        made_progress = True
        while made_progress and temp_available:
            made_progress = False

            # Deterministic station iteration order
            for label in sorted(station_queues.keys()):
                if not temp_available:
                    break
                if not station_queues[label]:
                    continue

                # Which compatible resources for this station are free now?
                compat_free = [r for r in station_resources.get(label, []) if r in temp_available]
                if not compat_free:
                    continue

                # Take the head of the station queue (strict FIFO at the station)
                pe = station_queues[label][0]

                # Choose the machine that's been idle the longest (tie-break by name)
                chosen_res = min(
                    compat_free,
                    key=lambda r: (resource_available_times.get(r, 0.0), r.name),
                )

                # Record assignment
                assignments.append((pe, chosen_res))
                temp_available.remove(chosen_res)

                # Remove from the station queue; Simulator will pop from unassigned_tasks after scheduling START
                station_queues[label].pop(0)

                made_progress = True

        return assignments

# -----------------------------
# Batch runner (multiprocessing)
# -----------------------------

def run_fifo_simulation(args):
    l, scenario_name, run_id = args

    processes = process_function(l, scenario_name)
    allocator = FIFO(processes)

    start_time = time.time()
    # Pass scenario_name for nicer log filenames and seed for reproducibility
    simulator = Simulator(
        simulation_run=run_id,
        process=allocator,
        scenario_name=scenario_name,
        log_only_case_completion=False,
        is_training=False,
        seed=run_id,
    )
    simulator.run(SIMULATION_RUN_TIME)
    runtime_seconds = time.time() - start_time

    df = simulator.event_log.get_dataframe()
    df["run"] = run_id
    df["l"] = l
    df["scenario"] = scenario_name
    df["method"] = allocator.allocation_method_name

    return df, runtime_seconds, run_id


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    os.makedirs("results", exist_ok=True)
    os.makedirs("runtime", exist_ok=True)
    os.makedirs("event_logs/train", exist_ok=True)
    os.makedirs("event_logs/test", exist_ok=True)

    runtime_csv_path = "runtime/FIFO_runtimes.csv"
    if not os.path.exists(runtime_csv_path):
        with open(runtime_csv_path, "w") as f:
            f.write("method,l,run,time\n")

    for l in ARRIVAL_RATES:
        for scenario_name in SCENARIO_NAMES:
            print(f"\n=== Running FIFO: scenario={scenario_name}, λ={l} ===")

            args_list = [(l, scenario_name, i) for i in range(SIMULATION_RUNS)]
            pool_size = min(max(1, mp.cpu_count() - 1), SIMULATION_RUNS, 8)

            with mp.Pool(processes=pool_size) as pool:
                results = list(tqdm(pool.imap(run_fifo_simulation, args_list), total=len(args_list)))

            combined_logs = []
            for df, runtime_seconds, run_id in results:
                combined_logs.append(df)
                with open(runtime_csv_path, "a") as f:
                    f.write(f"FIFO,{l},{run_id},{runtime_seconds:.2f}\n")

            combined_df = pd.concat(combined_logs, ignore_index=True)
            result_filename = f"results/251020FIFO_l{l}_{scenario_name}.csv"
            combined_df.to_csv(result_filename, index=False)
            print(f"Saved results to {result_filename}")
