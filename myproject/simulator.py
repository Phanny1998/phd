from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Dict, List, Tuple, Set
import heapq
import csv
import os
import random
import gc
import time

import pandas as pd

from myproject.processes import ProcessElement, Resource, Process, ProcessElementType


class SimulationItemType(Enum):
    CASE_ARRIVAL = auto()
    ACTIVATE_TASK = auto()
    COMPLETE_TASK = auto()
    COMPLETE_EVENT = auto()
    COMPLETE_CASE = auto()
    ASSIGN_RESOURCES = auto()
    START_TASK = auto()


class SimulationItem:
    def __init__(
        self,
        simulation_item_type: SimulationItemType,
        moment: float,
        process_element: Optional[ProcessElement],
        resource: Optional[Resource] = None,
    ):
        self.simulation_item_type = simulation_item_type
        self.moment = moment
        self.process_element = process_element
        self.resource = resource

    def __lt__(self, other):
        return self.moment < other.moment

    def __str__(self):
        return (
            f"{self.simulation_item_type} | time:{round(self.moment, 2)} | "
            f"{self.process_element} | {self.resource}"
        )


class EventLog:
    def __init__(
        self,
        run_id: int,
        scenario_name: str,
        allocation_method: str = "default",
        log_only_case_completion: bool = False,
        is_training: bool = False,
    ):
        self.is_training = is_training
        log_dir = "event_logs/train" if self.is_training else "event_logs/test"
        os.makedirs(log_dir, exist_ok=True)
        self.log_only_case_completion = log_only_case_completion
        self.filename = os.path.join(
            log_dir, f"log_{allocation_method}_run{run_id}_{scenario_name}.csv"
        )
        self.fields = [
            "method",
            "num_processes",
            "simulation_run",
            "timestamp",
            "process",
            "l",
            "status",
            "case_id",
            "activity",
            "resource",
            "end_time",
            "cycle_time",
            "data",
        ]
        with open(self.filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log_event(self, **kwargs):
        if self.log_only_case_completion and kwargs.get("status") != "COMPLETE":
            return
        filtered = {key: kwargs.get(key, "") for key in self.fields}
        if isinstance(filtered.get("data", ""), dict):
            filtered["data"] = str(filtered["data"])
        with open(self.filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(filtered)

    def get_dataframe(self):
        return pd.read_csv(self.filename)


class Simulator:
    def __init__(
        self,
        simulation_run: int,
        process: Process,
        scenario_name: str = "default",
        log_only_case_completion: bool = False,
        is_training: bool = False,
        seed: Optional[int] = None,
    ):
        """
        simulation_run: integer run id
        process:        instance of Process (e.g., FIFO allocator)
        scenario_name:  used in output filenames
        seed:           if provided, sets Python RNG seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.is_training = is_training
        self.scenario_name = scenario_name
        self.simulation_run = simulation_run
        self.process = process

        self.now: float = 0.0
        self.events: List[Tuple[float, SimulationItem]] = []
        self.unassigned_tasks: Dict[int, ProcessElement] = {}
        self.busy_resources: Dict[Resource, ProcessElement] = {}
        self.available_resources: Set[Resource] = set()
        self.resource_available_times: Dict[Resource, float] = {}

        self.case_start_times: Dict[int, float] = {}
        self.task_start_times: Dict[int, float] = {}
        self.busy_cases: Dict[int, List[ProcessElement]] = {}
        self.finalized_cases: int = 0
        self.total_cycle_time: float = 0.0

        allocation_method = getattr(process, "allocation_method_name", "default")
        self.event_log = EventLog(
            run_id=simulation_run,
            scenario_name=scenario_name,
            allocation_method=allocation_method,
            log_only_case_completion=log_only_case_completion,
            is_training=self.is_training,
        )
        process.set_simulator(self)

        self.initialize_simulation()
        # track when a resource becomes free (for FIFO tie-breaks)
        self.resource_available_times = {res: 0.0 for res in self.process.resources}

    # --------------- core loop ---------------

    def initialize_simulation(self):
        self.available_resources = set(self.process.resources)
        self.process.reset_all_process_parameter()
        initial_time, initial_event = self.process.next_case()
        self.schedule_event(
            initial_time,
            SimulationItem(
                simulation_item_type=SimulationItemType.CASE_ARRIVAL,
                moment=initial_time,
                process_element=initial_event,
            ),
        )

    def schedule_event(self, moment: float, simulation_item: SimulationItem):
        heapq.heappush(self.events, (moment, simulation_item))

    def run(self, running_time: float):
        while self.events and self.now <= running_time:
            self.now, current_event = heapq.heappop(self.events)
            self.handle_event(current_event)

            # occasional cleanup
            if self.finalized_cases and self.finalized_cases % 100 == 0:
                gc.collect()

    # --------------- handlers ---------------

    def handle_event(self, simulation_item: SimulationItem):
        pe = simulation_item.process_element
        t = simulation_item.simulation_item_type

        if t == SimulationItemType.CASE_ARRIVAL:
            self.handle_case_arrival(pe)

        elif t == SimulationItemType.COMPLETE_EVENT:
            self.handle_complete_event(pe)

        elif t == SimulationItemType.COMPLETE_TASK:
            self.handle_complete_task(simulation_item)

        elif t == SimulationItemType.START_TASK:
            self.handle_start_task(simulation_item)

        elif t == SimulationItemType.COMPLETE_CASE:
            self.handle_complete_case(pe)

        elif t == SimulationItemType.ASSIGN_RESOURCES:
            self.handle_assign_resources()

    def handle_case_arrival(self, process_element: ProcessElement):
        cid = process_element.case_id
        self.case_start_times[cid] = self.now
        self.busy_cases[cid] = []

        self.activate_element(process_element)

        next_time, next_event = self.process.next_case()
        self.schedule_event(
            next_time,
            SimulationItem(
                SimulationItemType.CASE_ARRIVAL, next_time, next_event
            ),
        )

        self.event_log.log_event(
            method=self.process.allocation_method_name,
            num_processes=len(self.process.case_types),
            simulation_run=self.simulation_run,
            timestamp=self.now,
            process=process_element.case_type,
            l=self.process.arrival_distributions[process_element.case_type],
            status="START",
            case_id=cid,
            activity=process_element.label,
            data=self.process.case_data[process_element.case_id],
        )

    def handle_complete_event(self, process_element: ProcessElement):
        next_elements = self.process.complete_element(process_element)

        if not next_elements:
            self.schedule_event(
                self.now,
                SimulationItem(
                    SimulationItemType.COMPLETE_CASE, self.now, process_element
                ),
            )
        else:
            if process_element.is_gateway:
                self.event_log.log_event(
                    method=self.process.allocation_method_name,
                    num_processes=len(self.process.case_types),
                    simulation_run=self.simulation_run,
                    timestamp=self.now,
                    process=process_element.case_type,
                    l=self.process.arrival_distributions[process_element.case_type],
                    status="gateway",
                    case_id=process_element.case_id,
                    activity=process_element.label,
                    data=self.process.case_data[process_element.case_id],
                )

        for nxt in next_elements:
            self.activate_element(nxt)

    def handle_complete_task(self, simulation_item: SimulationItem):
        resource = simulation_item.resource
        pe = simulation_item.process_element

        next_elements = self.process.complete_element(pe)

        # free resource
        self.busy_resources.pop(resource, None)
        self.available_resources.add(resource)
        self.resource_available_times[resource] = self.now

        if not next_elements:
            self.schedule_event(
                self.now,
                SimulationItem(SimulationItemType.COMPLETE_CASE, self.now, pe),
            )

        for nxt in next_elements:
            self.activate_element(nxt)

        # try assign immediately
        self.schedule_event(
            self.now, SimulationItem(SimulationItemType.ASSIGN_RESOURCES, self.now, None)
        )

    def handle_start_task(self, simulation_item: SimulationItem):
        resource = simulation_item.resource
        pe = simulation_item.process_element

        self.busy_resources[resource] = pe

        # -------------------------------
        # NEW: bind lane when MOULDING starts
        # This ensures ROUTE_TO_A1 will follow case_data['moulding_lane']
        # matching the actual MOULDING_MACHINE_i that was chosen.
        if pe.label == "MOULDING" and resource and resource.name.startswith("MOULDING_MACHINE_"):
            try:
                lane_idx = resource.name.rsplit("_", 1)[1]  # "1".."5"
                self.process.add_data(pe, {"moulding_lane": f"ASSEMBLY_1_{lane_idx}"})
            except Exception:
                # Don't break the sim if a name is unexpected
                pass
        # -------------------------------
        processing_time = self.process.processing_time_sample(resource, pe, self.now)
        self.task_start_times[pe.id] = self.now

        self.schedule_event(
            self.now + processing_time,
            SimulationItem(
                SimulationItemType.COMPLETE_TASK,
                self.now + processing_time,
                pe,
                resource,
            ),
        )

        self.event_log.log_event(
            method=self.process.allocation_method_name,
            num_processes=len(self.process.case_types),
            simulation_run=self.simulation_run,
            timestamp=self.now,
            process=pe.case_type,
            l=self.process.arrival_distributions[pe.case_type],
            status="running",
            case_id=pe.case_id,
            activity=pe.label,
            resource=resource.name,
            end_time=self.now + processing_time,
        )

    def handle_complete_case(self, process_element: ProcessElement):
        cid = process_element.case_id
        is_gateway = process_element.is_gateway

        self.finalized_cases += 1
        cycle_time = self.now - self.case_start_times[cid]
        self.total_cycle_time += cycle_time

        if not is_gateway:
            self.event_log.log_event(
                method=self.process.allocation_method_name,
                num_processes=len(self.process.case_types),
                simulation_run=self.simulation_run,
                timestamp=self.now,
                process=process_element.case_type,
                l=self.process.arrival_distributions[process_element.case_type],
                status="COMPLETE",
                case_id=cid,
                activity=process_element.label,
                cycle_time=cycle_time,
            )

        self.cleanup_case_state(cid)

    def handle_assign_resources(self):
        assignments = self.process.assign_resources(
            self.unassigned_tasks, self.available_resources, self.resource_available_times
        )
        for task, resource in assignments:
            self.schedule_event(
                self.now,
                SimulationItem(
                    SimulationItemType.START_TASK, self.now, task, resource
                ),
            )
            self.unassigned_tasks.pop(task.id, None)
            self.available_resources.discard(resource)

    # --------------- helpers ---------------

    def activate_element(self, process_element: ProcessElement):
        self.busy_cases[process_element.case_id].append(process_element)
        if process_element.is_event():
            self.schedule_event(
                process_element.occurrence_time,
                SimulationItem(
                    SimulationItemType.COMPLETE_EVENT,
                    process_element.occurrence_time,
                    process_element,
                ),
            )
        elif process_element.is_task():
            self.unassigned_tasks[process_element.id] = process_element
            if not hasattr(self.process, "task_first_ready_time"):
                self.process.task_first_ready_time = {}
            self.process.task_first_ready_time.setdefault(process_element.id, self.now)
            # Log true station-arrival (queue) time
            self.event_log.log_event(
                method=self.process.allocation_method_name,
                num_processes=len(self.process.case_types),
                simulation_run=self.simulation_run,
                timestamp=self.now,
                process=process_element.case_type,
                l=self.process.arrival_distributions[process_element.case_type],
                status="queued",
                case_id=process_element.case_id,
                activity=process_element.label,
)

            self.schedule_event(
                self.now,
                SimulationItem(SimulationItemType.ASSIGN_RESOURCES, self.now, None),
            )

    def cleanup_case_state(self, case_id: int):
        self.case_start_times.pop(case_id, None)
        self.busy_cases.pop(case_id, None)

        if hasattr(self.process, "case_type"):
            self.process.case_type.pop(case_id, None)
        if hasattr(self.process, "case_data"):
            self.process.case_data.pop(case_id, None)
        if hasattr(self.process, "completed_tasks"):
            self.process.completed_tasks.pop(case_id, None)
        if hasattr(self.process, "last_task_completion_time"):
            self.process.last_task_completion_time.pop(case_id, None)
        if hasattr(self.process, "task_first_ready_time"):
            for k in [k for k in list(self.process.task_first_ready_time.keys()) if self.get_case_id_from_task_id(k) == case_id]:
                self.process.task_first_ready_time.pop(k, None)
        if hasattr(self.process, "task_postpone_count"):
            for k in [k for k in list(self.process.task_postpone_count.keys()) if self.get_case_id_from_task_id(k) == case_id]:
                self.process.task_postpone_count.pop(k, None)

    def get_case_id_from_task_id(self, task_id: int):
        # not used in this setup
        return None
