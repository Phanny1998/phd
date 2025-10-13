from enum import Enum, auto
from typing import Optional, Dict
import random
import pandas as pd
import heapq
from framework.processes import ProcessElement, Resource, Process
import gc 
import tracemalloc
import csv
import os
import time

class SimulationItemType(Enum):
    CASE_ARRIVAL = auto()
    ACTIVATE_TASK = auto()
    COMPLETE_TASK = auto()
    COMPLETE_EVENT = auto()
    COMPLETE_CASE = auto()
    ASSIGN_RESOURCES = auto()
    START_TASK = auto()

class SimulationItem:
    def __init__(self, simulation_item_type: SimulationItemType, moment: float, process_element: Optional[ProcessElement], resource: Optional[Resource] = None):
        self.simulation_item_type = simulation_item_type
        self.moment = moment
        self.process_element = process_element
        self.resource = resource
        tracemalloc.start()

    def __lt__(self, other):
        return self.moment < other.moment

    def __str__(self):
        return (
            f"{self.simulation_item_type} | time:{round(self.moment, 2)} | "
            f"{self.process_element} | {self.resource}"
        )

# class EventLog:
#     def __init__(self):
#         self.log = []

#     def log_event(self, **kwargs):
#         self.log.append(kwargs)

#     def get_dataframe(self):
        
#         return pd.DataFrame(self.log)
    

class EventLog:
    def __init__(self, run_id: int, scenario_name: str, allocation_method: str = "default", log_dir: str = "event_logs", log_only_case_completion: bool = False, is_training = False):
        self.is_training=is_training
        log_dir = "event_logs/train" if self.is_training else "event_logs/test"
        os.makedirs(log_dir, exist_ok=True)
        self.log_only_case_completion = log_only_case_completion
        self.filename = os.path.join(log_dir, f"log_{allocation_method}_run{run_id}_{scenario_name}.csv")
        self.fields = ["method", "num_processes", "simulation_run", "timestamp", "process", "l", "status", "case_id", "activity", "resource", "end_time", "cycle_time", "data"]

        with open(self.filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log_event(self, **kwargs):
        if self.log_only_case_completion:
            if kwargs.get("status") != "COMPLETE":
                return  # Skip non-case-completion logs

        # Filter only known fields
        filtered = {key: kwargs.get(key, "") for key in self.fields}

        # Optional: serialize dicts
        if isinstance(filtered.get("data", ""), dict):
            filtered["data"] = str(filtered["data"])

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(filtered)


    def get_dataframe(self):
        import pandas as pd
        return pd.read_csv(self.filename)



class Simulator:
    def __init__(self, simulation_run: int, process: Process, scenario_name: str = "default", log_only_case_completion: bool = False, is_training=False):
        self.is_training = is_training
        self.simulation_run = simulation_run
        self.process = process
        self.now = 0
        self.events = []
        self.unassigned_tasks = {}
        self.busy_resources = {}
        self.available_resources = set()
        self.case_start_times = {}
        self.task_start_times = {}
        self.busy_cases = {}
        self.finalized_cases = 0
        self.total_cycle_time = 0
        #self.event_log = EventLog()
        allocation_method = getattr(process, "allocation_method_name", "default")
        self.event_log = EventLog(run_id=simulation_run, scenario_name=scenario_name, allocation_method=allocation_method, log_only_case_completion=log_only_case_completion, is_training=self.is_training)
        print("[Simulator Initialized]")
        process.set_simulator(self)
        
        self.initialize_simulation()
        self.resource_available_times = {res: 0.0 for res in self.process.resources}


    def initialize_simulation(self):
        #print("[Simulation Initialization]")
        self.available_resources = set(self.process.resources)
        #print(f"[Available Resources Initialized] {self.available_resources}")
        self.process.reset_all_process_parameter()
        initial_time, initial_event = self.process.next_case()
        #print(f"[Initial Event Scheduled] Time: {initial_time}, Event: {initial_event}")
        self.schedule_event(initial_time, SimulationItem(simulation_item_type=SimulationItemType.CASE_ARRIVAL, moment=initial_time, process_element=initial_event))
        #print("[Simulation Initialization End]")


    def schedule_event(self, moment: float, simulation_item: SimulationItem):
        #print(f"[Event Scheduled] Time: {moment}, Event: {simulation_item} ---------------------")
        #self.events.append((moment, simulation_item))
        heapq.heappush(self.events, (moment, simulation_item))
        #self.events.sort()

    def run(self, running_time: float):
        #print(f"[Simulation Start] Running Time: {running_time}")
        iteration_count = 0
        while self.now <= running_time:# and self.events:
            #self.now, current_event = self.events.pop(0)
            self.now, current_event = heapq.heappop(self.events)
            #print(f"[Processing Event] Time: {self.now}, Event: {current_event}")
            # if current_event.process_element:
            #     print(round(self.now,2)," | ",current_event.process_element.case_type," | ", current_event.process_element.case_id," | ",current_event.process_element.id," | ", current_event.process_element.label," | ", current_event.simulation_item_type," | ", current_event.resource, " | ", self.finalized_cases)
            # else:
            #     print(round(self.now,2)," | ", "---"," | ", "---"," | ", current_event.simulation_item_type," | ", current_event.resource)
            self.handle_event(current_event)
            #print("Number of scheduled events:", len(self.events))
            # Insert garbage collection every N events
            iteration_count += 1
            # if iteration_count % 500 == 0:
            #     # gc.collect()

            # Optional: monitor memory
            if self.finalized_cases % 100 == 0:
                # print(f"[Memory Cleanup] Cases: {self.finalized_cases} | Time: {self.now}")
                # gc.collect()
                current, peak = tracemalloc.get_traced_memory()
                #print(f"[Memory] {current/1024/1024:.2f}MB (Peak: {peak/1024/1024:.2f}MB)")
                gc.collect()

            #self.sort_events()
        #print(f"[Simulation End] Finalized Cases: {self.finalized_cases}, Total Cycle Time: {self.total_cycle_time}")


    def handle_event(self, simulation_item: SimulationItem):
        #print(f"[Handle Event] Type: {simulation_item.simulation_item_type}, Element: {simulation_item.process_element}")
        process_element = simulation_item.process_element

        if simulation_item.simulation_item_type == SimulationItemType.CASE_ARRIVAL:
            self.handle_case_arrival(process_element)

        elif simulation_item.simulation_item_type == SimulationItemType.COMPLETE_EVENT:
            self.handle_complete_event(process_element)

        elif simulation_item.simulation_item_type == SimulationItemType.COMPLETE_TASK:
            self.handle_complete_task(simulation_item)

        elif simulation_item.simulation_item_type == SimulationItemType.START_TASK:
            self.handle_start_task(simulation_item)

        elif simulation_item.simulation_item_type == SimulationItemType.COMPLETE_CASE:
            self.handle_complete_case(process_element)

        elif simulation_item.simulation_item_type == SimulationItemType.ASSIGN_RESOURCES:
            self.handle_assign_resources()

    def handle_case_arrival(self, process_element: ProcessElement):
        #print(f"[Case Arrival] Case ID: {process_element.case_id}, Time: {self.now}")
        case_id = process_element.case_id
        self.case_start_times[case_id] = self.now
        self.busy_cases[case_id] = []
        self.activate_element(process_element)
        next_time, next_event = self.process.next_case()
        #print(f"[Next Case Scheduled] Time: {next_time}, Event: {next_event}")
        self.schedule_event(next_time, SimulationItem(SimulationItemType.CASE_ARRIVAL, next_time, next_event))
        self.event_log.log_event(method=self.process.allocation_method_name, 
                                 num_processes = len(self.process.case_types),
                                 simulation_run=self.simulation_run, 
                                 timestamp=self.now, 
                                 process=process_element.case_type,
                                 l = self.process.arrival_distributions[process_element.case_type], 
                                 status = "START", 
                                 case_id=case_id,
                                 activity=process_element.label,
                                 data=self.process.case_data[process_element.case_id])
        
    def handle_complete_event(self, process_element: ProcessElement):
        #print(f"[Complete Event] Element: {process_element}")
        #self.busy_cases[process_element.case_id].remove(process_element)
        next_elements = self.process.complete_element(process_element)
        if not next_elements:
            self.schedule_event(self.now, SimulationItem(SimulationItemType.COMPLETE_CASE, self.now, process_element))
        elif process_element.is_gateway:
            pass
            self.event_log.log_event(method=self.process.allocation_method_name, 
                                 num_processes = len(self.process.case_types),
                                 simulation_run=self.simulation_run, 
                                 timestamp=self.now, 
                                 process=process_element.case_type,
                                 l = self.process.arrival_distributions[process_element.case_type], 
                                 status = "gateway", 
                                 case_id=process_element.case_id,
                                 activity=process_element.label,
                                 data=self.process.case_data[process_element.case_id])
        for next_element in next_elements:
            self.activate_element(next_element)
        
        

    def handle_complete_task(self, simulation_item: SimulationItem):
        #print(f"[Complete Task] Resource: {simulation_item.resource}, Element: {simulation_item.process_element}")
        resource = simulation_item.resource
        process_element = simulation_item.process_element

        next_elements = self.process.complete_element(process_element)

        self.busy_resources.pop(resource, None)
        self.available_resources.add(resource)
        self.resource_available_times[resource] = self.now
        #self.busy_cases[process_element.case_id].remove(process_element)

        
        
        if not next_elements:
            self.schedule_event(self.now, SimulationItem(SimulationItemType.COMPLETE_CASE, self.now, process_element))
        
        for next_element in next_elements:
            self.activate_element(next_element)
        
        self.schedule_event(self.now, SimulationItem(SimulationItemType.ASSIGN_RESOURCES, self.now, None))


    def handle_start_task(self, simulation_item: SimulationItem):
        #print(f"[Start Task] Resource: {simulation_item.resource}, Element: {simulation_item.process_element}")
        resource = simulation_item.resource
        process_element = simulation_item.process_element
        case_id = process_element.case_id
        self.busy_resources[resource] = process_element
        processing_time = self.process.processing_time_sample(resource, process_element, self.now)
        self.task_start_times[process_element.id] = self.now
        #print(f"[Task Processing Time] Resource: {resource}, Time: {processing_time}")
        self.schedule_event(self.now + processing_time, SimulationItem(SimulationItemType.COMPLETE_TASK, self.now + processing_time, process_element, resource))
        self.event_log.log_event(method=self.process.allocation_method_name, 
                                 num_processes = len(self.process.case_types),
                                 simulation_run=self.simulation_run, 
                                 timestamp=self.now, 
                                 process=process_element.case_type,
                                 l = self.process.arrival_distributions[process_element.case_type], 
                                 status = "running", 
                                 case_id=case_id,
                                 activity=process_element.label,
                                 resource=resource.name,
                                 end_time=self.now + processing_time)

    def handle_complete_case(self, process_element: ProcessElement):
        #print(f"[Complete Case] Case ID: {process_element.case_id}")
        case_id = process_element.case_id
        is_gateway = process_element.is_gateway
        self.finalized_cases += 1
        cycle_time = self.now - self.case_start_times[case_id]
        self.total_cycle_time += cycle_time

        if is_gateway:
            status='gateway'
        else:
            status='COMPLETE'
            #print('Final step. Case id: ', case_id, ' - Label: ', process_element.label)

            if hasattr(self.process, "handle_case_completion_reward"):
                self.process.handle_case_completion_reward(process_element)

            self.event_log.log_event(method=self.process.allocation_method_name, 
                                    num_processes = len(self.process.case_types),
                                    simulation_run=self.simulation_run, 
                                    timestamp=self.now, 
                                    process=process_element.case_type,
                                    l = self.process.arrival_distributions[process_element.case_type], 
                                    status = status, 
                                    case_id=case_id, 
                                    activity=process_element.label,
                                    cycle_time=cycle_time)
        
        self.cleanup_case_state(case_id)

            
    def cleanup_case_state(self, case_id: int):
        """Remove all tracking data related to a completed case to avoid memory bloat."""

        # Remove from core simulation state
        self.case_start_times.pop(case_id, None)
        self.busy_cases.pop(case_id, None)

        # Clean process-level tracking
        if hasattr(self.process, 'case_type'):
            self.process.case_type.pop(case_id, None)
        if hasattr(self.process, 'case_data'):
            self.process.case_data.pop(case_id, None)
        if hasattr(self.process, 'completed_tasks'):
            self.process.completed_tasks.pop(case_id, None)
        if hasattr(self.process, 'last_task_completion_time'):
            self.process.last_task_completion_time.pop(case_id, None)

        # Also remove task-level postpone tracking
        if hasattr(self.process, 'task_first_ready_time'):
            keys_to_remove = [k for k in self.process.task_first_ready_time if self.get_case_id_from_task_id(k) == case_id]
            for k in keys_to_remove:
                self.process.task_first_ready_time.pop(k, None)

        if hasattr(self.process, 'task_postpone_count'):
            keys_to_remove = [k for k in self.process.task_postpone_count if self.get_case_id_from_task_id(k) == case_id]
            for k in keys_to_remove:
                self.process.task_postpone_count.pop(k, None)

        #self.process.cleanup_case_data(case_id)


    def get_case_id_from_task_id(self, task_id: int):
        return None


    def handle_assign_resources(self):
        #print("[Assign Resources]")
        #assignments = self.process.assign_resources(self.unassigned_tasks, self.available_resources)
        #start = time.time()
        assignments = self.process.assign_resources(self.unassigned_tasks, self.available_resources, self.resource_available_times)
        #print(f"[DEBUG] Open cases: {len(self.busy_cases)}")
        #print(f"[TIME] assign_resources: {time.time() - start:.4f}s")

        for task, resource in assignments:
            self.schedule_event(self.now, SimulationItem(SimulationItemType.START_TASK, self.now, task, resource))
            self.unassigned_tasks.pop(task.id, None)
            self.available_resources.remove(resource)

       

    def activate_element(self, process_element: ProcessElement):
        #print(f"[Activate Element] Element: {process_element}")
        self.busy_cases[process_element.case_id].append(process_element)
        if process_element.is_event():
            self.schedule_event(process_element.occurrence_time, SimulationItem(SimulationItemType.COMPLETE_EVENT, process_element.occurrence_time, process_element))
        elif process_element.is_task():
            self.unassigned_tasks[process_element.id] = process_element
            self.schedule_event(self.now, SimulationItem(SimulationItemType.ASSIGN_RESOURCES, self.now, None))

    # def sort_events(self) -> None:
    #     """First start tasks (i.e. use resources) before another COMPLETE_EVENT comes into action"""
    #     self.events.sort(key = lambda k : (k[0], # time
	# 								 1 if k[1].simulation_item_type == SimulationItemType.COMPLETE_EVENT else
	# 								 0)
	# 	)
