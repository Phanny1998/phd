from typing import Dict, List, Optional
from dataclasses import dataclass, field
import random
import math

@dataclass(frozen=True) # Add frozen=True to make the dataclass immutable
class Resource:
    name: str
    execution_distribution: tuple[float,float] = None

@dataclass
class Task:
    name: str
    resources: List[Resource] = field(default_factory=list) 
    next_tasks: List[str] = field(default_factory=list) 

@dataclass
class ProcessStructure:
    name: str
    arrival_distribution: float = None
    data_options: Dict[str, Dict[str,int]] = field(default_factory=list) 
    tasks: Dict[str, Task] = field(default_factory=list) 

from enum import Enum, auto, StrEnum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import random

class Gateway:
    def __init__(self, name, gateway_type, conditions=None, next_tasks=None, merge_from=None):
        self.name = name
        self.gateway_type = gateway_type  # "AND", "XOR", etc.
        self.conditions = conditions 
        self.next_tasks = next_tasks or []  # For split
        self.merge_from = merge_from or []  # For join


class ProcessElementType(Enum):
    TASK = auto()
    EVENT = auto()

class ProcessElement:
    """
    An element is either a task or an event of a process.
    An element is part of a case.
    An element has a:
    - case_id: the id of the case to which the element belongs.
    - case_type: the type of the case to which the element belongs.
    - process_element_id: the id of the element.
    - label: the label of the element.
    - process_element_type: the type of the element, which is either a task or an event.
    - data: a dictionary of data that is associated with the element; the dictionary keys are the data types and the values are the data values.
    - occurrence_time: the time when the event should occur in absolute simulation time; used for events only and must be set for each event.
    """
    def __init__(self, case_id, case_type, process_element_id, label, process_element_type,is_gateway=False, occurrence_time=None):
        self.id = process_element_id
        self.case_id = case_id
        self.label = label
        self.case_type = case_type
        self.is_gateway = is_gateway
        self.process_element_type = process_element_type # task or event
        self.data = dict()
        self.occurrence_time = occurrence_time  # used for time-based events only, represents the time when the event should occur
        if self.is_event() and self.occurrence_time is None:
            raise ValueError("The occurrence time of an event must be set.")
        if self.is_task() and self.occurrence_time is not None:
            raise ValueError("The occurrence time of a task must not be set.")

    def is_event(self):
        return self.process_element_type == ProcessElementType.EVENT
    
    def is_task(self):
        return self.process_element_type == ProcessElementType.TASK

    def __str__(self):
        return 'label: ' + self.label + " | case_type: " + self.case_type + " case_id:" + str(self.case_id) + " | id: " + str(self.id) + " | "+ (str(self.data) if len(self.data) > 0 else "")

class Process(ABC):
    def __init__(self) -> None:
        self.simulator = None
        self.resources = []  # a list of resources that can be used in the problem.
        self.case_types = []  # a list of case types that can be generated in the problem.
        
        self.can_plan = dict()  # a dictionary of case_id to list of element labels that can be planned for that case.
        self.next_case_id = 0  # the id of the next case to be generated.
        self.next_case_arrival_time = dict()  # a dictionary of case type to the time of the next case arrival of that type.
        self.next_element_id = 0  # the id of the next element to be generated.
        self.case_type = dict()  # a dictionary of case_id to case_type
        self.case_data = dict()  # a dictionary of case_id to data, where data is a dictionary of data types to data values

    def add_data(self, process_element: ProcessElement, data):
        """
        Adds the given data to the given process_element and updates the case data as well.
        This method should be used to add data to the process_element. Data should not be added directly to the process_element, otherwise the case data will not be updated.
        """
        process_element.data = data
        if process_element.case_id not in self.case_data:
            self.case_data[process_element.case_id] = data
        else:
            self.case_data[process_element.case_id].update(data)
    
    def get_unique_element_id(self):
        unique_id = self.next_element_id
        self.next_element_id += 1
        return unique_id

    def set_simulator(self, simulator):
        self.simulator = simulator

    def on_element_completed(self, process_element: ProcessElement):
        """
        Hook method called when an element is completed.
        Subclasses can override this method to add custom behavior.
        """
        pass

class SimpleProcess(Process):
    def __init__(self, processes):
        super().__init__()
        self.process_definitions = {process.name: process for process in processes}
        self.case_types = list(self.process_definitions.keys())
        self.case_data_options = {process.name: process.data_options for process in processes}
        self.arrival_distributions = {process.name: process.arrival_distribution for process in processes}
        self.resources = self._extract_resources()
        self.completed_tasks = dict()  # case_id -> set of completed task names
        self.reset_all_process_parameter()
        

    
    def _extract_resources(self):
        resources = {}
        for process in self.process_definitions.values():
            for task in process.tasks.values():
                if isinstance(task, Task):
                    for res in task.resources:
                        if res.name not in resources:
                            resources[res.name] = res
        return list(resources.values())
    
    def reset_all_process_parameter(self):
        self.can_plan = dict()
        self.next_case_id = 0
        self.next_case_arrival_time = {ct: self.get_next_case_arrival_time(ct, True) for ct in self.case_types}
        self.next_element_id = 0
        self.case_type = dict()
        self.case_data = dict()
        self.completed_tasks = dict()
        try:
            self.last_task_completion_time = dict()
        except:
            pass

    def get_next_case_arrival_time(self, case_type, is_first_arrival=False):
        if case_type in self.arrival_distributions:
            parameter = self.arrival_distributions[case_type]
            return random.expovariate(parameter)
        else:
            raise ValueError(f"Unknown case type: {case_type}")
        
    # def get_next_case_arrival_time(self, case_type, is_first_arrival=False):
        
    #     def get_lambda(t):
    #         base_rate = 0.25# for all scenarios
    #         amplitude = 0.19
    #         period = 800  # controls how often it cycles
    #         return base_rate + amplitude * math.sin(2 * math.pi * t / period)
    #     if is_first_arrival:
    #         current_time=0    
    #     else:
    #         current_time = self.simulator.now
    #     lam = get_lambda(current_time)
    #     return random.expovariate(lam)
        
        
    def next_case(self):
        case_type, arrival_time, case_id = self.get_next_case_type()
        process_structure = self.process_definitions[case_type]
        start_task_label = "START"
        start_task = process_structure.tasks[start_task_label]
        
        initial_process_element = ProcessElement(
            case_id,
            case_type,
            self.get_unique_element_id(),
            start_task.name,
            ProcessElementType.EVENT,
            occurrence_time=arrival_time
        )
        self.add_data(initial_process_element, self.data_sample(initial_process_element))
        return arrival_time, initial_process_element
    
    def get_next_case_type(self):
        next_case_type = min(self.next_case_arrival_time, key=self.next_case_arrival_time.get)
        arrival_time = self.next_case_arrival_time[next_case_type]
        case_id = self.next_case_id
        self.next_case_arrival_time[next_case_type] += self.get_next_case_arrival_time(next_case_type)
        self.next_case_id += 1
        self.case_type[case_id] = next_case_type
        return next_case_type, arrival_time, case_id
    
    def generate_case_data(self, data_options):
        case_data = {}
        for key, value_probs in data_options.items():
            values = list(value_probs.keys())
            probs = list(value_probs.values())
            case_data[key] = random.choices(values, weights=probs, k=1)[0]
        return case_data
    
    def data_sample(self, process_element):
        try:
            case_data = self.generate_case_data(self.case_data_options[process_element.case_type])
        except:
            case_data = {}
        return case_data
    

    def complete_element(self, process_element):
        simulator_time = self.simulator.now
        process_structure = self.process_definitions[process_element.case_type]
        case_id = process_element.case_id
        tasks = process_structure.tasks
        #case_data = self.case_data[case_id]

        if case_id not in self.completed_tasks:
            self.completed_tasks[case_id] = set()
        self.completed_tasks[case_id].add(process_element.label)

        current_task = next((task for task in process_structure.tasks.values() if task.name == process_element.label), None)
        #current_task = process_structure.tasks.get(process_element.label)

        next_elements = []
        # If current element is a Gateway
        if isinstance(current_task, Gateway):
            if current_task.gateway_type == "AND" and current_task.merge_from:
                # This is an AND-join
                if all(t in self.completed_tasks[case_id] for t in current_task.merge_from):
                    # All required branches completed, proceed
                    for next_task_label in current_task.next_tasks:
                        next_task = tasks[next_task_label]
                        if isinstance(next_task, Gateway):
                            next_element_type = ProcessElementType.EVENT
                            is_gateway = True
                        else:
                            next_element_type = ProcessElementType.TASK if next_task.resources else ProcessElementType.EVENT
                            is_gateway = False
                        next_elements.append(ProcessElement(
                            case_id,
                            process_element.case_type,
                            self.get_unique_element_id(),
                            next_task_label,
                            next_element_type,
                            is_gateway,
                            occurrence_time=simulator_time if next_element_type == ProcessElementType.EVENT else None
                        ))
            # elif current_task.gateway_type == "XOR":
            #     for condition, next_task_label in zip(current_task.conditions, current_task.next_tasks):
            #         if condition(case_data):
            #             next_task = tasks[next_task_label]
            #             if isinstance(next_task, Gateway):
            #                 next_element_type = ProcessElementType.EVENT
            #                 is_gateway = True
            #             else:
            #                 next_element_type = ProcessElementType.TASK if next_task.resources else ProcessElementType.EVENT
            #                 is_gateway = False
            #             next_elements.append(ProcessElement(
            #                 case_id,
            #                 process_element.case_type,
            #                 self.get_unique_element_id(),
            #                 next_task_label,
            #                 next_element_type,
            #                 is_gateway,
            #                 occurrence_time=simulator_time if next_element_type == ProcessElementType.EVENT else None
            #             ))
            #             break  # Only one branch in XOR
            elif current_task.gateway_type == "XOR":
                attribute = current_task.conditions[0]  # e.g., "priority"
                distribution = process_structure.data_options[attribute]
                
                # Sample the next task label based on probabilities
                task_choice = random.choices(
                    population=current_task.next_tasks,
                    weights=[distribution[label] for label in current_task.next_tasks],
                    k=1
                )[0]
                
                next_task = tasks[task_choice]
                if isinstance(next_task, Gateway):
                    next_element_type = ProcessElementType.EVENT
                    is_gateway = True
                else:
                    next_element_type = ProcessElementType.TASK if next_task.resources else ProcessElementType.EVENT
                    is_gateway = False
                
                next_elements.append(ProcessElement(
                    case_id,
                    process_element.case_type,
                    self.get_unique_element_id(),
                    task_choice,
                    next_element_type,
                    is_gateway,
                    occurrence_time=simulator_time if next_element_type == ProcessElementType.EVENT else None
                ))

            elif current_task.gateway_type == "AND":
                # This is an AND-split
                for next_task_label in current_task.next_tasks:
                    next_task = tasks[next_task_label]
                    if isinstance(next_task, Gateway):
                            next_element_type = ProcessElementType.EVENT
                            is_gateway = True
                    else:
                        next_element_type = ProcessElementType.TASK if next_task.resources else ProcessElementType.EVENT
                        is_gateway = False
                    next_elements.append(ProcessElement(
                        case_id,
                        process_element.case_type,
                        self.get_unique_element_id(),
                        next_task_label,
                        next_element_type,
                        is_gateway,
                        occurrence_time=simulator_time if next_element_type == ProcessElementType.EVENT else None
                    ))

        # If current element is a normal Task or Event
        else:
            for next_task_label in current_task.next_tasks: 
                next_task = process_structure.tasks[next_task_label]
                if isinstance(next_task, Gateway):
                    next_element_type = ProcessElementType.EVENT
                    is_gateway = True
                else:
                    next_element_type = ProcessElementType.TASK if next_task.resources else ProcessElementType.EVENT
                    is_gateway = False
                next_process_element = ProcessElement(
                    process_element.case_id,
                    process_element.case_type,
                    self.get_unique_element_id(),
                    next_task_label,
                    next_element_type,
                    is_gateway,
                    occurrence_time=simulator_time if next_element_type == ProcessElementType.EVENT else None
                )
                next_elements.append(next_process_element)

        self.on_element_completed(process_element)
        #print([i.label for i in next_elements])
        return next_elements



    # def processing_time_sample(self, resource, task, simulation_time):
    #     try:
    #         mean, std_dev = resource.execution_distribution
    #         #return max(0, random.normalvariate(mean, std_dev))
    #         return random.expovariate(1 / mean)
    #     except Exception as e:
    #         print(f"Error in processing_time_sample: {e}, returning default value 0.0")
    #     return 0.0
    
    def processing_time_sample(self, resource, process_element, simulation_time):
        try:
            proc = self.process_definitions[process_element.case_type]
            task_def = proc.tasks[process_element.label]
            # find the distribution for THIS task and THIS resource name
            for r in task_def.resources:
                if r.name == resource.name:
                    mean, std_dev = r.execution_distribution
                    return random.expovariate(1 / mean)
            raise ValueError(f"No distribution for resource {resource.name} on task {process_element.label}")
        except Exception as e:
            print(f"Error in processing_time_sample: {e}, returning default value 0.0")
            return 0.0

    def resources_available(self, resource, simulator_time):
        resource_type = resource.type
        return self.resource_type_available(resource_type, simulator_time)

    def resource_type_available(self, resource_type, simulator_time):
        if resource_type.startswith("INTERN"):
            # Example: Intern resources are only available during working hours
            return self.is_working_time(simulator_time)
        elif resource_type.startswith("EXTERN"):
            # Example: External resources are always available
            return True
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")

    def is_working_time(self, simulator_time):
        # Example implementation of working hours: 8 AM to 6 PM
        #hours = int(simulator_time % 24)  # Get the hour of the day
        #return 8 <= hours < 18
        return True