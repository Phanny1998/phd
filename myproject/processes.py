from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union
import random
import math


# -----------------------------
# Core data structures
# -----------------------------

@dataclass(frozen=True)
class Resource:
    name: str
    # (mean, std_dev); simulator samples as Exp with mean
    execution_distribution: Tuple[float, float] | None = None


@dataclass
class Task:
    # IMPORTANT: In this rework, the key in the 'tasks' dict is treated
    # as the authoritative label/identifier. This 'name' is kept only
    # for logging / human readability and does not have to match the key.
    name: str
    resources: List[Resource] = field(default_factory=list)
    next_tasks: List[str] = field(default_factory=list)


class Gateway:
    """
    A BPMN-like gateway. We support:
      - XOR (probabilistic split using data_options[conditions[0]] as weights)
      - AND (split) via .next_tasks
      - AND-join via .merge_from (all listed labels must have completed)
    """
    def __init__(
        self,
        name: str,
        gateway_type: str,
        conditions: Optional[List[str]] = None,
        next_tasks: Optional[List[str]] = None,
        merge_from: Optional[List[str]] = None,
    ):
        self.name = name
        self.gateway_type = gateway_type  # "AND" or "XOR"
        self.conditions = conditions or []
        self.next_tasks = next_tasks or []     # for split
        self.merge_from = merge_from or []     # for join


@dataclass
class ProcessStructure:
    name: str
    arrival_distribution: float | None = None
    # e.g. {"qc_after_moulding": {"MOULDING":0.05,"ASSEMBLY_1":0.95}, ...}
    data_options: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Map of label -> (Task | Gateway)
    tasks: Dict[str, Union[Task, Gateway]] = field(default_factory=dict)


class ProcessElementType(Enum):
    TASK = auto()
    EVENT = auto()


class ProcessElement:
    """
    An element (task or event) belonging to a case instance.
    """
    def __init__(
        self,
        case_id: int,
        case_type: str,
        process_element_id: int,
        label: str,
        process_element_type: ProcessElementType,
        is_gateway: bool = False,
        occurrence_time: Optional[float] = None,
    ):
        self.id = process_element_id
        self.case_id = case_id
        self.label = label            # MUST equal the key in ProcessStructure.tasks
        self.case_type = case_type
        self.is_gateway = is_gateway
        self.process_element_type = process_element_type  # task or event
        self.data: Dict = dict()
        self.occurrence_time = occurrence_time

        if self.is_event() and self.occurrence_time is None:
            raise ValueError("The occurrence time of an event must be set.")
        if self.is_task() and self.occurrence_time is not None:
            raise ValueError("The occurrence time of a task must not be set.")

    def is_event(self) -> bool:
        return self.process_element_type == ProcessElementType.EVENT

    def is_task(self) -> bool:
        return self.process_element_type == ProcessElementType.TASK

    def __str__(self) -> str:
        return (
            'label: ' + self.label +
            " | case_type: " + self.case_type +
            " case_id:" + str(self.case_id) +
            " | id: " + str(self.id) +
            " | " + (str(self.data) if len(self.data) > 0 else "")
        )


# -----------------------------
# Abstract process base class
# -----------------------------

class Process:
    def __init__(self) -> None:
        self.simulator = None
        self.resources: List[Resource] = []
        self.case_types: List[str] = []

        self.can_plan: Dict[int, List[str]] = dict()
        self.next_case_id: int = 0
        self.next_case_arrival_time: Dict[str, float] = dict()
        self.next_element_id: int = 0
        self.case_type: Dict[int, str] = dict()
        self.case_data: Dict[int, Dict] = dict()

    def add_data(self, process_element: ProcessElement, data: Dict):
        process_element.data = data
        if process_element.case_id not in self.case_data:
            self.case_data[process_element.case_id] = data
        else:
            self.case_data[process_element.case_id].update(data)

    def get_unique_element_id(self) -> int:
        unique_id = self.next_element_id
        self.next_element_id += 1
        return unique_id

    def set_simulator(self, simulator):
        self.simulator = simulator

    def on_element_completed(self, process_element: ProcessElement):
        pass


# -----------------------------
# Concrete process: SimpleProcess
# -----------------------------

class SimpleProcess(Process):
    def __init__(self, processes: List[ProcessStructure]):
        super().__init__()
        self.process_definitions: Dict[str, ProcessStructure] = {
            process.name: process for process in processes
        }
        self.case_types = list(self.process_definitions.keys())
        self.case_data_options: Dict[str, Dict[str, Dict[str, float]]] = {
            process.name: process.data_options for process in processes
        }
        self.arrival_distributions: Dict[str, float] = {
            process.name: process.arrival_distribution for process in processes
        }

        # Build a canonical resource set (unique by resource name)
        self._resource_by_name: Dict[str, Resource] = self._extract_resources_unique()
        self.resources = list(self._resource_by_name.values())

        # Replace every Task.resources list to reference the canonical Resource objects
        self._canonicalize_task_resources()

        self.completed_tasks: Dict[int, set] = dict()
        self.reset_all_process_parameter()

    # --- internals: resources ---

    def _extract_resources_unique(self) -> Dict[str, Resource]:
        """Collect unique resources across all processes by name."""
        seen: Dict[str, Resource] = {}
        for proc in self.process_definitions.values():
            for node in proc.tasks.values():
                if isinstance(node, Task):
                    for r in node.resources:
                        if r.name not in seen:
                            seen[r.name] = r
        return seen

    def _canonicalize_task_resources(self) -> None:
        """Rewrite every Task.resources to point to the single canonical Resource per name."""
        for proc in self.process_definitions.values():
            for label, node in proc.tasks.items():
                if isinstance(node, Task) and node.resources:
                    canon_list: List[Resource] = []
                    for r in node.resources:
                        canon = self._resource_by_name.get(r.name)
                        if canon is None:
                            # Shouldn't happen, but keep safe
                            self._resource_by_name[r.name] = r
                            canon = r
                        canon_list.append(canon)
                    node.resources = canon_list

    # --- resets/arrivals ---

    def reset_all_process_parameter(self):
        self.can_plan = dict()
        self.next_case_id = 0
        self.next_case_arrival_time = {
            ct: self.get_next_case_arrival_time(ct, True) for ct in self.case_types
        }
        self.next_element_id = 0
        self.case_type = dict()
        self.case_data = dict()
        self.completed_tasks = dict()
        # NEW: true station-arrival timestamps for tasks
        self.task_first_ready_time = dict()
        try:
            self.last_task_completion_time = dict()
        except Exception:
            pass

    def get_next_case_arrival_time(self, case_type: str, is_first_arrival: bool = False) -> float:
        if case_type in self.arrival_distributions:
            lam = self.arrival_distributions[case_type]
            return random.expovariate(lam)
        else:
            raise ValueError(f"Unknown case type: {case_type}")

    # --- case generation ---

    def next_case(self) -> Tuple[float, ProcessElement]:
        case_type, arrival_time, case_id = self.get_next_case_type()
        process_structure = self.process_definitions[case_type]
        start_task_label = "START"
        start_task = process_structure.tasks[start_task_label]  # Task

        # NOTE: label uses the TASK KEY ("START"), not Task.name
        initial_process_element = ProcessElement(
            case_id=case_id,
            case_type=case_type,
            process_element_id=self.get_unique_element_id(),
            label=start_task_label,
            process_element_type=ProcessElementType.EVENT,
            occurrence_time=arrival_time,
        )
        self.add_data(initial_process_element, self.data_sample(initial_process_element))
        return arrival_time, initial_process_element

    def get_next_case_type(self) -> Tuple[str, float, int]:
        next_case_type = min(self.next_case_arrival_time, key=self.next_case_arrival_time.get)
        arrival_time = self.next_case_arrival_time[next_case_type]
        case_id = self.next_case_id
        self.next_case_arrival_time[next_case_type] += self.get_next_case_arrival_time(next_case_type)
        self.next_case_id += 1
        self.case_type[case_id] = next_case_type
        return next_case_type, arrival_time, case_id

    def generate_case_data(self, data_options: Dict[str, Dict[str, float]]) -> Dict:
        case_data: Dict[str, str] = {}
        for key, value_probs in data_options.items():
            if key.startswith("qc_") or key == "moulding_lane":
                continue
            values = list(value_probs.keys())
            probs = list(value_probs.values())
            case_data[key] = random.choices(values, weights=probs, k=1)[0]
        return case_data

    def data_sample(self, process_element: ProcessElement) -> Dict:
        try:
            case_data = self.generate_case_data(
                self.case_data_options[process_element.case_type]
            )
        except Exception:
            case_data = {}
        return case_data

    # --- flow logic ---

    def complete_element(self, process_element: ProcessElement) -> List[ProcessElement]:
        simulator_time = self.simulator.now
        process_structure = self.process_definitions[process_element.case_type]
        case_id = process_element.case_id

        if case_id not in self.completed_tasks:
            self.completed_tasks[case_id] = set()
        self.completed_tasks[case_id].add(process_element.label)

        # LOOKUP BY KEY (label), not by Task.name / Gateway.name
        current_node = process_structure.tasks.get(process_element.label)
        if current_node is None:
            raise KeyError(
                f"Unknown task/gateway label '{process_element.label}' in process '{process_element.case_type}'"
            )

        next_elements: List[ProcessElement] = []

        # Gateways
        if isinstance(current_node, Gateway):
            # AND-join
            if current_node.gateway_type == "AND" and current_node.merge_from:
                if all(t in self.completed_tasks[case_id] for t in current_node.merge_from):
                    for next_label in current_node.next_tasks:
                        next_node = process_structure.tasks[next_label]
                        is_gateway = isinstance(next_node, Gateway)
                        next_type = ProcessElementType.EVENT if is_gateway else (
                            ProcessElementType.TASK if getattr(next_node, "resources", None) else ProcessElementType.EVENT
                        )
                        next_elements.append(
                            ProcessElement(
                                case_id=case_id,
                                case_type=process_element.case_type,
                                process_element_id=self.get_unique_element_id(),
                                label=next_label,
                                process_element_type=next_type,
                                is_gateway=is_gateway,
                                occurrence_time=simulator_time if next_type == ProcessElementType.EVENT else None,
                            )
                        )

            elif current_node.gateway_type == "XOR":
                # Deterministic-if-available, else probabilistic split using data_options
                if not current_node.conditions:
                    raise ValueError(f"XOR gateway '{process_element.label}' missing conditions[]")

                attr = current_node.conditions[0]
                case_data_for_case = self.case_data.get(case_id, {})
                is_qc_key = attr.startswith("qc_")

                # If case data already specifies the route and it's valid, honor it.
                if (not is_qc_key
                    and attr in case_data_for_case 
                    and case_data_for_case[attr] in current_node.next_tasks):
                    chosen_label = case_data_for_case[attr]
                else:
                    distribution = process_structure.data_options.get(attr)
                    if distribution is None:
                        raise KeyError(
                            f"XOR gateway '{process_element.label}' expects data_options['{attr}'] not found."
                        )
                    try:
                        weights = [distribution[label] for label in current_node.next_tasks]
                    except KeyError as e:
                        raise KeyError(
                            f"XOR gateway '{process_element.label}' has next task '{e.args[0]}' "
                            f"not present in data_options['{attr}']."
                        )
                    chosen_label = random.choices(current_node.next_tasks, weights=weights, k=1)[0]

                next_node = process_structure.tasks[chosen_label]
                is_gateway = isinstance(next_node, Gateway)
                next_type = (
                    ProcessElementType.EVENT if is_gateway else
                    (ProcessElementType.TASK if getattr(next_node, "resources", None) else ProcessElementType.EVENT)
                )
                next_elements.append(
                    ProcessElement(
                        case_id=case_id,
                        case_type=process_element.case_type,
                        process_element_id=self.get_unique_element_id(),
                        label=chosen_label,
                        process_element_type=next_type,
                        is_gateway=is_gateway,
                        occurrence_time=simulator_time if next_type == ProcessElementType.EVENT else None,
                    )
                )

            elif current_node.gateway_type == "AND":
                # AND-split
                for next_label in current_node.next_tasks:
                    next_node = process_structure.tasks[next_label]
                    is_gateway = isinstance(next_node, Gateway)
                    next_type = ProcessElementType.EVENT if is_gateway else (
                        ProcessElementType.TASK if getattr(next_node, "resources", None) else ProcessElementType.EVENT
                    )
                    next_elements.append(
                        ProcessElement(
                            case_id=case_id,
                            case_type=process_element.case_type,
                            process_element_id=self.get_unique_element_id(),
                            label=next_label,
                            process_element_type=next_type,
                            is_gateway=is_gateway,
                            occurrence_time=simulator_time if next_type == ProcessElementType.EVENT else None,
                        )
                    )

        # Normal task or event (label exists but not a Gateway)
        else:
            for next_label in current_node.next_tasks:
                next_node = process_structure.tasks[next_label]
                is_gateway = isinstance(next_node, Gateway)
                next_type = ProcessElementType.EVENT if is_gateway else (
                    ProcessElementType.TASK if getattr(next_node, "resources", None) else ProcessElementType.EVENT
                )
                next_elements.append(
                    ProcessElement(
                        case_id=process_element.case_id,
                        case_type=process_element.case_type,
                        process_element_id=self.get_unique_element_id(),
                        label=next_label,
                        process_element_type=next_type,
                        is_gateway=is_gateway,
                        occurrence_time=simulator_time if next_type == ProcessElementType.EVENT else None,
                    )
                )

        self.on_element_completed(process_element)
        return next_elements

    # --- sampling ---

    def processing_time_sample(self, resource: Resource, process_element: ProcessElement, simulation_time: float) -> float:
        """
        Sample service time for the given task-resource pair.
        Currently uses an exponential distribution with the task/resource 'mean'.
        """
        try:
            proc = self.process_definitions[process_element.case_type]
            node = proc.tasks[process_element.label]
            if not isinstance(node, Task):
                raise ValueError(f"processing_time_sample called on non-Task label '{process_element.label}'")

            # Find THIS task's distribution for THIS resource name
            for r in node.resources:
                if r.name == resource.name:
                    mean, std_dev = r.execution_distribution
                    # Exponential with mean
                    return random.expovariate(1.0 / mean)

            raise ValueError(f"No distribution found for resource '{resource.name}' on task '{process_element.label}'")
        except Exception as e:
            print(f"[WARN] processing_time_sample error: {e}; returning 0.0")
            return 0.0

