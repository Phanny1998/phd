from framework.processes import ProcessStructure, Task, Resource
from framework.processes import ProcessStructure, Task, Resource, Gateway
import os
import random

def process_function(l:float, scenario:str):
    if scenario == 'bpi2020_2processes_massive_share':
        processes = [
            ProcessStructure(
                name="1_payment_request",
                arrival_distribution=l,
                data_options={
                    "1_approval_flow": {"1_submit_request": 0.15, "1_check_budget_approval": 0.85},
                    "1_budget_check": {"1_budget_approve": 0.35, "1_supervisor_approve": 0.65},
                    "1_final_payment_check": {"1_handle_payment": 0.01, "1_END": 0.99}
                },
                tasks={
                    "START": Task(name="1_start", next_tasks=["1_submit_request"]),

                    # ---- Process 1 (FASTER process) ----
                    # Early tasks: dedicated ≈ 0.95–1.10, X_SHARED = 0.75  (very attractive)
                    "1_submit_request": Task(name="1_submit_request", resources=[
                        Resource("1employee", (0.95, 0.1)),
                        Resource("1employee1", (0.75, 0.1)),
                    ], next_tasks=["1_check_request"]),

                    "1_check_request": Task(name="1_check_request", resources=[
                        Resource("112admin0", (1.10, 0.1)),
                        Resource("12admin1", (1.20, 0.1)),
                        Resource("1X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["1_XOR_admin_decision"]),

                    "1_XOR_admin_decision": Gateway(name="1_XOR_admin_decision", gateway_type="XOR",
                                                    next_tasks=["1_submit_request", "1_check_budget_approval"],
                                                    conditions=["1_approval_flow"]),

                    "1_check_budget_approval": Task(name="1_check_budget_approval", resources=[
                        Resource("12admin0", (1, 0.1)),
                        Resource("112admin1", (1.10, 0.1)),
                        Resource("1X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["1_XOR_budget_check"]),

                    "1_XOR_budget_check": Gateway(name="1_XOR_budget_check", gateway_type="XOR",
                                                next_tasks=["1_budget_approve", "1_supervisor_approve"],
                                                conditions=["1_budget_check"]),

                    "1_budget_approve": Task(name="1_budget_approve", resources=[
                        Resource("1budget_owner1", (1.05, 0.1)),
                        Resource("1budget_owner2", (1.2, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["1_supervisor_approve"]),

                    "1_supervisor_approve": Task(name="1_supervisor_approve", resources=[
                        Resource("1supervisor1", (1.1, 0.1)),
                        Resource("1supervisor2", (1.20, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["1_handle_payment"]),

                    # Final task: dedicated VERY SLOW (3.2/3.8), X_SHARED = 1.25 (best but not the fastest overall)
                    "1_handle_payment": Task(name="1_handle_payment", resources=[
                        Resource("12accounting0", (3.20, 0.1)),
                        Resource("12accounting1", (3.80, 0.1)),
                        Resource("X_SHARED", (1., 0.1)),
                    ], next_tasks=["1_XOR_payment_check"]),

                    "1_XOR_payment_check": Gateway(name="1_XOR_payment_check", gateway_type="XOR",
                                                next_tasks=["1_handle_payment", "1_END"],
                                                conditions=["1_final_payment_check"]),

                    "1_END": Task(name="1_END", next_tasks=[])
                }
            ),

            ProcessStructure(
                name="2_payment_request",
                arrival_distribution=l,
                data_options={
                    "2_approval_flow": {"2_submit_request": 0.15, "2_check_budget_approval": 0.85},
                    "2_budget_check": {"2_budget_approve": 0.35, "2_supervisor_approve": 0.65},
                    "3_final_payment_check": {"2_handle_payment": 0.01, "2_END": 0.99}
                },
                tasks={
                    "START": Task(name="2_start", next_tasks=["2_submit_request"]),

                    # ---- Process 2 (SLOWER process) ----
                    # Early tasks: dedicated ≈ 1.30–1.50, X_SHARED = 0.95 (still attractive)
                    "2_submit_request": Task(name="2_submit_request", resources=[
                        Resource("2employee", (1.30, 0.1)),
                        Resource("2employee1", (0.95, 0.1)),
                    ], next_tasks=["2_check_request"]),

                    "2_check_request": Task(name="2_check_request", resources=[
                        Resource("1212admin0", (1.30, 0.1)),
                        Resource("1212admin1", (1.60, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["2_XOR_admin_decision"]),

                    "2_XOR_admin_decision": Gateway(name="2_XOR_admin_decision", gateway_type="XOR",
                                                    next_tasks=["2_submit_request", "2_check_budget_approval"],
                                                    conditions=["2_approval_flow"]),

                    "2_check_budget_approval": Task(name="2_check_budget_approval", resources=[
                        Resource("212admin0", (1.30, 0.1)),
                        Resource("212admin1", (1.40, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["2_XOR_budget_check"]),

                    "2_XOR_budget_check": Gateway(name="2_XOR_budget_check", gateway_type="XOR",
                                                next_tasks=["2_budget_approve", "2_supervisor_approve"],
                                                conditions=["2_budget_check"]),

                    "2_budget_approve": Task(name="2_budget_approve", resources=[
                        Resource("21budget_owner1", (1.30, 0.1)),
                        Resource("21budget_owner2", (1.5, 0.1)),
                        Resource("X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["2_supervisor_approve"]),

                    "2_supervisor_approve": Task(name="2_supervisor_approve", resources=[
                        Resource("21supervisor1", (1.30, 0.1)),
                        Resource("21supervisor2", (1.5, 0.1)),
                        Resource("1X_SHARED", (0.9, 0.1)),
                    ], next_tasks=["2_handle_payment"]),

                    # Final task: dedicated EVEN SLOWER (4.4/5.0), X_SHARED = 1.25 (best)
                    "2_handle_payment": Task(name="2_handle_payment", resources=[
                        Resource("212accounting0", (4.40, 0.1)),
                        Resource("212accounting1", (3.00, 0.1)),
                        Resource("1X_SHARED", (1, 0.1)),
                    ], next_tasks=["2_XOR_payment_check"]),

                    "2_XOR_payment_check": Gateway(name="2_XOR_payment_check", gateway_type="XOR",
                                                next_tasks=["2_handle_payment", "2_END"],
                                                conditions=["3_final_payment_check"]),

                    "2_END": Task(name="2_END", next_tasks=[])
                }
            ),
        ]



    if scenario=='simple2XY':
        processes = [
            ProcessStructure(
                name="process_b",
                arrival_distribution=l,
                tasks={
                    "START": Task(name="B_RECEIVE_EVENT", next_tasks=["B0_TASK"]),
                    "B0_TASK": Task(name="B0_TASK",
                                    resources=[Resource(name="INTERN_B0_1",execution_distribution=(1.1, 0.2)),
                                               Resource(name="EXTERN_AB_1234",execution_distribution=(1, 1/8))],
                                            #Resource(name="INTERN_B0_2",execution_distribution=(1.6, 0.2))], 
                                    next_tasks=["B1_TASK"]),
                    "B1_TASK": Task(name="B1_TASK", 
                                    resources=[Resource(name="INTERN_B1_1",execution_distribution=(1.6, 1/2)),
                                            Resource(name="EXTERN_AB_1234",execution_distribution=(1, 1/2))], 
                                    next_tasks=["B_END_EVENT"]),
                    "B_END_EVENT": Task(name="B_END_EVENT", next_tasks=[])
                }
            ),
            ProcessStructure(
                name="process_a",
                arrival_distribution=l,
                tasks={
                    "START": Task(name="A_RECEIVE_EVENT", next_tasks=["A0_TASK"]),
                    "A0_TASK": Task(name="A0_TASK",
                                    resources=[Resource(name="INTERN_A0_1",execution_distribution=(1.1 , 1/2)),
                                               Resource(name="EXTERN_AB_1234",execution_distribution=(1, 1/8))],
                                            #Resource(name="INTERN_A0_2",execution_distribution=(1.4, 0.2))], 
                                    next_tasks=["A1_TASK"]),
                    "A1_TASK": Task(name="A1_TASK", 
                                    resources=[Resource(name="INTERN_A1_1",execution_distribution=(1.6, 1/8)),
                                            Resource(name="EXTERN_AB_1234",execution_distribution=(1, 1/8))], 
                                    next_tasks=["A_END_EVENT"]),
                    "A_END_EVENT": Task(name="A_END_EVENT", next_tasks=[])
                }
            ),     
        ]
    
    if scenario=='simple2XY_fast_slow':
            processes = [
                ProcessStructure(
                    name="process_b",
                    arrival_distribution=l,
                    tasks={
                        "START": Task(name="B_RECEIVE_EVENT", next_tasks=["B0_TASK"]),
                        "B0_TASK": Task(name="B0_TASK",
                                        resources=[Resource(name="INTERN_B0_1",execution_distribution=(1.1, 0.2)),#1.1, 0.2
                                                Resource(name="EXTERN_AB_1234",execution_distribution=(1, 1/8))],#1, 1/8
                                                #Resource(name="INTERN_B0_2",execution_distribution=(1.6, 0.2))], 
                                        next_tasks=["B1_TASK"]),
                        "B1_TASK": Task(name="B1_TASK", 
                                        resources=[Resource(name="INTERN_B1_1",execution_distribution=(1.6, 1/2)),#1.6, 1/2
                                                Resource(name="EXTERN_AB_1234",execution_distribution=(1, 1/2))], #1.2, 1/2
                                        next_tasks=["B_END_EVENT"]),
                        "B_END_EVENT": Task(name="B_END_EVENT", next_tasks=[])
                    }
                ),
                ProcessStructure(
                    name="process_a",
                    arrival_distribution=l,
                    tasks={
                        "START": Task(name="A_RECEIVE_EVENT", next_tasks=["A0_TASK"]),
                        "A0_TASK": Task(name="A0_TASK",
                                        resources=[Resource(name="INTERN_A0_1",execution_distribution=(2.1 , 1/2)),#1.6 , 1/2
                                                Resource(name="EXTERN_AB_1234",execution_distribution=(1, 1/8))],#1.5, 1/8
                                                #Resource(name="INTERN_A0_2",execution_distribution=(1.4, 0.2))], 
                                        next_tasks=["A1_TASK"]),
                        "A1_TASK": Task(name="A1_TASK", 
                                        resources=[Resource(name="INTERN_A1_1",execution_distribution=(2.6, 1/8)),#2.1, 1/8
                                                Resource(name="EXTERN_AB_1234",execution_distribution=(1, 1/8))], #1.7, 1/8
                                        next_tasks=["A_END_EVENT"]),
                        "A_END_EVENT": Task(name="A_END_EVENT", next_tasks=[])
                    }
                ), ]
    
    
    if scenario=='simple0':
        processes = [
            ProcessStructure(
                name="process_b",
                arrival_distribution=l,
                tasks={
                    "START": Task(name="B_RECEIVE_EVENT", next_tasks=["B0_TASK"]),
                    "B0_TASK": Task(name="B0_TASK",
                                    resources=[Resource(name="INTERN_B0_1",execution_distribution=(1.4, 0.2)),
                                               Resource(name="INTERN_B0_2",execution_distribution=(1.1, 1/8))],
                                            #Resource(name="INTERN_B0_2",execution_distribution=(1.6, 0.2))], 
                                    next_tasks=["B1_TASK"]),
                    "B1_TASK": Task(name="B1_TASK", 
                                    resources=[Resource(name="INTERN_B1_1",execution_distribution=(1.5, 1/2)),
                                            Resource(name="INTERN_B1_2",execution_distribution=(1.2, 1/2))], 
                                    next_tasks=["B_END_EVENT"]),
                    "B_END_EVENT": Task(name="B_END_EVENT", next_tasks=[])
                }
            ),
            ProcessStructure(
                name="process_a",
                arrival_distribution=l,
                tasks={
                    "START": Task(name="A_RECEIVE_EVENT", next_tasks=["A0_TASK"]),
                    "A0_TASK": Task(name="A0_TASK",
                                    resources=[Resource(name="INTERN_A0_1",execution_distribution=(1.5 , 1/2)),
                                               Resource(name="INTERN_A0_2",execution_distribution=(1.2, 1/8))],
                                            #Resource(name="INTERN_A0_2",execution_distribution=(1.4, 0.2))], 
                                    next_tasks=["A1_TASK"]),
                    "A1_TASK": Task(name="A1_TASK", 
                                    resources=[Resource(name="INTERN_A1_1",execution_distribution=(1.4, 1/8)),
                                            Resource(name="INTERN_A1_2",execution_distribution=(1.1, 1/8))], 
                                    next_tasks=["A_END_EVENT"]),
                    "A_END_EVENT": Task(name="A_END_EVENT", next_tasks=[])
                }
            ), 
             
        ]

    if scenario=='simple1':
        processes = [
            ProcessStructure(
                name="process_b",
                arrival_distribution=l,
                tasks={
                    "START": Task(name="B_RECEIVE_EVENT", next_tasks=["B0_TASK"]),
                    "B0_TASK": Task(name="B0_TASK",
                                    resources=[Resource(name="INTERN_B0_1",execution_distribution=(1.4, 0.2)),
                                               Resource(name="EXPERN_AB0",execution_distribution=(1.1, 1/8))],
                                            #Resource(name="INTERN_B0_2",execution_distribution=(1.6, 0.2))], 
                                    next_tasks=["B1_TASK"]),
                    "B1_TASK": Task(name="B1_TASK", 
                                    resources=[Resource(name="INTERN_B1_1",execution_distribution=(1.5, 1/2)),
                                            Resource(name="EXPERN_AB1",execution_distribution=(1.2, 1/2))], 
                                    next_tasks=["B_END_EVENT"]),
                    "B_END_EVENT": Task(name="B_END_EVENT", next_tasks=[])
                }
            ),
            ProcessStructure(
                name="process_a",
                arrival_distribution=l,
                tasks={
                    "START": Task(name="A_RECEIVE_EVENT", next_tasks=["A0_TASK"]),
                    "A0_TASK": Task(name="A0_TASK",
                                    resources=[Resource(name="INTERN_A0_1",execution_distribution=(1.5 , 1/2)),
                                               Resource(name="EXPERN_AB0",execution_distribution=(1.2, 1/8))],
                                            #Resource(name="INTERN_A0_2",execution_distribution=(1.4, 0.2))], 
                                    next_tasks=["A1_TASK"]),
                    "A1_TASK": Task(name="A1_TASK", 
                                    resources=[Resource(name="INTERN_A1_1",execution_distribution=(1.4, 1/8)),
                                            Resource(name="EXPERN_AB1",execution_distribution=(1.1, 1/8))], 
                                    next_tasks=["A_END_EVENT"]),
                    "A_END_EVENT": Task(name="A_END_EVENT", next_tasks=[])
                }
            ), 
             
        ]

    return processes
    

# scenario 1-4
ARRIVAL_RATES=[.2,.4,.6,.8]
SCENARIO_NAMES =['simple0', 'simple1','simple2XY','simple2XY_fast_slow']
SIMULATION_RUN_TIME = 2500
SIMULATION_RUNS = 50

# Scenario 5
ARRIVAL_RATES=[.2]
SCENARIO_NAMES =['bpi2020_2processes_massive_share']
SIMULATION_RUN_TIME = 500
SIMULATION_RUNS = 1
