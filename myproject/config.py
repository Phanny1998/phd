from myproject.processes import ProcessStructure, Task, Resource, Gateway
import random

def process_function(l: float, scenario: str):
    """
    l = arrival rate (batches per time unit)
    scenario = which manufacturing setup to simulate
    """
    
    if scenario == 'actuator_manufacturing_with_rework':
        processes = []
        
        # Define all 18 product variants
        for variant_id in range(1, 19):  # 18 variants
            variant_name = f"variant_{variant_id}"
            
            # Processing times (mean, std_dev) in minutes
            # Adjust these based on your actual manufacturing data
            moulding_time = (5.0, 0.5)
            assembly1_time = (8.0, 1.0)
            assembly2_time = (6.0, 0.8)
            sorting_time = (3.0, 0.3)
            packaging_time = (4.0, 0.5)
            
            # Variant-specific adjustments (optional)
            if variant_id <= 6:  # Simple variants
                assembly1_time = (7.0, 0.8)
            elif variant_id <= 12:  # Medium complexity
                assembly1_time = (8.5, 1.0)
            else:  # Complex variants
                assembly1_time = (10.0, 1.2)
            
            process = ProcessStructure(
                name=variant_name,
                arrival_distribution=l / 18,  # Split arrival rate among 18 variants
                
                # Quality check probabilities for each station
                data_options={
                    "qc_after_moulding": {
                        "ASSEMBLY_1": 0.95,        # 95% pass → go to next step
                        "MOULDING": 0.05           # 5% fail → rework (back to moulding)
                    },
                    "qc_after_assembly1": {
                        "ASSEMBLY_2": 0.95,
                        "ASSEMBLY_1": 0.05         # Back to Assembly 1
                    },
                    "qc_after_assembly2": {
                        "SORTING": 0.95,
                        "ASSEMBLY_2": 0.05         # Back to Assembly 2
                    },
                    "qc_after_sorting": {
                        "PACKAGING": 0.95,
                        "SORTING": 0.05            # Back to Sorting
                    },
                    "qc_after_packaging": {
                        "END": 0.95,
                        "PACKAGING": 0.05          # Back to Packaging
                    }
                },
                
                tasks={
                    "START": Task(
                        name=f"{variant_name}_START",
                        next_tasks=["MOULDING"]
                    ),
                    
                    # ========================================
                    # Process 1: MOULDING (5 parallel machines)
                    # ========================================
                    "MOULDING": Task(
                        name=f"{variant_name}_MOULDING",
                        resources=[
                            Resource("MOULDING_MACHINE_1", moulding_time),
                            Resource("MOULDING_MACHINE_2", moulding_time),
                            Resource("MOULDING_MACHINE_3", moulding_time),
                            Resource("MOULDING_MACHINE_4", moulding_time),
                            Resource("MOULDING_MACHINE_5", moulding_time),
                        ],
                        next_tasks=["QC_AFTER_MOULDING"]  # Go to quality check
                    ),
                    
                    # Quality Check Gateway after Moulding
                    "QC_AFTER_MOULDING": Gateway(
                        name=f"{variant_name}_QC_AFTER_MOULDING",
                        gateway_type="XOR",
                        conditions=["qc_after_moulding"],  # References data_options
                        next_tasks=["MOULDING", "ASSEMBLY_1"]  # Rework or proceed
                    ),
                    
                    # ========================================
                    # Process 2: ASSEMBLY 1 (5 lines)
                    # ========================================
                    "ASSEMBLY_1": Task(
                        name=f"{variant_name}_ASSEMBLY_1",
                        resources=[
                            Resource("LINE_1_ASSEMBLY1", assembly1_time),
                            Resource("LINE_2_ASSEMBLY1", assembly1_time),
                            Resource("LINE_3_ASSEMBLY1", assembly1_time),
                            Resource("LINE_4_ASSEMBLY1", assembly1_time),
                            Resource("LINE_5_ASSEMBLY1", assembly1_time),
                        ],
                        next_tasks=["QC_AFTER_ASSEMBLY1"]
                    ),
                    
                    # Quality Check Gateway after Assembly 1
                    "QC_AFTER_ASSEMBLY1": Gateway(
                        name=f"{variant_name}_QC_AFTER_ASSEMBLY1",
                        gateway_type="XOR",
                        conditions=["qc_after_assembly1"],
                        next_tasks=["ASSEMBLY_1", "ASSEMBLY_2"]  # Rework or proceed
                    ),
                    
                    # ========================================
                    # Process 3: ASSEMBLY 2 (2 lines)
                    # ========================================
                    "ASSEMBLY_2": Task(
                        name=f"{variant_name}_ASSEMBLY_2",
                        resources=[
                            Resource("ASSEMBLY2_LINE_1", assembly2_time),
                            Resource("ASSEMBLY2_LINE_2", assembly2_time),
                        ],
                        next_tasks=["QC_AFTER_ASSEMBLY2"]
                    ),
                    
                    # Quality Check Gateway after Assembly 2
                    "QC_AFTER_ASSEMBLY2": Gateway(
                        name=f"{variant_name}_QC_AFTER_ASSEMBLY2",
                        gateway_type="XOR",
                        conditions=["qc_after_assembly2"],
                        next_tasks=["ASSEMBLY_2", "SORTING"]  # Rework or proceed
                    ),
                    
                    # ========================================
                    # Process 4: SORTING (1 robot)
                    # ========================================
                    "SORTING": Task(
                        name=f"{variant_name}_SORTING",
                        resources=[
                            Resource("SORTING_ROBOT", sorting_time),
                        ],
                        next_tasks=["QC_AFTER_SORTING"]
                    ),
                    
                    # Quality Check Gateway after Sorting
                    "QC_AFTER_SORTING": Gateway(
                        name=f"{variant_name}_QC_AFTER_SORTING",
                        gateway_type="XOR",
                        conditions=["qc_after_sorting"],
                        next_tasks=["SORTING", "PACKAGING"]  # Rework or proceed
                    ),
                    
                    # ========================================
                    # Process 5: PACKAGING (3 automated lines)
                    # ========================================
                    "PACKAGING": Task(
                        name=f"{variant_name}_PACKAGING",
                        resources=[
                            Resource("PACKAGING_LINE_1", packaging_time),
                            Resource("PACKAGING_LINE_2", packaging_time),
                            Resource("PACKAGING_LINE_3", packaging_time),
                        ],
                        next_tasks=["QC_AFTER_PACKAGING"]
                    ),
                    
                    # Quality Check Gateway after Packaging
                    "QC_AFTER_PACKAGING": Gateway(
                        name=f"{variant_name}_QC_AFTER_PACKAGING",
                        gateway_type="XOR",
                        conditions=["qc_after_packaging"],
                        next_tasks=["PACKAGING", "END"]  # Rework or complete
                    ),
                    
                    # ========================================
                    # END
                    # ========================================
                    "END": Task(
                        name=f"{variant_name}_END",
                        next_tasks=[]
                    )
                }
            )
            
            processes.append(process)
        
        return processes
    
    
    elif scenario == 'actuator_manufacturing_no_rework':
        """Original version without rework for comparison"""
        processes = []
        
        for variant_id in range(1, 19):
            variant_name = f"variant_{variant_id}"
            
            moulding_time = (5.0, 0.5)
            assembly1_time = (8.0, 1.0)
            assembly2_time = (6.0, 0.8)
            sorting_time = (3.0, 0.3)
            packaging_time = (4.0, 0.5)
            
            if variant_id <= 6:
                assembly1_time = (7.0, 0.8)
            elif variant_id <= 12:
                assembly1_time = (8.5, 1.0)
            else:
                assembly1_time = (10.0, 1.2)
            
            process = ProcessStructure(
                name=variant_name,
                arrival_distribution=l / 18,
                data_options={},  # No quality checks
                tasks={
                    "START": Task(
                        name=f"{variant_name}_START",
                        next_tasks=["MOULDING"]
                    ),
                    
                    "MOULDING": Task(
                        name=f"{variant_name}_MOULDING",
                        resources=[
                            Resource("MOULDING_MACHINE_1", moulding_time),
                            Resource("MOULDING_MACHINE_2", moulding_time),
                            Resource("MOULDING_MACHINE_3", moulding_time),
                            Resource("MOULDING_MACHINE_4", moulding_time),
                            Resource("MOULDING_MACHINE_5", moulding_time),
                        ],
                        next_tasks=["ASSEMBLY_1"]  # Direct to next step
                    ),
                    
                    "ASSEMBLY_1": Task(
                        name=f"{variant_name}_ASSEMBLY_1",
                        resources=[
                            Resource("LINE_1_ASSEMBLY1", assembly1_time),
                            Resource("LINE_2_ASSEMBLY1", assembly1_time),
                            Resource("LINE_3_ASSEMBLY1", assembly1_time),
                            Resource("LINE_4_ASSEMBLY1", assembly1_time),
                            Resource("LINE_5_ASSEMBLY1", assembly1_time),
                        ],
                        next_tasks=["ASSEMBLY_2"]
                    ),
                    
                    "ASSEMBLY_2": Task(
                        name=f"{variant_name}_ASSEMBLY_2",
                        resources=[
                            Resource("ASSEMBLY2_LINE_1", assembly2_time),
                            Resource("ASSEMBLY2_LINE_2", assembly2_time),
                        ],
                        next_tasks=["SORTING"]
                    ),
                    
                    "SORTING": Task(
                        name=f"{variant_name}_SORTING",
                        resources=[
                            Resource("SORTING_ROBOT", sorting_time),
                        ],
                        next_tasks=["PACKAGING"]
                    ),
                    
                    "PACKAGING": Task(
                        name=f"{variant_name}_PACKAGING",
                        resources=[
                            Resource("PACKAGING_LINE_1", packaging_time),
                            Resource("PACKAGING_LINE_2", packaging_time),
                            Resource("PACKAGING_LINE_3", packaging_time),
                        ],
                        next_tasks=["END"]
                    ),
                    
                    "END": Task(
                        name=f"{variant_name}_END",
                        next_tasks=[]
                    )
                }
            )
            
            processes.append(process)
        
        return processes
    

    elif scenario == 'actuator_mfg_pooledM_dedicatedA1_with_rework':
        processes = []
        for variant_id in range(1, 19):
            variant_name = f"variant_{variant_id}"

            moulding_time = (5.0, 0.5)
            assembly1_time = (8.0, 1.0)
            assembly2_time = (6.0, 0.8)
            sorting_time   = (3.0, 0.3)
            packaging_time = (4.0, 0.5)

            if variant_id <= 6:
                assembly1_time = (7.0, 0.8)
            elif variant_id <= 12:
                assembly1_time = (8.5, 1.0)
            else:
                assembly1_time = (10.0, 1.2)

            data_options = {
                # QC AFTER MOULDING — SAME PROBS AS OLD SCENARIO
                # pass -> ROUTE_TO_A1 (instead of pooled ASSEMBLY_1), fail -> MOULDING rework
                "qc_after_moulding": {"ROUTE_TO_A1": 0.95, "MOULDING": 0.05},

                # Deterministic post-Moulding routing (weights are dummies; engine uses case_data)
                "moulding_lane": {
                    "ASSEMBLY_1_1": 0.2, "ASSEMBLY_1_2": 0.2, "ASSEMBLY_1_3": 0.2,
                    "ASSEMBLY_1_4": 0.2, "ASSEMBLY_1_5": 0.2
                },

                # QC AFTER A1 — SAME PROBS AS OLD SCENARIO, lane-specific loopback
                "qc_after_assembly1_1": {"ASSEMBLY_1_1": 0.05, "ASSEMBLY_2": 0.95},
                "qc_after_assembly1_2": {"ASSEMBLY_1_2": 0.05, "ASSEMBLY_2": 0.95},
                "qc_after_assembly1_3": {"ASSEMBLY_1_3": 0.05, "ASSEMBLY_2": 0.95},
                "qc_after_assembly1_4": {"ASSEMBLY_1_4": 0.05, "ASSEMBLY_2": 0.95},
                "qc_after_assembly1_5": {"ASSEMBLY_1_5": 0.05, "ASSEMBLY_2": 0.95},

                # Downstream QC — SAME AS OLD
                "qc_after_assembly2": {"SORTING": 0.95, "ASSEMBLY_2": 0.05},
                "qc_after_sorting":   {"PACKAGING": 0.95, "SORTING": 0.05},
                "qc_after_packaging": {"END": 0.95, "PACKAGING": 0.05},
            }

            tasks = {
                "START": Task(name=f"{variant_name}_START", next_tasks=["MOULDING"]),

                # POOLED MOULDING (M/M/5), then QC (same as old scenario)
                "MOULDING": Task(
                    name=f"{variant_name}_MOULDING",
                    resources=[
                        Resource("MOULDING_MACHINE_1", moulding_time),
                        Resource("MOULDING_MACHINE_2", moulding_time),
                        Resource("MOULDING_MACHINE_3", moulding_time),
                        Resource("MOULDING_MACHINE_4", moulding_time),
                        Resource("MOULDING_MACHINE_5", moulding_time),
                    ],
                    next_tasks=["QC_AFTER_MOULDING"]
                ),
                "QC_AFTER_MOULDING": Gateway(
                    name=f"{variant_name}_QC_AFTER_MOULDING",
                    gateway_type="XOR",
                    conditions=["qc_after_moulding"],
                    next_tasks=["MOULDING", "ROUTE_TO_A1"]
                ),

                # Deterministic routing by case_data['moulding_lane'] -> dedicated A1 lane
                "ROUTE_TO_A1": Gateway(
                    name=f"{variant_name}_ROUTE_TO_A1",
                    gateway_type="XOR",
                    conditions=["moulding_lane"],
                    next_tasks=["ASSEMBLY_1_1", "ASSEMBLY_1_2", "ASSEMBLY_1_3", "ASSEMBLY_1_4", "ASSEMBLY_1_5"]
                ),

                # FIVE DEDICATED A1 (M/M/1 each) with lane-specific QC
                "ASSEMBLY_1_1": Task(name=f"{variant_name}_ASSEMBLY_1_1",
                                    resources=[Resource("LINE_1_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["QC_AFTER_ASSEMBLY1_1"]),
                "QC_AFTER_ASSEMBLY1_1": Gateway(
                    name=f"{variant_name}_QC_AFTER_ASSEMBLY1_1", gateway_type="XOR",
                    conditions=["qc_after_assembly1_1"],
                    next_tasks=["ASSEMBLY_1_1", "ASSEMBLY_2"]
                ),
                "ASSEMBLY_1_2": Task(name=f"{variant_name}_ASSEMBLY_1_2",
                                    resources=[Resource("LINE_2_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["QC_AFTER_ASSEMBLY1_2"]),
                "QC_AFTER_ASSEMBLY1_2": Gateway(
                    name=f"{variant_name}_QC_AFTER_ASSEMBLY1_2", gateway_type="XOR",
                    conditions=["qc_after_assembly1_2"],
                    next_tasks=["ASSEMBLY_1_2", "ASSEMBLY_2"]
                ),
                "ASSEMBLY_1_3": Task(name=f"{variant_name}_ASSEMBLY_1_3",
                                    resources=[Resource("LINE_3_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["QC_AFTER_ASSEMBLY1_3"]),
                "QC_AFTER_ASSEMBLY1_3": Gateway(
                    name=f"{variant_name}_QC_AFTER_ASSEMBLY1_3", gateway_type="XOR",
                    conditions=["qc_after_assembly1_3"],
                    next_tasks=["ASSEMBLY_1_3", "ASSEMBLY_2"]
                ),
                "ASSEMBLY_1_4": Task(name=f"{variant_name}_ASSEMBLY_1_4",
                                    resources=[Resource("LINE_4_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["QC_AFTER_ASSEMBLY1_4"]),
                "QC_AFTER_ASSEMBLY1_4": Gateway(
                    name=f"{variant_name}_QC_AFTER_ASSEMBLY1_4", gateway_type="XOR",
                    conditions=["qc_after_assembly1_4"],
                    next_tasks=["ASSEMBLY_1_4", "ASSEMBLY_2"]
                ),
                "ASSEMBLY_1_5": Task(name=f"{variant_name}_ASSEMBLY_1_5",
                                    resources=[Resource("LINE_5_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["QC_AFTER_ASSEMBLY1_5"]),
                "QC_AFTER_ASSEMBLY1_5": Gateway(
                    name=f"{variant_name}_QC_AFTER_ASSEMBLY1_5", gateway_type="XOR",
                    conditions=["qc_after_assembly1_5"],
                    next_tasks=["ASSEMBLY_1_5", "ASSEMBLY_2"]
                ),

                # POOLED DOWNSTREAM (unchanged) with the SAME QC logic
                "ASSEMBLY_2": Task(
                    name=f"{variant_name}_ASSEMBLY_2",
                    resources=[Resource("ASSEMBLY2_LINE_1", assembly2_time),
                            Resource("ASSEMBLY2_LINE_2", assembly2_time)],
                    next_tasks=["QC_AFTER_ASSEMBLY2"]
                ),
                "QC_AFTER_ASSEMBLY2": Gateway(
                    name=f"{variant_name}_QC_AFTER_ASSEMBLY2", gateway_type="XOR",
                    conditions=["qc_after_assembly2"],
                    next_tasks=["SORTING", "ASSEMBLY_2"]
                ),
                "SORTING": Task(
                    name=f"{variant_name}_SORTING",
                    resources=[Resource("SORTING_ROBOT", sorting_time)],
                    next_tasks=["QC_AFTER_SORTING"]
                ),
                "QC_AFTER_SORTING": Gateway(
                    name=f"{variant_name}_QC_AFTER_SORTING", gateway_type="XOR",
                    conditions=["qc_after_sorting"],
                    next_tasks=["PACKAGING", "SORTING"]
                ),
                "PACKAGING": Task(
                    name=f"{variant_name}_PACKAGING",
                    resources=[Resource("PACKAGING_LINE_1", packaging_time),
                            Resource("PACKAGING_LINE_2", packaging_time),
                            Resource("PACKAGING_LINE_3", packaging_time)],
                    next_tasks=["QC_AFTER_PACKAGING"]
                ),
                "QC_AFTER_PACKAGING": Gateway(
                    name=f"{variant_name}_QC_AFTER_PACKAGING", gateway_type="XOR",
                    conditions=["qc_after_packaging"],
                    next_tasks=["END", "PACKAGING"]
                ),
                "END": Task(name=f"{variant_name}_END", next_tasks=[]),
            }

            processes.append(ProcessStructure(
                name=variant_name,
                arrival_distribution=l / 18,
                data_options=data_options,
                tasks=tasks
            ))

        return processes


    elif scenario == 'actuator_mfg_pooledM_dedicatedA1_no_rework':
        processes = []
        for variant_id in range(1, 19):
            variant_name = f"variant_{variant_id}"

            moulding_time = (5.0, 0.5)
            assembly1_time = (8.0, 1.0)
            assembly2_time = (6.0, 0.8)
            sorting_time   = (3.0, 0.3)
            packaging_time = (4.0, 0.5)

            if variant_id <= 6:
                assembly1_time = (7.0, 0.8)
            elif variant_id <= 12:
                assembly1_time = (8.5, 1.0)
            else:
                assembly1_time = (10.0, 1.2)

            data_options = {
                # dummies; engine will deterministically route by case_data['moulding_lane']
                "moulding_lane": {
                    "ASSEMBLY_1_1": 0.2, "ASSEMBLY_1_2": 0.2, "ASSEMBLY_1_3": 0.2,
                    "ASSEMBLY_1_4": 0.2, "ASSEMBLY_1_5": 0.2
                }
            }

            tasks = {
                "START": Task(name=f"{variant_name}_START", next_tasks=["MOULDING"]),
                "MOULDING": Task(
                    name=f"{variant_name}_MOULDING",
                    resources=[
                        Resource("MOULDING_MACHINE_1", moulding_time),
                        Resource("MOULDING_MACHINE_2", moulding_time),
                        Resource("MOULDING_MACHINE_3", moulding_time),
                        Resource("MOULDING_MACHINE_4", moulding_time),
                        Resource("MOULDING_MACHINE_5", moulding_time),
                    ],
                    next_tasks=["ROUTE_TO_A1"]  # NO QC ANYWHERE in "no_rework"
                ),
                "ROUTE_TO_A1": Gateway(
                    name=f"{variant_name}_ROUTE_TO_A1",
                    gateway_type="XOR",
                    conditions=["moulding_lane"],
                    next_tasks=["ASSEMBLY_1_1", "ASSEMBLY_1_2", "ASSEMBLY_1_3", "ASSEMBLY_1_4", "ASSEMBLY_1_5"]
                ),
                "ASSEMBLY_1_1": Task(name=f"{variant_name}_ASSEMBLY_1_1",
                                    resources=[Resource("LINE_1_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["ASSEMBLY_2"]),
                "ASSEMBLY_1_2": Task(name=f"{variant_name}_ASSEMBLY_1_2",
                                    resources=[Resource("LINE_2_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["ASSEMBLY_2"]),
                "ASSEMBLY_1_3": Task(name=f"{variant_name}_ASSEMBLY_1_3",
                                    resources=[Resource("LINE_3_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["ASSEMBLY_2"]),
                "ASSEMBLY_1_4": Task(name=f"{variant_name}_ASSEMBLY_1_4",
                                    resources=[Resource("LINE_4_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["ASSEMBLY_2"]),
                "ASSEMBLY_1_5": Task(name=f"{variant_name}_ASSEMBLY_1_5",
                                    resources=[Resource("LINE_5_ASSEMBLY1", assembly1_time)],
                                    next_tasks=["ASSEMBLY_2"]),
                "ASSEMBLY_2": Task(
                    name=f"{variant_name}_ASSEMBLY_2",
                    resources=[Resource("ASSEMBLY2_LINE_1", assembly2_time),
                            Resource("ASSEMBLY2_LINE_2", assembly2_time)],
                    next_tasks=["SORTING"]
                ),
                "SORTING": Task(
                    name=f"{variant_name}_SORTING",
                    resources=[Resource("SORTING_ROBOT", sorting_time)],
                    next_tasks=["PACKAGING"]
                ),
                "PACKAGING": Task(
                    name=f"{variant_name}_PACKAGING",
                    resources=[Resource("PACKAGING_LINE_1", packaging_time),
                            Resource("PACKAGING_LINE_2", packaging_time),
                            Resource("PACKAGING_LINE_3", packaging_time)],
                    next_tasks=["END"]
                ),
                "END": Task(name=f"{variant_name}_END", next_tasks=[]),
            }

            processes.append(ProcessStructure(
                name=variant_name,
                arrival_distribution=l / 18,
                data_options=data_options,
                tasks=tasks
            ))

        return processes
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

# ========================================
# Simulation Configuration
# ========================================
ARRIVAL_RATES = [0.28]#, 0.2, 0.3, 0.5]  # Batches per minute (adjust to your demand)
SCENARIO_NAMES = [
    'actuator_manufacturing_with_rework'#,
    #'actuator_manufacturing_no_rework'    
    #'actuator_mfg_pooledM_dedicatedA1_with_rework'
    # 'actuator_mfg_pooledM_dedicatedA1_no_rework',
  # For comparison
]
SIMULATION_RUN_TIME = 30000  # 48 hours in minutes (2 shifts)
SIMULATION_RUNS = 1  # Statistical reliability