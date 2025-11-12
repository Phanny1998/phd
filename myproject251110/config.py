# NEW
from __future__ import annotations
from myproject251110.processes import ProcessStructure, Task, Resource, Gateway
import random
import math

# ====== EXPERIMENT MENUS (fixed rates; we vary everything else) ======
QUEUE_STYLES = ["pooled", "hybrid30", "dedicated"]  # pooled vs per-machine vs ~30% dedicated

# Total number of activities per process (core 5 + extra INSPECTION_k)
ACTIVITY_TOTALS = [5, 8, 12, 16, 24, 32]

# Server-count presets
COUNT_PRESETS = {
    "C1": {"MOULD_C":5, "A1_C":5, "A2_C":2, "SORT_C":1, "PACK_C":3, "INSP_C":2},
    "C2": {"MOULD_C":5, "A1_C":5, "A2_C":3, "SORT_C":1, "PACK_C":3, "INSP_C":2},
    "C3": {"MOULD_C":5, "A1_C":5, "A2_C":2, "SORT_C":2, "PACK_C":3, "INSP_C":2},
    # C4: scale INSPECTION servers mildly with number of extra stages (computed later)
    "C4": {"MOULD_C":5, "A1_C":5, "A2_C":2, "SORT_C":1, "PACK_C":3, "INSP_C":"auto"},
    # Uniform controls
    "U1": {"UNIFORM_C":1},
    "U2": {"UNIFORM_C":2},
}

# Heterogeneity (multiplies *means*; 1.0 = identical)
HET_MULTIPLIERS = {
    "identical": {},
    "mild_all": {
        "MOULDING":[0.95,1.00,1.05,1.00,1.00],
        "A1":[0.90,1.00,1.10,1.00,1.00],
        "A2":[0.95,1.05],
        "SORTING":[1.00],
        "PACKAGING":[0.95,1.00,1.05],
    },
    "strong_A1": {
        "A1":[0.80,0.90,1.00,1.10,1.20],
    }
}

QC_PASS_LEVELS = [0.99, 0.97, 0.95]  # pass prob at each QC
VARIANT_COUNTS = [6, 18, 36]         # # product classes per run

# Keep your existing ARRIVAL_RATES list as-is. We’ll reuse it.
# Keep your BASE means as-is (5,8,6,3,4). We’ll derive INSPECTION mean from PACKAGING.
EXTRA_STAGE_ALPHA = 0.3  # INSPECTION mean = EXTRA_STAGE_ALPHA * PACKAGING mean (keeps extra stages “light”)

def _make_activity_line_for_variant(
    variant_name: str,
    l_share: float,
    counts: dict,
    qc_pass: float,
    queue_style: str,
    extra_stage_count: int,
    hetero_key: str,
    station_styles: dict | None = None,   # optional per-station overrides
):
    """
    Build ONE variant line with fixed base means (5,8,6,3,4 minutes).
    queue_style: default style if a station doesn't have an override
                 ("pooled" | "dedicated")
    station_styles: optional dict like {"MOULDING":"pooled","A1":"dedicated", ...}
    counts: either {"UNIFORM_C":1/2} OR explicit per-station counts (C1..C4)
    qc_pass: QC pass probability for all QCs (e.g. 0.97)
    hetero_key: key into HET_MULTIPLIERS
    """
    # ---- base means (keep yours) ----
    mould_m = (5.0, 0.5)
    a1_m    = (8.0, 1.0)
    a2_m    = (6.0, 0.8)
    sort_m  = (3.0, 0.3)
    pack_m  = (4.0, 0.5)
    insp_m  = (pack_m[0] * EXTRA_STAGE_ALPHA, 0.25)  # light extra checks

    # ---- per-station style (default to global queue_style) ----
    station_styles = station_styles or {}
    style_MOULD = station_styles.get("MOULDING",  queue_style)
    style_A1    = station_styles.get("A1",        queue_style)
    style_A2    = station_styles.get("A2",        queue_style)
    style_SORT  = station_styles.get("SORTING",   queue_style)
    style_PACK  = station_styles.get("PACKAGING", queue_style)
    style_INSP  = station_styles.get("INSPECTION", "pooled")  # default pooled

    # ---- counts (servers per station) ----
    if "UNIFORM_C" in counts:
        u = counts["UNIFORM_C"]
        c_mould = c_a1 = c_a2 = c_sort = c_pack = c_insp = max(1, int(u))
    else:
        c_mould = counts["MOULD_C"]
        c_a1    = counts["A1_C"]
        c_a2    = counts["A2_C"]
        c_sort  = counts["SORT_C"]
        c_pack  = counts["PACK_C"]
        c_insp  = counts["INSP_C"]
        if c_insp == "auto":
            c_insp = max(2, math.ceil(max(0, extra_stage_count) / 4))

    # ---- heterogeneity (mean multipliers) ----
    het = HET_MULTIPLIERS.get(hetero_key, {})

    def _dedicated_tuple(base_tuple: tuple[float,float], station_key: str, idx1based: int):
        """Apply heterogeneity to a dedicated machine i (by index)."""
        base_mean, base_std = base_tuple
        mults = het.get(station_key, None)
        if isinstance(mults, list) and 1 <= idx1based <= len(mults):
            return (base_mean * mults[idx1based - 1], base_std)
        elif isinstance(mults, (int, float)):
            return (base_mean * mults, base_std)
        else:
            return base_tuple

    def _mk_pool(prefix: str, base_tuple: tuple[float,float], count: int, station_key: str):
        """Create pooled resources; heterogeneity applies per resource if list or scalar."""
        base_mean, base_std = base_tuple
        mults = het.get(station_key, None)
        resources = []
        for i in range(1, count + 1):
            if isinstance(mults, list) and i <= len(mults):
                m = (base_mean * mults[i - 1], base_std)
            elif isinstance(mults, (int, float)):
                m = (base_mean * mults, base_std)
            else:
                m = base_tuple
            resources.append(Resource(f"{prefix}_{i}", m))
        return resources

    def qc_pair(pass_to: str, fail_to: str):
        return {pass_to: qc_pass, fail_to: 1.0 - qc_pass}

    # ==============================
    # build tasks + data_options
    # ==============================
    tasks: dict[str, Task | Gateway] = {}
    data_options: dict[str, dict[str, float]] = {}

    # ---- START (respect MOULDING style for first hop) ----
    tasks["START"] = Task(
        name=f"{variant_name}_START",
        next_tasks=["MOULDING" if style_MOULD == "pooled" else "ROUTE_TO_MOULD"]
    )

    # ===================
    # MOULDING
    # ===================
    if style_MOULD == "pooled":
        tasks["MOULDING"] = Task(
            name=f"{variant_name}_MOULDING",
            resources=_mk_pool("MOULDING_MACHINE", mould_m, c_mould, "MOULDING"),
            next_tasks=["QC_AFTER_MOULDING"],
        )
        tasks["QC_AFTER_MOULDING"] = Gateway(
            name=f"{variant_name}_QC_AFTER_MOULDING",
            gateway_type="XOR",
            conditions=["qc_after_moulding"],
            next_tasks=[
                "MOULDING",
                ("ASSEMBLY_1" if style_A1 == "pooled" else "ROUTE_TO_A1")
            ],
        )
        data_options["qc_after_moulding"] = qc_pair(
            ("ASSEMBLY_1" if style_A1 == "pooled" else "ROUTE_TO_A1"),
            "MOULDING"
        )
    else:
        # dedicated MOULDING lanes
        tasks["ROUTE_TO_MOULD"] = Gateway(
            name=f"{variant_name}_ROUTE_TO_MOULD",
            gateway_type="XOR",
            conditions=["mould_lane"],
            next_tasks=[f"MOULDING_{i}" for i in range(1, c_mould + 1)],
        )
        data_options["mould_lane"] = {f"MOULDING_{i}": 1.0 for i in range(1, c_mould + 1)}
        for i in range(1, c_mould + 1):
            tasks[f"MOULDING_{i}"] = Task(
                name=f"{variant_name}_MOULDING_{i}",
                resources=[Resource(f"MOULDING_MACHINE_{i}", _dedicated_tuple(mould_m, "MOULDING", i))],
                next_tasks=[f"QC_AFTER_MOULDING_{i}"],
            )
            tasks[f"QC_AFTER_MOULDING_{i}"] = Gateway(
                name=f"{variant_name}_QC_AFTER_MOULDING_{i}",
                gateway_type="XOR",
                conditions=[f"qc_after_moulding_{i}"],
                next_tasks=[
                    f"MOULDING_{i}",
                    ("ASSEMBLY_1" if style_A1 == "pooled" else "ROUTE_TO_A1")
                ],
            )
            data_options[f"qc_after_moulding_{i}"] = qc_pair(
                ("ASSEMBLY_1" if style_A1 == "pooled" else "ROUTE_TO_A1"),
                f"MOULDING_{i}"
            )

    # ===================
    # ASSEMBLY 1 (A1)
    # ===================
    if style_A1 == "pooled":
        tasks["ASSEMBLY_1"] = Task(
            name=f"{variant_name}_ASSEMBLY_1",
            resources=_mk_pool("LINE_ASSEMBLY1", a1_m, c_a1, "A1"),
            next_tasks=["QC_AFTER_A1"],
        )
        tasks["QC_AFTER_A1"] = Gateway(
            name=f"{variant_name}_QC_AFTER_A1",
            gateway_type="XOR",
            conditions=["qc_after_a1"],
            next_tasks=[
                "ASSEMBLY_1",
                ("ASSEMBLY_2" if style_A2 == "pooled" else "ROUTE_TO_A2")
            ],
        )
        data_options["qc_after_a1"] = qc_pair(
            ("ASSEMBLY_2" if style_A2 == "pooled" else "ROUTE_TO_A2"),
            "ASSEMBLY_1"
        )
    else:
        tasks["ROUTE_TO_A1"] = Gateway(
            name=f"{variant_name}_ROUTE_TO_A1",
            gateway_type="XOR",
            conditions=["a1_lane"],
            next_tasks=[f"ASSEMBLY_1_{i}" for i in range(1, c_a1 + 1)],
        )
        data_options["a1_lane"] = {f"ASSEMBLY_1_{i}": 1.0 for i in range(1, c_a1 + 1)}
        for i in range(1, c_a1 + 1):
            tasks[f"ASSEMBLY_1_{i}"] = Task(
                name=f"{variant_name}_ASSEMBLY_1_{i}",
                resources=[Resource(f"LINE_{i}_ASSEMBLY1", _dedicated_tuple(a1_m, "A1", i))],
                next_tasks=[f"QC_AFTER_A1_{i}"],
            )
            tasks[f"QC_AFTER_A1_{i}"] = Gateway(
                name=f"{variant_name}_QC_AFTER_A1_{i}",
                gateway_type="XOR",
                conditions=[f"qc_after_a1_{i}"],
                next_tasks=[
                    f"ASSEMBLY_1_{i}",
                    ("ASSEMBLY_2" if style_A2 == "pooled" else "ROUTE_TO_A2")
                ],
            )
            data_options[f"qc_after_a1_{i}"] = qc_pair(
                ("ASSEMBLY_2" if style_A2 == "pooled" else "ROUTE_TO_A2"),
                f"ASSEMBLY_1_{i}"
            )

    # ===================
    # ASSEMBLY 2 (A2)
    # ===================
    if style_A2 == "pooled":
        tasks["ASSEMBLY_2"] = Task(
            name=f"{variant_name}_ASSEMBLY_2",
            resources=_mk_pool("ASSEMBLY2_LINE", a2_m, c_a2, "A2"),
            next_tasks=["QC_AFTER_A2"],
        )
        next_from_a2_pass = ("SORTING" if (style_SORT == "pooled" or c_sort == 1) else "ROUTE_TO_SORT")
        tasks["QC_AFTER_A2"] = Gateway(
            name=f"{variant_name}_QC_AFTER_A2",
            gateway_type="XOR",
            conditions=["qc_after_a2"],
            next_tasks=[next_from_a2_pass, "ASSEMBLY_2"],
        )
        data_options["qc_after_a2"] = qc_pair(next_from_a2_pass, "ASSEMBLY_2")
    else:
        tasks["ROUTE_TO_A2"] = tasks.get("ROUTE_TO_A2") or Gateway(
            name=f"{variant_name}_ROUTE_TO_A2",
            gateway_type="XOR",
            conditions=["a2_lane"],
            next_tasks=[f"ASSEMBLY_2_{i}" for i in range(1, c_a2 + 1)],
        )
        data_options["a2_lane"] = {f"ASSEMBLY_2_{i}": 1.0 for i in range(1, c_a2 + 1)}
        for i in range(1, c_a2 + 1):
            tasks[f"ASSEMBLY_2_{i}"] = Task(
                name=f"{variant_name}_ASSEMBLY_2_{i}",
                resources=[Resource(f"ASSEMBLY2_LINE_{i}", _dedicated_tuple(a2_m, "A2", i))],
                next_tasks=[f"QC_AFTER_A2_{i}"],
            )
            next_pass = ("SORTING" if (style_SORT == "pooled" or c_sort == 1) else "ROUTE_TO_SORT")
            tasks[f"QC_AFTER_A2_{i}"] = Gateway(
                name=f"{variant_name}_QC_AFTER_A2_{i}",
                gateway_type="XOR",
                conditions=[f"qc_after_a2_{i}"],
                next_tasks=[next_pass, f"ASSEMBLY_2_{i}"],
            )
            data_options[f"qc_after_a2_{i}"] = qc_pair(next_pass, f"ASSEMBLY_2_{i}")

    # ===================
    # SORTING
    # ===================
    if style_SORT == "pooled" or c_sort == 1:
        tasks["SORTING"] = Task(
            name=f"{variant_name}_SORTING",
            resources=_mk_pool("SORTING_ROBOT", sort_m, c_sort, "SORTING"),
            next_tasks=["QC_AFTER_SORTING"],
        )
        next_from_sort = ("PACKAGING" if style_PACK == "pooled" else "ROUTE_TO_PACKAGING")
        tasks["QC_AFTER_SORTING"] = Gateway(
            name=f"{variant_name}_QC_AFTER_SORTING",
            gateway_type="XOR",
            conditions=["qc_after_sorting"],
            next_tasks=[next_from_sort, "SORTING"],
        )
        data_options["qc_after_sorting"] = qc_pair(next_from_sort, "SORTING")
    else:
        # dedicated SORTING lanes
        tasks["ROUTE_TO_SORT"] = Gateway(
            name=f"{variant_name}_ROUTE_TO_SORT",
            gateway_type="XOR",
            conditions=["sort_lane"],
            next_tasks=[f"SORTING_{i}" for i in range(1, c_sort + 1)],
        )
        data_options["sort_lane"] = {f"SORTING_{i}": 1.0 for i in range(1, c_sort + 1)}
        for i in range(1, c_sort + 1):
            tasks[f"SORTING_{i}"] = Task(
                name=f"{variant_name}_SORTING_{i}",
                resources=[Resource(f"SORTING_ROBOT_{i}", _dedicated_tuple(sort_m, "SORTING", i))],
                next_tasks=[f"QC_AFTER_SORTING_{i}"],
            )
            next_from_sort = ("PACKAGING" if style_PACK == "pooled" else "ROUTE_TO_PACKAGING")
            tasks[f"QC_AFTER_SORTING_{i}"] = Gateway(
                name=f"{variant_name}_QC_AFTER_SORTING_{i}",
                gateway_type="XOR",
                conditions=[f"qc_after_sorting_{i}"],
                next_tasks=[next_from_sort, f"SORTING_{i}"],
            )
            data_options[f"qc_after_sorting_{i}"] = qc_pair(next_from_sort, f"SORTING_{i}")

    # ===================
    # PACKAGING
    # ===================
    if style_PACK == "pooled":
        tasks["PACKAGING"] = Task(
            name=f"{variant_name}_PACKAGING",
            resources=_mk_pool("PACKAGING_LINE", pack_m, c_pack, "PACKAGING"),
            next_tasks=["QC_AFTER_PACKAGING"],
        )
        # TEMP pass target; will be rewired if we add INSPECTION
        tasks["QC_AFTER_PACKAGING"] = Gateway(
            name=f"{variant_name}_QC_AFTER_PACKAGING",
            gateway_type="XOR",
            conditions=["qc_after_packaging"],
            next_tasks=["END", "PACKAGING"],  # pass -> END (will be changed), fail -> PACKAGING
        )
        data_options["qc_after_packaging"] = qc_pair("END", "PACKAGING")
    else:
        tasks["ROUTE_TO_PACKAGING"] = Gateway(
            name=f"{variant_name}_ROUTE_TO_PACKAGING",
            gateway_type="XOR",
            conditions=["pack_lane"],
            next_tasks=[f"PACKAGING_{i}" for i in range(1, c_pack + 1)],
        )
        data_options["pack_lane"] = {f"PACKAGING_{i}": 1.0 for i in range(1, c_pack + 1)}
        for i in range(1, c_pack + 1):
            tasks[f"PACKAGING_{i}"] = Task(
                name=f"{variant_name}_PACKAGING_{i}",
                resources=[Resource(f"PACKAGING_LINE_{i}", _dedicated_tuple(pack_m, "PACKAGING", i))],
                next_tasks=[f"QC_AFTER_PACKAGING_{i}"],
            )
            tasks[f"QC_AFTER_PACKAGING_{i}"] = Gateway(
                name=f"{variant_name}_QC_AFTER_PACKAGING_{i}",
                gateway_type="XOR",
                conditions=[f"qc_after_packaging_{i}"],
                next_tasks=["END", f"PACKAGING_{i}"],  # pass->END (will be changed)
            )
            data_options[f"qc_after_packaging_{i}"] = qc_pair("END", f"PACKAGING_{i}")

    # ===================
    # INSPECTION_k (extra stages)
    # ===================
    # Create END first (we’ll rewire PACKAGING pass to INSPECTION_1 if any)
    tasks["END"] = Task(name=f"{variant_name}_END", next_tasks=[])

    if extra_stage_count > 0:
        # determine entry from PACKAGING to INSPECTION_1 regardless of PACKAGING style
        def _set_pack_pass_to(label_target: str):
            # pooled PACKAGING:
            if "QC_AFTER_PACKAGING" in tasks and "qc_after_packaging" in data_options:
                tasks["QC_AFTER_PACKAGING"].next_tasks = [label_target, "PACKAGING"]
                data_options["qc_after_packaging"] = qc_pair(label_target, "PACKAGING")
            # dedicated PACKAGING lanes:
            for i in range(1, c_pack + 1):
                qcl = f"QC_AFTER_PACKAGING_{i}"
                if qcl in tasks and f"qc_after_packaging_{i}" in data_options:
                    tasks[qcl].next_tasks = [label_target, f"PACKAGING_{i}"]
                    data_options[f"qc_after_packaging_{i}"] = qc_pair(label_target, f"PACKAGING_{i}")

        # INSPECTION chain
        # support pooled OR dedicated INSPECTION (based on style_INSP)
        if style_INSP == "pooled":
            # rewire PACK pass to INSPECTION_1
            _set_pack_pass_to("INSPECTION_1")
            for k in range(1, extra_stage_count + 1):
                lab = f"INSPECTION_{k}"
                tasks[lab] = Task(
                    name=f"{variant_name}_{lab}",
                    resources=_mk_pool(f"{lab}_LINE", insp_m, c_insp, "INSPECTION"),
                    next_tasks=[f"QC_AFTER_{lab}"],
                )
                next_ok = (f"INSPECTION_{k+1}" if k < extra_stage_count else "END")
                tasks[f"QC_AFTER_{lab}"] = Gateway(
                    name=f"{variant_name}_QC_AFTER_{lab}",
                    gateway_type="XOR",
                    conditions=[f"qc_after_{lab}"],
                    next_tasks=[next_ok, lab],
                )
                data_options[f"qc_after_{lab}"] = qc_pair(next_ok, lab)
        else:
            # dedicated INSPECTION lanes
            _set_pack_pass_to("ROUTE_TO_INSP")
            tasks["ROUTE_TO_INSP"] = Gateway(
                name=f"{variant_name}_ROUTE_TO_INSP",
                gateway_type="XOR",
                conditions=["insp_lane"],
                next_tasks=[f"INSPECTION_{i}_1" for i in range(1, c_insp + 1)],
            )
            data_options["insp_lane"] = {f"INSPECTION_{i}_1": 1.0 for i in range(1, c_insp + 1)}
            # Build a chain per lane: INSPECTION_i_1 -> ... -> INSPECTION_i_K
            for i in range(1, c_insp + 1):
                for k in range(1, extra_stage_count + 1):
                    lab = f"INSPECTION_{i}_{k}"
                    tasks[lab] = Task(
                        name=f"{variant_name}_{lab}",
                        resources=[Resource(f"{lab}_LINE", _dedicated_tuple(insp_m, "INSPECTION", i))],
                        next_tasks=[f"QC_AFTER_{lab}"],
                    )
                    next_ok = (f"INSPECTION_{i}_{k+1}" if k < extra_stage_count else "END")
                    tasks[f"QC_AFTER_{lab}"] = Gateway(
                        name=f"{variant_name}_QC_AFTER_{lab}",
                        gateway_type="XOR",
                        conditions=[f"qc_after_{lab}"],
                        next_tasks=[next_ok, lab],
                    )
                    data_options[f"qc_after_{lab}"] = qc_pair(next_ok, lab)

    return ProcessStructure(
        name=variant_name,
        arrival_distribution=l_share,
        data_options=data_options,
        tasks=tasks,
    )

# Map high-level queue_style to actual per-station styles.
def _resolve_styles(queue_style: str, station_styles: dict | None):
    if station_styles:                 # explicit per-station wins
        return ("pooled", station_styles)

    if queue_style == "pooled":
        return ("pooled", {})          # all pooled

    if queue_style == "dedicated":
        return ("dedicated", {         # all dedicated
            "MOULDING":"dedicated", "A1":"dedicated", "A2":"dedicated",
            "SORTING":"dedicated", "PACKAGING":"dedicated", "INSPECTION":"dedicated",
        })

    if queue_style == "hybrid30":
        # Example “hybrid”: A1 dedicated (bottleneck-style), rest pooled
        return ("pooled", {
            "A1":"dedicated",          # dedicated where we want stickiness
            # every other station inherits "pooled" from the global default
        })

    raise ValueError(f"Unknown queue_style: {queue_style}")

def process_function_experiment(l: float,
                                queue_style: str,
                                variant_count: int,
                                count_preset_key: str,
                                activity_total: int,
                                qc_pass: float,
                                hetero_key: str,
                                station_styles: dict | None = None):

    """
    Build 'variant_count' processes that share the same line structure (per your factors),
    dividing total arrival rate l equally across variants (fixed rates kept).
    """
    processes = []
    extra_stage_count = max(0, activity_total - 5)  # we add INSPECTION_k to reach total

    counts = COUNT_PRESETS[count_preset_key]
    l_share = l / float(variant_count)

     # NEW: normalize the style inputs
    base_style, resolved_station_styles = _resolve_styles(queue_style, station_styles)

    for variant_id in range(1, variant_count+1):
        variant_name = f"variant_{variant_id}"
        proc = _make_activity_line_for_variant(
            variant_name=variant_name,
            l_share=l_share,
            counts=counts,
            qc_pass=qc_pass,
            queue_style=base_style,                 # <- use resolved base style
            extra_stage_count=extra_stage_count,
            hetero_key=hetero_key,
            station_styles=resolved_station_styles, # <- use resolved per-station overrides
        )

        processes.append(proc)
    return processes

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
    elif scenario == 'all_dedicated_sticky_with_rework':
        processes = []
        for variant_id in range(1, 19):
            variant_name = f"variant_{variant_id}"

            # --- base times (keep your A1 variant tweak like before) ---
            moulding_time  = (5.0, 0.5)
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

            # --- lane-specific QC (sticky) + deterministic lane keys we will set in simulator ---
            data_options = {
                # Lane keys (deterministic XOR; values are dummies — engine uses case_data set at first start)
                "mould_lane": {f"MOULDING_{i}": 1 for i in range(1, 6)},
                "a1_lane":    {f"ASSEMBLY_1_{i}": 1 for i in range(1, 6)},
                "a2_lane":    {f"ASSEMBLY_2_{i}": 1 for i in range(1, 2+1)},
                "pack_lane":  {f"PACKAGING_{i}": 1 for i in range(1, 3+1)},
                "always": {"SORTING": 1},  # trivial router

                # Lane-specific QC (fail -> same lane; pass -> next router)
                **{f"qc_after_moulding_{i}": {f"MOULDING_{i}": 0.05, "ROUTE_TO_A1": 0.95} for i in range(1, 6)},
                **{f"qc_after_a1_{i}": {f"ASSEMBLY_1_{i}": 0.05, "ROUTE_TO_A2": 0.95} for i in range(1, 6)},
                "qc_after_a2_1": {"ASSEMBLY_2_1": 0.05, "ROUTE_TO_SORTING": 0.95},
                "qc_after_a2_2": {"ASSEMBLY_2_2": 0.05, "ROUTE_TO_SORTING": 0.95},
                "qc_after_packaging_1": {"END": 0.95, "PACKAGING_1": 0.05},
                "qc_after_packaging_2": {"END": 0.95, "PACKAGING_2": 0.05},
                "qc_after_packaging_3": {"END": 0.95, "PACKAGING_3": 0.05},
                "qc_after_sorting": {"ROUTE_TO_PACKAGING": 0.95, "SORTING": 0.05},

            }

            # --- tasks/gateways (all dedicated; sticky via lane-specific QC) ---
            tasks = {
                "START": Task(name=f"{variant_name}_START", next_tasks=["ROUTE_TO_MOULDING"]),

                # MOULDING dedicated (5)
                "ROUTE_TO_MOULDING": Gateway(
                    name=f"{variant_name}_ROUTE_TO_MOULDING",
                    gateway_type="XOR", conditions=["mould_lane"],
                    next_tasks=[f"MOULDING_{i}" for i in range(1, 6)]
                ),
            }
            # MOULD lanes + QC
            for i in range(1, 6):
                tasks[f"MOULDING_{i}"] = Task(
                    name=f"{variant_name}_MOULDING_{i}",
                    resources=[Resource(f"MOULDING_MACHINE_{i}", moulding_time)],
                    next_tasks=[f"QC_AFTER_MOULDING_{i}"]
                )
                tasks[f"QC_AFTER_MOULDING_{i}"] = Gateway(
                    name=f"{variant_name}_QC_AFTER_MOULDING_{i}",
                    gateway_type="XOR", conditions=[f"qc_after_moulding_{i}"],
                    next_tasks=[f"MOULDING_{i}", "ROUTE_TO_A1"]
                )

            # A1 dedicated (5)
            tasks["ROUTE_TO_A1"] = Gateway(
                name=f"{variant_name}_ROUTE_TO_A1",
                gateway_type="XOR", conditions=["a1_lane"],
                next_tasks=[f"ASSEMBLY_1_{i}" for i in range(1, 6)]
            )
            for i in range(1, 6):
                tasks[f"ASSEMBLY_1_{i}"] = Task(
                    name=f"{variant_name}_ASSEMBLY_1_{i}",
                    resources=[Resource(f"LINE_{i}_ASSEMBLY1", assembly1_time)],
                    next_tasks=[f"QC_AFTER_A1_{i}"]
                )
                tasks[f"QC_AFTER_A1_{i}"] = Gateway(
                    name=f"{variant_name}_QC_AFTER_A1_{i}",
                    gateway_type="XOR", conditions=[f"qc_after_a1_{i}"],
                    next_tasks=[f"ASSEMBLY_1_{i}", "ROUTE_TO_A2"]
                )

            # A2 dedicated (2)
            tasks["ROUTE_TO_A2"] = Gateway(
                name=f"{variant_name}_ROUTE_TO_A2",
                gateway_type="XOR", conditions=["a2_lane"],
                next_tasks=["ASSEMBLY_2_1", "ASSEMBLY_2_2"]
            )
            tasks["ASSEMBLY_2_1"] = Task(
                name=f"{variant_name}_ASSEMBLY_2_1",
                resources=[Resource("ASSEMBLY2_LINE_1", assembly2_time)],
                next_tasks=["QC_AFTER_A2_1"]
            )
            tasks["QC_AFTER_A2_1"] = Gateway(
                name=f"{variant_name}_QC_AFTER_A2_1",
                gateway_type="XOR", conditions=["qc_after_a2_1"],
                next_tasks=["ASSEMBLY_2_1", "ROUTE_TO_SORTING"]
            )
            tasks["ASSEMBLY_2_2"] = Task(
                name=f"{variant_name}_ASSEMBLY_2_2",
                resources=[Resource("ASSEMBLY2_LINE_2", assembly2_time)],
                next_tasks=["QC_AFTER_A2_2"]
            )
            tasks["QC_AFTER_A2_2"] = Gateway(
                name=f"{variant_name}_QC_AFTER_A2_2",
                gateway_type="XOR", conditions=["qc_after_a2_2"],
                next_tasks=["ASSEMBLY_2_2", "ROUTE_TO_SORTING"]
            )

            # SORTING (single)
            tasks["ROUTE_TO_SORTING"] = Gateway(
                name=f"{variant_name}_ROUTE_TO_SORTING",
                gateway_type="XOR", conditions=["always"],
                next_tasks=["SORTING"]
            )
            tasks["SORTING"] = Task(
                name=f"{variant_name}_SORTING",
                resources=[Resource("SORTING_ROBOT", sorting_time)],
                next_tasks=["QC_AFTER_SORTING"]
            )

            tasks["QC_AFTER_SORTING"] = Gateway(
                name=f"{variant_name}_QC_AFTER_SORTING",
                gateway_type="XOR",
                conditions=["qc_after_sorting"],
                next_tasks=["ROUTE_TO_PACKAGING", "SORTING"]  # pass → route, fail → rework Sorting
            )

            # PACKAGING dedicated (3)
            tasks["ROUTE_TO_PACKAGING"] = Gateway(
                name=f"{variant_name}_ROUTE_TO_PACKAGING",
                gateway_type="XOR", conditions=["pack_lane"],
                next_tasks=["PACKAGING_1","PACKAGING_2","PACKAGING_3"]
            )
            for i in range(1, 3+1):
                tasks[f"PACKAGING_{i}"] = Task(
                    name=f"{variant_name}_PACKAGING_{i}",
                    resources=[Resource(f"PACKAGING_LINE_{i}", packaging_time)],
                    next_tasks=[f"QC_AFTER_PACKAGING_{i}"]
                )
                tasks[f"QC_AFTER_PACKAGING_{i}"] = Gateway(
                    name=f"{variant_name}_QC_AFTER_PACKAGING_{i}",
                    gateway_type="XOR", conditions=[f"qc_after_packaging_{i}"],
                    next_tasks=["END", f"PACKAGING_{i}"]
                )

            tasks["END"] = Task(name=f"{variant_name}_END", next_tasks=[])

            processes.append(ProcessStructure(
                name=variant_name,
                arrival_distribution=l/18,
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
    #'actuator_manufacturing_with_rework',
    #'actuator_manufacturing_no_rework'    
    #'actuator_mfg_pooledM_dedicatedA1_with_rework'
    #'actuator_mfg_pooledM_dedicatedA1_no_rework',
    'all_dedicated_sticky_with_rework'
  # For comparison
]
SIMULATION_RUN_TIME = 30000  # 48 hours in minutes (2 shifts)
SIMULATION_RUNS = 1  # Statistical reliability