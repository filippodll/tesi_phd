import csv
import math
import pathlib
import pickle
import shutil
import threading
import json
import pandas as pd
from collections import deque
import random
import dsf
from dsf import mobility

dsf.set_log_level(dsf.LogLevel.ERROR)

DT_AGENT = 10  # seconds
USE_OD_PROFILES = True  # set False to ignore OD pickles and spawn fully random

NORM_WEIGHTS = False
SMOOTHING_HOURS = 3  # Number of hours to average over (odd number recommended)
BASE_AGENT_COUNT = 50  # Agents to inject every DT_AGENT seconds
CHARGE_INCREMENT = 10  # Permanent increment added to BASE_AGENT_COUNT when triggered

# Stability evaluation settings (mean density of the network)
STABILITY_WINDOW = 24  # Number of recent macroscopic records to evaluate (e.g., ~2 hours if saved every 5 minutes)
STABILITY_MIN_POINTS = 8  # Minimum points required before computing statistics
STABILITY_ALPHA = 0.05  # Significance for the trend test (two-sided)
STABILITY_CV_THRESHOLD = 0.10  # Coefficient of variation threshold for declaring stability (relative dispersion)

print(f"Using dsf version {dsf.__version__}")

origin_nodes = []
destination_nodes = []
if USE_OD_PROFILES:
    origin_nodes = pickle.load(open("./input/origin_dicts.pkl", "rb"))
    destination_nodes = pickle.load(open("./input/destination_dicts.pkl", "rb"))

    # Make all weights 1
    if NORM_WEIGHTS:
        for origin_dict in origin_nodes:
            for key in origin_dict:
                origin_dict[key] = 1
        for dest_dict in destination_nodes:
            for key in dest_dict:
                dest_dict[key] = 1

rn = mobility.RoadNetwork()
rn.importEdges("./input/edges_tl.csv")
rn.importNodeProperties("./input/node_props_tl.csv")
rn.makeRoundabout(72)
rn.initTrafficLights()

def _normalize_id(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return val


print(f"Bologna's road network has {rn.nNodes()} nodes and {rn.nEdges()} edges.")
print(
    f"There are {rn.nCoils()} magnetic coils, {rn.nTrafficLights()} traffic lights and {rn.nRoundabouts()} roundabouts"
)

# Clear output directory
output_dir = pathlib.Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)
for file in output_dir.glob("*"):
    if file.is_file():
        file.unlink()

rn.adjustNodeCapacities()
rn.autoMapStreetLanes()

# Copy edges file to output directory for reference
shutil.copy("./input/edges.csv", "./output/edges.csv")

print(f"Maximum capacity of the network is {rn.capacity()} agents.")

simulator = mobility.Dynamics(rn, False, 69, 0.8)
simulator.killStagnantAgents(5.0)
#simulator.setWeightFunction(mobility.PathWeight.TRAVELTIME, weightThreshold=1.5)

#simulator.setErrorProbability(0.15)
# simulator.initTurnCounts()


# Get the epoch time for 2022-01-31 00:00:00 UTC
epoch_time = 1643587200
simulator.setInitTime(epoch_time)

# Combine all hourly origins/destinations into a single always-on set
def merge_weighted_dicts(dicts):
    merged = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def _parse_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) or math.isinf(parsed) else parsed


def _load_mean_density_series(path, window_size):
    """
    Read the last `window_size` mean density observations from the macroscopic CSV.
    """
    series = deque(maxlen=window_size)
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                value = _parse_float(row.get("mean_density_vpk"))
                if value is not None:
                    series.append(value)
    except FileNotFoundError:
        return []
    return list(series)


def _mann_kendall_test(values):
    """
    Non-parametric trend test. Returns S statistic, Z score and two-sided p-value.
    Implemented explicitly to avoid external dependencies.
    """
    n = len(values)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = values[j] - values[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    # Variance with tie correction
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    var_s = n * (n - 1) * (2 * n + 5)
    for t in counts.values():
        if t > 1:
            var_s -= t * (t - 1) * (2 * t + 5)
    var_s /= 18
    if var_s == 0:
        return s, 0.0, 1.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    # Two-sided p-value using the complementary error function
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return s, z, p_value


def _sen_slope(values):
    """Median pairwise slope (Sen's slope)."""
    slopes = []
    n = len(values)
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((values[j] - values[i]) / (j - i))
    if not slopes:
        return 0.0
    slopes.sort()
    mid = len(slopes) // 2
    if len(slopes) % 2 == 0:
        return 0.5 * (slopes[mid - 1] + slopes[mid])
    return slopes[mid]


def evaluate_density_stability(path, alpha=STABILITY_ALPHA, cv_threshold=STABILITY_CV_THRESHOLD):
    """
    Evaluate network stability using the mean density time series:
      - Mann-Kendall test for monotonic trends (p-value)
      - Sen's slope (robust trend magnitude)
      - Coefficient of variation as dispersion measure
    Returns a dict with metrics or None if not enough data.
    """
    series = _load_mean_density_series(path, STABILITY_WINDOW)
    n = len(series)
    if n < STABILITY_MIN_POINTS:
        return None

    mean_val = sum(series) / n
    variance = sum((x - mean_val) ** 2 for x in series) / (n - 1)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_val if mean_val else float("inf")

    s, z, p_value = _mann_kendall_test(series)
    sen_slope = _sen_slope(series)
    stable = p_value > alpha and cv <= cv_threshold

    return {
        "samples": n,
        "mean_density_vpk": mean_val,
        "std_density_vpk": std_dev,
        "coefficient_of_variation": cv,
        "mann_kendall_s": s,
        "z_score": z,
        "p_value": p_value,
        "sen_slope_per_step": sen_slope,
        "is_stable": stable,
    }


def report_density_stability(path):
    stats = evaluate_density_stability(path)
    if not stats:
        return
    with open("./output/stability.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(
        "[stability] "
        f"n={stats['samples']} mean={stats['mean_density_vpk']:.3f} "
        f"CV={stats['coefficient_of_variation']:.3f} "
        f"p={stats['p_value']:.3f} slope={stats['sen_slope_per_step']:.3e} "
        f"stable={stats['is_stable']}"
    )

if USE_OD_PROFILES and origin_nodes and destination_nodes:
    _nodes_df = pd.read_csv("./input/node_props_tl.csv", sep=';')
    _tl_ids = {
        _normalize_id(i)
        for i in _nodes_df.loc[
            _nodes_df["type"].str.contains("traffic_signals", case=False, na=False), "id"
        ]
    }
    _allowed_nodes = [
        _normalize_id(i) for i in _nodes_df["id"] if _normalize_id(i) not in _tl_ids
    ]
    def _skip_tl(d):
        if not _allowed_nodes:
            return d
        return {
            (random.choice(_allowed_nodes) if _normalize_id(k) in _tl_ids else k): v
            for k, v in d.items()
        }
    combined_origins = _skip_tl(merge_weighted_dicts(origin_nodes))
    combined_destinations = _skip_tl(merge_weighted_dicts(destination_nodes))
    print(
        f"Using combined origins ({len(combined_origins)}) and destinations ({len(combined_destinations)}) for all time."
    )
    simulator.setOriginNodes(combined_origins)
    simulator.setDestinationNodes(combined_destinations)
else:
    print("OD profiles disabled; agents will spawn with fully random origins/destinations.")


# Mutable base count protected by a lock; grows permanently when user types "c"
charge_lock = threading.Lock()
base_agent_count = BASE_AGENT_COUNT

# Background listener to trigger a permanent base increase when user types "c" + Enter
def listen_for_charge_more():
    global base_agent_count
    try:
        while True:
            cmd = input().strip().lower()
            if cmd == "c":
                with charge_lock:
                    base_agent_count += CHARGE_INCREMENT
                    updated = base_agent_count
                print(f"Charge boost applied. New BASE_AGENT_COUNT: {updated}")
    except EOFError:
        pass

threading.Thread(target=listen_for_charge_more, daemon=True).start()

try:
    i = 0
    simulator.updatePaths()
    while True:
        #if i % 300 == 0:
            #simulator.updatePaths()
            # print(f"Updated paths")

        if i >= 0:
            if i % 3600 == 0:
                simulator.saveCoilCounts("./output/counts.csv", True)
            if i % 300 == 0:
                simulator.saveStreetDensities("./output/densities.csv", True)
                simulator.saveTravelData("./output/speeds.csv")
                # if i % 1500 == 0:
                simulator.saveMacroscopicObservables("./output/data.csv")
                report_density_stability("./output/data.csv")
        if i % DT_AGENT == 0:
            with charge_lock:
                n_agents = base_agent_count
            simulator.addAgentsRandomly(n_agents if n_agents > 0 else 0)
        simulator.evolve(False)
        i += 1
except KeyboardInterrupt:
    print("Simulation stopped by user.")

# counts = simulator.normalizedTurnCounts()
# # Counts is a dict like {edge_id: {turn_edge_id: count, ...}, ...}
# # Format floats as strings with two decimal places
# import json
# def format_floats(d):
#     if isinstance(d, dict):
#         return {k: format_floats(v) for k, v in d.items()}
#     elif isinstance(d, float):
#         return f"{d:.2f}"
#     else:
#         return d
# counts_formatted = format_floats(counts)
# with open("./output/turn_counts.json", "w") as f:
#     json.dump(counts_formatted, f)
