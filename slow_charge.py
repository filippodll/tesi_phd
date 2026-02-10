#!/usr/bin/env python3
"""
Traffic simulation entry point for the Bologna road network using DSF.

This version favors explicit, in-code configuration. Edit the CONFIG object
below to adjust cartography, solver, runtime, or demand parameters. Agents can
be spawned from an Origin-Destination (OD) matrix, from random mobility, or a
mix of the two.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely import wkt

from dsf import mobility
from dsf.python.cartography import graph_from_gdfs


@dataclass(frozen=True)
class ODPair:
    """Single OD entry expressed as agents per minute."""

    origin: int
    destination: int
    rate_per_minute: float


@dataclass
class CartographyConfig:
    place: str = "Bologna, Emilia-Romagna, Italy"
    network_type: str = "drive"
    consolidate: float = 15.0
    infer_speeds: bool = True
    default_speed: float = 35.0
    edges_path: Path = Path("input/edges.csv")
    nodes_path: Path = Path("input/node_props.csv")
    roundabout_node: int | None = 72


@dataclass
class RuntimeConfig:
    duration_minutes: int = 180
    max_steps: int | None = None
    spawn_interval: int = 60  # evolution steps between injections
    log_period: int = 600
    save_period: int = 300
    output_dir: Path | None = None
    preview_html: Path | None = Path("outputs/bologna_cartography.html")
    preview_only: bool = False


@dataclass
class DynamicsConfig:
    seed: int = 1234
    alpha: float = 0.6
    error_probability: float = 0.2
    reinsert_finished: bool = False
    traffic_light_threshold: int = 4
    traffic_light_cycle: int = 120
    traffic_light_offset: int = 0
    min_green: int = 30


@dataclass
class ODMatrixConfig:
    path: Path | None = None
    separator: str = ","
    demand_scale: float = 1.0


@dataclass
class RandomMobilityConfig:
    enabled: bool = False
    agents_per_minute: float = 0.0
    origin_weights: dict[int, float] | None = None
    destination_weights: dict[int, float] | None = None


@dataclass
class SimulationConfig:
    cartography: CartographyConfig = field(default_factory=CartographyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    od: ODMatrixConfig = field(default_factory=ODMatrixConfig)
    random: RandomMobilityConfig = field(default_factory=RandomMobilityConfig)
    log_level: str = "INFO"


# --------------------------------------------------------------------------- #
# Customize the simulation by editing CONFIG directly.
# --------------------------------------------------------------------------- #
CONFIG = SimulationConfig(
    od=ODMatrixConfig(
        path=Path("data/bologna_od.csv"),
        separator=",",
        demand_scale=1.0,
    ),
    random=RandomMobilityConfig(
        enabled=True,
        agents_per_minute=50.0,
        origin_weights=None,  # e.g. {123456789: 2.0, 987654321: 1.0}
        destination_weights=None,
    ),
    runtime=RuntimeConfig(
        duration_minutes=240,
        spawn_interval=60,
        log_period=600,
        save_period=300,
        output_dir=Path("outputs/bologna_example"),
    ),
    dynamics=DynamicsConfig(
        seed=42,
        alpha=0.6,
        error_probability=0.05,
        reinsert_finished=False,
        traffic_light_threshold=4,
        traffic_light_cycle=120,
        traffic_light_offset=0,
        min_green=30,
    ),
    cartography=CartographyConfig(
        place="Bologna, Emilia-Romagna, Italy",
        network_type="drive",
        consolidate=15.0,
        infer_speeds=True,
        default_speed=35.0,
    ),
    log_level="INFO",
)


def load_od_matrix(config: ODMatrixConfig) -> list[ODPair]:
    if config.path is None:
        logging.info("No OD matrix configured; skipping OD demand.")
        return []
    if not config.path.exists():
        raise FileNotFoundError(f"OD matrix file not found: {config.path}")
    required_cols = {"origin_id", "destination_id", "agents_per_min"}
    logging.info("Loading OD matrix from %s", config.path)
    df = pd.read_csv(config.path, sep=config.separator)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"OD matrix is missing columns: {sorted(missing)}")
    od_pairs: list[ODPair] = []
    for row in df.itertuples():
        rate = float(row.agents_per_min)
        if rate <= 0:
            continue
        od_pairs.append(
            ODPair(origin=int(row.origin_id), destination=int(row.destination_id), rate_per_minute=rate)
        )
    logging.info("Loaded %d OD pairs.", len(od_pairs))
    return od_pairs


def download_bologna_cartography(config: CartographyConfig) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    edges_path = Path(config.edges_path)
    nodes_path = Path(config.nodes_path)
    if not edges_path.exists() or not nodes_path.exists():
        raise FileNotFoundError(f"Cartography files not found: {edges_path} and/or {nodes_path}")

    logging.info("Loading cartography from %s and %s", edges_path, nodes_path)
    edges_df = pd.read_csv(edges_path, sep=";")
    nodes_df = pd.read_csv(nodes_path, sep=";")

    for col in ("id", "source", "target"):
        edges_df[col] = edges_df[col].astype(int)
    nodes_df["id"] = nodes_df["id"].astype(int)

    edges_df["nlanes"] = pd.to_numeric(edges_df.get("nlanes"), errors="coerce").fillna(1).astype(int)
    edges_df["maxspeed"] = pd.to_numeric(edges_df.get("maxspeed"), errors="coerce").fillna(config.default_speed)
    if "length" in edges_df:
        edges_df["length"] = pd.to_numeric(edges_df["length"], errors="coerce")

    edges_df["geometry"] = edges_df["geometry"].apply(wkt.loads)
    nodes_df["geometry"] = nodes_df["geometry"].apply(wkt.loads)

    edges = gpd.GeoDataFrame(edges_df, geometry="geometry", crs="EPSG:4326")
    nodes = gpd.GeoDataFrame(nodes_df, geometry="geometry", crs="EPSG:4326")
    logging.info("Retrieved %d edges and %d nodes from local cartography.", len(edges), len(nodes))
    return edges, nodes


def export_cartography_html(edges: pd.DataFrame, nodes: pd.DataFrame, html_path: Path) -> None:
    """Write a self-contained Leaflet map with edges and nodes."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    edges_json = edges.to_json()
    nodes_json = nodes.to_json()
    template = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>DSF Cartography Preview</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ width: 100%; height: 100%; margin: 0; padding: 0; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    const edges = {edges_json};
    const nodes = {nodes_json};
    const map = L.map('map');
    const edgeLayer = L.geoJSON(edges, {{
      style: () => ({{ color: '#2364AA', weight: 2, opacity: 0.8 }}),
    }}).addTo(map);
    const nodeLayer = L.geoJSON(nodes, {{
      pointToLayer: (_, latlng) => L.circleMarker(latlng, {{
        radius: 3,
        color: '#F3A712',
        weight: 1,
        fillColor: '#F3DFA2',
        fillOpacity: 0.9,
      }}),
    }}).addTo(map);
    const bounds = edgeLayer.getBounds().isValid() ? edgeLayer.getBounds() : nodeLayer.getBounds();
    map.fitBounds(bounds, {{ padding: [20, 20] }});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    }}).addTo(map);
    L.control.layers(null, {{ Edges: edgeLayer, Nodes: nodeLayer }}).addTo(map);
  </script>
</body>
</html>
"""
    html_path.write_text(template, encoding="utf-8")
    logging.info("Cartography preview written to %s", html_path)


def save_cartography_data(edges: pd.DataFrame, nodes: pd.DataFrame, out_dir: Path) -> None:
    """Persist edges and nodes to CSV for external export/reuse."""
    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_dir / "edges.csv", index=False)
    nodes.to_csv(out_dir / "nodes.csv", index=False)
    logging.info("Cartography data written to %s (edges.csv, nodes.csv)", out_dir)


def build_network(edges: pd.DataFrame, nodes: pd.DataFrame, config: CartographyConfig | None = None) -> mobility.RoadNetwork:
    cfg = config or CONFIG.cartography
    graph = mobility.RoadNetwork()
    graph.importEdges(str(cfg.edges_path))
    graph.importNodeProperties(str(cfg.nodes_path))
    if cfg.roundabout_node is not None:
        try:
            graph.makeRoundabout(int(cfg.roundabout_node))
        except RuntimeError as exc:
            logging.warning("Unable to create roundabout at node %s: %s", cfg.roundabout_node, exc)
    graph.adjustNodeCapacities()
    graph.autoMapStreetLanes()
    logging.info("RoadNetwork built with %d nodes and %d streets.", graph.nNodes(), graph.nEdges())
    return graph


def detect_signalized_nodes(edges: pd.DataFrame, threshold: int) -> list[int]:
    outgoing_counts = edges.groupby("source").size()
    return [int(node_id) for node_id, degree in outgoing_counts.items() if degree >= threshold]


def setup_controls(graph: mobility.RoadNetwork, edges: pd.DataFrame, config: DynamicsConfig) -> None:
    for node_id in detect_signalized_nodes(edges, config.traffic_light_threshold):
        try:
            graph.makeTrafficLight(node_id, config.traffic_light_cycle, config.traffic_light_offset)
        except RuntimeError as exc:
            logging.debug("Skipping node %s for traffic light creation: %s", node_id, exc)
    b_has_coilcode = "coilcode" in edges.columns
    for _, row in edges.iterrows():
        street_id = int(row["id"])
        if b_has_coilcode and pd.notna(row.get("coilcode")) and str(row.get("coilcode")).lower() not in {"", "nan", "null"}:
            continue
        try:
            graph.addCoil(street_id)
        except RuntimeError as exc:
            logging.debug("Skipping coil on street %s: %s", street_id, exc)
            continue
    graph.adjustNodeCapacities()
    graph.initTrafficLights(config.min_green)


def configure_dynamics(
    graph: mobility.RoadNetwork,
    od_pairs: Iterable[ODPair],
    config: DynamicsConfig,
) -> mobility.Dynamics:
    dynamics = mobility.Dynamics(
        graph,
        useCache=True,
        seed=config.seed,
        alpha=config.alpha,
    )
    if config.error_probability:
        dynamics.setErrorProbability(config.error_probability)

    origin_weights: Counter[int] = Counter()
    destination_weights: Counter[int] = Counter()
    for entry in od_pairs:
        origin_weights[entry.origin] += entry.rate_per_minute
        destination_weights[entry.destination] += entry.rate_per_minute

    del origin_weights[174]
    del destination_weights[174]
    
    if origin_weights:
        dynamics.setOriginNodes(dict(origin_weights))
    if destination_weights:
        dynamics.setDestinationNodes(dict(destination_weights))
    dynamics.updatePaths(throw_on_empty=False)
    return dynamics


def spawn_from_od(
    dynamics: mobility.Dynamics,
    od_pairs: Iterable[ODPair],
    demand_scale: float,
    minutes_per_interval: float,
    accumulators: dict[tuple[int, int], float],
) -> int:
    spawned = 0
    if demand_scale <= 0:
        return spawned
    for entry in od_pairs:
        key = (entry.origin, entry.destination)
        accumulators[key] += entry.rate_per_minute * demand_scale * minutes_per_interval
        n_agents = math.floor(accumulators[key])
        if n_agents == 0:
            continue
        accumulators[key] -= n_agents
        try:
            dynamics.addAgentsRandomly(
                int(n_agents),
                {entry.origin: 1.0},
                {entry.destination: 1.0},
            )
        except OverflowError as exc:
            logging.error("Capacity reached while inserting OD agents: %s", exc)
            raise
        spawned += n_agents
    return spawned


def spawn_random_agents(
    dynamics: mobility.Dynamics,
    random_cfg: RandomMobilityConfig,
    minutes_per_interval: float,
    accumulator: float,
) -> tuple[int, float]:
    if not random_cfg.enabled or random_cfg.agents_per_minute <= 0:
        return 0, accumulator

    accumulator += random_cfg.agents_per_minute * minutes_per_interval
    n_agents = math.floor(accumulator)
    if n_agents == 0:
        return 0, accumulator
    accumulator -= n_agents
    try:
        if random_cfg.origin_weights and random_cfg.destination_weights:
            dynamics.addAgentsRandomly(
                int(n_agents),
                random_cfg.origin_weights,
                random_cfg.destination_weights,
            )
        else:
            dynamics.addAgentsRandomly(int(n_agents))
    except OverflowError as exc:
        logging.error("Capacity reached while inserting random agents: %s", exc)
        raise
    return n_agents, accumulator


def dump_diagnostics(dynamics: mobility.Dynamics, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics.saveStreetDensities(str(output_dir / "street_densities.csv"), True)
    dynamics.saveCoilCounts(str(output_dir / "coil_counts.csv"))
    dynamics.saveMacroscopicObservables(str(output_dir / "macroscopic.csv"))


def ensure_od_in_giant_component(edges: pd.DataFrame, nodes: pd.DataFrame, od_pairs: list[ODPair]) -> None:
    """Abort if any OD node is not in the giant strongly connected component."""
    if not od_pairs:
        return
    G = graph_from_gdfs(edges, nodes)
    scc = list(nx.strongly_connected_components(G))
    if not scc:
        raise SystemExit("Graph has no strongly connected components; aborting.")
    giant = max(scc, key=len)
    od_nodes = {p.origin for p in od_pairs} | {p.destination for p in od_pairs}
    unreachable = sorted(n for n in od_nodes if n not in giant)
    if unreachable:
        raise SystemExit(
            f"OD nodes not in the giant strongly connected component: {unreachable}"
        )


def run_simulation(
    dynamics: mobility.Dynamics,
    od_pairs: list[ODPair],
    config: SimulationConfig,
) -> None:
    runtime = config.runtime
    minutes_per_interval = runtime.spawn_interval / 60.0
    total_steps = runtime.max_steps or runtime.duration_minutes * 60
    od_accumulators: dict[tuple[int, int], float] = defaultdict(float)
    random_accumulator = 0.0
    stats = {"od": 0, "random": 0}

    for step in range(total_steps):
        if step % runtime.spawn_interval == 0:
            if od_pairs:
                stats["od"] += spawn_from_od(
                    dynamics,
                    od_pairs,
                    config.od.demand_scale,
                    minutes_per_interval,
                    od_accumulators,
                )
            spawned_random, random_accumulator = spawn_random_agents(
                dynamics,
                config.random,
                minutes_per_interval,
                random_accumulator,
            )
            stats["random"] += spawned_random

        dynamics.evolve(config.dynamics.reinsert_finished)

        if runtime.output_dir and step % runtime.save_period == 0:
            dump_diagnostics(dynamics, runtime.output_dir)

        if step % runtime.log_period == 0:
            logging.info(
                "t=%d steps, active agents=%d, spawned_od=%d, spawned_random=%d",
                dynamics.time_step(),
                dynamics.nAgents(),
                stats["od"],
                stats["random"],
            )

    logging.info("Simulation finished after %d steps.", total_steps)
    logging.info("Total spawned agents -> OD: %d, random: %d", stats["od"], stats["random"])
    if stats["od"] + stats["random"] > 0:
        logging.info("Mean travel time: %.2f s", dynamics.meanTravelTime(clearData=False).mean)
        logging.info("Mean travel speed: %.2f km/h", dynamics.meanTravelSpeed(clearData=False).mean)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, CONFIG.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    edges, nodes = download_bologna_cartography(CONFIG.cartography)
    save_cartography_data(edges, nodes, Path("data"))
    if CONFIG.runtime.preview_html:
        export_cartography_html(edges, nodes, CONFIG.runtime.preview_html)
        if CONFIG.runtime.preview_only:
            logging.info("Preview-only mode enabled; exiting before simulation.")
            return

    od_pairs = load_od_matrix(CONFIG.od)
    if not od_pairs and not CONFIG.random.enabled:
        raise SystemExit("No OD matrix available and random demand disabled; nothing to simulate.")

    ensure_od_in_giant_component(edges, nodes, od_pairs)

    graph = build_network(edges, nodes)
    setup_controls(graph, edges, CONFIG.dynamics)
    dynamics = configure_dynamics(graph, od_pairs, CONFIG.dynamics)
    run_simulation(dynamics, od_pairs, CONFIG)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Simulation interrupted by user.")
