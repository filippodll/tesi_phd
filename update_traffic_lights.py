"""
Fetch OSM nodes, save them to a CSV, and update `input/node_props.csv`
setting the `type` column to `traffic_light` when a nearby OSM node is
tagged with `highway=traffic_signals`.
"""

import argparse
import csv
import json
import math
import pathlib
import sys
from typing import Iterable, List, Tuple

import requests


def parse_point_wkt(wkt: str) -> Tuple[float, float]:
    """
    Parse strings like `POINT (lon lat)` into (lat, lon).
    """
    if not wkt.startswith("POINT"):
        raise ValueError(f"Unsupported geometry: {wkt}")
    start = wkt.find("(")
    end = wkt.find(")", start)
    if start == -1 or end == -1:
        raise ValueError(f"Malformed POINT WKT: {wkt}")
    coords = wkt[start + 1 : end].split()
    if len(coords) != 2:
        raise ValueError(f"Malformed POINT WKT: {wkt}")
    lon, lat = map(float, coords)
    return lat, lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in meters.
    """
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2
    ) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_node_props(path: pathlib.Path) -> Tuple[List[dict], List[str]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = []
        for row in reader:
            lat, lon = parse_point_wkt(row["geometry"])
            row["_lat"] = lat
            row["_lon"] = lon
            rows.append(row)
        return rows, reader.fieldnames  # type: ignore[arg-type]


def bbox_from_nodes(rows: Iterable[dict], pad: float = 0.001) -> Tuple[float, float, float, float]:
    lats = [r["_lat"] for r in rows]
    lons = [r["_lon"] for r in rows]
    south = min(lats) - pad
    north = max(lats) + pad
    west = min(lons) - pad
    east = max(lons) + pad
    return south, west, north, east


def fetch_osm_nodes(
    bbox: Tuple[float, float, float, float], overpass_url: str
) -> List[dict]:
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:180];
    (
      node["highway"]({south},{west},{north},{east});
    );
    out body;
    """
    response = requests.post(overpass_url, data={"data": query})
    response.raise_for_status()
    data = response.json()
    return data.get("elements", [])


def write_osm_nodes_csv(nodes: Iterable[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "lat", "lon", "highway", "tags"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for node in nodes:
            tags = node.get("tags", {})
            writer.writerow(
                {
                    "id": node.get("id"),
                    "lat": node.get("lat"),
                    "lon": node.get("lon"),
                    "highway": tags.get("highway"),
                    "tags": json.dumps(tags, ensure_ascii=True, sort_keys=True),
                }
            )


def update_node_props_with_signals(
    node_props: List[dict], osm_nodes: List[dict], tolerance_m: float
) -> int:
    signals = [
        (n["lat"], n["lon"])
        for n in osm_nodes
        if n.get("tags", {}).get("highway") == "traffic_signals"
    ]
    updated = 0
    for row in node_props:
        lat, lon = row["_lat"], row["_lon"]
        if any(haversine_m(lat, lon, s_lat, s_lon) <= tolerance_m for s_lat, s_lon in signals):
            if row.get("type") != "traffic_signals":
                row["type"] = "traffic_signals"
                updated += 1
    return updated


def write_node_props(rows: List[dict], fieldnames: List[str], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            row_copy = {k: row[k] for k in fieldnames}
            writer.writerow(row_copy)


def parse_bbox(arg: str) -> Tuple[float, float, float, float]:
    parts = [float(p) for p in arg.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be south,west,north,east")
    return parts[0], parts[1], parts[2], parts[3]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch OSM nodes and update node_props.csv traffic lights."
    )
    parser.add_argument(
        "--node-props",
        type=pathlib.Path,
        default=pathlib.Path("input/node_props.csv"),
        help="Path to the node_props.csv file to read.",
    )
    parser.add_argument(
        "--output-node-props",
        type=pathlib.Path,
        default=None,
        help="Where to write the updated node_props.csv (default: in-place).",
    )
    parser.add_argument(
        "--save-osm-nodes",
        type=pathlib.Path,
        default=pathlib.Path("output/osm_nodes.csv"),
        help="Where to save the fetched OSM nodes CSV.",
    )
    parser.add_argument(
        "--overpass-url",
        default="https://overpass-api.de/api/interpreter",
        help="Overpass API endpoint.",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        help="Optional bounding box 'south,west,north,east'. "
        "Defaults to the extent of node_props with a small pad.",
    )
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=10.0,
        help="Maximum distance in meters to consider a node a traffic light match.",
    )

    args = parser.parse_args()

    node_props_path: pathlib.Path = args.node_props
    output_node_props: pathlib.Path = args.output_node_props or node_props_path
    node_props, fieldnames = load_node_props(node_props_path)

    bbox = args.bbox or bbox_from_nodes(node_props)
    osm_nodes = fetch_osm_nodes(bbox, args.overpass_url)
    write_osm_nodes_csv(osm_nodes, args.save_osm_nodes)

    updated_count = update_node_props_with_signals(
        node_props, osm_nodes, args.tolerance_m
    )
    write_node_props(node_props, fieldnames, output_node_props)

    print(
        f"Updated {updated_count} nodes to traffic_light. "
        f"OSM nodes saved to {args.save_osm_nodes}. "
        f"node_props written to {output_node_props}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
