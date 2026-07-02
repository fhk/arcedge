#!/usr/bin/env python3
"""Converts Overture Maps transportation segments (GeoParquet) into an arcedge
static street graph.

Each LineString endpoint becomes a node (deduplicated by rounded coordinate,
so segments that meet at an intersection share a node), and every segment
produces arcs in BOTH directions weighted by the geodesic length in meters of
the full polyline. Only the largest connected component is kept, since
commodities must be routable between any sampled origin/destination pair.

Output format (read by the arcedge CLI's `gen --street`):
    g <num_nodes> <num_arcs>
    v <node_id> <lon> <lat>
    e <tail> <head> <length_m>
"""
import argparse
import collections
import math
import sys

import pyarrow.parquet as pq
from shapely import from_wkb

# Pedestrian-only classes excluded by default; the solver models street
# networks (fiber/vehicle routing along roads). Override with --all-classes.
DEFAULT_EXCLUDED_CLASSES = {"footway", "steps", "path", "bridleway"}


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def polyline_length_m(coords):
    return sum(haversine_m(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out")
    ap.add_argument("--all-classes", action="store_true",
                    help="keep pedestrian-only classes (footway, steps, ...)")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="only keep segments whose endpoints are inside this lon/lat box")
    ap.add_argument("--round", type=int, default=7,
                    help="coordinate rounding (decimal places) for node dedup")
    args = ap.parse_args()

    table = pq.read_table(args.parquet, columns=["subtype", "class", "geometry"])
    subtypes = table["subtype"].to_pylist()
    classes = table["class"].to_pylist()
    geoms = from_wkb(table["geometry"].to_pylist())

    excluded = set() if args.all_classes else DEFAULT_EXCLUDED_CLASSES
    node_id = {}
    node_coord = []
    edges = []  # (u, v, length_m), undirected
    kept = 0
    skipped = collections.Counter()

    def node_of(lon, lat):
        key = (round(lon, args.round), round(lat, args.round))
        nid = node_id.get(key)
        if nid is None:
            nid = len(node_coord)
            node_id[key] = nid
            node_coord.append(key)
        return nid

    for subtype, cls, geom in zip(subtypes, classes, geoms):
        if subtype != "road":
            skipped[f"subtype:{subtype}"] += 1
            continue
        if cls in excluded:
            skipped[f"class:{cls}"] += 1
            continue
        if geom is None or geom.geom_type != "LineString" or len(geom.coords) < 2:
            skipped["bad-geometry"] += 1
            continue
        coords = list(geom.coords)
        (lon_a, lat_a), (lon_b, lat_b) = coords[0][:2], coords[-1][:2]
        if args.bbox:
            xmin, ymin, xmax, ymax = args.bbox
            if not (xmin <= lon_a <= xmax and ymin <= lat_a <= ymax and
                    xmin <= lon_b <= xmax and ymin <= lat_b <= ymax):
                skipped["outside-bbox"] += 1
                continue
        u, v = node_of(lon_a, lat_a), node_of(lon_b, lat_b)
        if u == v:
            skipped["degenerate-loop"] += 1
            continue
        edges.append((u, v, polyline_length_m([(c[0], c[1]) for c in coords])))
        kept += 1

    # Largest connected component (undirected == strongly connected here,
    # because every edge is emitted in both directions).
    adj = collections.defaultdict(list)
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = {}
    best_comp, best_size = -1, 0
    comp = 0
    for start in range(len(node_coord)):
        if start in seen or start not in adj:
            continue
        stack, size = [start], 0
        seen[start] = comp
        while stack:
            x = stack.pop()
            size += 1
            for y in adj[x]:
                if y not in seen:
                    seen[y] = comp
                    stack.append(y)
        if size > best_size:
            best_comp, best_size = comp, size
        comp += 1
    keep_edges = [(u, v, w) for u, v, w in edges if seen.get(u) == best_comp]

    remap = {}
    out_nodes = []
    for u, v, _ in keep_edges:
        for x in (u, v):
            if x not in remap:
                remap[x] = len(out_nodes)
                out_nodes.append(node_coord[x])
    with open(args.out, "w") as f:
        f.write(f"c arcedge static street graph from {args.parquet}\n")
        f.write(f"c bidirectional; cost = geodesic polyline length in meters\n")
        f.write(f"g {len(out_nodes)} {2 * len(keep_edges)}\n")
        for i, (lon, lat) in enumerate(out_nodes):
            f.write(f"v {i} {lon} {lat}\n")
        for u, v, w in keep_edges:
            f.write(f"e {remap[u]} {remap[v]} {w:.2f}\n")
            f.write(f"e {remap[v]} {remap[u]} {w:.2f}\n")

    total_km = sum(w for _, _, w in keep_edges) / 1000.0
    print(f"segments kept: {kept} (skipped: {dict(skipped)})")
    print(f"components: {comp}; largest has {best_size} nodes, "
          f"{len(keep_edges)} segments ({100.0 * len(keep_edges) / max(kept, 1):.1f}% of kept)")
    print(f"wrote {args.out}: {len(out_nodes)} nodes, {2 * len(keep_edges)} directed arcs, "
          f"{total_km:.1f} km of street")
    return 0


if __name__ == "__main__":
    sys.exit(main())
