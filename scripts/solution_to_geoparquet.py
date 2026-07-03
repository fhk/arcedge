#!/usr/bin/env python3
"""Maps arcedge solutions back onto street-graph geometry and writes GeoParquet.

Two modes:

  design  --graph G --pois P --solution S --out F.parquet
      Hub-location solution: hubs as Points, used edges as LineStrings split
      into kind='cable' (street) and kind='drop' (POI connector) with their
      load and length, and POIs as Points carrying their serving hub.

  flow    --street G --instance I --flow F --out F.parquet [--check-ub R]
      Time-expanded MCF solution: per-arc flows are mapped back to the static
      street graph (movement arcs aggregated over all time layers onto their
      street edge; waiting flow aggregated per node) and written as
      LineStrings (kind='street_flow') and Points (kind='wait'). With
      --check-ub, verifies sum(flow * cost) equals the solver's reported UB.

Geometry is WGS84 lon/lat (OGC:CRS84, the GeoParquet default), edges as
straight chords between node coordinates -- the same geometry the graph file
carries. Output opens directly in QGIS / geopandas / DuckDB spatial.
"""
import argparse
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import to_wkb
from shapely.geometry import LineString, Point


def read_graph(path):
    nodes = []
    edges = {}  # (min,max) -> length
    with open(path) as f:
        for line in f:
            if not line.strip() or line[0] == "c":
                continue
            p = line.split()
            if p[0] == "g":
                nodes = [None] * int(p[1])
            elif p[0] == "v":
                nodes[int(p[1])] = (float(p[2]), float(p[3]))
            elif p[0] == "e":
                u, v, w = int(p[1]), int(p[2]), float(p[3])
                edges.setdefault((min(u, v), max(u, v)), w)
    return nodes, edges


def write_geoparquet(path, columns, geoms):
    geom_types = sorted({g.geom_type for g in geoms})
    table = pa.table({**columns, "geometry": pa.array(to_wkb(geoms))})
    geo_meta = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "geometry_types": geom_types}},
    }
    table = table.replace_schema_metadata(
        {b"geo": json.dumps(geo_meta).encode(), b"arcedge": b"solution export"})
    pq.write_table(table, path)
    return len(geoms)


def run_design(args):
    nodes, edge_len = read_graph(args.graph)
    poi_set = set(int(l) for l in open(args.pois))
    hubs, used, assign = [], [], []
    with open(args.solution) as f:
        for line in f:
            p = line.split()
            if not p or p[0] == "c":
                continue
            if p[0] == "h":
                hubs.append(int(p[1]))
            elif p[0] == "e":
                used.append((int(p[1]), int(p[2]), float(p[3])))
            elif p[0] == "p":
                assign.append((int(p[1]), int(p[2])))

    kind, u_col, v_col, load, length, hub_col, geoms = [], [], [], [], [], [], []
    for h in hubs:
        kind.append("hub"); u_col.append(h); v_col.append(-1)
        load.append(None); length.append(None); hub_col.append(h)
        geoms.append(Point(nodes[h]))
    for u, v, ld in used:
        is_drop = u in poi_set or v in poi_set
        kind.append("drop" if is_drop else "cable")
        u_col.append(u); v_col.append(v); load.append(ld)
        length.append(edge_len.get((min(u, v), max(u, v))))
        hub_col.append(-1)
        geoms.append(LineString([nodes[u], nodes[v]]))
    for poi, h in assign:
        kind.append("poi"); u_col.append(poi); v_col.append(-1)
        load.append(None); length.append(None); hub_col.append(h)
        geoms.append(Point(nodes[poi]))

    n = write_geoparquet(args.out, {
        "kind": kind, "u": u_col, "v": v_col, "load": load,
        "length_m": length, "hub": hub_col}, geoms)

    cable_m = sum(l for k, l in zip(kind, length) if k in ("cable", "drop") and l)
    drops = sum(1 for k in kind if k == "drop")
    print(f"wrote {args.out}: {n} features "
          f"({len(hubs)} hubs, {len(used) - drops} cable edges, {drops} drops, "
          f"{len(assign)} POIs); mapped cable {cable_m:.0f} m")
    unassigned = [p for p, h in assign if h < 0]
    if unassigned:
        print(f"ERROR: {len(unassigned)} POIs have no serving hub", file=sys.stderr)
        return 1
    return 0


def run_flow(args):
    nodes, edge_len = read_graph(args.street)
    # Street arcs in file order define the time-expansion arc layout:
    # per transition t: all directed street arcs, then one wait arc per node.
    street_arcs = []
    with open(args.street) as f:
        for line in f:
            p = line.split()
            if p and p[0] == "e":
                street_arcs.append((int(p[1]), int(p[2]), float(p[3])))
    n_static, n_move = len(nodes), len(street_arcs)

    te_nodes = te_arcs = None
    arc_cost = []
    with open(args.instance) as f:
        for line in f:
            p = line.split()
            if not p or p[0] == "c":
                continue
            if p[0] == "p":
                te_nodes, te_arcs = int(p[1]), int(p[2])
            elif p[0] == "a" and args.check_ub:
                arc_cost.append(float(p[3]))
    T = te_nodes // n_static
    layer = n_move + n_static  # arcs per time transition
    assert te_nodes == n_static * T, "instance does not match street graph"
    assert te_arcs == layer * (T - 1), "arc layout mismatch: not a street TE instance"

    flows = {}
    with open(args.flow) as f:
        for line in f:
            p = line.split()
            if p and p[0] == "a":
                flows[int(p[1])] = float(p[2])

    if args.check_ub:
        cost = sum(arc_cost[a] * fl for a, fl in flows.items())
        ub = dict(l.split() for l in open(args.check_ub))
        ub = float(ub["ub"])
        rel = abs(cost - ub) / max(abs(ub), 1.0)
        print(f"flow cost check: sum(flow*cost) = {cost:.6f} vs solver ub = {ub:.6f} "
              f"({'OK' if rel < 1e-9 else 'MISMATCH'})")
        if rel >= 1e-9:
            return 1

    # Aggregate across time layers back onto the static graph.
    street_flow = np.zeros(n_move)
    wait_flow = np.zeros(n_static)
    for a, fl in flows.items():
        i = a % layer
        if i < n_move:
            street_flow[i] += fl
        else:
            wait_flow[i - n_move] += fl

    # Combine the two directions of each street edge into one feature.
    edge_flow = {}
    for i, (u, v, _) in enumerate(street_arcs):
        if street_flow[i] > 0:
            edge_flow[(min(u, v), max(u, v))] = \
                edge_flow.get((min(u, v), max(u, v)), 0.0) + street_flow[i]

    kind, u_col, v_col, flow_col, length, geoms = [], [], [], [], [], []
    for (u, v), fl in sorted(edge_flow.items()):
        kind.append("street_flow"); u_col.append(u); v_col.append(v)
        flow_col.append(fl); length.append(edge_len.get((u, v)))
        geoms.append(LineString([nodes[u], nodes[v]]))
    for vtx in np.nonzero(wait_flow)[0]:
        kind.append("wait"); u_col.append(int(vtx)); v_col.append(-1)
        flow_col.append(float(wait_flow[vtx])); length.append(None)
        geoms.append(Point(nodes[vtx]))

    n = write_geoparquet(args.out, {
        "kind": kind, "u": u_col, "v": v_col, "flow": flow_col,
        "length_m": length}, geoms)
    print(f"wrote {args.out}: {n} features ({len(edge_flow)} street edges with "
          f"flow, {int((wait_flow > 0).sum())} wait nodes) from {T} time layers")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("design")
    d.add_argument("--graph", required=True)
    d.add_argument("--pois", required=True)
    d.add_argument("--solution", required=True)
    d.add_argument("--out", required=True)
    fl = sub.add_parser("flow")
    fl.add_argument("--street", required=True)
    fl.add_argument("--instance", required=True)
    fl.add_argument("--flow", required=True)
    fl.add_argument("--out", required=True)
    fl.add_argument("--check-ub", help="result file to verify sum(flow*cost) == ub")
    args = ap.parse_args()
    return run_design(args) if args.mode == "design" else run_flow(args)


if __name__ == "__main__":
    sys.exit(main())
