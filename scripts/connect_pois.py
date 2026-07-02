#!/usr/bin/env python3
"""Connects Overture Places POIs to an arcedge street graph.

For every POI point, finds the nearest street edge, drops a perpendicular
(straight line) to it, BREAKS the edge at the intersection (foot) point, and
adds the drop as a new bidirectional edge from a new POI node to the new
split node. Edges hit by several POIs are split at every foot point. Foot
points landing within --snap meters of an existing node reuse that node
instead of creating a degenerate sliver edge.

Edge geometry in the .graph file is the straight chord between endpoints, but
the stored length is the original polyline length; child edges of a split
inherit lengths proportional to the split parameter so total street length is
preserved.

Outputs:
  <out>.graph  augmented street graph (same g/v/e format)
  <out>.pois   one POI node id per line
"""
import argparse
import math
import sys

import numpy as np
import pyarrow.parquet as pq
from shapely import from_wkb, STRtree
from shapely.geometry import LineString, Point

LAT0 = 37.77  # projection reference latitude (SF); equirectangular is fine
KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110540.0


def project(lon, lat):
    return lon * KX, lat * KY


def unproject(x, y):
    return x / KX, y / KY


def read_graph(path):
    nodes = []  # (lon, lat)
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
    return nodes, sorted(edges.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("places_parquet")
    ap.add_argument("out_prefix")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="only connect POIs inside this lon/lat box")
    ap.add_argument("--max-pois", type=int, default=0,
                    help="deterministically subsample to at most this many POIs (0 = all)")
    ap.add_argument("--snap", type=float, default=0.5,
                    help="snap foot points within this many meters of an existing node")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    nodes, edge_items = read_graph(args.graph)
    xy = [project(lon, lat) for lon, lat in nodes]

    table = pq.read_table(args.places_parquet, columns=["geometry", "confidence"])
    pois = []
    for geom, conf in zip(from_wkb(table["geometry"].to_pylist()),
                          table["confidence"].to_pylist()):
        if geom is None or geom.geom_type != "Point":
            continue
        if conf is not None and conf < args.min_confidence:
            continue
        lon, lat = geom.x, geom.y
        if args.bbox:
            xmin, ymin, xmax, ymax = args.bbox
            if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
                continue
        pois.append((lon, lat))
    if args.max_pois and len(pois) > args.max_pois:
        step = len(pois) / args.max_pois
        pois = [pois[int(i * step)] for i in range(args.max_pois)]
    if not pois:
        print("no POIs after filtering", file=sys.stderr)
        return 1

    # Nearest street edge per POI via an STRtree over projected chords.
    chords = [LineString([xy[u], xy[v]]) for (u, v), _ in edge_items]
    tree = STRtree(chords)
    poi_pts = [Point(project(lon, lat)) for lon, lat in pois]
    nearest = tree.query_nearest(poi_pts, all_matches=False)
    # query_nearest returns (input_idx, tree_idx) pairs
    edge_of_poi = {int(i): int(e) for i, e in zip(nearest[0], nearest[1])}

    # Collect split parameters per edge: t in (0,1) along the chord.
    splits = {}  # edge_idx -> list of (t, poi_idx, foot_xy, drop_len)
    drop_lens = []
    for p_idx, pt in enumerate(poi_pts):
        e_idx = edge_of_poi[p_idx]
        chord = chords[e_idx]
        t = chord.project(pt) / max(chord.length, 1e-12)
        foot = chord.interpolate(chord.project(pt))
        drop = pt.distance(foot)
        drop_lens.append(drop)
        splits.setdefault(e_idx, []).append((t, p_idx, (foot.x, foot.y), drop))

    new_nodes = list(nodes)  # (lon, lat)
    out_edges = []           # (u, v, length_m)
    poi_node_of = {}

    def add_node(x, y):
        new_nodes.append(unproject(x, y))
        return len(new_nodes) - 1

    n_split_nodes = 0
    for e_idx, ((u, v), length) in enumerate(edge_items):
        if e_idx not in splits:
            out_edges.append((u, v, length))
            continue
        chord_len = max(chords[e_idx].length, 1e-12)
        snap_t = args.snap / chord_len
        # Walk the chord u -> v, creating a split node at each distinct foot.
        prev_node, prev_t = u, 0.0
        for t, p_idx, (fx, fy), drop in sorted(splits[e_idx]):
            if t <= snap_t:
                foot_node = u
            elif t >= 1.0 - snap_t:
                foot_node = v
            elif prev_node != u and (t - prev_t) * chord_len < args.snap:
                foot_node = prev_node  # merge with the previous split point
            else:
                foot_node = add_node(fx, fy)
                n_split_nodes += 1
                out_edges.append((prev_node, foot_node, max(t - prev_t, 0.0) * length))
                prev_node, prev_t = foot_node, t
            poi_node = add_node(*project(*pois[p_idx]))
            poi_node_of[p_idx] = poi_node
            out_edges.append((poi_node, foot_node, max(drop, 0.1)))
        if prev_node != v:
            out_edges.append((prev_node, v, max(1.0 - prev_t, 0.0) * length))

    with open(args.out_prefix + ".graph", "w") as f:
        f.write(f"c arcedge access graph: {args.graph} + POI drops from {args.places_parquet}\n")
        f.write(f"g {len(new_nodes)} {2 * len(out_edges)}\n")
        for i, (lon, lat) in enumerate(new_nodes):
            f.write(f"v {i} {lon:.7f} {lat:.7f}\n")
        for u, v, w in out_edges:
            f.write(f"e {u} {v} {w:.2f}\n")
            f.write(f"e {v} {u} {w:.2f}\n")
    with open(args.out_prefix + ".pois", "w") as f:
        for p_idx in range(len(pois)):
            f.write(f"{poi_node_of[p_idx]}\n")

    d = np.array(drop_lens)
    print(f"POIs connected: {len(pois)}; edges split at {n_split_nodes} new nodes")
    print(f"drop length m: median {np.median(d):.1f}, mean {d.mean():.1f}, "
          f"p95 {np.percentile(d, 95):.1f}, max {d.max():.1f}")
    print(f"wrote {args.out_prefix}.graph: {len(new_nodes)} nodes, "
          f"{2 * len(out_edges)} directed arcs")
    print(f"wrote {args.out_prefix}.pois: {len(pois)} POI node ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
