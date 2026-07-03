#!/usr/bin/env python3
"""arcedge model compiler (R3-7a).

Compiles a declarative YAML network model (see docs/model-config.md) into the
flat artifacts the C++ solvers consume, plus a canonical model.json and a
manifest with provenance and suggested commands.

v1 scope:
  layers      street layers from .graph files; demand layers from CSV
              (lon,lat[,demand]) or GeoParquet points
  couplings   nearest_edge_split (perpendicular drop, edge broken at the
              foot point) from a demand layer onto a street layer
  capacities  edge rules by layer selector; hard, or soft with
              penalty_per_unit (compiled as parallel overflow arcs) and
              optional hard ceiling max
  facilities  one tier -> design mode (open_cost, hard serving cap)
  commodities assignment (-> design), pairs / sampled (-> MCF, with
              optional declarative time expansion)
  objective   unserved_penalty.per_unit -> per-commodity relief arcs
Determinism: all sampling uses Python's random.Random(seed), which is
platform-portable (unlike C++ std::uniform_* distributions).
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import yaml

LAT0 = 37.77  # keep consistent with connect_pois.py / design.cpp
KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110540.0


class ModelError(Exception):
    pass


def fail(msg):
    raise ModelError(msg)


def project(lon, lat):
    return lon * KX, lat * KY


def unproject(x, y):
    return x / KX, y / KY


# --------------------------------------------------------------- loading --

def read_graph(path):
    nodes, edges = [], []
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
                edges.append((int(p[1]), int(p[2]), float(p[3])))
    dedup = {}
    for u, v, w in edges:
        dedup.setdefault((min(u, v), max(u, v)), w)
    return nodes, sorted(dedup.items())  # nodes, [((u,v), len_m)]


def read_demand_csv(path):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append((float(row["lon"]), float(row["lat"]),
                        float(row.get("demand", 1) or 1)))
    return out


def read_demand_geoparquet(path, spec):
    import pyarrow.parquet as pq
    from shapely import from_wkb
    cols = ["geometry"]
    table = pq.read_table(path, columns=cols)
    out = []
    bbox = spec.get("bbox")
    for geom in from_wkb(table["geometry"].to_pylist()):
        if geom is None or geom.geom_type != "Point":
            continue
        if bbox and not (bbox[0] <= geom.x <= bbox[2] and bbox[1] <= geom.y <= bbox[3]):
            continue
        out.append((geom.x, geom.y, 1.0))
    limit = spec.get("max_points")
    if limit and len(out) > limit:
        step = len(out) / limit
        out = [out[int(i * step)] for i in range(limit)]
    return out


# ------------------------------------------------------------- coupling --

def nearest_edge_split(street_nodes, street_edges, demand_pts, snap_m=0.5,
                       max_length_m=None):
    """Ports connect_pois.py: each demand point drops perpendicularly to its
    nearest street edge, which is broken at the foot point. Returns
    (nodes, undirected_edges, poi_node_ids, drop_edge_set, n_split_nodes,
    skipped_long)."""
    from shapely import STRtree
    from shapely.geometry import LineString, Point

    xy = [project(lon, lat) for lon, lat in street_nodes]
    chords = [LineString([xy[u], xy[v]]) for u, v, _ in street_edges]
    tree = STRtree(chords)
    pts = [Point(project(lon, lat)) for lon, lat, _ in demand_pts]
    nearest = tree.query_nearest(pts, all_matches=False)
    edge_of = {int(i): int(e) for i, e in zip(nearest[0], nearest[1])}

    splits = {}
    skipped_long = 0
    for p_idx, pt in enumerate(pts):
        e_idx = edge_of[p_idx]
        chord = chords[e_idx]
        t = chord.project(pt) / max(chord.length, 1e-12)
        foot = chord.interpolate(chord.project(pt))
        drop = pt.distance(foot)
        if max_length_m is not None and drop > max_length_m:
            skipped_long += 1
            continue
        splits.setdefault(e_idx, []).append((t, p_idx, (foot.x, foot.y), drop))

    nodes = list(street_nodes)
    edges = []          # (u, v, len_m)
    drop_edges = set()  # indices into edges
    poi_node_of = {}
    n_split = 0

    def add_node(x, y):
        nodes.append(unproject(x, y))
        return len(nodes) - 1

    for e_idx, (u, v, length) in enumerate(street_edges):
        if e_idx not in splits:
            edges.append((u, v, length))
            continue
        chord_len = max(chords[e_idx].length, 1e-12)
        snap_t = snap_m / chord_len
        prev_node, prev_t = u, 0.0
        for t, p_idx, (fx, fy), drop in sorted(splits[e_idx]):
            if t <= snap_t:
                foot_node = u
            elif t >= 1.0 - snap_t:
                foot_node = v
            elif prev_node != u and (t - prev_t) * chord_len < snap_m:
                foot_node = prev_node
            else:
                foot_node = add_node(fx, fy)
                n_split += 1
                edges.append((prev_node, foot_node, max(t - prev_t, 0.0) * length))
                prev_node, prev_t = foot_node, t
            poi_node = add_node(*project(demand_pts[p_idx][0], demand_pts[p_idx][1]))
            poi_node_of[p_idx] = poi_node
            drop_edges.add(len(edges))
            edges.append((poi_node, foot_node, max(drop, 0.1)))
        if prev_node != v:
            edges.append((prev_node, v, max(1.0 - prev_t, 0.0) * length))
    return nodes, edges, poi_node_of, drop_edges, n_split, skipped_long


# ----------------------------------------------------------- capacities --

def resolve_edge_capacity(rules, layer_of_edge):
    """Returns per-edge dicts {kind, cap, penalty, max}; later rules win."""
    def match(rule_where, layer):
        sel = rule_where.get("edges") if isinstance(rule_where, dict) else None
        if sel is None:
            return False
        want = sel.get("layer", "*")
        return want == "*" or want == layer

    resolved = {}
    for i, layer in enumerate(layer_of_edge):
        spec = None
        for rule in rules:
            # `where` is canonical; `on` (which YAML 1.1 parses as boolean
            # True) is tolerated for hand-written configs.
            sel = rule.get("where", rule.get("on", rule.get(True, {})))
            if match(sel, layer):
                spec = rule["capacity"]
        if spec is None:
            resolved[i] = {"kind": "none"}
        elif "hard" in spec:
            resolved[i] = {"kind": "hard", "cap": float(spec["hard"])}
        elif "soft" in spec:
            resolved[i] = {"kind": "soft", "cap": float(spec["soft"]),
                           "penalty": float(spec["penalty_per_unit"]),
                           "max": float(spec["max"]) if spec.get("max") else None}
        else:
            fail(f"capacity rule needs 'hard' or 'soft': {spec}")
    return resolved


# ------------------------------------------------------------ emitters ---

def write_street_graph(path, nodes, edges):
    with open(path, "w") as f:
        f.write("c compiled by arcedge_modelc\n")
        f.write(f"g {len(nodes)} {2 * len(edges)}\n")
        for i, (lon, lat) in enumerate(nodes):
            f.write(f"v {i} {lon:.7f} {lat:.7f}\n")
        for u, v, w in edges:
            f.write(f"e {u} {v} {w:.2f}\ne {v} {u} {w:.2f}\n")


def sample_commodities(spec, adj, n_nodes, T, rng):
    """Ports the C++ sampler with a portable RNG: BFS hop reachability within
    T-1 steps, hub destinations shared by hub_frac of commodities."""
    def bfs(src):
        hops = [-1] * n_nodes
        hops[src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if hops[v] < 0:
                        hops[v] = hops[u] + 1
                        nxt.append(v)
            frontier = nxt
        return hops

    def sources_for(dst):
        hops = bfs(dst)
        return [v for v in range(n_nodes)
                if v != dst and 1 <= hops[v] <= T - 1]

    hubs = []
    min_sources = min(8, n_nodes // 2)
    guard = 0
    while len(hubs) < spec.get("hubs", 0) and guard < 1000:
        guard += 1
        h = rng.randrange(n_nodes)
        cands = sources_for(h)
        if len(cands) >= min_sources:
            hubs.append((h, cands))
    out = []
    for k in range(spec["count"]):
        if hubs and rng.random() < spec.get("hub_frac", 0.6):
            dst, cands = hubs[k % len(hubs)]
        else:
            g2 = 0
            while True:
                g2 += 1
                dst = rng.randrange(n_nodes)
                cands = sources_for(dst)
                if len(cands) >= 2 or g2 >= 1000:
                    break
            if len(cands) < 2:
                fail("cannot sample reachable commodity; increase time steps")
        src = cands[rng.randrange(len(cands))]
        out.append((src, dst, float(rng.randint(1, 3))))
    return out


def emit_mcf(out_dir, model, nodes, edges, edge_caps, report):
    te = model.get("time_expansion") or {}
    T = int(te.get("steps", 2))
    wait_cost = float(te.get("wait", {}).get("cost", 0.01))
    n_static = len(nodes)
    unserved = (model.get("objective", {}).get("unserved_penalty") or {}).get("per_unit")

    adj = [[] for _ in range(n_static)]
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)

    com_spec = model["commodities"][0]
    rng = random.Random(int(com_spec.get("seed", model.get("seed", 1))))
    if "pairs" in com_spec:
        commodities = [(int(p["src"]), int(p["dst"]), float(p.get("demand", 1)))
                       for p in com_spec["pairs"]]
    elif "sampled" in com_spec:
        commodities = sample_commodities(com_spec["sampled"], adj, n_static, T, rng)
    else:
        fail("MCF commodities need 'pairs' or 'sampled'")

    arcs = []  # (tail, head, cost, cap)
    overflow = 0
    for t in range(T - 1):
        base, nxt = t * n_static, (t + 1) * n_static
        for i, (u, v, w) in enumerate(edges):
            spec = edge_caps[i]
            for a, b in ((u, v), (v, u)):
                if spec["kind"] == "none":
                    arcs.append((base + a, nxt + b, w, -1.0))
                elif spec["kind"] == "hard":
                    arcs.append((base + a, nxt + b, w, spec["cap"]))
                else:  # soft: base arc at cap + overflow arc at cost+penalty
                    arcs.append((base + a, nxt + b, w, spec["cap"]))
                    over_cap = (spec["max"] - spec["cap"]) if spec["max"] else -1.0
                    arcs.append((base + a, nxt + b, w + spec["penalty"], over_cap))
                    overflow += 1
        for vtx in range(n_static):
            arcs.append((base + vtx, nxt + vtx, wait_cost, -1.0))

    coms = [(src, (T - 1) * n_static + dst, q) for src, dst, q in commodities]
    if unserved is not None:
        for src, dst_te, _ in coms:
            arcs.append((src, dst_te, float(unserved), -1.0))

    inst = out_dir / "instance.txt"
    with open(inst, "w") as f:
        f.write("c compiled by arcedge_modelc (time-expanded MCF)\n")
        f.write(f"p {n_static * T} {len(arcs)} {len(coms)}\n")
        for a in arcs:
            f.write(f"a {a[0]} {a[1]} {a[2]:.6g} {a[3]:.6g}\n")
        for src, dst, q in coms:
            f.write(f"k {src} {dst} {q:.6g}\n")

    report.update(mode="mcf", te_nodes=n_static * T, te_arcs=len(arcs),
                  commodities=len(coms), time_steps=T,
                  overflow_arcs=overflow,
                  total_demand=sum(q for _, _, q in coms))
    return {"instance": str(inst),
            "solve": f"./build/arcedge solve {inst} --tol 0.01"}


def resolve_design_edge_cap(edge_caps):
    """Uniform hard edge capacity from the rules; 0 = uncapacitated."""
    kinds = {edge_caps[i]["kind"] for i in edge_caps}
    if kinds == {"none"} or not kinds:
        return 0.0
    if kinds - {"hard", "none"}:
        fail("v1 design mode needs uniform HARD edge capacities "
             "(soft-capacity design lands with R3-7c)")
    caps = {edge_caps[i]["cap"] for i in edge_caps
            if edge_caps[i]["kind"] == "hard"}
    if len(caps) != 1 or "none" in kinds:
        fail(f"v1 design mode needs one uniform edge capacity, got {caps}")
    return caps.pop()


def emit_design(out_dir, model, nodes, edges, edge_caps, poi_nodes,
                poi_demands, report):
    """One or more facility tiers. Tier order = declaration order; each tier
    serves the previous tier's opened facilities (demand = units served),
    starting from the coupled demand points. Multi-tier solving is chained
    single-tier passes (spec R3-7d approximation), driven by --solve."""
    tiers = model.get("facilities", {})
    if not tiers:
        fail("design mode needs at least one facility tier")
    default_edge_cap = resolve_design_edge_cap(edge_caps)
    cable_cost = float((model.get("cable") or {}).get("fixed_cost_per_m", 10.0))

    graph = out_dir / "access.graph"
    pois = out_dir / "tier0.pois"
    write_street_graph(graph, nodes, edges)
    with open(pois, "w") as f:
        for n, q in zip(poi_nodes, poi_demands):
            f.write(f"{n} {q:g}\n")

    passes = []
    for i, (tier_name, tier) in enumerate(tiers.items()):
        cap_spec = tier.get("capacity", {})
        edge_cap = tier.get("edge_capacity", {}).get(
            "hard", default_edge_cap if i == 0 else 0.0)
        passes.append({
            "tier": tier_name,
            "hub_cost": float(tier["open_cost"]),
            "hub_cap": float(cap_spec.get("hard", 0)),
            "edge_cap": float(edge_cap),
            "cable_cost": float(tier.get("cable_cost_per_m", cable_cost)),
            "demand_transit": i > 0,  # upper tiers sit on street cabinets
        })

    report.update(mode="design", nodes=len(nodes), undirected_edges=len(edges),
                  pois=len(poi_nodes), tiers=[p["tier"] for p in passes])
    t0 = passes[0]
    cmd = (f"./build/arcedge design --graph {graph} --pois {pois} "
           f"--cap {t0['edge_cap']:g} --hub-cap {t0['hub_cap']:g} "
           f"--hub-cost {t0['hub_cost']:g} --cable-cost {t0['cable_cost']:g}")
    if len(passes) > 1:
        cmd += "   # tier 1 of {}; use --solve to run the whole chain".format(
            len(passes))
    return {"graph": str(graph), "pois": str(pois), "passes": passes,
            "design": cmd}


# ----------------------------------------------------------- tier chain ---

def solve_design_chain(out_dir, artifacts, arcedge_bin):
    """Runs the facility tiers bottom-up: each pass's opened hubs become the
    next pass's demand points (demand = units served)."""
    import subprocess
    graph = artifacts["graph"]
    pois_path = artifacts["pois"]
    passes = artifacts["passes"]
    summary = {"tiers": [], "total_cost": 0.0}
    for i, ps in enumerate(passes):
        result = out_dir / f"tier{i + 1}.result"
        solution = out_dir / f"tier{i + 1}.solution"
        cmd = [arcedge_bin, "design", "--graph", str(graph),
               "--pois", str(pois_path),
               "--cap", f"{ps['edge_cap']:g}", "--hub-cap", f"{ps['hub_cap']:g}",
               "--hub-cost", f"{ps['hub_cost']:g}",
               "--cable-cost", f"{ps['cable_cost']:g}",
               "--quiet", "--result", str(result), "--solution", str(solution)]
        if ps["demand_transit"]:
            cmd.append("--demand-transit")
        print(f"[tier {i + 1}/{len(passes)}: {ps['tier']}] "
              f"hub_cap {ps['hub_cap']:g}, open {ps['hub_cost']:g}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout + proc.stderr, file=sys.stderr)
            fail(f"tier {ps['tier']} design failed")
        res = {k: float(v) for k, v in
               (l.split() for l in open(result) if not l.startswith("hub_nodes"))}
        # Demands of this pass's POIs, to weight the next tier's demand.
        dem_of = {}
        for line in open(pois_path):
            parts = line.split()
            if parts:
                dem_of[int(parts[0])] = float(parts[1]) if len(parts) > 1 else 1.0
        served = {}
        for line in open(solution):
            parts = line.split()
            if parts and parts[0] == "p":
                served[int(parts[2])] = served.get(int(parts[2]), 0.0) + \
                    dem_of.get(int(parts[1]), 1.0)
        tier_sum = dict(tier=ps["tier"], hubs=int(res["hubs"]),
                        cable_m=res["cable_m"], cost=res["total_cost"],
                        result=str(result), solution=str(solution))
        summary["tiers"].append(tier_sum)
        summary["total_cost"] += res["total_cost"]
        print(f"  -> {tier_sum['hubs']} {ps['tier']}(s), "
              f"{tier_sum['cable_m']:.0f} m cable, cost {tier_sum['cost']:.0f}",
              flush=True)
        if i + 1 < len(passes):
            pois_path = out_dir / f"tier{i + 1}.pois"
            with open(pois_path, "w") as f:
                for hub, q in sorted(served.items()):
                    f.write(f"{hub} {q:g}\n")
    json.dump(summary, open(out_dir / "tiers_summary.json", "w"), indent=2)
    breakdown = " + ".join(
        "{:.0f} {}".format(t["cost"], t["tier"]) for t in summary["tiers"])
    print("\nCHAIN TOTAL COST {:.0f}  ({})".format(summary["total_cost"], breakdown))
    return summary


# ---------------------------------------------------------------- main ----

def compile_model(cfg_path, out_dir):
    model = yaml.safe_load(open(cfg_path))
    if model.get("model_version") != 1:
        fail("model_version must be 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"name": model.get("name", cfg_path.stem)}

    layers = model.get("layers") or fail("layers section required")
    street_layers = {n: s for n, s in layers.items()
                     if s.get("role") != "terminals"}
    demand_layers = {n: s for n, s in layers.items()
                     if s.get("role") == "terminals"}
    if len(street_layers) != 1:
        fail("v1 supports exactly one street layer")
    (street_name, street_spec), = street_layers.items()
    src = street_spec.get("source", {})
    if "file" not in src:
        fail("v1 street layer source must be a .graph file")
    nodes, ue = read_graph(src["file"])
    edges = [(u, v, w) for (u, v), w in ue]
    layer_of_edge = [street_name] * len(edges)
    report["street_layer"] = {"name": street_name, "nodes": len(nodes),
                              "undirected_edges": len(edges)}

    poi_nodes = []
    poi_demands = []
    for coup in model.get("couplings", []):
        if coup.get("method") != "nearest_edge_split":
            fail(f"v1 coupling method must be nearest_edge_split: {coup}")
        dl = coup["from"]
        if dl not in demand_layers:
            fail(f"coupling 'from' must be a terminals layer: {dl}")
        dsrc = demand_layers[dl]["source"]
        if "csv" in dsrc:
            pts = read_demand_csv(dsrc["csv"])
        elif "geoparquet" in dsrc:
            pts = read_demand_geoparquet(dsrc["geoparquet"], dsrc)
        else:
            fail("demand layer source must be csv or geoparquet")
        nodes, edges, poi_of, drop_edges, n_split, skipped = nearest_edge_split(
            nodes, edges, pts, snap_m=float(coup.get("snap_m", 0.5)),
            max_length_m=coup.get("max_length_m"))
        # Street edges (including split children) keep the street layer tag;
        # drops belong to the demand layer.
        layer_of_edge = [dl if i in drop_edges else street_name
                         for i in range(len(edges))]
        poi_nodes = [poi_of[i] for i in sorted(poi_of)]
        poi_demands = [pts[i][2] for i in sorted(poi_of)]
        report["coupling"] = {"from": dl, "points": len(pts),
                              "connected": len(poi_nodes),
                              "edge_splits": n_split,
                              "skipped_too_long": skipped}

    edge_caps = resolve_edge_capacity(model.get("capacities", []), layer_of_edge)

    commodities = model.get("commodities") or fail("commodities section required")
    if len(commodities) != 1:
        fail("v1 supports exactly one commodity block")
    com = commodities[0]

    if "assignment" in com:
        if not poi_nodes:
            fail("assignment commodities need a coupled terminals layer")
        artifacts = emit_design(out_dir, model, nodes, edges, edge_caps,
                                poi_nodes, poi_demands, report)
    else:
        artifacts = emit_mcf(out_dir, model, nodes, edges, edge_caps, report)

    json.dump(model, open(out_dir / "model.json", "w"), indent=2)
    manifest = {"model": report["name"], "report": report,
                "artifacts": artifacts,
                "node_ranges": {"street": [0, report["street_layer"]["nodes"]],
                                "compiled_total": len(nodes)}}
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\ncompiled to {out_dir}/ -- next:")
    for k, v in artifacts.items():
        if k in ("solve", "design"):
            print(f"  {v}")
    return artifacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("-o", "--out", required=True, help="output directory")
    ap.add_argument("--solve", action="store_true",
                    help="after compiling, run the solver (multi-tier designs "
                         "run the whole facility chain)")
    ap.add_argument("--arcedge", default="./build/arcedge",
                    help="path to the arcedge binary (with --solve)")
    args = ap.parse_args()
    try:
        artifacts = compile_model(Path(args.config), Path(args.out))
        if args.solve:
            if "passes" in artifacts:
                solve_design_chain(Path(args.out), artifacts, args.arcedge)
            else:
                import subprocess
                cmd = artifacts["solve"].split()
                cmd[0] = args.arcedge
                print(flush=True)
                subprocess.run(cmd, check=True)
        return 0
    except ModelError as e:
        print(f"model error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
