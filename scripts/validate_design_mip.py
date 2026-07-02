#!/usr/bin/env python3
"""Validates the arcedge design heuristic against an exact MIP (HiGHS).

Builds the aggregated single-commodity fixed-charge model: every POI injects
one unit at its drop's foot node (drop edges are leaves that must be paid for
regardless, so they enter the objective as a constant), flow may only use a
street edge if its binary y_e is open (paying cable cost * length once), total
two-way flow on an edge is capped, and any node with an open hub binary z_v
absorbs flow (paying hub cost). HiGHS solves the MIP to a gap or time limit;
the heuristic's total must be >= the MIP lower bound, and close to the MIP
incumbent.
"""
import argparse
import sys

import numpy as np
import highspy


def read_graph(path):
    nodes = 0
    edges = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line[0] == "c":
                continue
            p = line.split()
            if p[0] == "g":
                nodes = int(p[1])
            elif p[0] == "e":
                u, v, w = int(p[1]), int(p[2]), float(p[3])
                edges.setdefault((min(u, v), max(u, v)), w)
    return nodes, sorted(edges.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("pois")
    ap.add_argument("result")
    ap.add_argument("--cap", type=float, default=500.0)
    ap.add_argument("--hub-cost", type=float, default=20000.0)
    ap.add_argument("--cable-cost", type=float, default=10.0)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--mip-gap", type=float, default=0.005)
    args = ap.parse_args()

    num_nodes, edge_items = read_graph(args.graph)
    poi_nodes = set(int(l) for l in open(args.pois))
    heur = {}
    for line in open(args.result):
        parts = line.split()
        if parts[0] != "hub_nodes":
            heur[parts[0]] = float(parts[1])

    # Split edges into drops (incident to a POI leaf) and street edges;
    # aggregate POI injections at foot nodes.
    inject = np.zeros(num_nodes)
    drop_const = 0.0
    street = []
    for (u, v), w in edge_items:
        if u in poi_nodes or v in poi_nodes:
            drop_const += args.cable_cost * w
            foot = v if u in poi_nodes else u
            inject[foot] += 1.0
        else:
            street.append((u, v, w))
    n_flow = float(len(poi_nodes))
    m = len(street)
    real_nodes = [v for v in range(num_nodes) if v not in poi_nodes]
    node_row = {v: i for i, v in enumerate(real_nodes)}
    n = len(real_nodes)

    # Columns: x_uv, x_vu per street edge, s_v (absorption), y_e, z_v.
    # Rows: balance per node (== inject), capacity per edge (x_uv + x_vu -
    # cap*y <= 0), absorption activation (s_v - N*z_v <= 0).
    ncol = 2 * m + n + m + n
    x_uv, x_vu = 0, m
    s0, y0, z0 = 2 * m, 2 * m + n, 2 * m + n + m
    nrow = n + m + n
    bal0, cap0, act0 = 0, n, n + m

    cols = [[] for _ in range(ncol)]  # (row, coef)
    for e, (u, v, w) in enumerate(street):
        ru, rv = node_row[u], node_row[v]
        cols[x_uv + e] += [(bal0 + ru, 1.0), (bal0 + rv, -1.0), (cap0 + e, 1.0)]
        cols[x_vu + e] += [(bal0 + rv, 1.0), (bal0 + ru, -1.0), (cap0 + e, 1.0)]
        cols[y0 + e].append((cap0 + e, -min(args.cap, n_flow)))
    for i in range(n):
        cols[s0 + i] += [(bal0 + i, 1.0), (act0 + i, 1.0)]
        cols[z0 + i].append((act0 + i, -n_flow))

    inf = highspy.kHighsInf
    row_lb = np.zeros(nrow)
    row_ub = np.zeros(nrow)
    for v in real_nodes:
        row_lb[bal0 + node_row[v]] = row_ub[bal0 + node_row[v]] = inject[v]
    row_lb[cap0:cap0 + m] = -inf
    row_lb[act0:act0 + n] = -inf

    cost = np.zeros(ncol)
    lb = np.zeros(ncol)
    ub = np.full(ncol, inf)
    for e, (u, v, w) in enumerate(street):
        cost[y0 + e] = args.cable_cost * w
    cost[z0:z0 + n] = args.hub_cost
    ub[y0:y0 + m] = 1.0
    ub[z0:z0 + n] = 1.0

    h = highspy.Highs()
    h.silent()
    h.addRows(nrow, row_lb, row_ub, 0,
              np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0))
    starts = np.zeros(ncol, dtype=np.int64)
    idx, val = [], []
    for c in range(ncol):
        starts[c] = len(idx)
        for r, coef in cols[c]:
            idx.append(r)
            val.append(coef)
    h.addCols(ncol, cost, lb, ub, len(idx), starts,
              np.array(idx, dtype=np.int32), np.array(val))
    integrality = np.zeros(ncol, dtype=np.int32)
    integrality[y0:y0 + m] = 1
    integrality[z0:z0 + n] = 1
    h.changeColsIntegrality(ncol, np.arange(ncol, dtype=np.int32), integrality)
    h.setOptionValue("time_limit", args.time_limit)
    h.setOptionValue("mip_rel_gap", args.mip_gap)
    h.run()

    info = h.getInfo()
    mip_obj = info.objective_function_value + drop_const
    mip_bound = info.mip_dual_bound + drop_const
    heur_total = heur["total_cost"]
    print(f"MIP incumbent : {mip_obj:.0f}   (dual bound {mip_bound:.0f}, "
          f"status {h.getModelStatus()})")
    print(f"heuristic     : {heur_total:.0f}  ({heur['hubs']:.0f} hubs, "
          f"{heur['cable_m']:.0f} m cable)")
    lb_ok = heur_total >= mip_bound - 1e-4 * abs(mip_bound)
    gap = (heur_total - mip_bound) / heur_total
    print(f"heuristic vs MIP bound gap: {100 * gap:.2f}%  "
          f"({'OK' if lb_ok else 'VIOLATION: heuristic beats a valid lower bound'})")
    if lb_ok:
        print("VALIDATION PASSED")
        return 0
    print("VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
