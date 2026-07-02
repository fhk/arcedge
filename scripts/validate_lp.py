#!/usr/bin/env python3
"""Validates arcedge solver bounds against the exact LP optimum from HiGHS.

Builds the full capacitated multicommodity min-cost flow LP for an arcedge
instance file (one flow variable per commodity-arc pair, per-commodity flow
conservation, joint capacity rows), solves it with HiGHS, and checks the
solver's Lagrangian lower bound and heuristic upper bound sandwich it:

    lb <= LP* <= ub   (within tolerance)

Exit code 0 iff both inequalities hold.
"""
import argparse
import sys

import numpy as np
import highspy


def read_instance(path):
    num_nodes = 0
    arcs = []  # (tail, head, cost, cap)
    commodities = []  # (src, dst, demand)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "c":
                continue
            parts = line.split()
            if parts[0] == "p":
                num_nodes = int(parts[1])
            elif parts[0] == "a":
                arcs.append((int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])))
            elif parts[0] == "k":
                commodities.append((int(parts[1]), int(parts[2]), float(parts[3])))
    return num_nodes, arcs, commodities


def read_result(path):
    vals = {}
    with open(path) as f:
        for line in f:
            key, val = line.split()
            vals[key] = float(val)
    return vals


def solve_lp(num_nodes, arcs, commodities):
    n, m, K = num_nodes, len(arcs), len(commodities)
    cap_arcs = [a for a in range(m) if arcs[a][3] >= 0]
    cap_row = {a: K * n + i for i, a in enumerate(cap_arcs)}
    num_rows = K * n + len(cap_arcs)
    num_cols = K * m

    # Row bounds: conservation equalities (out - in = b), then capacities.
    row_lb = np.zeros(num_rows)
    row_ub = np.zeros(num_rows)
    for k, (src, dst, demand) in enumerate(commodities):
        row_lb[k * n + src] = row_ub[k * n + src] = demand
        row_lb[k * n + dst] = row_ub[k * n + dst] = -demand
    for i, a in enumerate(cap_arcs):
        row_lb[K * n + i] = -highspy.kHighsInf
        row_ub[K * n + i] = arcs[a][3]

    h = highspy.Highs()
    h.silent()
    h.addRows(num_rows, row_lb, row_ub, 0,
              np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0))

    # Columns: x[k, a] >= 0 with coefficients +1 at tail row, -1 at head row,
    # +1 in the arc's capacity row.
    costs = np.empty(num_cols)
    starts = np.empty(num_cols, dtype=np.int64)
    indices = []
    values = []
    nnz = 0
    for k in range(K):
        for a, (tail, head, cost, cap) in enumerate(arcs):
            col = k * m + a
            costs[col] = cost
            starts[col] = nnz
            indices.extend((k * n + tail, k * n + head))
            values.extend((1.0, -1.0))
            nnz += 2
            if cap >= 0:
                indices.append(cap_row[a])
                values.append(1.0)
                nnz += 1
    h.addCols(num_cols, costs, np.zeros(num_cols),
              np.full(num_cols, highspy.kHighsInf), nnz,
              starts, np.array(indices, dtype=np.int32), np.array(values))
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS did not reach optimality: {status}")
    return h.getInfo().objective_function_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("result")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="relative tolerance on the bound checks")
    args = ap.parse_args()

    num_nodes, arcs, commodities = read_instance(args.instance)
    res = read_result(args.result)
    print(f"instance: {num_nodes} nodes, {len(arcs)} arcs, {len(commodities)} commodities"
          f" -> LP with {len(commodities) * len(arcs)} variables")
    lp_opt = solve_lp(num_nodes, arcs, commodities)

    lb, ub = res["lb"], res["ub"]
    scale = max(abs(lp_opt), 1.0)
    lb_ok = lb <= lp_opt + args.tol * scale
    ub_ok = ub >= lp_opt - args.tol * scale
    ub_gap = (ub - lp_opt) / scale
    print(f"HiGHS LP optimum : {lp_opt:.6f}")
    print(f"arcedge lower bnd: {lb:.6f}  ({'OK' if lb_ok else 'VIOLATION'}: lb <= LP*)")
    print(f"arcedge upper bnd: {ub:.6f}  ({'OK' if ub_ok else 'VIOLATION'}: ub >= LP*)")
    print(f"ub vs LP* gap    : {100.0 * ub_gap:.4f}%")
    print(f"lb vs LP* gap    : {100.0 * (lp_opt - lb) / scale:.4f}%")
    if lb_ok and ub_ok:
        print("VALIDATION PASSED: lb <= LP* <= ub")
        return 0
    print("VALIDATION FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
