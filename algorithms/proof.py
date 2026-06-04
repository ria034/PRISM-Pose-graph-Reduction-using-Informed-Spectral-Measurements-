import os
os.add_dll_directory(r'C:\Users\xingyue\.conda\envs\mac\Lib\site-packages') 
os.add_dll_directory(r'C:\Users\xingyue\.conda\envs\mac\Library\bin')
os.add_dll_directory(r'E:\Git\SE-Sync\C++\build\lib\Release')
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import sys
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from pose_graph_utils import (split_edges, read_g2o_file, rpm_to_mac)

from mac.solvers import MAC, NaiveGreedy
from mac.utils.graphs import weight_graph_lap_from_edges
from mac.utils.rounding import round_madow


def compute_effective_resistance(candidate_edges, num_nodes):
    edges_array = np.array([[e.i, e.j] for e in candidate_edges])
    weights = np.array([e.weight for e in candidate_edges])
    L = weight_graph_lap_from_edges(edges_array, weights, num_nodes).toarray()
    L_pinv = np.linalg.pinv(L)
    m = len(candidate_edges)
    B = np.zeros((m, num_nodes))
    for k, edge in enumerate(candidate_edges):
        B[k, edge.i] = 1
        B[k, edge.j] = -1
    return np.diag(B @ L_pinv @ B.T)


def build_MST(candidate_edges, weights, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for w, e in zip(weights, candidate_edges):
        G.add_edge(e.i, e.j, weight=w)
    mst = nx.maximum_spanning_tree(G)
    mst_edges = set(mst.edges())
    x = np.zeros(len(candidate_edges))
    for k, e in enumerate(candidate_edges):
        if (e.i, e.j) in mst_edges or (e.j, e.i) in mst_edges:
            x[k] = 1.0
    return x


def lambda2(edge_list, num_nodes):
    if len(edge_list) == 0:
        return 0.0
    arr = np.array([[e.i, e.j] for e in edge_list])
    wts = np.array([e.weight for e in edge_list])
    L = weight_graph_lap_from_edges(arr, wts, num_nodes).toarray()
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python proof.py <.g2o file>")
        sys.exit()

    dataset_name = sys.argv[1].split('\\')[-1].split('/')[-1].split('.')[0]
    print("Loading dataset: %s" % dataset_name)

    start = timer()
    measurements, num_poses = read_g2o_file(sys.argv[1])
    print("Loaded in %.3fs" % (timer() - start))

    odom_measurements, lc_measurements = split_edges(measurements)
    odom_edges = rpm_to_mac(odom_measurements)
    lc_edges   = rpm_to_mac(lc_measurements)
    all_edges  = rpm_to_mac(measurements)

    print("  %d odometry edges" % len(odom_measurements))
    print("  %d loop closure edges" % len(lc_measurements))
    print("  %d total edges" % len(measurements))
    print("  %d poses" % num_poses)

    # -- Build backbones --
    backbones = {}

    # 1) ODM = odometry as fixed (the chain)
    backbones["ODM"] = {
        "fixed": list(odom_edges),
        "candidate": list(lc_edges),
    }

    # 2) MST by effective resistance (from all edges)
    eff_res = compute_effective_resistance(all_edges, num_poses)
    x_mst = build_MST(all_edges, eff_res, num_poses)
    mst_idx = np.where(x_mst == 1.0)[0]
    non_idx = np.where(x_mst == 0.0)[0]
    backbones["MST"] = {
        "fixed":     [all_edges[i] for i in mst_idx],
        "candidate": [all_edges[i] for i in non_idx],
    }
    # 3) what u want
    """
    just add this after build the back bone
    backbones["MST"] = {
        "fixed":     [all_edges[i] for i in mst_idx],
        "candidate": [all_edges[i] for i in non_idx],
    }"""

    # -- Print base stats --
    print("")
    print("%-10s | %5s %5s %10s" % ("Backbone", "|A|", "|B|", "lam2(A)"))
    print("-" * 38)
    for name, bb in backbones.items():
        lam = lambda2(bb["fixed"], num_poses)
        print("%-10s | %5d %5d %10.6f" % (name, len(bb["fixed"]), len(bb["candidate"]), lam))
    lam_full = lambda2(all_edges, num_poses)
    print("%-10s | %5d %5s %10.6f" % ("Full", len(all_edges), "--", lam_full))

    # -- Build MAC solvers --
    mac_solvers = {}
    naive_solvers = {}
    for name, bb in backbones.items():
        if len(bb["candidate"]) > 0:
            mac_solvers[name] = MAC(
                bb["fixed"], bb["candidate"], num_poses,
                fiedler_method="tracemin_cholesky", fiedler_module="standard")
            naive_solvers[name] = NaiveGreedy(bb["candidate"])

    # -- Run MAC at each budget --
    budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []

    for pct in budgets:
        for name, bb in backbones.items():
            if name not in mac_solvers:
                continue

            target_total = len(odom_edges) + int(pct * len(lc_edges))
            num_cand = max(target_total - len(bb["fixed"]), 0)
            num_cand = min(num_cand, len(bb["candidate"]))
            if num_cand <= 0:
                continue

            w_init = naive_solvers[name].subset(num_cand)
            rounded, x_unrounded, dual_bound, _ = mac_solvers[name].solve(
                num_cand, w_init, max_iters=20,
                rounding="nearest", return_rounding_time=True, use_cache=True)

            # Relaxed objective directly from MAC
            lam_relaxed = mac_solvers[name].evaluate_objective(x_unrounded)

            # Madow rounded
            x_madow = round_madow(x_unrounded, num_cand,
                                   seed=np.random.RandomState(42))
            selected_madow = [bb["candidate"][i]
                              for i in np.where(x_madow == 1.0)[0]]
            lam_madow = lambda2(list(bb["fixed"]) + selected_madow, num_poses)

            # Nearest rounded
            selected_near = [bb["candidate"][i]
                             for i in np.where(rounded == 1.0)[0]]
            lam_nearest = lambda2(list(bb["fixed"]) + selected_near, num_poses)

            rows.append({
                "backbone": name,
                "pct": pct,
                "K": num_cand,
                "num_cand": len(bb["candidate"]),
                "lam_relaxed": lam_relaxed,
                "lam_madow": lam_madow,
                "lam_nearest": lam_nearest,
                "dual_bound": dual_bound,
            })

    # -- Print results --
    print("")
    header = "%-10s %4s %5s | %12s %12s %12s %12s | %10s" % (
        "Backbone", "%", "K", "lam2 RELAXED", "lam2 Madow", "lam2 Nearest", "Dual UB", "D(relax)")
    print(header)
    print("=" * len(header))

    for pct in budgets:
        pct_rows = [r for r in rows if r["pct"] == pct]
        if not pct_rows:
            continue

        odm_row = next((r for r in pct_rows if r["backbone"] == "ODM"), None)

        for r in pct_rows:
            if odm_row and r["backbone"] != "ODM":
                delta = odm_row["lam_relaxed"] - r["lam_relaxed"]
                delta_str = "%+10.6f" % delta
            else:
                delta_str = "%10s" % "ref"

            print("%-10s %3d%% %5d | %12.6f %12.6f %12.6f %12.6f | %s" % (
                r["backbone"], int(r["pct"]*100), r["K"],
                r["lam_relaxed"], r["lam_madow"],
                r["lam_nearest"], r["dual_bound"], delta_str))
        print("")

    # -- Summary --
    print("=" * 60)
    print("SUMMARY: ODM vs MST at the RELAXED level")
    print("=" * 60)
    odm_wins = 0
    mst_wins = 0
    ties = 0
    for pct in budgets:
        odm_r = next((r for r in rows if r["pct"] == pct and r["backbone"] == "ODM"), None)
        mst_r = next((r for r in rows if r["pct"] == pct and r["backbone"] == "MST"), None)
        if odm_r and mst_r:
            diff = odm_r["lam_relaxed"] - mst_r["lam_relaxed"]
            if abs(diff) < 1e-8:
                ties += 1
                verdict = "TIE"
            elif diff > 0:
                odm_wins += 1
                verdict = "ODM wins"
            else:
                mst_wins += 1
                verdict = "MST wins"
            print("  K=%3d%%: ODM=%.6f  MST=%.6f  D=%+.6f  -> %s" % (
                int(pct*100), odm_r["lam_relaxed"], mst_r["lam_relaxed"], diff, verdict))

    print("")
    print("ODM wins: %d   MST wins: %d   Ties: %d" % (odm_wins, mst_wins, ties))