import os
import sys
import warnings
warnings.filterwarnings("ignore")

# HACK: Necessary in Windows
DLL_PATHS = [
    r"C:\Users\lenovo\Downloads\SE-Sync\C++\build\lib",
    r"C:\Users\lenovo\anaconda3\envs\repo_env\Library\bin",
    r"C:/Users/neilk/.vscode/SE-Sync/C++/build/lib/RelWithDebInfo",
]

for path in DLL_PATHS:
    if os.path.isdir(path):
        os.add_dll_directory(path)
        if "SE-Sync" in path:
            sys.path.insert(0, path)

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import sys
import random
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from pose_graph_utils import split_edges, read_g2o_file, plot_poses, rpm_to_mac, RelativePoseMeasurement, poses_ate_tran, poses_rpe_rot

from mac.solvers import MAC, NaiveGreedy
from mac.utils.graphs import Edge
from mac.utils.graphs import weight_graph_lap_from_edges
from mac.utils.rounding import round_madow
from mac.utils.rounding import round_nearest

import matplotlib.pyplot as plt

import PySESync
PERCENTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def is_connected(w, all_edges, num_poses):
    G_check = nx.Graph()
    G_check.add_nodes_from(range(num_poses))
    for idx in np.where(w == 1.0)[0]:
        G_check.add_edge(all_edges[idx].i, all_edges[idx].j)
    is_conn = nx.is_connected(G_check)
    return is_conn


def compute_effective_resistance(candidate_edges, num_nodes):
    # Build Laplacian from candidate edges
    edges_array = np.array([[edge.i, edge.j] for edge in candidate_edges])
    weights = np.array([edge.weight for edge in candidate_edges])
    L = weight_graph_lap_from_edges(edges_array, weights, num_nodes)
   
    L_dense = L.toarray()
    # Pseudoinverse of Laplacian
    L_pinv = np.linalg.pinv(L_dense)
    #build the basis 
    m = len(candidate_edges)
    B = np.zeros((m, num_nodes))
    for k, edge in enumerate(candidate_edges):
        B[k, edge.i] = 1
        B[k, edge.j] = -1
    
    # Effective resistance for each edge = diag(B @ L_pinv @ B.T)
    x_eff_resistance = np.diag(B @ L_pinv @ B.T)
    return x_eff_resistance


def build_MST(candidate_edges, weights, num_nodes):
    """
    Builds an Maximum Spanning Tree (MST) from the candidate edges using NetworkX. 
    The weights of the edges are determined by the input `weights` array.

    Parameters:
    - candidate_edges: List of Edge objects representing the candidate edges.
    - weights: Array of weights corresponding to each candidate edge. Higher weights indicate stronger preference for selection.
    - num_nodes: Total number of nodes in the graph.

    Returns:
    - w_mst: Binary array indicating which candidate edges are included in the MST (1 for included, 0 for not included).
    """

    # Build a NetworkX graph with combined weights
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    for weight, edge in zip(weights, candidate_edges):
        G.add_edge(edge.i, edge.j, weight=weight)

    # Build MST (Kruskal's by default)
    # Note: nx.minimum_spanning_tree minimizes weight, so higher combined
    # weight edges are preferred if you use maximum_spanning_tree
    mst = nx.maximum_spanning_tree(G)

    # Extract selected edges as a binary vector
    mst_edges = set(mst.edges())
    x_mst = np.zeros(len(candidate_edges))
    for k, edge in enumerate(candidate_edges):
        if (edge.i, edge.j) in mst_edges or (edge.j, edge.i) in mst_edges:
            x_mst[k] = 1.0

    return x_mst

# Algorithm 3: Iterative MAC with fixed edges (IterativeMAC)
def iterative_mac_with_fixed(fixed_edges, candidate_edges, num_poses, K_total, fix_per_iter_frac=0.2, max_mac_iters=20):

    m = len(candidate_edges)
    K_total = min(K_total, m)

    committed = set() # the set that get slowly get built as mac gets run iteratively
    all_idx = set(range(m))
    iter_log = []

    it = 0
    while len(committed) < K_total:
        it += 1

        remaining = K_total - len(committed)
        fix_now = max(1, int(remaining * fix_per_iter_frac))
        fix_now = min(fix_now, remaining)

        # Split: already committed + original fixed = new fixed set
        # Remaining candidates = everything not yet committed or in fixed
        cand_idx = sorted(all_idx - committed)
        cand_edges_it = [candidate_edges[i] for i in cand_idx]
        committed_edges = [candidate_edges[i] for i in sorted(committed)]
        fixed_edges_it = fixed_edges + committed_edges

        # Solve MAC
        mac_it = MAC(fixed_edges_it, cand_edges_it, num_poses,
                     fiedler_method="tracemin_cholesky")

        naive_it = NaiveGreedy(cand_edges_it)
        w_init = naive_it.subset(fix_now)

        start_it = timer()
        _, unrounded, upper, _ = mac_it.solve(
            fix_now, w_init, max_iters=max_mac_iters,
            rounding="nearest", return_rounding_time=True,
            use_cache=True
        )
        time_it = timer() - start_it

        # Commit top edges by unrounded weight
        top_local = np.argsort(unrounded)[-fix_now:]
        for li in top_local:
            committed.add(cand_idx[li])

        iter_log.append({
            "iter": it,
            "committed": len(committed),
            "remaining_cand": len(cand_idx) - fix_now,
            "fixed_this_round": fix_now,
            "upper_bound": upper,
            "time": time_it,
        })

    # Assemble binary vector over candidate_edges
    w_final = np.zeros(m)
    for idx in committed:
        w_final[idx] = 1.0

    info = {
        "iter_log": iter_log,
        "num_committed": len(committed),
    }

    return w_final, info 

# Algorithm 2: Iterative MAC with MST-based fixing (MadowEffr)
def iterative_mac_solve(all_edges, num_poses, K_total, fix_per_iter_frac=0.2, max_mac_iters=20):
    """
    Iterative MAC: run MAC -> fix top edges -> shrink candidates -> repeat -> MST from remainder.
    Returns binary selection vector and diagnostics dict.
    """
    m = len(all_edges)
    mst_slots = num_poses - 1
    num_fix_target = K_total - mst_slots # edges available to pick, leaving room for the MST step

    if num_fix_target <= 0:
        # print(f"  algo2: budget K={K_total} <= MST size {mst_slots}, just building MST")
        w_mst = build_MST(all_edges, np.ones(m), num_poses)
        return w_mst, {"iter_log": [], "num_fixed": 0, "num_mst": int(np.sum(w_mst)),
                       "num_total": int(np.sum(w_mst)), "fixed_indices": set()}

    fixed_set = set()
    all_idx = set(range(m))
    iter_log = []
    unrounded_final = {}

    it = 0
    while len(fixed_set) < num_fix_target: # while we still have room to fix more edges before the MST step
        it += 1
        
        # edges to be picked by this iteration of MAC (fraction of remaining budget, but at least 1)
        remaining_to_fix = num_fix_target - len(fixed_set) # how many more edges we can fix before hitting the target for the MST step
        fix_now = max(1, int(remaining_to_fix * fix_per_iter_frac))
        fix_now = min(fix_now, remaining_to_fix)

        # candidates are all edges that aren't in the fixed set yet
        cand_idx = sorted(all_idx - fixed_set)
        cand_edges = [all_edges[i] for i in cand_idx]
        fixed_edges = [all_edges[i] for i in sorted(fixed_set)]

        # mac_budget = K_total - len(fixed_set)
        # mac_budget = min(mac_budget, len(cand_edges))

        # create MAC instance
        mac_it = MAC(fixed_edges, cand_edges, num_poses, fiedler_method="tracemin_cholesky", fiedler_module="enhanced")

        # initial iterate for MAC
        naive_it = NaiveGreedy(cand_edges)
        w_init_it = naive_it.subset(fix_now)
        
        # Run MAC
        start_it = timer()
        _, unrounded_it, upper_it, _ = mac_it.solve(
            fix_now, w_init_it, max_iters=max_mac_iters,
            rounding="nearest", return_rounding_time=True, use_cache=True
        )
        time_it = timer() - start_it

        unrounded_final = {}
        for k, gi in enumerate(cand_idx):
            unrounded_final[gi] = unrounded_it[k]

        # Pick fix_now edges (highest weights)
        top_local_idx = np.argsort(unrounded_it)[-fix_now:]
        for li in top_local_idx:
            gi = cand_idx[li]
            fixed_set.add(gi)

        log_entry = {
            "iter": it,
            "fixed_count": len(fixed_set),
            "cand_count": len(cand_idx) - fix_now,
            "fixed_this_round": fix_now,
            "mac_budget": fix_now,
            "upper_bound": upper_it,
            "time": time_it,
            "top_weight_fixed": max(unrounded_it[li] for li in top_local_idx),
            "min_weight_fixed": min(unrounded_it[li] for li in top_local_idx),
        }
        iter_log.append(log_entry)

    # After the iterative rounds, we have created a fixed set
    # Now, the final step, to ensure connectivity, we build a MST using the edges that are not in the fixed set
    final_cand_idx = sorted(all_idx - fixed_set)
    final_cand_edges = [all_edges[i] for i in final_cand_idx]

    # Since we are running several iterations of MAC, there is no one iteration we could simply pick
    # Get weights for the final candidate edges
    mst_weights = np.zeros(len(final_cand_edges))
    for k, gi in enumerate(final_cand_idx):
        mst_weights[k] = unrounded_final.get(gi, 0.0)

    # Add the fixed edges (with inf weight to ensure they are always picked)
    G = nx.Graph()
    G.add_nodes_from(range(num_poses))
    for idx in fixed_set:
        G.add_edge(all_edges[idx].i, all_edges[idx].j, weight=float('inf'))

    # Now add the candidate edges with their weights
    for k, edge in enumerate(final_cand_edges):
        if not G.has_edge(edge.i, edge.j):
            G.add_edge(edge.i, edge.j, weight=mst_weights[k], local_idx=k)
        else:
            existing_w = G[edge.i][edge.j].get('weight', 0)
            if existing_w != float('inf') and mst_weights[k] > existing_w:
                G[edge.i][edge.j]['weight'] = mst_weights[k]
                G[edge.i][edge.j]['local_idx'] = k

    mst_tree = nx.maximum_spanning_tree(G)

    w_mst_local = np.zeros(len(final_cand_edges))
    for u, v, data in mst_tree.edges(data=True):
        if data.get('weight') == float('inf'):
            continue
        li = data.get('local_idx', None)
        if li is not None:
            w_mst_local[li] = 1.0

    # edit: if MST used fewer than mst_slots candidate edges, fill with best remaining
    num_mst_cand_used = int(np.sum(w_mst_local))
    slots_left = mst_slots - num_mst_cand_used
    if slots_left > 0:
        unused = [(k, mst_weights[k]) for k in range(len(final_cand_edges)) if w_mst_local[k] == 0.0]
        unused.sort(key=lambda x: x[1], reverse=True)
        for k, _ in unused[:slots_left]:
            w_mst_local[k] = 1.0

    # assemble final binary vector
    w_final = np.zeros(m)
    for idx in fixed_set:
        w_final[idx] = 1.0
    for k, idx in enumerate(final_cand_idx):
        if w_mst_local[k] == 1.0:
            w_final[idx] = 1.0

    num_selected = int(np.sum(w_final))
    num_from_mst = int(np.sum(w_mst_local))

    info = {
        "iter_log": iter_log,
        "num_fixed": len(fixed_set),
        "num_mst": num_from_mst,
        "num_total": num_selected,
        "fixed_indices": fixed_set,
    }

    return w_final, info


def orbit_distance_dS(X, Y, compute_G_S=False):
    d = X.shape[0]
    n = int(X.shape[1] / d)

    XYt = X @ Y.T

    u, s, vh = np.linalg.svd(XYt)

    uvh = u @ vh
    Sigma = np.diag(s)

    Xi_diag = np.diag(np.ones(d))
    Xi_diag[d-1, d-1] = np.copysign(1.0, np.linalg.det(uvh))

    dS = np.sqrt(np.abs(
        2 * d * n - 2 * np.trace(Xi_diag @ Sigma)
        ))

    if compute_G_S:
        G_S = u @ Xi_diag @ vh
        return dS, G_S
    return dS

def construct_LGrho(measurements):
    d = len(measurements[0].t) if len(measurements) > 0 else 0

    num_poses = 0
    for measurement in measurements:
        max_pair = max(measurement.i, measurement.j)
        if max_pair > num_poses:
            num_poses = max_pair
    num_poses = num_poses + 1

    LGrho = np.zeros([d * num_poses, d * num_poses])

    for measurement in measurements:
        i = measurement.i
        j = measurement.j

        for k in range(0, d):
            LGrho[d * i + k, d * i + k] += measurement.kappa

        for k in range(0, d):
            LGrho[d * j + k, d * j + k] += measurement.kappa

        for r in range(0, d):
            for c in range(0, d):
                LGrho[i * d + r, j * d + c] += -measurement.kappa * measurement.R[r, c]

        for r in range(0, d):
            for c in range(0, d):
                LGrho[j * d + r, i * d + c] += -measurement.kappa * measurement.R[c, r]

    return LGrho

def evaluate_sesync_rotation_objective(LGrho, R):
    return np.trace(R @ LGrho @ R.T)

def construct_sesync_quadratic_form_matrix(measurements):
    d = len(measurements[0].t) if len(measurements) > 0 else 0

    num_poses = 0
    for measurement in measurements:
        max_pair = max(measurement.i, measurement.j)
        if max_pair > num_poses:
            num_poses = max_pair
    num_poses = num_poses + 1

    M = np.zeros([(d + 1) * num_poses, (d + 1) * num_poses])

    for measurement in measurements:
        i = measurement.i
        j = measurement.j

        M[i, i] += measurement.tau
        M[j, j] += measurement.tau
        M[i, j] += -measurement.tau
        M[j, i] += -measurement.tau

        for k in range(0, d):
            M[i, num_poses + i * d + k] += measurement.tau * measurement.t[k]
        for k in range(0, d):
            M[j, num_poses + i * d + k] += -measurement.tau * measurement.t[k]

        for k in range(0, d):
            M[num_poses + i * d + k, i] += measurement.tau * measurement.t[k]
        for k in range(0, d):
            M[num_poses + i * d + k, j] += -measurement.tau * measurement.t[k]

        for k in range(0, d):
            M[num_poses + d * i + k,
              num_poses + d * i + k] += measurement.kappa

        for k in range(0, d):
            M[num_poses + d * j + k,
              num_poses + d * j + k] += measurement.kappa

        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + i * d + r, num_poses + j * d +
                  c] += -measurement.kappa * measurement.R[r, c]

        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + j * d + r, num_poses + i * d +
                  c] += -measurement.kappa * measurement.R[c, r]

        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + i * d + r, num_poses + i * d +
                  c] += measurement.tau * measurement.t[r] * measurement.t[c]

    return M

def evaluate_sesync_objective(M, Xhat):
    return np.trace(Xhat @ M @ Xhat.T)


def nx_rot_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        i = measurement.i
        j = measurement.j
        w = measurement.kappa
        G.add_edge(i, j, weight=w)
    return G

def nx_tran_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        i = measurement.i
        j = measurement.j
        w = measurement.tau
        G.add_edge(i, j, weight=w)
    return G

def select_measurements(measurements, w):
    assert(len(measurements) == len(w))
    meas_out = []
    for i, meas in enumerate(measurements):
        if w[i] == 1.0:
            meas_out.append(meas)
    return meas_out

def to_sesync_format(measurements):
    sesync_measurements = []
    for meas in measurements:
        sesync_meas = PySESync.RelativePoseMeasurement()
        sesync_meas.i = meas.i
        sesync_meas.j = meas.j
        sesync_meas.kappa = meas.kappa
        sesync_meas.tau = meas.tau
        sesync_meas.R = meas.R
        sesync_meas.t = meas.t
        sesync_measurements.append(sesync_meas)
    return sesync_measurements

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [.g2o file] [optional: --run-greedy]")
        sys.exit()
        pass

    dataset_name = sys.argv[1].split('/')[-1].split('.')[0]
    print(f"Loading dataset: {dataset_name}")

    print("Reading g2o file")
    start = timer()
    measurements, num_poses = read_g2o_file(sys.argv[1])
    end = timer()
    print("Success! elapsed time: ", (end - start))

    odom_measurements, lc_measurements = split_edges(measurements)

    odom_edges = rpm_to_mac(odom_measurements)
    lc_edges = rpm_to_mac(lc_measurements)
    all_edges = rpm_to_mac(measurements)

    print(f"Loaded {len(measurements)} total measurements with: ")
    print(f"\t {len(odom_measurements)} base (odometry) measurements and")
    print(f"\t {len(lc_measurements)} candidate (loop closure) measurements")

    #############################
    # Test 1: Standard MAC at varying budgets (MAC Paper)
    # Required output: AC(MAC + Nearest) at different budgets, viz., the list mac_standard['mac_nearest'] 
    # Experiment Setup: 
    #   - Fix Edge Set: odometry edges (base graph)
    #   - Candidate Edge Set: loop closure edges
    #   - Budgets: Varying percentage of loop closures to accept (10%, 20%, ..., 100%)
    #   - Rounding: Nearest neighbor and Madow
    #############################

    print("="*25 + "\n  TEST-1: Standard MAC with Fixed Odometric Edges \n" + "="*25)

    mac = MAC(odom_edges, lc_edges, num_poses, fiedler_method="tracemin_cholesky", fiedler_module="standard")
    naive = NaiveGreedy(lc_edges)

    mac_standard = {
        "percent_lc": PERCENTS,
        "naive": [],
        "mac_nearest": [],
        "mac_madow": [],
        "unrounded": [],
        "upper_bounds": [],
        "times": [],
        "madow_times": [],
        "greedy_esp": [],
        "greedy_esp_times": [],
    }

    for pct_lc in PERCENTS:
        num_lc = int(pct_lc * len(lc_measurements))
        print("Num LC to accept: ", num_lc)

        # Naive solution
        naive_result = naive.subset(num_lc)
        mac_standard["naive"].append(naive_result)

        # MAC solution with nearest rounding
        start = timer()
        w_init = naive_result
        nearest_result, unrounded, upper, rtime = mac.solve(num_lc,
                                                            w_init,
                                                            max_iters=20,
                                                            rounding="nearest",
                                                            return_rounding_time=True,
                                                            use_cache=True)
        end = timer()
        solve_time = end - start

        mac_standard["times"].append(solve_time)
        mac_standard["mac_nearest"].append(nearest_result)
        mac_standard["upper_bounds"].append(upper)
        mac_standard["unrounded"].append(unrounded)

        # MAC solution with Madow Rounding
        start = timer()
        madow_rounded = round_madow(unrounded, num_lc, seed=np.random.RandomState(42))
        end = timer()

        mac_standard["mac_madow"].append(madow_rounded)
        mac_standard["madow_times"].append(solve_time + (end - start) - rtime)

    #############################
    # Test 2: Enhanced MAC - Best Version (MadowEffr)
    # Required output: mac_enhanced['mac_madow']
    # Experiment Setup: 
    #   - Fix Edge Set: build MST using edge effective resistance weights
    #   - Candidate Edge Set: All the remaining edges (basically the edges not in the MST)
    #   - Budgets: Varying percentage of loop closures to accept (10%, 20%, ..., 100%) - but now this is the percentage of the remaining edges after fixing
    #   - Rounding: Madow
    # #############################

    print("="*25 + "\n  TEST-2: Enhanced MAC with MST-based Fixing (MadowEffr) \n" + "="*25)

    # step 1: compute effective resistance
    if os.path.exists(f"{dataset_name}_eff_res.npy"):
        eff_resistance = np.load(f"{dataset_name}_eff_res.npy")
    else:
        eff_resistance = compute_effective_resistance(all_edges, num_poses)
        np.save(f"{dataset_name}_eff_res.npy", eff_resistance)

    # step 2: build MST using effective resistance as weights
    weights_mod = [e.weight * eff_res for e, eff_res in zip(all_edges, eff_resistance)]
    x_mst = build_MST(all_edges, weights_mod, num_poses)
    fixed_edges = [all_edges[i] for i in range(len(all_edges)) if x_mst[i] == 1.0]

    # step 3: remaining candidates are the edges not in the MST
    remaining_idx = np.where(x_mst == 0.0)[0]
    remaining_edges = [all_edges[i] for i in remaining_idx]

    # step 4: run MAC with the MST edges fixed, and the remaining edges as candidates
    mac = MAC(fixed_edges, remaining_edges, num_poses, fiedler_method="tracemin_cholesky", fiedler_module="enhanced")
    naive = NaiveGreedy(lc_edges)

    # step 5: run MAC at varying budgets (but now this is the percentage of the remaining edges after fixing)
    mac_enhanced = {
        "percent_lc": PERCENTS,
        "naive": [],
        "mac_madow": [],
        "unrounded": [],
        "upper_bounds": [],
        "times": [],
        "madow_times": [],
    }

    for pct in PERCENTS:
        pick_now = int(pct * len(remaining_edges))
        print("Num edges to accept: ", pick_now)

        # Naive solution
        naive_result = naive.subset(pick_now)
        mac_enhanced["naive"].append(naive_result)

        # Enhanced MAC solution with Madow Rounding
        start = timer()
        w_init = naive_result
        madow_result, unrounded, upper, rtime = mac.solve(pick_now,
                                                          w_init,
                                                          max_iters=20,
                                                          rounding="madow",
                                                          return_rounding_time=True,
                                                          use_cache=True)
        end = timer()
        solve_time = end - start

        mac_enhanced["times"].append(solve_time)
        mac_enhanced["upper_bounds"].append(upper)
        mac_enhanced["unrounded"].append(unrounded)

        mac_enhanced["mac_madow"].append(madow_result)
        mac_enhanced["madow_times"].append(solve_time + (end - start) - rtime)

    #############################
    # Test 3: Iterative MAC
    # Required output: mac_iterative['mac_highest_wt']
    # Experiment Setup: 
    #   - All edges are candidates
    #   - Edge Budget = MST slots + (Edges remaining) * pct, where pct is the varying percentage of edges to accept (10%, 20%, ..., 100%)
    # #############################

    print("="*25 + "\n  TEST-3: Iterative MAC (No fixed) \n" + "="*25)

    mac_iterative = {
        "percent_lc": PERCENTS,
        "mac_highest_wt": [],
        "times": [],
    }
    mst_slots_needed = num_poses - 1

    mac = MAC([], all_edges, num_poses, fiedler_method="tracemin_cholesky", fiedler_module="enhanced")
    naive = NaiveGreedy(lc_edges)

    for pct in PERCENTS:
        
        # Edge Budget = MST slots + (Edges remaining) * pct
        K_algo2 = mst_slots_needed + int(pct * (len(all_edges) - mst_slots_needed))
        print(f"Budget breakdown: MST slots = {mst_slots_needed}, Additional Edges to accept = {K_algo2 - mst_slots_needed}")

        start_algo2 = timer()
        w_algo2, info_algo2 = iterative_mac_solve(
            all_edges,
            num_poses,
            K_total=K_algo2,
            fix_per_iter_frac=0.3,
            max_mac_iters=20,
        )
        time_algo2 = timer() - start_algo2

        mac_iterative['mac_highest_wt'].append(w_algo2)
        mac_iterative['times'].append(time_algo2)
        
        if not is_connected(w_algo2, all_edges, num_poses):
            print("  WARNING: algo2 solution is not connected!")

    #############################
    # Test 4: Iterative MAC with odometric edges as the fixed edges
    # Required output: iterative_fixed_results
    # Experiment Setup: 
    #   - Fixed Edge Set: odometry edges (base graph)
    #   - Candidate Edge Set: loop closure edges
    # #############################
    print("="*25 + "\n  TEST-4: Iterative MAC with odometric edges \n" + "="*25)

    # Run iterative MAC with odom fixed
    iterative_fixed_results = []
    iterative_fixed_times = []

    for pct in PERCENTS:
        num_lc = int(pct * len(lc_edges))
        print("Num LC to accept: ", num_lc)

        start_t4 = timer()
        w_result, info = iterative_mac_with_fixed(
            fixed_edges=odom_edges,
            candidate_edges=lc_edges,
            num_poses=num_poses,
            K_total=num_lc,
            fix_per_iter_frac=0.3,
            max_mac_iters=20
        )
        time_t4 = timer() - start_t4

        iterative_fixed_results.append(w_result)
        iterative_fixed_times.append(time_t4)


    #############################
    # Summary Plots
    #############################
    # Evaluate AC for each method at each budget
    def eval_ac(mac_obj, results_list):
        return [mac_obj.evaluate_objective(r) for r in results_list]

    # Standard MAC (uses odom fixed edges)
    mac_std = MAC(odom_edges, lc_edges, num_poses, fiedler_method="tracemin_cholesky", fiedler_module="standard")

    # ---- Plot 1: AC vs. budget ----
    fig_ac, ax = plt.subplots(figsize=(8, 5))

    ax.plot(PERCENTS, eval_ac(mac_std, mac_standard["mac_nearest"]),  '-o', color='#534AB7', label='Standard MAC + Nearest', markersize=4)
    ax.plot(PERCENTS, eval_ac(mac_std, mac_standard["mac_madow"]),    '-s', color='#7F77DD', label='Standard MAC + Madow',   markersize=4)

    # Enhanced MAC (uses MST fixed edges)
    mac_enh = MAC(fixed_edges, remaining_edges, num_poses, fiedler_method="tracemin_cholesky", fiedler_module="enhanced")
    ax.plot(PERCENTS, eval_ac(mac_enh, mac_enhanced["mac_madow"]),    '-^', color='#1D9E75', label='Enhanced MAC + Madow',   markersize=4)

    # Iterative MAC (no fixed edges, all candidates)
    mac_iter = MAC([], all_edges, num_poses, fiedler_method="tracemin_cholesky", fiedler_module="enhanced")
    ax.plot(PERCENTS, eval_ac(mac_iter, mac_iterative["mac_highest_wt"]), '-d', color='#D85A30', label='Algorithm-2', markersize=4)
    
    # Odom Fixed Iterative MAC
    ax.plot(PERCENTS, eval_ac(mac_std, iterative_fixed_results),'-x', color='#993556', label='Algorithm-3', markersize=5)

    ax.set_xlabel('Budget (fraction of candidates)')
    ax.set_ylabel('Algebraic connectivity')
    ax.set_title(f'{dataset_name} — AC vs. budget')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_ac.tight_layout()
    plt.savefig(f'./results/performance/{dataset_name}.png', dpi=150)
    # plt.show()

    # ---- Plot 2: Time vs. budget ----
    fig_time, ax_time = plt.subplots(figsize=(8, 5))

    ax_time.plot(PERCENTS, mac_standard["times"],       '-o', color='#534AB7', label='Standard MAC + Nearest', markersize=4)
    ax_time.plot(PERCENTS, mac_standard["madow_times"], '-s', color='#7F77DD', label='Standard MAC + Madow',   markersize=4)
    ax_time.plot(PERCENTS, mac_enhanced["times"],       '-^', color='#1D9E75', label='Enhanced MAC + Madow',   markersize=4)
    ax_time.plot(PERCENTS, mac_iterative["times"],      '-d', color='#D85A30', label='Algorithm-2',            markersize=4)
    ax_time.plot(PERCENTS, iterative_fixed_times,       '-x', color='#993556', label='Algorithm-3',            markersize=5)

    ax_time.set_xlabel('Budget (fraction of candidates)')
    ax_time.set_ylabel('Time (seconds)')
    ax_time.set_title(f'{dataset_name} — Solve time vs. budget')
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3)
    fig_time.tight_layout()
    plt.savefig(f'./results/time/{dataset_name}.png', dpi=150)
    # plt.show()