import os
import sys
import warnings
warnings.filterwarnings("ignore")

import os
os.add_dll_directory(r"C:\Users\lenovo\Downloads\SE-Sync\C++\build\lib")
os.add_dll_directory(r"C:\Users\lenovo\anaconda3\envs\repo_env\Library\bin")

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import sys
import numpy as np
import networkx as nx
from pose_graph_utils import split_edges, read_g2o_file, rpm_to_mac, RelativePoseMeasurement, poses_ate_tran, poses_rpe_rot

# MAC requirements
from mac.solvers import MAC, NaiveGreedy
from mac.utils.graphs import Edge
from mac.utils.graphs import weight_graph_lap_from_edges
from mac.utils.rounding import round_madow
from mac.utils.rounding import round_nearest

import matplotlib.pyplot as plt

import PySESync


def compute_effective_resistance(all_edges, x_weights, num_nodes):
    edges_array = np.array([[edge.i, edge.j] for edge in all_edges])
    weights = x_weights  # continuous weights from MAC solve
    
    L = weight_graph_lap_from_edges(edges_array, weights, num_nodes)
    L_dense = L.toarray()
    L_pinv = np.linalg.pinv(L_dense)
    
    m = len(all_edges)
    B = np.zeros((m, num_nodes))
    for k, edge in enumerate(all_edges):
        B[k, edge.i] = 1
        B[k, edge.j] = -1
    
    x_eff_resistance = np.diag(B @ L_pinv @ B.T)
    return x_eff_resistance


def build_MST(candidate_edges, weights, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for weight, edge in zip(weights, candidate_edges):
        G.add_edge(edge.i, edge.j, weight=weight)
    mst = nx.maximum_spanning_tree(G)
    mst_edges = set(mst.edges())
    x_mst = np.zeros(len(candidate_edges))
    for k, edge in enumerate(candidate_edges):
        if (edge.i, edge.j) in mst_edges or (edge.j, edge.i) in mst_edges:
            x_mst[k] = 1.0
    return x_mst


def orbit_distance_dS(X, Y, compute_G_S=False):
    d = X.shape[0]
    n = int(X.shape[1] / d)
    XYt = X @ Y.T
    u, s, vh = np.linalg.svd(XYt)
    uvh = u @ vh
    Sigma = np.diag(s)
    Xi_diag = np.diag(np.ones(d))
    Xi_diag[d-1, d-1] = np.copysign(1.0, np.linalg.det(uvh))
    dS = np.sqrt(np.abs(2 * d * n - 2 * np.trace(Xi_diag @ Sigma)))
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
            M[num_poses + d * i + k, num_poses + d * i + k] += measurement.kappa
        for k in range(0, d):
            M[num_poses + d * j + k, num_poses + d * j + k] += measurement.kappa
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + i * d + r, num_poses + j * d + c] += -measurement.kappa * measurement.R[r, c]
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + j * d + r, num_poses + i * d + c] += -measurement.kappa * measurement.R[c, r]
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + i * d + r, num_poses + i * d + c] += measurement.tau * measurement.t[r] * measurement.t[c]
    return M


def evaluate_sesync_objective(M, Xhat):
    return np.trace(Xhat @ M @ Xhat.T)


def nx_rot_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        G.add_edge(measurement.i, measurement.j, weight=measurement.kappa)
    return G


def nx_tran_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        G.add_edge(measurement.i, measurement.j, weight=measurement.tau)
    return G


def select_measurements(measurements, w):
    assert(len(measurements) == len(w))
    return [meas for i, meas in enumerate(measurements) if w[i] == 1.0]


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

    run_greedy = False
    if len(sys.argv) > 2:
        if sys.argv[2] == "--run-greedy":
            run_greedy = True
        else:
            print(f"Unknown argument: {sys.argv[2]}")
            print(f"Usage: {sys.argv[0]} [.g2o file] [optional: --run-greedy]")
            sys.exit()

    dataset_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    print(f"Loading dataset: {dataset_name}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    print("Reading g2o file")
    measurements, num_poses = read_g2o_file(sys.argv[1])
    print(f"Dataset has {num_poses} poses and {len(measurements)} measurements")

    odom_measurements, lc_measurements = split_edges(measurements)
    odom_edges = rpm_to_mac(odom_measurements)
    lc_edges   = rpm_to_mac(lc_measurements)
    all_edges  = rpm_to_mac(measurements)

    mac = MAC(fixed_edges=[],
              candidate_edges=all_edges,
              num_nodes=num_poses,
              fiedler_method="tracemin_cholesky", 
              fiedler_module="enhanced")
    
    # Make a Naive Solver object (picks top k edges)
    

    naive = NaiveGreedy(all_edges)

    percents  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    naive_results           = []
    mac_mst_nearest_results = []
    mac_mst_madow_results   = []
    mac_mst_results         = []
    budgets                 = []

    mac_mst_time         = []
    mac_mst_nearest_time = []
    mac_mst_madow_time   = []
    mac_nearest_time     = []
    mac_madow_time       = []
    effrMadow_round_time = []


    allowed_budget = len(all_edges) - (num_poses - 1)

    for pct in percents:
        num_mst_edges = num_poses - 1
        budget = int(pct * allowed_budget)
        budgets.append(budget)
        print(f"\nBudget: {budget} edges")

        naive_result = naive.subset(budget + num_mst_edges)
        naive_results.append(naive_result)

        w_init = naive_result
       
        mac_nearest_result, x_unrounded, upper, rtime = mac.solve(
            k=budget + num_mst_edges,
            x_init=w_init,
            max_iters=20,
            rounding="nearest",
            return_rounding_time=True,
            use_cache=True)
       
        # mac_solve = end - start
        print(x_unrounded)
        x_eff_resistance = compute_effective_resistance(all_edges, x_unrounded, num_poses)



        
        x_mst = build_MST(all_edges,x_eff_resistance *x_unrounded, num_poses)
      
        x_non_mst_idx = np.where(x_mst == 0)[0]
        x_mst_idx     = np.where(x_mst == 1)[0]

        non_mst_weights = x_unrounded[x_non_mst_idx]
        non_mst_combined_weights_norm = non_mst_weights / np.sum(non_mst_weights) * budget
        non_mst_combined_weights_norm[-1] += budget - np.sum(non_mst_combined_weights_norm)

        
        nearest_remaining = round_nearest(non_mst_combined_weights_norm, budget)

        madow_remaining = round_madow(non_mst_combined_weights_norm, budget, seed=np.random.RandomState(42))


        x_final_nearest_combined = x_mst.copy()
        x_final_nearest_combined[x_non_mst_idx] = nearest_remaining

        x_final_madow_combined = x_mst.copy()
        x_final_madow_combined[x_non_mst_idx] = madow_remaining

        #Implementing MAC-rerun 

        # start = timer()
        # x_mst = build_MST(all_edges, x_eff_resistance, num_poses)
        # end = timer()
        # mst_time = end - start

        # x_non_mst_idx = np.where(x_mst == 0)[0]
        # x_mst_idx     = np.where(x_mst == 1)[0]

        # # Get the actual edge objects
        # mst_edge_list     = [all_edges[i] for i in x_mst_idx]
        # non_mst_edge_list = [all_edges[i] for i in x_non_mst_idx]

        # # Run MAC again on non-MST edges, with MST edges fixed
        # mac_rerun = MAC(fixed_edges=mst_edge_list,
        #                 candidate_edges=non_mst_edge_list,
        #                 num_nodes=num_poses,
        #                 fiedler_method="tracemin_lu")

        # naive_rerun = NaiveGreedy(non_mst_edge_list)
        # w_init_rerun = naive_rerun.subset(budget)

        # _, x_unrounded_rerun, _, _ = mac_rerun.solve(
        #     k=budget,
        #     x_init=w_init_rerun,
        #     max_iters=20,
        #     rounding="nearest",
        #     return_rounding_time=True,
        #     use_cache=True)

        # start = timer()
        # nearest_remaining = round_nearest(x_unrounded_rerun, budget)
        # end = timer()
        # n_time = end - start

        # start = timer()
        # madow_remaining = round_madow(x_unrounded_rerun, budget, seed=np.random.RandomState(42))
        # end = timer()
        # m_time = end - start

        # x_final_nearest_combined = x_mst.copy()
        # x_final_nearest_combined[x_non_mst_idx] = nearest_remaining
        
        # x_final_madow_combined = x_mst.copy()
        # x_final_madow_combined[x_non_mst_idx] = madow_remaining

        mac_mst_nearest_results.append(x_final_nearest_combined)
        mac_mst_madow_results.append(x_final_madow_combined)
        mac_mst_results.append(x_mst)

    

    # ---- Baselines ----
    MAC_original_nearest_results = []
    MAC_original_madow_results= []
    odom, candidates = split_edges(measurements)
    mac_original = MAC(fixed_edges=odom, candidate_edges=candidates, num_nodes=num_poses, fiedler_method="tracemin_lu", fiedler_module="standard")

    naive_original = NaiveGreedy(candidates)
    percent_lc     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for pct_lc in percent_lc:
        num_lc = int(pct_lc * len(candidates))
        w_init = naive_original.subset(num_lc)
        result, unrounded, upper, rtime = mac_original.solve(
            num_lc, w_init, max_iters=20, rounding="nearest",
            return_rounding_time=True, use_cache=True)

        
        MAC_original_nearest_rounding = round_nearest(unrounded, num_lc)

        MAC_original_madow_rounding = round_madow(unrounded, num_lc, seed=np.random.RandomState(42))


        MAC_original_nearest_results.append(MAC_original_nearest_rounding)
        MAC_original_madow_results.append(MAC_original_madow_rounding)

    # ---- EffrMadow baseline ----
    effr_Madow_rounding_results = []

    edge_weights         = np.array([edge.weight for edge in all_edges])
    weights_enhanced_MAC = x_eff_resistance * edge_weights

  
    x_MST_Madow_eff = build_MST(measurements,weights_enhanced_MAC, num_poses) # build MST based on effective resistance
    
    #compute the Non-MST candidate edges for the MAC solver 
    mst_edges = [e for i, e in enumerate(measurements) if x_MST_Madow_eff[i] == 1]
    non_mst_edges = [e for i, e in enumerate(measurements) if x_MST_Madow_eff[i] == 0]
    mac_effrMadow = MAC(fixed_edges=mst_edges, candidate_edges=non_mst_edges, num_nodes=num_poses, fiedler_method="tracemin_lu", fiedler_module="standard")

    naive_effrMadow = NaiveGreedy(non_mst_edges)

    for pct_effrMadow in percent_lc:
        num_effrMadow    = int(pct_effrMadow * len(non_mst_edges))
        w_init_effrMadow = naive_effrMadow.subset(num_effrMadow)
        result_effrMadow, unrounded_effrMadow, upper_effrMadow, rtime_effrMadow = mac_effrMadow.solve(
            num_effrMadow, w_init_effrMadow, max_iters=20, rounding="nearest",
            return_rounding_time=True, use_cache=True)

      
        effrMadow_madow_rounding = round_madow(unrounded_effrMadow, num_effrMadow, seed=np.random.RandomState(42))
   
        effr_Madow_rounding_results.append(effrMadow_madow_rounding)
 

    num_mst_edges_fixed = num_poses - 1
    num_odom            = len(odom)
    num_mst_effrMadow   = len(mst_edges)

    print(f"num_poses - 1: {num_poses - 1}")
    print(f"len(odom): {num_odom}")
    print(f"len(candidates): {len(candidates)}")
    print(f"len(all_edges): {len(all_edges)}")

    col = 18
    print("\n" + "=" * (14 + col * 5))
    print(f"{'%':<6} {'Budget':<8} {'Alg1+Nearest':<{col}} {'Alg1+Madow':<{col}} {'StdMAC+Nearest':<{col}} {'StdMAC+Madow':<{col}} {'EnhMAC+Madow':<{col}}")
    print(f"{'':6} {'':8} {'AC    Edges':<{col}} {'AC    Edges':<{col}} {'AC    Edges':<{col}} {'AC    Edges':<{col}} {'AC    Edges':<{col}}")
    print("=" * (14 + col * 5))

    fiedler_mst_vals         = []
    fiedler_mst_nearest_vals = []
    fiedler_mst_madow_vals   = []
    fiedler_mac_nearest_vals = []
    fiedler_mac_madow_vals   = []
    fiedler_effrMadow_vals   = []

    for i, percent in enumerate(percents):
        budget        = budgets[i]
        num_lc        = int(percent * len(candidates))
        num_effrMadow = int(percent * len(non_mst_edges))

        fiedler_mst         = mac.evaluate_objective(mac_mst_results[i])
        fiedler_mst_nearest = mac.evaluate_objective(mac_mst_nearest_results[i])
        fiedler_mst_madow   = mac.evaluate_objective(mac_mst_madow_results[i])
        fiedler_mac_nearest = mac_original.evaluate_objective(MAC_original_nearest_results[i])
        fiedler_mac_madow   = mac_original.evaluate_objective(MAC_original_madow_results[i])
        fiedler_effrMadow   = mac_effrMadow.evaluate_objective(effr_Madow_rounding_results[i])

        fiedler_mst_vals.append(fiedler_mst)
        fiedler_mst_nearest_vals.append(fiedler_mst_nearest)
        fiedler_mst_madow_vals.append(fiedler_mst_madow)
        fiedler_mac_nearest_vals.append(fiedler_mac_nearest)
        fiedler_mac_madow_vals.append(fiedler_mac_madow)
        fiedler_effrMadow_vals.append(fiedler_effrMadow)

        edges_mst_nearest = num_mst_edges_fixed + budget
        edges_mst_madow   = num_mst_edges_fixed + budget
        edges_mac_nearest = num_odom + num_lc
        edges_mac_madow   = num_odom + num_lc
        edges_effrMadow   = num_mst_effrMadow + num_effrMadow

        def cell(ac, edges): return f"{ac:.3f} {edges}"
        print(f"{int(percent*100):<6} {budget:<8} "
              f"{cell(fiedler_mst_nearest, edges_mst_nearest):<{col}} "
              f"{cell(fiedler_mst_madow, edges_mst_madow):<{col}} "
              f"{cell(fiedler_mac_nearest, edges_mac_nearest):<{col}} "
              f"{cell(fiedler_mac_madow, edges_mac_madow):<{col}} "
              f"{cell(fiedler_effrMadow, edges_effrMadow):<{col}}")
    print("=" * (14 + col * 5))

    #############################
    # Style constants
    #############################
    LBL_NEAREST     = 'Algorithm 1 + Nearest'
    LBL_MADOW       = 'Algorithm 1 + Madow'
    LBL_STD_NEAREST = 'Standard MAC + Nearest'
    LBL_STD_MADOW   = 'Standard MAC + Madow'
    LBL_ENH_MADOW   = 'Enhanced MAC + Madow'

    CLR_NEAREST     = '#D85A30'
    CLR_MADOW       = '#F0997B'
    CLR_STD_NEAREST = '#534AB7'
    CLR_STD_MADOW   = '#7F77DD'
    CLR_ENH_MADOW   = '#1D9E75'

    MRK_NEAREST     = '-o'
    MRK_MADOW       = '-v'
    MRK_STD_NEAREST = '-o'
    MRK_STD_MADOW   = '-s'
    MRK_ENH_MADOW   = '-^'

    MS = 4

    # Title for plots — underscores replaced with spaces
    title_name = dataset_name.replace('_', ' ')

    #############################
    # Plot 1: Algebraic Connectivity
    #############################
    plt.figure()
    plt.plot(100.0 * np.array(percents), fiedler_mst_nearest_vals, MRK_NEAREST,     color=CLR_NEAREST,     label=LBL_NEAREST,     markersize=MS)
    plt.plot(100.0 * np.array(percents), fiedler_mst_madow_vals,   MRK_MADOW,       color=CLR_MADOW,       label=LBL_MADOW,       markersize=MS)
    plt.plot(100.0 * np.array(percents), fiedler_mac_nearest_vals, MRK_STD_NEAREST, color=CLR_STD_NEAREST, label=LBL_STD_NEAREST, markersize=MS)
    plt.plot(100.0 * np.array(percents), fiedler_mac_madow_vals,   MRK_STD_MADOW,   color=CLR_STD_MADOW,   label=LBL_STD_MADOW,   markersize=MS)
    plt.plot(100.0 * np.array(percents), fiedler_effrMadow_vals,   MRK_ENH_MADOW,   color=CLR_ENH_MADOW,   label=LBL_ENH_MADOW,   markersize=MS)
    plt.title(title_name)
    plt.xlabel(r'% Edges Added')
    plt.ylabel(r'Algebraic Connectivity $\lambda_2$')
    plt.xlim([0.0, 100.0])
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"alg_conn_{dataset_name}.png"), dpi=600, bbox_inches='tight')
    plt.close()

    #############################
    # Plot 2: Computation Time
    #############################
    plt.figure()
    plt.semilogy(100.0 * np.array(percents[:-1]), mac_mst_nearest_time[:-1], MRK_NEAREST,     color=CLR_NEAREST,     label=LBL_NEAREST,     markersize=MS)
    plt.semilogy(100.0 * np.array(percents[:-1]), mac_mst_madow_time[:-1],   MRK_MADOW,       color=CLR_MADOW,       label=LBL_MADOW,       markersize=MS)
    plt.semilogy(100.0 * np.array(percents[:-1]), mac_nearest_time[:-1],     MRK_STD_NEAREST, color=CLR_STD_NEAREST, label=LBL_STD_NEAREST, markersize=MS)
    plt.semilogy(100.0 * np.array(percents[:-1]), mac_madow_time[:-1],       MRK_STD_MADOW,   color=CLR_STD_MADOW,   label=LBL_STD_MADOW,   markersize=MS)
    plt.semilogy(100.0 * np.array(percents[:-1]), effrMadow_round_time[:-1], MRK_ENH_MADOW,   color=CLR_ENH_MADOW,   label=LBL_ENH_MADOW,   markersize=MS)
    plt.title(title_name)
    plt.xlim([0.0, 100.0])
    plt.xlabel(r'% Edges Added')
    plt.ylabel(r'Time (s)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"comp_time_{dataset_name}.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # #############################
    # # Run SE-Sync
    # #############################
    # non_mst_measurements     = [measurements[i] for i in x_non_mst_idx]
    # mst_measurements         = [measurements[i] for i in x_mst_idx]
    # non_mst_measurements_eff = [measurements[i] for i in x_non_mst_idx2]
    # mst_measurements_eff     = [measurements[i] for i in x_mst_idx2]

    # print(f"non_mst_measurements_eff length: {len(non_mst_measurements_eff)}")
    # print(f"mst_measurements_eff length: {len(mst_measurements_eff)}")

    # mst_edges      = [e for i, e in enumerate(all_edges) if x_MST_Madow_eff[i] == 1]
    # non_mst_edges  = [e for i, e in enumerate(all_edges) if x_MST_Madow_eff[i] == 0]
    # x_non_mst_idx2 = np.where(x_MST_Madow_eff == 0)[0]
    # x_mst_idx2     = np.where(x_MST_Madow_eff == 1)[0]

    # d = mst_measurements[0].R.shape[0]
    # opts = PySESync.SESyncOpts()
    # opts.num_threads = 4
    # opts.verbose = True
    # opts.r0 = d + 1

    # sesync_nearest         = []
    # sesync_madow           = []
    # sesync_mst             = []
    # sesync_effrMadow       = []
    # sesync_mac_org_nearest = []
    # sesync_mac_org_madow   = []

    # for i in range(len(naive_results)):
    #     madow_selected_lc = select_measurements(measurements, mac_mst_madow_results[i])
    #     sesync_madow.append(PySESync.SESync(to_sesync_format(mst_measurements + madow_selected_lc), opts))

    #     nearest_selected_lc = select_measurements(measurements, mac_mst_nearest_results[i])
    #     sesync_nearest.append(PySESync.SESync(to_sesync_format(mst_measurements + nearest_selected_lc), opts))

    #     sesync_mst.append(PySESync.SESync(to_sesync_format(measurements), opts))

    # for i in range(len(effr_Madow_rounding_results)):
    #     effr_madow_selected_lc = select_measurements(non_mst_measurements_eff, effr_Madow_rounding_results[i])
    #     sesync_effrMadow.append(PySESync.SESync(to_sesync_format(mst_measurements_eff + effr_madow_selected_lc), opts))

    # for i in range(len(MAC_original_nearest_results)):
    #     mac_org_nearest_selected_lc = select_measurements(lc_measurements, MAC_original_nearest_results[i])
    #     sesync_mac_org_nearest.append(PySESync.SESync(to_sesync_format(odom_measurements + mac_org_nearest_selected_lc), opts))

    #     mac_org_madow_selected_lc = select_measurements(lc_measurements, MAC_original_madow_results[i])
    #     sesync_mac_org_madow.append(PySESync.SESync(to_sesync_format(odom_measurements + mac_org_madow_selected_lc), opts))

    # #############################
    # # Plot 3: SE-Sync Computation Time
    # #############################
    # plt.figure()
    # plt.plot(100.0 * np.array(percents), [r.total_computation_time for r in sesync_nearest],         MRK_NEAREST,     color=CLR_NEAREST,     label=LBL_NEAREST,     markersize=MS)
    # plt.plot(100.0 * np.array(percents), [r.total_computation_time for r in sesync_madow],           MRK_MADOW,       color=CLR_MADOW,       label=LBL_MADOW,       markersize=MS)
    # plt.plot(100.0 * np.array(percents), [r.total_computation_time for r in sesync_mac_org_nearest], MRK_STD_NEAREST, color=CLR_STD_NEAREST, label=LBL_STD_NEAREST, markersize=MS)
    # plt.plot(100.0 * np.array(percents), [r.total_computation_time for r in sesync_mac_org_madow],   MRK_STD_MADOW,   color=CLR_STD_MADOW,   label=LBL_STD_MADOW,   markersize=MS)
    # plt.plot(100.0 * np.array(percents), [r.total_computation_time for r in sesync_effrMadow],       MRK_ENH_MADOW,   color=CLR_ENH_MADOW,   label=LBL_ENH_MADOW,   markersize=MS)
    # plt.title(title_name)
    # plt.xlabel(r'% Edges Added')
    # plt.ylabel(r'SE-Sync Time (s)')
    # plt.xlim([0.0, 100.0])
    # plt.legend()
    # plt.savefig(os.path.join(output_dir, f"sesync_comp_time_{dataset_name}.png"), dpi=600, bbox_inches='tight')
    # plt.close()

    # # Full SE-Sync reference
    # M_full      = construct_sesync_quadratic_form_matrix(measurements)
    # LGrho       = construct_LGrho(measurements)
    # sesync_full = PySESync.SESync(to_sesync_format(measurements), opts)

    # mst_rot_costs = [];     mst_full_costs = [];     mst_SOd_orbdists = [];     mst_ate_trans = [];     mst_rpe_rots = []
    # madow_rot_costs = [];   madow_full_costs = [];   madow_SOd_orbdists = [];   madow_ate_trans = [];   madow_rpe_rots = []
    # nearest_rot_costs = []; nearest_full_costs = []; nearest_SOd_orbdists = []; nearest_ate_trans = []; nearest_rpe_rots = []
    # effMadow_rot_costs = [];effMadow_full_costs = [];effMadow_SOd_orbdists = [];effMadow_ate_trans = [];effMadow_rpe_rots = []
    # org_nearest_rot_costs = [];org_nearest_full_costs = [];org_nearest_SOd_orbdists = [];org_nearest_ate_trans = [];org_nearest_rpe_rots = []
    # org_madow_rot_costs = []; org_madow_full_costs = []; org_madow_SOd_orbdists = []; org_madow_ate_trans = []; org_madow_rpe_rots = []

    # for i in range(len(percents)):
    #     print(f"Percent LC: {percents[i]}")

    #     xhat_nearest         = sesync_nearest[i].xhat
    #     xhat_madow           = sesync_madow[i].xhat
    #     xhat_mst             = sesync_mst[i].xhat
    #     xhat_effrMadow       = sesync_effrMadow[i].xhat
    #     xhat_mac_org_nearest = sesync_mac_org_nearest[i].xhat
    #     xhat_mac_org_madow   = sesync_mac_org_madow[i].xhat

    #     madow_selected_lc          = select_measurements(measurements, mac_mst_madow_results[i])
    #     madow_meas                 = mst_measurements + madow_selected_lc

    #     nearest_selected_lc        = select_measurements(measurements, mac_mst_nearest_results[i])
    #     nearest_meas               = mst_measurements + nearest_selected_lc

    #     mst_meas                   = measurements

    #     effr_madow_selected_lc     = select_measurements(non_mst_measurements_eff, effr_Madow_rounding_results[i])
    #     effrMadow_meas             = mst_measurements_eff + effr_madow_selected_lc

    #     mac_org_nearest_selected_lc = select_measurements(lc_measurements, MAC_original_nearest_results[i])
    #     mac_org_nearest_meas       = odom_measurements + mac_org_nearest_selected_lc

    #     mac_org_madow_selected_lc  = select_measurements(lc_measurements, MAC_original_madow_results[i])
    #     mac_org_madow_meas         = odom_measurements + mac_org_madow_selected_lc

    #     pct_str = str(percents[i])

    #     # --- Madow ---
    #     madow_rot_costs.append(evaluate_sesync_rotation_objective(LGrho, xhat_madow[:, num_poses:]))
    #     madow_full_costs.append(evaluate_sesync_objective(M_full, xhat_madow))
    #     madow_SOd_orbdists.append(orbit_distance_dS(sesync_full.xhat[:, num_poses:], xhat_madow[:, num_poses:]))
    #     madow_ate_trans.append(poses_ate_tran(xhat_madow, sesync_full.xhat))
    #     madow_rpe_rots.append(poses_rpe_rot(xhat_madow, sesync_full.xhat))
    #     plt.figure()
    #     plot_poses(xhat_madow, madow_meas, show=False, color=CLR_MADOW)
    #     plt.title(f"{title_name} — {LBL_MADOW} ({pct_str})")
    #     plt.savefig(os.path.join(output_dir, f"madow_{dataset_name}_{pct_str}.png"), dpi=600)
    #     plt.close()

    #     # --- Nearest ---
    #     nearest_rot_costs.append(evaluate_sesync_rotation_objective(LGrho, xhat_nearest[:, num_poses:]))
    #     nearest_full_costs.append(evaluate_sesync_objective(M_full, xhat_nearest))
    #     nearest_SOd_orbdists.append(orbit_distance_dS(sesync_full.xhat[:, num_poses:], xhat_nearest[:, num_poses:]))
    #     nearest_ate_trans.append(poses_ate_tran(xhat_nearest, sesync_full.xhat))
    #     nearest_rpe_rots.append(poses_rpe_rot(xhat_nearest, sesync_full.xhat))
    #     plt.figure()
    #     plot_poses(xhat_nearest, nearest_meas, show=False, color=CLR_NEAREST)
    #     plt.title(f"{title_name} — {LBL_NEAREST} ({pct_str})")
    #     plt.savefig(os.path.join(output_dir, f"nearest_{dataset_name}_{pct_str}.png"), dpi=600)
    #     plt.close()

    #     # --- MST ---
    #     mst_rot_costs.append(evaluate_sesync_rotation_objective(LGrho, xhat_mst[:, num_poses:]))
    #     mst_full_costs.append(evaluate_sesync_objective(M_full, xhat_mst))
    #     mst_SOd_orbdists.append(orbit_distance_dS(sesync_full.xhat[:, num_poses:], xhat_mst[:, num_poses:]))
    #     mst_ate_trans.append(poses_ate_tran(xhat_mst, sesync_full.xhat))
    #     mst_rpe_rots.append(poses_rpe_rot(xhat_mst, sesync_full.xhat))
    #     plt.figure()
    #     plot_poses(xhat_mst, mst_meas, show=False, color=CLR_NEAREST)
    #     plt.title(f"{title_name} — MST ({pct_str})")
    #     plt.savefig(os.path.join(output_dir, f"mst_{dataset_name}_{pct_str}.png"), dpi=600)
    #     plt.close()

    #     # --- EffrMadow ---
    #     effMadow_rot_costs.append(evaluate_sesync_rotation_objective(LGrho, xhat_effrMadow[:, num_poses:]))
    #     effMadow_full_costs.append(evaluate_sesync_objective(M_full, xhat_effrMadow))
    #     effMadow_SOd_orbdists.append(orbit_distance_dS(sesync_full.xhat[:, num_poses:], xhat_effrMadow[:, num_poses:]))
    #     effMadow_ate_trans.append(poses_ate_tran(xhat_effrMadow, sesync_full.xhat))
    #     effMadow_rpe_rots.append(poses_rpe_rot(xhat_effrMadow, sesync_full.xhat))
    #     plt.figure()
    #     plot_poses(xhat_effrMadow, effrMadow_meas, show=False, color=CLR_ENH_MADOW)
    #     plt.title(f"{title_name} — {LBL_ENH_MADOW} ({pct_str})")
    #     plt.savefig(os.path.join(output_dir, f"effrMadow_{dataset_name}_{pct_str}.png"), dpi=600)
    #     plt.close()

    #     # --- Org Nearest ---
    #     org_nearest_rot_costs.append(evaluate_sesync_rotation_objective(LGrho, xhat_mac_org_nearest[:, num_poses:]))
    #     org_nearest_full_costs.append(evaluate_sesync_objective(M_full, xhat_mac_org_nearest))
    #     org_nearest_SOd_orbdists.append(orbit_distance_dS(sesync_full.xhat[:, num_poses:], xhat_mac_org_nearest[:, num_poses:]))
    #     org_nearest_ate_trans.append(poses_ate_tran(xhat_mac_org_nearest, sesync_full.xhat))
    #     org_nearest_rpe_rots.append(poses_rpe_rot(xhat_mac_org_nearest, sesync_full.xhat))
    #     plt.figure()
    #     plot_poses(xhat_mac_org_nearest, mac_org_nearest_meas, show=False, color=CLR_STD_NEAREST)
    #     plt.title(f"{title_name} — {LBL_STD_NEAREST} ({pct_str})")
    #     plt.savefig(os.path.join(output_dir, f"org_nearest_{dataset_name}_{pct_str}.png"), dpi=600)
    #     plt.close()

    #     # --- Org Madow ---
    #     org_madow_rot_costs.append(evaluate_sesync_rotation_objective(LGrho, xhat_mac_org_madow[:, num_poses:]))
    #     org_madow_full_costs.append(evaluate_sesync_objective(M_full, xhat_mac_org_madow))
    #     org_madow_SOd_orbdists.append(orbit_distance_dS(sesync_full.xhat[:, num_poses:], xhat_mac_org_madow[:, num_poses:]))
    #     org_madow_ate_trans.append(poses_ate_tran(xhat_mac_org_madow, sesync_full.xhat))
    #     org_madow_rpe_rots.append(poses_rpe_rot(xhat_mac_org_madow, sesync_full.xhat))
    #     plt.figure()
    #     plot_poses(xhat_mac_org_madow, mac_org_madow_meas, show=False, color=CLR_STD_MADOW)
    #     plt.title(f"{title_name} — {LBL_STD_MADOW} ({pct_str})")
    #     plt.savefig(os.path.join(output_dir, f"org_madow_{dataset_name}_{pct_str}.png"), dpi=600)
    #     plt.close()

    # #############################
    # # Plot 4: ATE Translation
    # #############################
    # plt.figure()
    # plt.semilogy(100.0 * np.array(percents[:-1]), nearest_ate_trans[:-1],     MRK_NEAREST,     color=CLR_NEAREST,     label=LBL_NEAREST,     markersize=MS)
    # plt.semilogy(100.0 * np.array(percents[:-1]), madow_ate_trans[:-1],       MRK_MADOW,       color=CLR_MADOW,       label=LBL_MADOW,       markersize=MS)
    # plt.semilogy(100.0 * np.array(percents[:-1]), org_nearest_ate_trans[:-1], MRK_STD_NEAREST, color=CLR_STD_NEAREST, label=LBL_STD_NEAREST, markersize=MS)
    # plt.semilogy(100.0 * np.array(percents[:-1]), org_madow_ate_trans[:-1],   MRK_STD_MADOW,   color=CLR_STD_MADOW,   label=LBL_STD_MADOW,   markersize=MS)
    # plt.semilogy(100.0 * np.array(percents[:-1]), effMadow_ate_trans[:-1],    MRK_ENH_MADOW,   color=CLR_ENH_MADOW,   label=LBL_ENH_MADOW,   markersize=MS)
    # plt.title(title_name)
    # plt.xlabel(r'% Edges Added')
    # plt.ylabel(r'ATE Translation [m]')
    # plt.xlim([0.0, 100.0])
    # plt.legend()
    # plt.savefig(os.path.join(output_dir, f"ate_tran_{dataset_name}_log.png"), dpi=600, bbox_inches='tight')
    # plt.close()

    # #############################
    # # Plot 5: RPE Rotation
    # #############################
    # plt.figure()
    # plt.semilogy(100.0 * np.array(percents[:-1]), nearest_rpe_rots[:-1],     MRK_NEAREST,     color=CLR_NEAREST,     label=LBL_NEAREST,     markersize=MS)
    # plt.semilogy(100.0 * np.array(percents[:-1]), madow_rpe_rots[:-1],       MRK_MADOW,       color=CLR_MADOW,       label=LBL_MADOW,       markersize=MS)
    # plt.semilogy(100.0 * np.array(percents[:-1]), org_nearest_rpe_rots[:-1], MRK_STD_NEAREST, color=CLR_STD_NEAREST, label=LBL_STD_NEAREST, markersize=MS)
    # plt.semilogy(100.0 * np.array(percents[:-1]), org_madow_rpe_rots[:-1],   MRK_STD_MADOW,   color=CLR_STD_MADOW,   label=LBL_STD_MADOW,   markersize=MS)
    # plt.semilogy(100.0 * np.array(percents[:-1]), effMadow_rpe_rots[:-1],    MRK_ENH_MADOW,   color=CLR_ENH_MADOW,   label=LBL_ENH_MADOW,   markersize=MS)
    # plt.title(title_name)
    # plt.xlabel(r'% Edges Added')
    # plt.ylabel(r'RPE Rotation [deg]')
    # plt.xlim([0.0, 100.0])
    # plt.legend()
    # plt.savefig(os.path.join(output_dir, f"rpe_rot_{dataset_name}_log.png"), dpi=600, bbox_inches='tight')
    # plt.close()

    # #############################
    # # Plot 6: SE-Sync Objective Value
    # #############################
    # plt.figure()
    # plt.semilogy(100.0 * np.array(percents), nearest_full_costs,     MRK_NEAREST,     color=CLR_NEAREST,     label=LBL_NEAREST,     markersize=MS)
    # plt.semilogy(100.0 * np.array(percents), madow_full_costs,       MRK_MADOW,       color=CLR_MADOW,       label=LBL_MADOW,       markersize=MS)
    # plt.semilogy(100.0 * np.array(percents), org_nearest_full_costs, MRK_STD_NEAREST, color=CLR_STD_NEAREST, label=LBL_STD_NEAREST, markersize=MS)
    # plt.semilogy(100.0 * np.array(percents), org_madow_full_costs,   MRK_STD_MADOW,   color=CLR_STD_MADOW,   label=LBL_STD_MADOW,   markersize=MS)
    # plt.semilogy(100.0 * np.array(percents), effMadow_full_costs,    MRK_ENH_MADOW,   color=CLR_ENH_MADOW,   label=LBL_ENH_MADOW,   markersize=MS)
    # plt.title(title_name)
    # plt.xlabel(r'% Edges Added')
    # plt.ylabel(r'Objective Value')
    # plt.xlim([0.0, 100.0])
    # plt.legend()
    # plt.savefig(os.path.join(output_dir, f"obj_val_{dataset_name}.png"), dpi=600, bbox_inches='tight')
    # plt.close()

    # #############################
    # # Plot 7: SO(d) Orbit Distance
    # #############################
    # plt.figure()
    # plt.plot(100.0 * np.array(percents), nearest_SOd_orbdists,     MRK_NEAREST,     color=CLR_NEAREST,     label=LBL_NEAREST,     markersize=MS)
    # plt.plot(100.0 * np.array(percents), madow_SOd_orbdists,       MRK_MADOW,       color=CLR_MADOW,       label=LBL_MADOW,       markersize=MS)
    # plt.plot(100.0 * np.array(percents), org_nearest_SOd_orbdists, MRK_STD_NEAREST, color=CLR_STD_NEAREST, label=LBL_STD_NEAREST, markersize=MS)
    # plt.plot(100.0 * np.array(percents), org_madow_SOd_orbdists,   MRK_STD_MADOW,   color=CLR_STD_MADOW,   label=LBL_STD_MADOW,   markersize=MS)
    # plt.plot(100.0 * np.array(percents), effMadow_SOd_orbdists,    MRK_ENH_MADOW,   color=CLR_ENH_MADOW,   label=LBL_ENH_MADOW,   markersize=MS)
    # plt.title(title_name)
    # plt.xlabel(r'% Edges Added')
    # plt.ylabel(r'$\mathrm{SO}(d)$ orbit distance')
    # plt.xlim([0.0, 100.0])
    # plt.legend()
    # plt.savefig(os.path.join(output_dir, f"orbdist_{dataset_name}.png"), dpi=600, bbox_inches='tight')
    # plt.close()