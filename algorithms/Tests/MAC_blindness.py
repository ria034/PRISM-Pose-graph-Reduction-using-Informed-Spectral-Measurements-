import warnings
warnings.filterwarnings("ignore")

import os
import sys

# --- DYNAMIC PATH INJECTION ---
# 1. Get the directory where THIS script is (the 'algorithms' folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the root directory (the parent of 'algorithms')
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

# 3. Add both to sys.path so Python can see everything
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now standard DLL loads for SE-Sync
os.add_dll_directory(r"C:\Users\lenovo\Downloads\SE-Sync\C++\build\lib")
os.add_dll_directory(r"C:\Users\lenovo\anaconda3\envs\repo_env\Library\bin")

import matplotlib
matplotlib.rcParams['text.usetex'] = False
import random
import numpy as np
import networkx as nx
from timeit import default_timer as timer

# --- CORRECTED LOCAL IMPORT ---
# Since pose_graph_utils.py is in the SAME folder as this script
from pose_graph_utils import (
    split_edges, read_g2o_file, plot_poses, rpm_to_mac, 
    RelativePoseMeasurement, poses_ate_tran, poses_rpe_rot
)

# MAC requirements
from mac.solvers import MAC, NaiveGreedy
from mac.utils.graphs import weight_graph_lap_from_edges
from mac.utils.rounding import round_madow, round_nearest
from mac.utils.eigen_spectrum import find_eigen_spectrum 

import matplotlib.pyplot as plt

sesync_lib_path = r"C:\Users\lenovo\Downloads\SE-Sync\C++\build\lib"
if sesync_lib_path not in sys.path:
    sys.path.insert(0, sesync_lib_path)

import PySESync

# --- VISUALIZATION HELPER ---
def plot_spectral_comparison(lambdas_before, lambdas_after, vecs_before, vecs_after, budget_pct):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Eigenvalues (Scree Plot)
    ax1.plot(range(1, len(lambdas_before)+1), lambdas_before, 'o--', color='gray', label='MST Only')
    ax1.plot(range(1, len(lambdas_after)+1), lambdas_after, 's-', color='blue', label=f'MAC Optimized ({budget_pct})')
    # ax1.set_yscale('log')
    ax1.set_title("Eigenvalue Spectrum")
    ax1.set_xlabel("k-th Eigenvalue")
    ax1.legend()

    # Plot Fiedler Vector Entries (The 2nd Eigenvector)
    ax2.scatter(range(len(vecs_before)), vecs_before[:, 1], s=10, color='gray', alpha=0.5, label='MST')
    ax2.scatter(range(len(vecs_after)), vecs_after[:, 1], s=10, color='blue', alpha=0.5, label='MAC')
    ax2.set_title("Fiedler Vector Entries")
    ax2.set_xlabel("Node Index")
    ax2.legend()
    plt.show()

def compute_effective_resistance(candidate_edges, num_nodes):
    edges_array = np.array([[edge.i, edge.j] for edge in candidate_edges])
    weights = np.array([edge.weight for edge in candidate_edges])
    L = weight_graph_lap_from_edges(edges_array, weights, num_nodes)
    L_dense = L.toarray()
    L_pinv = np.linalg.pinv(L_dense)
    m = len(candidate_edges)
    B = np.zeros((m, num_nodes))
    for k, edge in enumerate(candidate_edges):
        B[k, edge.i] = 1
        B[k, edge.j] = -1
    return np.diag(B @ L_pinv @ B.T)

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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python MAC_blindness.py [path_to_g2o]")
        sys.exit()

    g2o_path = sys.argv[1]
    if not os.path.exists(g2o_path):
        print(f"Error: Path does not exist: {g2o_path}")
        sys.exit()

    measurements, num_poses = read_g2o_file(g2o_path)
    measurements = rpm_to_mac(measurements)
    
    
    # --- EffrMadow Baseline Logic ---
    x_eff_resistance = compute_effective_resistance(measurements, num_poses)
    edge_weights = np.array([edge.weight for edge in measurements])
    weights_enhanced_MAC = x_eff_resistance * edge_weights
    x_MST_Madow_eff = build_MST(measurements, weights_enhanced_MAC, num_poses)
    
    mst_edges = [e for i, e in enumerate(measurements) if x_MST_Madow_eff[i] == 1]
    non_mst_edges = [e for i, e in enumerate(measurements) if x_MST_Madow_eff[i] == 0]
    
    # Initial spectrum (MST Only)
    weights_MST = np.array([edge.weight for edge in mst_edges])
    # Ensure weight_graph_lap_from_edges gets an array of [i, j]
    mst_edges_array = np.array([[e.i, e.j] for e in mst_edges])
    L_x_MST_Madow = weight_graph_lap_from_edges(mst_edges_array, weights_MST, num_poses)
    lambdas_before, vecs_before = find_eigen_spectrum(L_x_MST_Madow, sigma=1e-5, k=5)

    mac_effrMadow = MAC(fixed_edges=mst_edges, candidate_edges=non_mst_edges, num_nodes=num_poses, fiedler_method="tracemin_lu", fiedler_module="standard")
    naive_effrMadow = NaiveGreedy(non_mst_edges)
    percent_lc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for pct_effrMadow in percent_lc:
        num_effrMadow = int(pct_effrMadow * len(non_mst_edges))
        w_init_effrMadow = naive_effrMadow.subset(num_effrMadow)

        _, unrounded_effrMadow, _ = mac_effrMadow.solve(num_effrMadow, w_init_effrMadow, max_iters=20, use_cache=True)
        effrMadow_madow_mask = round_madow(unrounded_effrMadow, num_effrMadow, seed=np.random.RandomState(42))

        selected_non_mst = [non_mst_edges[i] for i, val in enumerate(effrMadow_madow_mask) if val == 1.0]
        whole_set = mst_edges + selected_non_mst
        
        weights_whole = np.array([edge.weight for edge in whole_set])
        edges_array_whole = np.array([[edge.i, edge.j] for edge in whole_set])

        L_whole_set = weight_graph_lap_from_edges(edges_array_whole, weights_whole, num_poses)
        lambdas_sorted, vector_sorted = find_eigen_spectrum(L_whole_set, sigma=1e-5, k=5)
        
        plot_spectral_comparison(lambdas_before, lambdas_sorted, vecs_before, vector_sorted, pct_effrMadow)