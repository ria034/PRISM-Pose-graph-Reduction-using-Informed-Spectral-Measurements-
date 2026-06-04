import numpy as np
import sys
sys.path.insert(0, r"C:\Users\lenovo\Desktop\MAC-Explorer")
import networkx as nx
import os
import json
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from mac.solvers import MAC, NaiveGreedy
from mac.utils.graphs import weight_graph_lap_from_edges
from mac.utils.rounding import round_nearest
from pose_graph_utils import split_edges, read_g2o_file, rpm_to_mac


PERCENTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def read_data_file(g2o_path):
    dataset_name = os.path.splitext(os.path.basename(g2o_path))[0]
    print(f"Loading dataset: {dataset_name}")
    measurements, num_poses = read_g2o_file(g2o_path)
    print(f"Dataset has {num_poses} poses and {len(measurements)} measurements")
    odom_measurements, lc_measurements = split_edges(measurements)
    odom_edges = rpm_to_mac(odom_measurements)
    lc_edges   = rpm_to_mac(lc_measurements)
    all_edges  = rpm_to_mac(measurements)
    return odom_edges, lc_edges, all_edges, num_poses


def get_critical_set(json_path, num_poses):
    with open(json_path, 'r') as f:
        edge_calcs = json.load(f)

    sorted_edges = sorted(edge_calcs['edges'],
                          key=lambda e: e['delta_lambda2'],
                          reverse=True)
    critical_edges = sorted_edges[:num_poses - 1]

    lc_count   = sum(1 for e in critical_edges if not e['is_odom'])
    odom_count = sum(1 for e in critical_edges if e['is_odom'])

    print(f"\nCritical edges (top n-1={num_poses-1} by delta_lambda2):")
    print(f"  Critical odom: {odom_count} ({100*odom_count/(num_poses-1):.1f}%)")
    print(f"  Critical LC:   {lc_count}   ({100*lc_count/(num_poses-1):.1f}%)")

    critical_set = set()
    for e in critical_edges:
        critical_set.add((min(e['i'], e['j']), max(e['i'], e['j'])))

    return critical_set, lc_count, odom_count


def compute_effective_resistance(all_edges, num_poses):
    edge_weights = np.array([e.weight for e in all_edges])
    edges_array  = np.array([[e.i, e.j] for e in all_edges])
    L            = weight_graph_lap_from_edges(edges_array, edge_weights, num_poses)
    L_dense      = L.toarray()
    L_pinv       = np.linalg.pinv(L_dense)
    m = len(all_edges)
    B = np.zeros((m, num_poses))
    for k, edge in enumerate(all_edges):
        B[k, edge.i] =  1
        B[k, edge.j] = -1
    return np.diag(B @ L_pinv @ B.T)


def build_MST(all_edges, weights, num_poses):
    G = nx.Graph()
    G.add_nodes_from(range(num_poses))
    for weight, edge in zip(weights, all_edges):
        G.add_edge(edge.i, edge.j, weight=weight)
    mst       = nx.maximum_spanning_tree(G, weight='weight')
    mst_edges = set(mst.edges())
    x_mst     = np.zeros(len(all_edges))
    for k, edge in enumerate(all_edges):
        if (edge.i, edge.j) in mst_edges or (edge.j, edge.i) in mst_edges:
            x_mst[k] = 1.0
    return x_mst


def get_edge_set(edges):
    return set((min(e.i, e.j), max(e.i, e.j)) for e in edges)


def compute_critical_coverage(backbone_set, mac_selected_set, critical_set):
    combined = backbone_set | mac_selected_set
    overlap  = combined & critical_set
    return len(overlap), 100 * len(overlap) / len(critical_set)


def run_mac_and_measure(backbone_edges, lc_edges, num_poses,
                        critical_set, backbone_set, label):
    print(f"\n{'='*60}")
    print(f"Running MAC on {label} backbone")
    print(f"{'='*60}")

    mac   = MAC(backbone_edges, lc_edges, num_poses,
                fiedler_method="tracemin_cholesky")
    naive = NaiveGreedy(lc_edges)

    coverage_counts  = []
    coverage_percents = []
    lambda2_vals     = []

    # baseline: backbone alone, no MAC edges
    base_count, base_pct = compute_critical_coverage(
        backbone_set, set(), critical_set)
    print(f"\nBackbone alone (budget=0):")
    print(f"  Critical edges covered: {base_count}/{len(critical_set)} ({base_pct:.1f}%)")

    for pct in PERCENTS:
        num_lc = int(pct * len(lc_edges))

        w_init  = naive.subset(num_lc)
        result, unrounded, upper, _ = mac.solve(
            num_lc, w_init, max_iters=20,
            rounding="nearest", return_rounding_time=True,
            use_cache=True)

        # get MAC selected edges
        mac_selected_set = set()
        for k, edge in enumerate(lc_edges):
            if result[k] == 1.0:
                mac_selected_set.add((min(edge.i, edge.j),
                                      max(edge.i, edge.j)))

        # measure critical coverage
        count, pct_covered = compute_critical_coverage(
            backbone_set, mac_selected_set, critical_set)

        # measure lambda2
        lv = mac.evaluate_objective(result)

        coverage_counts.append(count)
        coverage_percents.append(pct_covered)
        lambda2_vals.append(lv)

        print(f"  Budget {int(pct*100):3d}%: "
              f"critical covered={count}/{len(critical_set)} "
              f"({pct_covered:.1f}%)  "
              f"lambda2={lv:.4f}")

    return coverage_counts, coverage_percents, lambda2_vals, base_count, base_pct


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} [.g2o file] [.json scores file]")
        sys.exit()

    g2o_path     = sys.argv[1]
    json_path    = sys.argv[2]
    dataset_name = os.path.splitext(os.path.basename(g2o_path))[0]

    output_dir = os.path.join(os.path.dirname(os.path.abspath(g2o_path)), "results")
    os.makedirs(output_dir, exist_ok=True)

    # load data
    odom_edges, lc_edges, all_edges, num_poses = read_data_file(g2o_path)

    # get critical edges
    critical_set, lc_count, odom_count = get_critical_set(json_path, num_poses)

    # build odom backbone set
    odom_set = get_edge_set(odom_edges)

    # build MST backbone
    print("\nComputing effective resistance...")
    x_eff_resistance = compute_effective_resistance(all_edges, num_poses)
    edge_weights     = np.array([e.weight for e in all_edges])
    x_mst            = build_MST(all_edges, x_eff_resistance * edge_weights, num_poses)

    mst_set = set()
    for k, edge in enumerate(all_edges):
        if x_mst[k] == 1.0:
            mst_set.add((min(edge.i, edge.j), max(edge.i, edge.j)))

    mst_backbone_edges = [all_edges[k] for k in range(len(all_edges))
                          if x_mst[k] == 1.0]

    # baseline coverage
    odom_base_count = len(odom_set & critical_set)
    mst_base_count  = len(mst_set  & critical_set)
    print(f"\nBackbone baseline coverage (before MAC):")
    print(f"  Odom backbone covers: {odom_base_count}/{len(critical_set)} "
          f"({100*odom_base_count/len(critical_set):.1f}%)")
    print(f"  MST  backbone covers: {mst_base_count}/{len(critical_set)} "
          f"({100*mst_base_count/len(critical_set):.1f}%)")

    # run MAC on odom backbone
    odom_counts, odom_pcts, odom_lv, odom_base, odom_base_pct = \
        run_mac_and_measure(odom_edges, lc_edges, num_poses,
                            critical_set, odom_set, "ODOM")

    # run MAC on MST backbone
    # MST backbone edges are fixed, remaining edges are candidates
    remaining_idx   = [k for k in range(len(all_edges)) if x_mst[k] == 0.0]
    remaining_edges = [all_edges[k] for k in remaining_idx]

    mst_counts, mst_pcts, mst_lv, mst_base, mst_base_pct = \
        run_mac_and_measure(mst_backbone_edges, remaining_edges, num_poses,
                            critical_set, mst_set, "MST")

    # --------------------------------------------------------
    # PLOT 1: Critical edge coverage vs budget
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # plot coverage
    ax = axes[0]
    budgets = [0] + [int(p*100) for p in PERCENTS]

    odom_pcts_full = [100*odom_base/len(critical_set)] + odom_pcts
    mst_pcts_full  = [100*mst_base/len(critical_set)]  + mst_pcts

    ax.plot(budgets, odom_pcts_full, '-o', color='#1D9E75',
            label='Odom backbone + MAC', markersize=5)
    ax.plot(budgets, mst_pcts_full,  '-s', color='#D85A30',
            label='MST backbone + MAC',  markersize=5)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% coverage')
    ax.set_xlabel('Budget (% of LC edges)')
    ax.set_ylabel('Critical edges covered (%)')
    ax.set_title(f'{dataset_name}: Critical Edge Recovery vs Budget')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])

    # plot lambda2
    ax2 = axes[1]
    ax2.plot([int(p*100) for p in PERCENTS], odom_lv, '-o', color='#1D9E75',
             label='Odom backbone + MAC', markersize=5)
    ax2.plot([int(p*100) for p in PERCENTS], mst_lv,  '-s', color='#D85A30',
             label='MST backbone + MAC',  markersize=5)
    ax2.set_xlabel('Budget (% of LC edges)')
    ax2.set_ylabel('Algebraic Connectivity λ₂')
    ax2.set_title(f'{dataset_name}: Lambda2 vs Budget')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{dataset_name}_critical_recovery.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Dataset:                     {dataset_name}")
    print(f"Total critical edges (n-1):  {len(critical_set)}")
    print(f"  Critical odom:             {odom_count} ({100*odom_count/len(critical_set):.1f}%)")
    print(f"  Critical LC:               {lc_count}   ({100*lc_count/len(critical_set):.1f}%)")

    print(f"\nBaseline (backbone alone):")
    print(f"  Odom backbone covers:      {odom_base_count}/{len(critical_set)} ({100*odom_base_count/len(critical_set):.1f}%)")
    print(f"  MST  backbone covers:      {mst_base_count}/{len(critical_set)}  ({100*mst_base_count/len(critical_set):.1f}%)")

    print(f"\nAfter MAC at 40% budget:")
    idx_40 = PERCENTS.index(0.4)
    print(f"  Odom + MAC covers:         {odom_counts[idx_40]}/{len(critical_set)} ({odom_pcts[idx_40]:.1f}%)")
    print(f"  MST  + MAC covers:         {mst_counts[idx_40]}/{len(critical_set)}  ({mst_pcts[idx_40]:.1f}%)")

    print(f"\nAfter MAC at 100% budget:")
    print(f"  Odom + MAC covers:         {odom_counts[-1]}/{len(critical_set)} ({odom_pcts[-1]:.1f}%)")
    print(f"  MST  + MAC covers:         {mst_counts[-1]}/{len(critical_set)}  ({mst_pcts[-1]:.1f}%)")

    print(f"\nKey finding:")
    if odom_pcts[-1] > mst_pcts[-1]:
        print(f"  Odom + MAC recovers MORE critical edges than MST + MAC")
        print(f"  MST backbone actively prevents MAC from finding critical edges")
        print(f"  even at 100% budget")
    else:
        print(f"  MST + MAC recovers more critical edges than Odom + MAC")
        print(f"  on this dataset MST backbone is better foundation")