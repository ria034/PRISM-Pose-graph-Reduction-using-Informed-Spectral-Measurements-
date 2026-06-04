import numpy as np
import sys
sys.path.insert(0, r"C:\Users\lenovo\Desktop\MAC-Explorer")
import networkx as nx
import os
import json
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from mac.utils.graphs import weight_graph_lap_from_edges
from pose_graph_utils import split_edges, read_g2o_file, rpm_to_mac


def read_data_file(g2o_path):
    dataset_name = os.path.splitext(os.path.basename(g2o_path))[0]
    print(f"Loading dataset: {dataset_name}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(g2o_path)), "results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    print("Reading g2o file")
    measurements, num_poses = read_g2o_file(g2o_path)
    print(f"Dataset has {num_poses} poses and {len(measurements)} measurements")

    odom_measurements, lc_measurements = split_edges(measurements)
    odom_edges = rpm_to_mac(odom_measurements)
    lc_edges   = rpm_to_mac(lc_measurements)
    all_edges  = rpm_to_mac(measurements)
    return odom_edges, lc_edges, all_edges, num_poses


def get_critical_edges(json_path, num_poses, dataset_name):
    with open(json_path, 'r') as f:
        edge_calcs = json.load(f)

    # sort descending — highest delta_lambda2 first
    sorted_edges = sorted(edge_calcs['edges'],
                          key=lambda e: e['delta_lambda2'],
                          reverse=True)

    # top n-1 are the critical edges
    critical_edges = sorted_edges[:num_poses - 1]

    # count LC and odom
    lc_count   = sum(1 for e in critical_edges if not e['is_odom'])
    odom_count = sum(1 for e in critical_edges if e['is_odom'])

    print(f"\nCritical edges (top n-1={num_poses-1} by delta_lambda2):")
    print(f"  Critical LC edges:   {lc_count}  ({100*lc_count/(num_poses-1):.1f}%)")
    print(f"  Critical odom edges: {odom_count}  ({100*odom_count/(num_poses-1):.1f}%)")
    print(f"  Dataset: {dataset_name}")

    # build set of (i,j) pairs for fast lookup
    critical_set = set()
    for e in critical_edges:
        critical_set.add((min(e['i'], e['j']), max(e['i'], e['j'])))

    return critical_edges, critical_set


def compute_effective_resistance(all_edges, num_poses):
    edge_weights = np.array([e.weight for e in all_edges])
    edges_array  = np.array([[e.i, e.j] for e in all_edges])
    L            = weight_graph_lap_from_edges(edges_array, edge_weights, num_poses)

    L_dense = L.toarray()
    L_pinv  = np.linalg.pinv(L_dense)

    m = len(all_edges)
    B = np.zeros((m, num_poses))
    for k, edge in enumerate(all_edges):
        B[k, edge.i] =  1
        B[k, edge.j] = -1

    x_eff_resistance = np.diag(B @ L_pinv @ B.T)
    return x_eff_resistance


def build_MST(all_edges, weights, num_poses):
    G = nx.Graph()
    G.add_nodes_from(range(num_poses))
    for weight, edge in zip(weights, all_edges):
        G.add_edge(edge.i, edge.j, weight=weight)
    mst        = nx.maximum_spanning_tree(G, weight='weight')
    mst_edges  = set(mst.edges())
    x_mst      = np.zeros(len(all_edges))
    for k, edge in enumerate(all_edges):
        if (edge.i, edge.j) in mst_edges or (edge.j, edge.i) in mst_edges:
            x_mst[k] = 1.0
    return x_mst


def build_laplacian_from_edge_objects(edges, num_poses):
    rows, cols, vals = [], [], []
    for edge in edges:
        w = edge.weight
        i, j = edge.i, edge.j
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]
    return csr_matrix((vals, (rows, cols)), shape=(num_poses, num_poses))


def compute_fiedler(edges, num_poses):
    L = build_laplacian_from_edge_objects(edges, num_poses)
    try:
        vals, vecs = eigsh(L, k=2, which='SM', tol=1e-5,
                           maxiter=num_poses * 20,
                           ncv=min(50, num_poses))
        idx = np.argsort(vals)
        return vals[idx[1]], vecs[:, idx[1]]
    except Exception:
        print("  (falling back to dense eigh)")
        L_dense = L.toarray()
        vals, vecs = eigh(L_dense, subset_by_index=[0, 1])
        return vals[1], vecs[:, 1]


def compare_MST_bottlenecks(all_edges, x_mst, critical_set, num_poses):
    mst_set = set()
    for k, edge in enumerate(all_edges):
        if x_mst[k] == 1.0:
            mst_set.add((min(edge.i, edge.j), max(edge.i, edge.j)))

    overlap    = mst_set & critical_set
    mst_missed = critical_set - mst_set
    mst_extra  = mst_set - critical_set

    print("\n" + "="*60)
    print("MST vs CRITICAL EDGES COMPARISON")
    print("="*60)
    print(f"MST size:             {len(mst_set)}")
    print(f"Critical edges:       {len(critical_set)}")
    print(f"Overlap:              {len(overlap)}/{num_poses-1} ({100*len(overlap)/(num_poses-1):.1f}%)")
    print(f"Missed by MST:        {len(mst_missed)}/{num_poses-1} ({100*len(mst_missed)/(num_poses-1):.1f}%)")
    print(f"MST non-critical:     {len(mst_extra)}/{len(mst_set)} ({100*len(mst_extra)/len(mst_set):.1f}%)")

    missed_lc = missed_odom = 0
    for edge_key in mst_missed:
        for edge in all_edges:
            if (min(edge.i, edge.j), max(edge.i, edge.j)) == edge_key:
                if abs(edge.i - edge.j) == 1:
                    missed_odom += 1
                else:
                    missed_lc += 1
                break

    print(f"\nMissed critical edges breakdown:")
    print(f"  odom: {missed_odom}")
    print(f"  LC:   {missed_lc}")

    if len(mst_missed) > 0:
        print(f"\n→ MST misses {100*len(mst_missed)/(num_poses-1):.1f}% of critical edges")
        print(f"→ MST selects {100*len(mst_extra)/len(mst_set):.1f}% non-critical edges")
        print(f"→ PROVEN: MST does not select critical edges")
    else:
        print(f"\n→ MST correctly selects all critical edges on this dataset")

    return mst_set, mst_missed, mst_extra


def compare_odom_bottlenecks(odom_edges, critical_set, num_poses):
    odom_set = set()
    for edge in odom_edges:
        odom_set.add((min(edge.i, edge.j), max(edge.i, edge.j)))

    overlap     = odom_set & critical_set
    odom_missed = critical_set - odom_set
    odom_extra  = odom_set - critical_set

    print("\n" + "="*60)
    print("ODOM BACKBONE vs CRITICAL EDGES COMPARISON")
    print("="*60)
    print(f"Odom size:            {len(odom_set)}")
    print(f"Critical edges:       {len(critical_set)}")
    print(f"Overlap:              {len(overlap)}/{num_poses-1} ({100*len(overlap)/(num_poses-1):.1f}%)")
    print(f"Missed by odom:       {len(odom_missed)}/{num_poses-1} ({100*len(odom_missed)/(num_poses-1):.1f}%)")
    print(f"Odom non-critical:    {len(odom_extra)}/{len(odom_set)} ({100*len(odom_extra)/len(odom_set):.1f}%)")
    print(f"\nMissed critical edges (must be LC): {len(odom_missed)}")

    if len(odom_missed) > 0:
        print(f"\n→ Odom misses {100*len(odom_missed)/(num_poses-1):.1f}% of critical edges")
        print(f"→ These are all LC edges that odom cannot provide")
        print(f"→ Odom has {len(odom_extra)} redundant non-critical edges")
    else:
        print(f"\n→ Odom correctly covers all critical edges")

    return odom_set, odom_missed, odom_extra


def compare_fiedler_recovery(all_edges, odom_edges, mst_set,
                              critical_edges, critical_set, num_poses):
    print("\n" + "="*60)
    print("FIEDLER VALUE AND RECOVERY OF EACH BACKBONE")
    print("="*60)

    # full graph Fiedler — ground truth
    lv_full, fv_full = compute_fiedler(all_edges, num_poses)
    print(f"\nFull graph lambda2:      {lv_full:.6f}  (ground truth)")

    # MST backbone
    mst_edge_list = [edge for edge in all_edges
                     if (min(edge.i, edge.j), max(edge.i, edge.j)) in mst_set]
    lv_mst, fv_mst = compute_fiedler(mst_edge_list, num_poses)
    if np.dot(fv_mst, fv_full) < 0:
        fv_mst = -fv_mst
    corr_mst = np.corrcoef(fv_full, fv_mst)[0, 1]
    l2_mst   = np.linalg.norm(fv_full - fv_mst)
    print(f"MST backbone lambda2:    {lv_mst:.6f}  "
          f"corr: {corr_mst:.4f}  "
          f"L2: {l2_mst:.4f}")

    # odom backbone
    lv_odom, fv_odom = compute_fiedler(odom_edges, num_poses)
    if np.dot(fv_odom, fv_full) < 0:
        fv_odom = -fv_odom
    corr_odom = np.corrcoef(fv_full, fv_odom)[0, 1]
    l2_odom   = np.linalg.norm(fv_full - fv_odom)
    print(f"Odom backbone lambda2:   {lv_odom:.6f}  "
          f"corr: {corr_odom:.4f}  "
          f"L2: {l2_odom:.4f}")

    # ideal backbone (top n-1 by delta_lambda2)
    critical_edge_list = []
    for e in critical_edges:
        key = (min(e['i'], e['j']), max(e['i'], e['j']))
        for edge in all_edges:
            if (min(edge.i, edge.j), max(edge.i, edge.j)) == key:
                critical_edge_list.append(edge)
                break
    lv_crit, fv_crit = compute_fiedler(critical_edge_list, num_poses)
    if np.dot(fv_crit, fv_full) < 0:
        fv_crit = -fv_crit
    corr_crit = np.corrcoef(fv_full, fv_crit)[0, 1]
    l2_crit   = np.linalg.norm(fv_full - fv_crit)
    print(f"Ideal backbone lambda2:  {lv_crit:.6f}  "
          f"corr: {corr_crit:.4f}  "
          f"L2: {l2_crit:.4f}")

    print(f"\nFiedler recovery ranking (higher corr = better):")
    backbones = [
        ("Ideal",  corr_crit, lv_crit),
        ("Odom",   corr_odom, lv_odom),
        ("MST",    corr_mst,  lv_mst),
    ]
    backbones_sorted = sorted(backbones, key=lambda x: x[1], reverse=True)
    for rank, (name, corr, lv) in enumerate(backbones_sorted):
        print(f"  rank {rank+1}: {name:<8} corr={corr:.4f}  lambda2={lv:.6f}")

    print(f"\nKey insight:")
    if corr_odom > corr_mst:
        print(f"  Odom backbone better preserves Fiedler structure than MST")
        print(f"  MST heuristic distorts the spectral structure")
    else:
        print(f"  MST backbone better preserves Fiedler structure than Odom")

    return lv_full, lv_mst, lv_odom, lv_crit, corr_mst, corr_odom, corr_crit


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} [.g2o file] [.json scores file]")
        sys.exit()

    g2o_path     = sys.argv[1]
    json_path    = sys.argv[2]
    dataset_name = os.path.splitext(os.path.basename(g2o_path))[0]

    # load data
    odom_edges, lc_edges, all_edges, num_poses = read_data_file(g2o_path)

    # get critical edges from saved delta_lambda2
    critical_edges, critical_set = get_critical_edges(json_path, num_poses, dataset_name)

    # compute effective resistance and build MST
    print("\nComputing effective resistance...")
    x_eff_resistance = compute_effective_resistance(all_edges, num_poses)
    print("Building MST...")
    edge_weights = np.array([e.weight for e in all_edges])
    x_mst = build_MST(all_edges, x_eff_resistance * edge_weights, num_poses)

    # compare MST vs critical edges
    mst_set, mst_missed, mst_extra = compare_MST_bottlenecks(
        all_edges, x_mst, critical_set, num_poses)

    # compare odom vs critical edges
    odom_set, odom_missed, odom_extra = compare_odom_bottlenecks(
        odom_edges, critical_set, num_poses)

    # compare Fiedler recovery
    lv_full, lv_mst, lv_odom, lv_crit, corr_mst, corr_odom, corr_crit = \
        compare_fiedler_recovery(all_edges, odom_edges, mst_set,
                                 critical_edges, critical_set, num_poses)

    # final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Dataset:                    {dataset_name}")
    print(f"Total edges:                {len(all_edges)}")
    print(f"Critical edges (n-1):       {num_poses-1}")
    print(f"")
    print(f"MST overlap with ideal:     {100*len(mst_set & critical_set)/(num_poses-1):.1f}%")
    print(f"Odom overlap with ideal:    {100*len(odom_set & critical_set)/(num_poses-1):.1f}%")
    print(f"")
    print(f"MST missed:                 {100*len(mst_missed)/(num_poses-1):.1f}%")
    print(f"Odom missed:                {100*len(odom_missed)/(num_poses-1):.1f}%")
    print(f"")
    print(f"Full graph lambda2:         {lv_full:.6f}")
    print(f"Ideal backbone lambda2:     {lv_crit:.6f}")
    print(f"Odom backbone lambda2:      {lv_odom:.6f}")
    print(f"MST backbone lambda2:       {lv_mst:.6f}")
    print(f"")
    print(f"Ideal backbone Fiedler corr: {corr_crit:.4f}")
    print(f"Odom backbone Fiedler corr:  {corr_odom:.4f}")
    print(f"MST backbone Fiedler corr:   {corr_mst:.4f}")

    if len(odom_set & critical_set) > len(mst_set & critical_set):
        print(f"\n→ ODOM backbone covers more critical edges than MST")
        print(f"→ MST heuristic is WORSE than simply using odom edges")
    else:
        print(f"\n→ MST covers more critical edges than odom on this dataset")

    if corr_odom > corr_mst:
        print(f"→ Odom backbone better preserves Fiedler structure than MST")
    else:
        print(f"→ MST backbone better preserves Fiedler structure than Odom")