import os
import sys
sys.path.insert(0, r"C:\Users\lenovo\Desktop\MAC-Explorer")

import numpy as np
import networkx as nx
import json
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from pose_graph_utils import split_edges, read_g2o_file, rpm_to_mac

# ─────────────────────────────────────────
# Laplacian builder  (weight = kappa = i33)
# ─────────────────────────────────────────
def build_laplacian(edges, num_poses):
    rows, cols, vals = [], [], []
    for edge in edges:
        w = edge.weight
        i, j = edge.i, edge.j
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]
    return csr_matrix((vals, (rows, cols)), shape=(num_poses, num_poses))

# ─────────────────────────────────────────
# Fiedler value
# ─────────────────────────────────────────
def compute_lambda2(edges, num_poses, v0=None):
    L = build_laplacian(edges, num_poses)
    try:
        kwargs = dict(k=2, which='SM', tol=1e-5,
                      maxiter=num_poses * 20, ncv=min(50, num_poses))
        if v0 is not None:
            kwargs['v0'] = v0
        vals, _ = eigsh(L, **kwargs)
        return float(sorted(vals)[1])
    except Exception:
        vals, _ = eigh(L.toarray(), subset_by_index=[0, 1])
        return float(vals[1])

def get_fiedler_vec(edges, num_poses):
    L = build_laplacian(edges, num_poses)
    try:
        vals, vecs = eigsh(L, k=2, which='SM', tol=1e-5,
                           maxiter=num_poses*20, ncv=min(50, num_poses))
        return vecs[:, np.argsort(vals)[1]]
    except Exception:
        vals, vecs = eigh(L.toarray(), subset_by_index=[0, 1])
        return vecs[:, 1]

# ─────────────────────────────────────────
# Build MST backbone using R_eff * weight
# ─────────────────────────────────────────
def build_mst_backbone(all_edges, num_poses):
    rows, cols, vals = [], [], []
    for e in all_edges:
        w = e.weight
        rows += [e.i, e.j, e.i, e.j]
        cols += [e.j, e.i, e.i, e.j]
        vals += [-w, -w, w, w]
    L      = csr_matrix((vals, (rows, cols)), shape=(num_poses, num_poses))
    L_pinv = np.linalg.pinv(L.toarray())

    m = len(all_edges)
    B = np.zeros((m, num_poses))
    for k, e in enumerate(all_edges):
        B[k, e.i] =  1
        B[k, e.j] = -1
    r_eff    = np.diag(B @ L_pinv @ B.T)
    enhanced = r_eff * np.array([e.weight for e in all_edges])

    G = nx.Graph()
    G.add_nodes_from(range(num_poses))
    for k, e in enumerate(all_edges):
        G.add_edge(e.i, e.j, weight=enhanced[k])
    mst     = nx.maximum_spanning_tree(G, weight='weight')
    mst_set = set((min(u,v), max(u,v)) for u, v in mst.edges())
    return [e for e in all_edges if (min(e.i,e.j), max(e.i,e.j)) in mst_set]

# ─────────────────────────────────────────
# Save / load results
# ─────────────────────────────────────────
def save_results(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"Results saved to: {path}")

def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main(g2o_path):
    dataset_name = os.path.splitext(os.path.basename(g2o_path))[0]
    save_path    = os.path.join(os.path.dirname(g2o_path),
                                f"{dataset_name}_edge_importance_cases.json")
    plot_path    = os.path.join(os.path.dirname(g2o_path),
                                f"{dataset_name}_edge_importance_cases.png")

    # ── load if already computed ───────────
    if os.path.exists(save_path):
        print(f"Found saved results at: {save_path}")
        print("Loading from disk — skipping recomputation.")
        data   = load_results(save_path)
        case1  = np.array(data['case1'])
        case2  = np.array(data['case2'])
        case3  = np.array(data['case3'])
        case4  = np.array(data['case4'])
        edges_info = data['edges']
    else:
        # ── load graph ────────────────────────
        print(f"Loading: {g2o_path}")
        measurements, num_poses = read_g2o_file(g2o_path)
        odom_meas, lc_meas      = split_edges(measurements)

        odom_edges = rpm_to_mac(odom_meas)
        lc_edges   = rpm_to_mac(lc_meas)
        all_edges  = rpm_to_mac(measurements)

        print(f"Poses: {num_poses}  Odom: {len(odom_edges)}  LC: {len(lc_edges)}  Total: {len(all_edges)}")

        print("\nBuilding MST backbone...")
        mst_edges = build_mst_backbone(all_edges, num_poses)
        print(f"MST edges: {len(mst_edges)}")

        print("Computing baseline lambda2 values...")
        lv_full = compute_lambda2(all_edges,  num_poses)
        lv_odom = compute_lambda2(odom_edges, num_poses)
        lv_mst  = compute_lambda2(mst_edges,  num_poses)
        print(f"lambda2 — full: {lv_full:.6f}  odom: {lv_odom:.6f}  MST: {lv_mst:.6f}")

        v0_full = get_fiedler_vec(all_edges,  num_poses)
        v0_odom = get_fiedler_vec(odom_edges, num_poses)
        v0_mst  = get_fiedler_vec(mst_edges,  num_poses)

        mst_set = set((min(e.i,e.j), max(e.i,e.j)) for e in mst_edges)

        # ── per-edge computation for ALL edges ─
        print(f"\nComputing all 4 cases for ALL {len(all_edges)} edges...")
        print("Saving after every 50 edges so progress is not lost.\n")

        case1, case2, case3, case4 = [], [], [], []
        edges_info = []

        for idx, edge in enumerate(all_edges):
            if idx % 50 == 0:
                print(f"  edge {idx}/{len(all_edges)}")
                # save intermediate progress
                save_results(save_path, {
                    'dataset':    dataset_name,
                    'num_poses':  num_poses,
                    'lv_full':    lv_full,
                    'lv_odom':    lv_odom,
                    'lv_mst':     lv_mst,
                    'complete':   False,
                    'edges':      edges_info,
                    'case1':      case1,
                    'case2':      case2,
                    'case3':      case3,
                    'case4':      case4,
                })

            key     = (min(edge.i, edge.j), max(edge.i, edge.j))
            is_odom = key not in set((min(e.i,e.j), max(e.i,e.j)) for e in lc_edges)
            in_mst  = key in mst_set

            # full graph without this edge
            full_minus = [e for e in all_edges
                          if (min(e.i,e.j), max(e.i,e.j)) != key]

            # Case 1: remove from full graph
            lv_c1    = compute_lambda2(full_minus, num_poses, v0=v0_full)
            delta_c1 = lv_full - lv_c1

            # Case 2: add back to full-minus (sanity check)
            lv_c2    = compute_lambda2(full_minus + [edge], num_poses, v0=v0_full)
            delta_c2 = lv_c2 - lv_c1

            # Case 3: add to odom backbone
            lv_c3    = compute_lambda2(odom_edges + [edge], num_poses, v0=v0_odom)
            delta_c3 = lv_c3 - lv_odom

            # Case 4: add to MST backbone
            lv_c4    = compute_lambda2(mst_edges + [edge], num_poses, v0=v0_mst)
            delta_c4 = lv_c4 - lv_mst

            case1.append(delta_c1)
            case2.append(delta_c2)
            case3.append(delta_c3)
            case4.append(delta_c4)
            edges_info.append({
                'i':       edge.i,
                'j':       edge.j,
                'weight':  edge.weight,
                'is_odom': is_odom,
                'in_mst':  in_mst,
            })

        # final save
        save_results(save_path, {
            'dataset':   dataset_name,
            'num_poses': num_poses,
            'lv_full':   lv_full,
            'lv_odom':   lv_odom,
            'lv_mst':    lv_mst,
            'complete':  True,
            'edges':     edges_info,
            'case1':     case1,
            'case2':     case2,
            'case3':     case3,
            'case4':     case4,
        })

        case1 = np.array(case1)
        case2 = np.array(case2)
        case3 = np.array(case3)
        case4 = np.array(case4)

    # ── correlations ──────────────────────
    sp12, _ = spearmanr(case1, case2)
    sp13, _ = spearmanr(case1, case3)
    sp14, _ = spearmanr(case1, case4)
    sp34, _ = spearmanr(case3, case4)
    pe12, _ = pearsonr(case1, case2)
    pe13, _ = pearsonr(case1, case3)
    pe14, _ = pearsonr(case1, case4)
    pe34, _ = pearsonr(case3, case4)

    is_odom = np.array([e['is_odom'] for e in edges_info])

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Case 1 (remove from full):       mean={case1.mean():.6f}  std={case1.std():.6f}")
    print(f"Case 2 (add back to full-minus):  mean={case2.mean():.6f}  std={case2.std():.6f}")
    print(f"Case 3 (add to odom backbone):   mean={case3.mean():.6f}  std={case3.std():.6f}")
    print(f"Case 4 (add to MST backbone):    mean={case4.mean():.6f}  std={case4.std():.6f}")

    diff12 = np.abs(case1 - case2)
    print(f"\nCase1 vs Case2 max abs diff:  {diff12.max():.2e}  mean: {diff12.mean():.2e}")
    if diff12.max() < 1e-4:
        print("  → removal and addition SYMMETRIC on same graph ✓")
    else:
        print("  → WARNING: removal and addition differ — numerical issue?")

    print("\n── Spearman / Pearson correlations ──")
    print(f"Case 1 vs Case 2 (sanity):  spearman={sp12:.4f}  pearson={pe12:.4f}")
    print(f"Case 1 vs Case 3:           spearman={sp13:.4f}  pearson={pe13:.4f}")
    print(f"Case 1 vs Case 4:           spearman={sp14:.4f}  pearson={pe14:.4f}")
    print(f"Case 3 vs Case 4:           spearman={sp34:.4f}  pearson={pe34:.4f}")

    # split by odom vs LC
    print("\n── Odom edges only ──")
    if is_odom.sum() > 0:
        sp_odom, _ = spearmanr(case1[is_odom], case3[is_odom])
        sp_odom34,_ = spearmanr(case3[is_odom], case4[is_odom])
        print(f"  Case1 vs Case3: {sp_odom:.4f}")
        print(f"  Case3 vs Case4: {sp_odom34:.4f}")

    print("\n── LC edges only ──")
    lc_mask = ~is_odom
    if lc_mask.sum() > 0:
        sp_lc,  _ = spearmanr(case1[lc_mask], case3[lc_mask])
        sp_lc34,_ = spearmanr(case3[lc_mask], case4[lc_mask])
        print(f"  Case1 vs Case3: {sp_lc:.4f}")
        print(f"  Case3 vs Case4: {sp_lc34:.4f}")

    # top-k overlap
    k = max(10, len(case1) // 5)
    top1 = set(np.argsort(case1)[-k:])
    top2 = set(np.argsort(case2)[-k:])
    top3 = set(np.argsort(case3)[-k:])
    top4 = set(np.argsort(case4)[-k:])
    print(f"\n── Top-{k} (20%) overlap ──")
    print(f"Case1 ∩ Case2: {len(top1&top2)}/{k} ({100*len(top1&top2)/k:.1f}%)")
    print(f"Case1 ∩ Case3: {len(top1&top3)}/{k} ({100*len(top1&top3)/k:.1f}%)")
    print(f"Case1 ∩ Case4: {len(top1&top4)}/{k} ({100*len(top1&top4)/k:.1f}%)")
    print(f"Case3 ∩ Case4: {len(top3&top4)}/{k} ({100*len(top3&top4)/k:.1f}%)")

    print("\n── Interpretation ──")
    if diff12.max() < 1e-4:
        print("✓ Case1 == Case2: removal = addition on same graph")
    if sp13 > 0.7:
        print("✓ Case1 vs Case3: full graph removal valid proxy for odom backbone")
    else:
        print("✗ Case1 vs Case3: context dependent — full graph != odom backbone")
    if sp34 < 0.7:
        print("✗ Case3 vs Case4: same edge contributes DIFFERENTLY on odom vs MST")
        print("  → H3 CONFIRMED: backbone choice changes edge contribution")
    else:
        print("✓ Case3 vs Case4: edge contribution similar on both backbones")

    # ── plots ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = np.where(is_odom, 'blue', 'red')

    for ax, (cx, cy, title) in zip(axes.flat, [
        (case1, case2, f"Sanity: C1 vs C2  sp={sp12:.3f}"),
        (case1, case3, f"C1 vs C3  sp={sp13:.3f}"),
        (case1, case4, f"C1 vs C4  sp={sp14:.3f}"),
        (case3, case4, f"C3 vs C4  sp={sp34:.3f}"),
    ]):
        ax.scatter(cx, cy, c=colors, alpha=0.4, s=8)
        ax.set_title(title)

    fig.legend(handles=[
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='blue', label='odom'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='red',  label='LC'),
    ], loc='lower center', ncol=2)

    plt.suptitle(f"Edge importance across contexts — {dataset_name}")
    plt.tight_layout(rect=[0,0.04,1,1])
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved: {plot_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python edge_importance_cases.py <path_to.g2o>")
        sys.exit(1)
    main(sys.argv[1])