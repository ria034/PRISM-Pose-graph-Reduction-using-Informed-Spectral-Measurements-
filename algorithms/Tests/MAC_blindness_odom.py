import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- DYNAMIC DIRECTORY RESOLUTION ---
current_file = Path(__file__).resolve()
current_dir  = current_file.parent
root_dir     = current_dir.parent

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    os.add_dll_directory(r"C:\Users\lenovo\Downloads\SE-Sync\C++\build\lib")
    os.add_dll_directory(r"C:\Users\lenovo\anaconda3\envs\repo_env\Library\bin")
except AttributeError:
    pass

from pose_graph_utils import split_edges, read_g2o_file, rpm_to_mac
from mac.solvers import MAC, NaiveGreedy
from mac.utils.graphs import weight_graph_lap_from_edges
from mac.utils.rounding import round_madow
from mac.utils.eigen_spectrum import find_eigen_spectrum

EDGE_COLORS = {
    'odom':       ('#2196F3', 'Odometry Edges'),
    'mst':        ('#FF9800', 'MST Edges'),
    'lc_sampled': ('#FF00FF', 'MAC Sampled LC'),
}

def make_legend(ax, keys):
    patches = [mpatches.Patch(color=EDGE_COLORS[k][0], label=EDGE_COLORS[k][1]) for k in keys]
    ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.8)

# ─────────────────────────────────────────────────────────────────────
# Pose reading — auto-detects SE2 (2D) vs SE3 (3D)
# ─────────────────────────────────────────────────────────────────────
def get_poses_dict(filepath):
    """
    Returns poses dict and is_3d flag.
    SE2  -> {'x', 'y'}
    SE3  -> {'x', 'y', 'z'}
    """
    poses = {}
    is_3d = False
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'VERTEX_SE2':
                poses[int(parts[1])] = {
                    'x': float(parts[2]),
                    'y': float(parts[3])
                }
            elif parts[0] in ('VERTEX_SE3:QUAT', 'VERTEX_SE3'):
                is_3d = True
                poses[int(parts[1])] = {
                    'x': float(parts[2]),
                    'y': float(parts[3]),
                    'z': float(parts[4])
                }
    return poses, is_3d

def get_coords(poses, is_3d):
    ids = sorted(poses.keys())
    if is_3d:
        return np.array([[poses[i]['x'], poses[i]['y'], poses[i]['z']] for i in ids])
    else:
        return np.array([[poses[i]['x'], poses[i]['y']] for i in ids])

def compute_fiedler(edges, num_poses):
    edges_arr   = np.array([[e.i, e.j] for e in edges])
    weights_arr = np.array([e.weight   for e in edges])
    L = weight_graph_lap_from_edges(edges_arr, weights_arr, num_poses)
    _, vecs = find_eigen_spectrum(L, sigma=1e-5, k=2)
    return vecs[:, 1]

# ─────────────────────────────────────────────────────────────────────
# Unified scatter — works for both 2D and 3D axes
# ─────────────────────────────────────────────────────────────────────
def scatter(ax, coords, is_3d, **kwargs):
    if is_3d:
        return ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **kwargs)
    else:
        return ax.scatter(coords[:, 0], coords[:, 1], **kwargs)

def plot_edge(ax, poses, e, is_3d, **kwargs):
    xi, xj = poses[e.i]['x'], poses[e.j]['x']
    yi, yj = poses[e.i]['y'], poses[e.j]['y']
    if is_3d:
        zi, zj = poses[e.i]['z'], poses[e.j]['z']
        ax.plot([xi, xj], [yi, yj], [zi, zj], **kwargs)
    else:
        ax.plot([xi, xj], [yi, yj], **kwargs)

def style_ax(ax, is_3d, title):
    ax.set_title(title, fontsize=8, pad=5)
    if not is_3d:
        ax.set_aspect('equal')
    ax.axis('off') if not is_3d else ax.set_axis_off()

# ─────────────────────────────────────────────────────────────────────
# FIGURE 1 — Full graph Fiedler heatmap
# ─────────────────────────────────────────────────────────────────────
def plot_full_heatmap(ax, poses, all_meas, num_poses, is_3d):
    fiedler_vec = compute_fiedler(all_meas, num_poses)
    coords      = get_coords(poses, is_3d)
    sc = scatter(ax, coords, is_3d, c=fiedler_vec, cmap='coolwarm', s=20, linewidths=0)
    plt.colorbar(sc, ax=ax, label='Fiedler Value', shrink=0.8)
    style_ax(ax, is_3d, "Full Pose Graph — Fiedler Heatmap\n(All Loop Closures Included)")

# ─────────────────────────────────────────────────────────────────────
# FIGURE 2 — Edges only
# ─────────────────────────────────────────────────────────────────────
def plot_edges_only(ax, poses, backbone_edges, sampled_edges,
                    num_poses, title, backbone_type, is_3d):
    coords   = get_coords(poses, is_3d)
    bb_color = EDGE_COLORS['odom'][0] if backbone_type == 'odom' else EDGE_COLORS['mst'][0]

    for e in backbone_edges:
        plot_edge(ax, poses, e, is_3d, color=bb_color, lw=0.7, alpha=0.4)

    for e in sampled_edges:
        plot_edge(ax, poses, e, is_3d, color=EDGE_COLORS['lc_sampled'][0], lw=1.8, alpha=0.9)

    scatter(ax, coords, is_3d, c='black', s=4, linewidths=0, alpha=0.4)
    style_ax(ax, is_3d, title)
    make_legend(ax, ['odom', 'lc_sampled'] if backbone_type == 'odom' else ['mst', 'lc_sampled'])

# ─────────────────────────────────────────────────────────────────────
# FIGURE 3 — Heatmap only, evolves with budget
# ─────────────────────────────────────────────────────────────────────
def plot_heatmap_only(ax, poses, backbone_edges, sampled_edges,
                      num_poses, title, is_3d):
    combined    = list(backbone_edges) + list(sampled_edges)
    fiedler_vec = compute_fiedler(combined, num_poses)
    coords      = get_coords(poses, is_3d)
    sc = scatter(ax, coords, is_3d, c=fiedler_vec, cmap='coolwarm', s=16, linewidths=0)
    plt.colorbar(sc, ax=ax, label='Fiedler Value', shrink=0.7, pad=0.02)
    style_ax(ax, is_3d, title)

# ─────────────────────────────────────────────────────────────────────
# Helper — create subplot grid with optional 3D projection
# ─────────────────────────────────────────────────────────────────────
def make_subplots(nrows, ncols, is_3d, figsize):
    if is_3d:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 subplot_kw={'projection': '3d'})
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: python script.py [path_to_g2o]")

    g2o_path   = sys.argv[1]
    poses_dict, is_3d = get_poses_dict(g2o_path)
    print(f"Dataset type: {'3D (SE3)' if is_3d else '2D (SE2)'}")

    raw_meas, num_poses = read_g2o_file(g2o_path)
    all_meas   = rpm_to_mac(raw_meas)

    odom_raw, lc_raw = split_edges(raw_meas)
    odom_set      = rpm_to_mac(odom_raw)
    lc_candidates = rpm_to_mac(lc_raw)

    # MST backbone
    L_all  = weight_graph_lap_from_edges(
                 np.array([[e.i, e.j] for e in all_meas]),
                 np.array([e.weight   for e in all_meas]), num_poses)
    L_pinv = np.linalg.pinv(L_all.toarray())
    B = np.zeros((len(all_meas), num_poses))
    for k, e in enumerate(all_meas):
        B[k, e.i], B[k, e.j] = 1, -1
    eff_res = np.diag(B @ L_pinv @ B.T)

    G_mst = nx.Graph()
    for w, e in zip(eff_res * np.array([e.weight for e in all_meas]), all_meas):
        G_mst.add_edge(e.i, e.j, weight=w, obj=e)
    mst_set     = [data['obj'] for _, _, data in nx.maximum_spanning_tree(G_mst).edges(data=True)]
    non_mst_set = [e for e in all_meas if e not in mst_set]

    budgets   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_budgets = len(budgets)

    # MAC objects created once
    mac_o = MAC(fixed_edges=odom_set, candidate_edges=lc_candidates, num_nodes=num_poses)
    mac_m = MAC(fixed_edges=mst_set,  candidate_edges=non_mst_set,   num_nodes=num_poses)

    results = {}
    for b_pct in budgets:
        b_o = max(1, int(b_pct * len(lc_candidates)))
        _, un_o, _ = mac_o.solve(b_o, NaiveGreedy(lc_candidates).subset(b_o))
        sampled_o = [lc_candidates[i] for i in np.where(round_madow(un_o, b_o) > 0.5)[0]]

        b_m = max(1, int(b_pct * len(non_mst_set)))
        _, un_m, _ = mac_m.solve(b_m, NaiveGreedy(non_mst_set).subset(b_m))
        sampled_m = [non_mst_set[i] for i in np.where(round_madow(un_m, b_m) > 0.5)[0]]

        results[b_pct] = {'sampled_o': sampled_o, 'sampled_m': sampled_m}
        print(f"Budget {int(b_pct*100)}%: odom={len(sampled_o)} LC, mst={len(sampled_m)} LC")

    # ── Figure 1: Full graph Fiedler heatmap ────────────────────────
    fig1 = plt.figure(figsize=(10, 9))
    ax1  = fig1.add_subplot(111, projection='3d') if is_3d else fig1.add_subplot(111)
    plot_full_heatmap(ax1, poses_dict, all_meas, num_poses, is_3d)
    fig1.tight_layout()
    fig1.savefig("fig1_full_heatmap.png", dpi=150, bbox_inches='tight')
    print("Saved fig1_full_heatmap.png")

    # ── Figure 2: Edges only ─────────────────────────────────────────
    fig2, axes2 = make_subplots(2, n_budgets, is_3d, figsize=(4 * n_budgets, 10))
    fig2.suptitle(
        "MAC Loop Closure Selection — Edges Only\n"
        "Blue = Odom backbone  |  Orange = MST backbone  |  Magenta = MAC-selected LC",
        fontsize=12, fontweight='bold', y=1.01)

    for col, b_pct in enumerate(budgets):
        r = results[b_pct]
        plot_edges_only(axes2[0, col], poses_dict, odom_set, r['sampled_o'], num_poses,
                        f"Odom {int(b_pct*100)}%\n({len(r['sampled_o'])}/{len(lc_candidates)} LC)",
                        backbone_type='odom', is_3d=is_3d)
        plot_edges_only(axes2[1, col], poses_dict, mst_set, r['sampled_m'], num_poses,
                        f"MST {int(b_pct*100)}%\n({len(r['sampled_m'])}/{len(non_mst_set)} LC)",
                        backbone_type='mst', is_3d=is_3d)

    fig2.tight_layout()
    fig2.savefig("fig2_edges_only.png", dpi=150, bbox_inches='tight')
    print("Saved fig2_edges_only.png")

    # ── Figure 3: Heatmap only ───────────────────────────────────────
    fig3, axes3 = make_subplots(2, n_budgets, is_3d, figsize=(4 * n_budgets, 10))
    fig3.suptitle(
        "MAC Loop Closure Selection — Fiedler Heatmap Evolution\n"
        "Fiedler computed from backbone + sampled LC at each budget",
        fontsize=12, fontweight='bold', y=1.01)

    for col, b_pct in enumerate(budgets):
        r = results[b_pct]
        plot_heatmap_only(axes3[0, col], poses_dict, odom_set, r['sampled_o'], num_poses,
                          f"Odom {int(b_pct*100)}%", is_3d=is_3d)
        plot_heatmap_only(axes3[1, col], poses_dict, mst_set, r['sampled_m'], num_poses,
                          f"MST {int(b_pct*100)}%", is_3d=is_3d)

    fig3.tight_layout()
    fig3.savefig("fig3_heatmap_only.png", dpi=150, bbox_inches='tight')
    print("Saved fig3_heatmap_only.png")

    plt.show()