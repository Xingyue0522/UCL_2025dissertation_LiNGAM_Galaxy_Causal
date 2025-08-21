# plot_ancestors_by_scenario.py
"""
Plot ancestor subgraphs (upstream variables) for selected targets in each scenario.

Workflow
--------
- Read per-scenario edge lists from results/<category>/{edges_*, edges_all_*}.csv
- Build a DiGraph and extract the ancestor subgraph for each plot target
- Render PNGs with edge widths proportional to |weight| and color by sign

Notes
-----
- Targets default to 'plot_targets' in a scenario; if absent, fall back to
  'forbid_outgoing'; if still empty, try a small heuristic for arms_* variables.
- This script does not change node names in the graph; only display labels
  are prettified.
"""

from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

try:
    from networkx.drawing.nx_agraph import graphviz_layout
    _HAS_GRAPHVIZ = True
except Exception:
    _HAS_GRAPHVIZ = False

from scenarios_with_control_var import SCENARIOS  # use 'forbid_outgoing' as plot targets

RESULTS_DIR = Path("results")

# ---------------------------------------------------------------------
# Pretty labels for nodes (add/remove as needed)
# ---------------------------------------------------------------------
NODE_LABEL_MAP = {
    "logMass_median": "Mass",
    "color_pc1": "color",
    # Common fields (optional examples; enable/modify as needed):
    "v_disp": "Vdisp",
    "oh_p50": "O/H",
    "ssfr_mean": "sSFR",
    "age_mean": "Age",
    "metallicity_mean": "Metallicity",
    "elliptical_prob": "Elliptical",
    "spiral_prob_eff": "Spiral",
    "bar_prob": "Bar",
    "bulge_ev": "Bulge",
    "odd_prob": "Odd",
    "arms_num_ev": "Arms#",
    "arms_wind_ev": "Winding",
    "z": "z",
}

def pretty(name: str) -> str:
    """Map raw node name to a prettier display label; graph node IDs remain unchanged."""
    return NODE_LABEL_MAP.get(name, name)

def _layout(G: nx.DiGraph):
    """Choose a layout: Graphviz 'dot' if available, else spring layout."""
    if _HAS_GRAPHVIZ and G.number_of_nodes() > 0:
        try:
            return graphviz_layout(G, prog="dot")
        except Exception:
            pass
    return nx.spring_layout(G, seed=42)

def load_filtered_edges_graph(path_csv: Path, weight_col: str = "mean_weight") -> nx.DiGraph:
    """
    Load the filtered edge set (edges_<SCENARIO>.csv). These edges are expected to satisfy:
      - frequency >= MIN_FREQ
      - significant (bootstrap CI does not cross 0)
    """
    df = pd.read_csv(path_csv)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        w = float(r.get(weight_col, 0.0))
        G.add_edge(str(r["source"]), str(r["target"]),
                   weight=w,
                   freq=float(r.get("frequency", 0.0)))
    return G

def ancestors_subgraph(G: nx.DiGraph, center: str) -> nx.DiGraph:
    """Return the induced subgraph consisting of all ancestors of 'center' plus 'center' itself."""
    if center not in G:
        return nx.DiGraph()
    nodes = nx.ancestors(G, center) | {center}
    return G.subgraph(nodes).copy()

def draw(G: nx.DiGraph, title_target: str, out_png: Path):
    """Render and save the ancestor subgraph to a PNG file."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if G.number_of_edges() == 0:
        print(f"[WARN] empty subgraph -> {out_png} (skip drawing).")
        return

    pos = _layout(G)
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)

    edges = list(G.edges(data=True))
    widths = [max(1.0, 6.0 * abs(d.get("weight", 0.0))) for _, _, d in edges]
    colors = ["tab:blue" if d.get("weight", 0.0) > 0 else "tab:orange" for _, _, d in edges]

    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="#E6F2FF", edgecolors="#333")

    # Use prettified node labels for display only
    nice_labels = {n: pretty(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=nice_labels, font_size=12)

    nx.draw_networkx_edges(G, pos, width=widths, edge_color=colors, arrowsize=20, alpha=0.9)

    edge_labels = {(u, v): f"{d.get('weight', 0.0):.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title(f"Ancestors → {pretty(title_target)}")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {out_png}")

# ---------------------------------------------------------------------
# Target selection helper
# ---------------------------------------------------------------------
def _get_plot_targets(scen: dict) -> list[str]:
    """
    Determine targets to plot for a scenario:
      1) Prefer 'plot_targets' if present
      2) Else fall back to 'forbid_outgoing'
      3) If still empty, use a heuristic: select arms_* if present
    """
    # 1) New key first
    targets = scen.get("plot_targets", None)
    if targets:
        return list(targets)

    # 2) Backward-compatible behavior
    targets = scen.get("forbid_outgoing", []) or []
    if targets:
        return list(targets)

    # 3) Heuristic (optional): auto-detect arm metrics
    vars_in = set(scen.get("vars", []))
    candidates = ["arms_num_ev", "arms_wind_ev"]
    auto = [c for c in candidates if c in vars_in]
    return auto

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    """Iterate over scenarios, build ancestor subgraphs for selected targets, and save PNGs."""
    for scen in SCENARIOS:
        name = scen["name"]
        cat  = scen["category"]

        targets = _get_plot_targets(scen)
        if not targets:
            print(f"[INFO] Scenario {name}: no plot targets; skipped.")
            continue

        # Prefer filtered edges; if missing, fall back to all-edges
        edges_csv = RESULTS_DIR / cat / f"edges_{name}.csv"
        if not edges_csv.exists():
            print(f"[WARN] {edges_csv} not found; fallback to edges_all_{name}.csv")
            edges_csv = RESULTS_DIR / cat / f"edges_all_{name}.csv"
            if not edges_csv.exists():
                print(f"[WARN] {edges_csv} not found; run lingam_batch1.py first.")
                continue

        G = load_filtered_edges_graph(edges_csv, weight_col="mean_weight")

        for tgt in targets:
            sub = ancestors_subgraph(G, center=tgt)
            out_png = RESULTS_DIR / cat / f"ancestors_{name}_{tgt}.png"
            draw(sub, tgt, out_png)


if __name__ == "__main__":
    main()
