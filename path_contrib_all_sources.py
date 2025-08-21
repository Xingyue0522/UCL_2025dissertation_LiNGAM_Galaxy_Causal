# path_contrib_all_sources.py
"""
Path decomposition across all ancestor sources within a scenario.
Keeps the original per-target outputs and adds a scenario-level
"ALL PATHS" consolidated table.

Overview
--------
- Load stable edges for a scenario and build a DAG.
- For each chosen target, enumerate all simple paths (up to max length) from
  every ancestor source to the target and compute path contributions
  (product of edge weights).
- Produce:
  * Per-source→target detailed path CSVs (Top-K after importance filtering).
  * Per-target source summary CSVs (counts, TE, top +/- examples, TE_matrix check).
  * One scenario-level "ALL PATHS" CSV combining all targets.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import networkx as nx

RESULTS_DIR = Path("results")

# Friendly display names (for tables/legends). Filenames use safe_slug.
DISPLAY = {
    "z": "z",
    "logMass_median": "Mass",
    "color_pc1": "color",
    "oh_p50": "O/H",
    "v_disp": "Vdisp",
    "elliptical_prob": "Elliptical",
    "spiral_prob_eff": "Spiral",
    "bar_prob": "Bar",
    "bulge_ev": "Bulge",
    "arms_num_ev": "Arms#",
    "arms_wind_ev": "Winding",
}
ARROW = "→"

def pretty(x: str) -> str:
    """Return a human-friendly label for variable x (used only for display)."""
    return DISPLAY.get(x, x)

def safe_slug(x: str) -> str:
    """Return a filename-safe slug derived from the pretty label of x."""
    s = pretty(x)
    s = s.replace("/", "-").replace("#", "num").replace("%", "pct").replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s or "var"

# ---------------------------------------------------------------------
# I/O and graph construction
# ---------------------------------------------------------------------

def load_stable_edges_csv(category: str, scen_name: str, weight_col="mean_weight") -> pd.DataFrame:
    """
    Load the filtered/stable edges CSV for a scenario.

    Parameters
    ----------
    category : str
        Scenario category (subfolder under results/).
    scen_name : str
        Scenario name used in filenames.
    weight_col : str
        Preferred weight column name ('mean_weight' by default).

    Returns
    -------
    pd.DataFrame
        Columns: ['source', 'target', 'weight'].

    Notes
    -----
    - If the requested weight_col is not found, fall back to 'avg_weight' if present.
    - Raises FileNotFoundError if the stable edges file is missing.
    """
    path = RESULTS_DIR / category / f"edges_{scen_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Please run lingam_batch1.py first to generate stable edges.")
    df = pd.read_csv(path)
    if weight_col not in df.columns:
        if "avg_weight" in df.columns:
            weight_col = "avg_weight"
        else:
            raise ValueError(f"{path} is missing the '{weight_col}'/'avg_weight' column.")
    df = df[["source", "target", weight_col]].copy()
    df.rename(columns={weight_col: "weight"}, inplace=True)
    return df

def build_dag(df_edges: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed acyclic graph (DAG) from an edge list with weights.

    Behavior
    --------
    - If multiple edges between the same (u, v) appear, average their weights.
    - Verifies acyclicity and raises RuntimeError if cycles are detected.
    """
    G = nx.DiGraph()
    for _, r in df_edges.iterrows():
        u, v, w = str(r["source"]), str(r["target"]), float(r["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] = 0.5 * (G[u][v]["weight"] + w)
        else:
            G.add_edge(u, v, weight=w)
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("Constructed graph contains cycles (not a DAG). Please check priors/input edges.")
    return G

# ---------------------------------------------------------------------
# Paths and total effects
# ---------------------------------------------------------------------

def all_simple_paths_contrib(G: nx.DiGraph, s: str, t: str, maxlen: int = 4) -> list[tuple[list[str], float]]:
    """
    Enumerate all simple paths from source s to target t with length ≤ maxlen,
    and compute each path's contribution as the product of edge weights.

    Returns
    -------
    list of (path_nodes, contribution)
        path_nodes: list[str], contribution: float
    """
    if s not in G or t not in G:
        return []
    paths = nx.all_simple_paths(G, source=s, target=t, cutoff=maxlen)
    out = []
    for path in paths:
        prod = 1.0
        ok = True
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not G.has_edge(u, v):
                ok = False
                break
            prod *= float(G[u][v]["weight"])
        if ok:
            out.append((path, float(prod)))
    return out

def te_matrix(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute the total-effect matrix:
        T = (I - B)^(-1) - I
    where B_{u,v} = β_{u->v} from the DAG edge weights.
    """
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)
    B = np.zeros((p, p), dtype=float)
    for u, v, d in G.edges(data=True):
        B[idx[u], idx[v]] = float(d.get("weight", 0.0))
    I = np.eye(p)
    T = np.linalg.inv(I - B) - I
    return pd.DataFrame(T, index=nodes, columns=nodes)

def path_str(path: list[str]) -> str:
    """Format a list of node names into a 'A→B→C' string using pretty labels."""
    return ARROW.join(pretty(n) for n in path)

# ---------------------------------------------------------------------
# Per-target summary (kept from original functionality)
# ---------------------------------------------------------------------

def source_summary_for_target(G: nx.DiGraph, target: str, maxlen=4,
                              abs_thr=0.02, rel_thr=0.10, topk=8,
                              save_dir: Path | None = None) -> pd.DataFrame:
    """
    Summarize contributions from all ancestor sources to a given target.

    Steps
    -----
    - For each ancestor source s, enumerate all simple paths (≤ maxlen).
    - Compute TE(s→target) as the sum of path contributions.
    - Keep important paths via absolute/relative thresholds; take Top-K.
    - Optionally save per-source path CSVs in save_dir/<source_slug>.

    Returns
    -------
    pd.DataFrame
        One row per source with TE, positive/negative sums, top +/- examples,
        TE from matrix (for sanity check), and their difference.
    """
    if target not in G:
        return pd.DataFrame()

    ancestors = sorted(nx.ancestors(G, target))
    dfT = te_matrix(G)

    rows = []
    for s in ancestors:
        contribs = all_simple_paths_contrib(G, s, target, maxlen=maxlen)
        if not contribs:
            continue
        te_sum = float(np.sum([c for _, c in contribs]))

        items = []
        for path, c in contribs:
            items.append({
                "path": path_str(path),
                "contribution": c,
                "sign": "+" if c >= 0 else "-",
                "share_of_TE": (c / te_sum) if te_sum != 0 else np.nan,
                "length": len(path) - 1
            })
        df = pd.DataFrame(items)
        df["share_of_TE(%)"] = df["share_of_TE"] * 100.0

        # Importance filtering
        if te_sum != 0:
            df_keep = df[(df["contribution"].abs() >= abs_thr) | (df["contribution"].abs() >= rel_thr * abs(te_sum))].copy()
        else:
            df_keep = df[df["contribution"].abs() >= abs_thr].copy()
        if df_keep.empty:
            df_keep = df.copy()
        df_keep = df_keep.sort_values(by="contribution", key=lambda s: s.abs(), ascending=False)
        if topk and len(df_keep) > topk:
            df_keep = df_keep.head(topk).copy()

        S_pos = float((df["contribution"][df["contribution"] > 0]).sum())
        S_neg = float((df["contribution"][df["contribution"] < 0]).sum())

        top_pos = df[df["contribution"] > 0].sort_values("contribution", ascending=False).head(1)
        top_neg = df[df["contribution"] < 0].sort_values("contribution", ascending=True).head(1)
        pos_example = f"{top_pos.iloc[0]['path']} ({top_pos.iloc[0]['contribution']:+.3f})" if len(top_pos) else ""
        neg_example = f"{top_neg.iloc[0]['path']} ({top_neg.iloc[0]['contribution']:+.3f})" if len(top_neg) else ""

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            src_slug = safe_slug(s)
            tgt_slug = safe_slug(target)
            out_csv = save_dir / f"paths_{src_slug}_to_{tgt_slug}.csv"
            df_keep.to_csv(out_csv, index=False)

        te_mat = float(dfT.loc[s, target]) if (s in dfT.index and target in dfT.columns) else np.nan
        rows.append({
            "source": s, "source_disp": pretty(s),
            "target": target, "target_disp": pretty(target),
            "n_paths": len(df),
            "TE": te_sum,
            "S_plus": S_pos,
            "S_minus": S_neg,
            "pos_share_of_TE": (S_pos / te_sum) if te_sum != 0 else np.nan,
            "neg_share_of_TE": (S_neg / te_sum) if te_sum != 0 else np.nan,
            "top_pos_path": pos_example,
            "top_neg_path": neg_example,
            "TE_matrix": te_mat,
            "TE_diff": te_sum - te_mat if not np.isnan(te_mat) else np.nan
        })

    df_sum = pd.DataFrame(rows)
    if not df_sum.empty:
        df_sum["abs_TE"] = df_sum["TE"].abs()
        df_sum = df_sum.sort_values(by="abs_TE", ascending=False)
    return df_sum

# ---------------------------------------------------------------------
# NEW: scenario-level "ALL PATHS" consolidated table
# ---------------------------------------------------------------------

def collect_all_paths_for_scenario(G: nx.DiGraph, scen_name: str, category: str,
                                   targets: list[str], maxlen=4) -> pd.DataFrame:
    """
    Collect all simple paths (≤ maxlen) from every ancestor source to each target
    in this scenario and return a single DataFrame (one row per path).

    Returns
    -------
    pd.DataFrame
        Columns include scenario, category, target, source, path, length,
        contribution, sign, TE_source_to_target, share_of_TE, TE_matrix.
    """
    dfT = te_matrix(G)  # to provide TE(source->target) and share_of_TE

    rows = []
    for tgt in targets:
        if tgt not in G: 
            continue
        ancestors = sorted(nx.ancestors(G, tgt))
        for s in ancestors:
            contribs = all_simple_paths_contrib(G, s, tgt, maxlen=maxlen)
            if not contribs:
                continue
            te_sum = float(np.sum([c for _, c in contribs]))
            for path, c in contribs:
                rows.append({
                    "scenario": scen_name,
                    "category": category,
                    "target": pretty(tgt),
                    "source": pretty(s),
                    "path": path_str(path),
                    "length": len(path) - 1,
                    "contribution": c,
                    "sign": "+" if c >= 0 else "−",
                    "TE_source_to_target": te_sum,
                    "share_of_TE": (c / te_sum) if te_sum != 0 else np.nan,
                    "TE_matrix": float(dfT.loc[s, tgt]) if (s in dfT.index and tgt in dfT.columns) else np.nan
                })
    df_all = pd.DataFrame(rows)
    if not df_all.empty:
        # Sort primarily by target, then by |contribution| descending
        df_all["abs_contribution"] = df_all["contribution"].abs()
        df_all = df_all.sort_values(by=["target", "abs_contribution"], ascending=[True, False]).drop(columns=["abs_contribution"])
    return df_all

# ---------------------------------------------------------------------
# Node-level "net channel flow" (optional)
# ---------------------------------------------------------------------

def node_flow_for_target(G: nx.DiGraph, target: str, maxlen=4) -> pd.DataFrame:
    """
    Aggregate path contributions passing through each mediator node on routes
    from ancestor sources to a given target (excluding endpoints).

    Returns
    -------
    pd.DataFrame
        Columns: node, node_disp, throughflow, abs_throughflow (sorted by |throughflow| desc).
    """
    if target not in G:
        return pd.DataFrame()
    ancestors = sorted(nx.ancestors(G, target))
    contrib_by_node = {}
    for s in ancestors:
        for path, c in all_simple_paths_contrib(G, s, target, maxlen=maxlen):
            for m in path[1:-1]:  # accumulate only mediator nodes
                contrib_by_node[m] = contrib_by_node.get(m, 0.0) + c
    rows = [{"node": n, "node_disp": pretty(n), "throughflow": v, "abs_throughflow": abs(v)}
            for n, v in contrib_by_node.items()]
    df = pd.DataFrame(rows).sort_values("abs_throughflow", ascending=False)
    return df

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    """Command-line interface: per-target outputs plus a scenario-level ALL PATHS table."""
    parser = argparse.ArgumentParser(description="Path decomposition over all ancestor sources; also write a per-scenario ALL PATHS summary CSV.")
    parser.add_argument("--scenarios", type=str, default="ALL",
                        help="Comma-separated list or ALL. e.g., SPIRAL_PRESENCE,BULGE_PROMINENCE_DISC")
    parser.add_argument("--targets", type=str, default="AUTO",
                        help="AUTO = use forbid_outgoing; ALL = nodes with out-degree 0; or provide a comma-separated list.")
    parser.add_argument("--maxlen", type=int, default=4, help="Max path length (edge count) to enumerate; default 4.")
    parser.add_argument("--abs-thr", type=float, default=0.02, help="Used only for per-source detail filtering (absolute contribution).")
    parser.add_argument("--rel-thr", type=float, default=0.10, help="Used only for per-source detail filtering (relative to |TE|).")
    parser.add_argument("--topk", type=int, default=8, help="Top-K rows per source→target detail output.")
    parser.add_argument("--node-flow", action="store_true", help="Also write node-level net channel flow ranking per target.")
    args = parser.parse_args()

    from scenarios_with_control_var import SCENARIOS

    # Scenario selection
    if args.scenarios == "ALL":
        scen_list = SCENARIOS
    else:
        want = set(s.strip() for s in args.scenarios.split(",") if s.strip())
        scen_list = [s for s in SCENARIOS if s["name"] in want]
        missing = want - set(s["name"] for s in scen_list)
        if missing:
            raise ValueError(f"Missing scenarios: {missing}")

    for scen in scen_list:
        name, cat = scen["name"], scen["category"]
        try:
            df_edges = load_stable_edges_csv(cat, name)
        except Exception as e:
            print(f"[WARN] {e}")
            continue
        G = build_dag(df_edges)

        # Target selection
        if args.targets == "AUTO":
            targets = (scen.get("forbid_outgoing", []) or [])
            if not targets:
                targets = [n for n in G.nodes() if G.out_degree(n) == 0]
        elif args.targets == "ALL":
            targets = [n for n in G.nodes() if G.out_degree(n) == 0]
        else:
            targets = [t.strip() for t in args.targets.split(",") if t.strip()]

        if not targets:
            print(f"[INFO] Scenario {name}: no targets (AUTO/ALL); skipped.")
            continue

        out_dir = RESULTS_DIR / cat / f"path_tables_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ===== Original per-target outputs (kept) =====
        for tgt in targets:
            tgt_dir = out_dir / safe_slug(tgt)
            df_sum = source_summary_for_target(
                G, target=tgt, maxlen=args.maxlen,
                abs_thr=args.abs_thr, rel_thr=args.rel_thr, topk=args.topk,
                save_dir=tgt_dir
            )
            if not df_sum.empty:
                summary_csv = out_dir / f"sources_summary_to_{safe_slug(tgt)}.csv"
                df_sum[["source_disp","target_disp","n_paths","TE","S_plus","S_minus",
                        "pos_share_of_TE","neg_share_of_TE","top_pos_path","top_neg_path",
                        "TE_matrix","TE_diff"]].to_csv(summary_csv, index=False)
                print(f"[OK] {name}: saved source summary -> {summary_csv}")
            else:
                print(f"[INFO] {name}: {pretty(tgt)} has no ancestors or paths.")

            if args.node_flow:
                df_nf = node_flow_for_target(G, target=tgt, maxlen=args.maxlen)
                if not df_nf.empty:
                    nf_csv = out_dir / f"node_flow_to_{safe_slug(tgt)}.csv"
                    df_nf.to_csv(nf_csv, index=False)
                    print(f"[OK] {name}: saved node net throughflow -> {nf_csv}")

        # ===== NEW: scenario-level ALL PATHS table =====
        df_all = collect_all_paths_for_scenario(G, scen_name=name, category=cat,
                                               targets=targets, maxlen=args.maxlen)
        if not df_all.empty:
            all_csv = out_dir / f"all_paths_{name}.csv"
            df_all.to_csv(all_csv, index=False)
            print(f"[OK] {name}: saved scenario ALL PATHS -> {all_csv}")
        else:
            print(f"[INFO] {name}: no paths to export.")

if __name__ == "__main__":
    main()
