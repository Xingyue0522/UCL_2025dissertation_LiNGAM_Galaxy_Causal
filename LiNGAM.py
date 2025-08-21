# -*- coding: utf-8 -*-
"""
lingam_batch1.py
Batch runner for DirectLiNGAM + bootstrap stability. It produces:
  - edges_all_<SCENARIO>.csv : ALL edges (no filtering; used for ancestor subgraphs)
  - edges_<SCENARIO>.csv     : stable edges (filtered by frequency and significance)
  - effects_report_<SCENARIO>.csv : frequency, mean coefficients, CI, p-values, and partial R²
                                   (no additional thresholding on effect size)

Dependencies:
  pip install causallearn scikit-learn statsmodels networkx matplotlib pandas numpy

Works together with scenarios_with_control_var.py in the same directory
(which contains scenario configurations and gates).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from causallearn.search.FCMBased.lingam import DirectLiNGAM
import statsmodels.api as sm

# ==== Import scenarios ====
from scenarios import SCENARIOS  # Import scenarios without control variables
# If you want to use scenarios with control variables, uncomment the next line:
# from scenarios_with_control_var import SCENARIOS


# ---------------------------------------------------------------------
# ====== Global parameters (tune as needed) ===========================
# ---------------------------------------------------------------------
INPUT_FILE  = "data/main.csv"
RESULTS_DIR = Path("results")

N_RUNS      = 200
SAMPLE_FRAC = 0.9

# -- Stability criteria --
MIN_FREQ          = 0.90     # frequency threshold (proportion of bootstrap runs with |w| > threshold)
WEIGHT_THRESHOLD  = 0.05     # |w| threshold used when counting "frequency"
RANK_BY           = "avg_weight"  # sorting key for edges_* files ("avg_weight" / "mean_weight" / "frequency")

# Significance / bootstrap CI
ALPHA     = 0.05
N_BOOT_CI = 2000
RAND_BASE = 0

GATE_STRICT = True  # when gate column is missing: True=raise error; False=warn and skip

# ---------------------------------------------------------------------
# ====== Utilities =====================================================
# ---------------------------------------------------------------------
def bootstrap_mean_ci(x, n_boot=2000, alpha=0.05, random_state=0):
    """
    Compute a percentile bootstrap confidence interval for the mean of array x,
    and return (ci_low, ci_high, two_sided_p_value).

    Parameters
    ----------
    x : array-like
        Sample of values whose mean CI is desired.
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level for the CI (e.g., 0.05 for a 95% CI).
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    (float, float, float)
        (lower bound, upper bound, two-sided p-value for mean ≠ 0 based on bootstrap distribution).
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    boot_means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = np.percentile(boot_means, 100 * (alpha / 2))
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    p_two = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())
    return lo, hi, float(np.clip(p_two, 0.0, 1.0))


def save_edges_to_csv(edge_stats, n_runs, filename, min_freq, rank_by,
                      save_all=False, require_significant=True):
    """
    Export edges to CSV.

    Modes
    -----
    - save_all=True : export everything (no filtering)
    - save_all=False: keep edges with frequency >= min_freq AND (optionally) significant (CI not crossing 0)

    Parameters
    ----------
    edge_stats : dict
        Aggregated edge statistics from bootstrap_lingam_stability().
    n_runs : int
        Number of bootstrap runs.
    filename : Path
        Output CSV path.
    min_freq : float
        Frequency threshold when filtering (ignored if save_all=True).
    rank_by : str
        Sorting key: "avg_weight", "mean_weight", or "frequency".
    save_all : bool
        Whether to export all edges regardless of filtering.
    require_significant : bool
        If True, require CI not crossing zero to keep an edge (when save_all=False).
    """
    rows = []
    for (src, tgt), s in edge_stats.items():
        cond = True if save_all else (s["frequency"] >= min_freq and (not require_significant or s["significant"]))
        if cond:
            rows.append({
                "source": src, "target": tgt,
                "count": s["count"], "frequency": s["frequency"],
                "avg_weight": s["avg_weight"],
                "mean_weight": s["mean_weight"],
                "mean_abs_weight": s["mean_abs_weight"],
                "ci_low": s["ci_low"], "ci_high": s["ci_high"],
                "p_value": s["p_value"], "significant": s["significant"],
                "n_runs": n_runs
            })
    df_out = pd.DataFrame(rows)

    if not df_out.empty:
        if rank_by == "mean_weight":
            df_out = df_out.reindex(df_out["mean_weight"].abs().sort_values(ascending=False).index)
        else:
            df_out = df_out.sort_values(by=rank_by, ascending=False)

    filename.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(filename, index=False)
    print(f"[OK] Saved edges -> {filename} (save_all={save_all}, min_freq={min_freq}, rank_by={rank_by})")


def build_prior_knowledge(var_names, layer=None, mutually_exclusive=None,
                          extra_forbid=None, extra_require=None):
    """
    Build a prior-knowledge matrix for DirectLiNGAM.

    Encoding
    --------
    -1 = unknown, 0 = forbid, 1 = require, diagonal = 0.

    Rules
    -----
    - layer: smaller layer index is more upstream (forbid edges from lower to higher layers).
    - mutually_exclusive: list of (a, b), forbids both a->b and b->a.
    - extra_forbid / extra_require: explicit lists of (src, tgt).
    - Special handling: if 'z' exists, force it to be a root (only outgoing, no incoming).

    Parameters
    ----------
    var_names : list[str]
        Variable names in the model.
    layer : dict[str, int] or None
        Layer index per variable.
    mutually_exclusive : list[tuple[str, str]] or None
        Pairs of variables with mutual exclusion (no edges both ways).
    extra_forbid : list[tuple[str, str]] or None
        Additional edges to forbid.
    extra_require : list[tuple[str, str]] or None
        Additional edges to require.

    Returns
    -------
    np.ndarray (p x p, int)
        Prior-knowledge matrix.
    """
    p = len(var_names)
    P = np.full((p, p), -1, dtype=int)
    np.fill_diagonal(P, 0)

    if layer:
        for i, src in enumerate(var_names):
            for j, tgt in enumerate(var_names):
                li, lj = layer.get(src, None), layer.get(tgt, None)
                if li is not None and lj is not None and li > lj:
                    P[i, j] = 0

    if mutually_exclusive:
        for a, b in mutually_exclusive:
            if a in var_names and b in var_names:
                ia, ib = var_names.index(a), var_names.index(b)
                P[ia, ib] = 0
                P[ib, ia] = 0

    if extra_forbid:
        for a, b in extra_forbid:
            if a in var_names and b in var_names:
                P[var_names.index(a), var_names.index(b)] = 0

    if extra_require:
        for a, b in extra_require:
            if a in var_names and b in var_names:
                P[var_names.index(a), var_names.index(b)] = 1

    if "z" in var_names:
        zi = var_names.index("z")
        for j in range(p):
            if j != zi:
                P[zi, j] = 1
                P[j, zi] = 0
    return P


def bootstrap_lingam_stability(df: pd.DataFrame, var_names: list, n_runs: int, sample_frac: float,
                               threshold: float, prior_knowledge, alpha=0.05, n_boot_ci=2000,
                               random_state_base=0):
    """
    Fit DirectLiNGAM with bootstrap resampling to obtain edge stability statistics.

    For each bootstrap run:
      1) Sample rows with replacement (fraction = sample_frac).
      2) Standardize variables.
      3) Fit DirectLiNGAM with the given prior_knowledge.
      4) Record adjacency weights; count edges with |w| > threshold.

    Aggregates
    ----------
    For each directed pair (src, tgt), compute:
      - count, frequency (= count / n_runs)
      - avg_weight: average absolute weight contribution per run used for counting (sum|w| over runs / n_runs)
      - mean_weight, mean_abs_weight over all runs (including near-zero)
      - bootstrap CI and two-sided p-value for mean_weight
      - significance flag: CI not crossing 0

    Returns
    -------
    dict[(str, str) -> dict]
        Edge statistics keyed by (source, target).
    """
    edge_counts = defaultdict(int)
    edge_sum_abs_w = defaultdict(float)
    edge_weights_allruns = defaultdict(list)

    for run in range(n_runs):
        df_sample = df.sample(frac=sample_frac, replace=True, random_state=run)
        X_sample = StandardScaler().fit_transform(df_sample[var_names].values)

        model = DirectLiNGAM(random_state=run + random_state_base, prior_knowledge=prior_knowledge)
        model.fit(X_sample)
        B = model.adjacency_matrix_

        for i, src in enumerate(var_names):
            for j, tgt in enumerate(var_names):
                if i == j:
                    continue
                w = float(B[i, j])
                edge = (src, tgt)
                edge_weights_allruns[edge].append(w)
                if abs(w) > threshold:
                    edge_counts[edge] += 1
                    edge_sum_abs_w[edge] += abs(w)

    edge_stats = {}
    for edge, w_list in edge_weights_allruns.items():
        count = edge_counts[edge]
        freq  = count / n_runs
        avg_abs_w = edge_sum_abs_w[edge] / n_runs
        mean_w    = float(np.mean(w_list))
        mean_abs  = float(np.mean(np.abs(w_list)))
        ci_lo, ci_hi, pval = bootstrap_mean_ci(w_list, n_boot=n_boot_ci, alpha=alpha, random_state=random_state_base)
        significant = (ci_lo > 0) or (ci_hi < 0)
        edge_stats[edge] = {
            "count": count, "frequency": freq,
            "avg_weight": avg_abs_w,
            "mean_weight": mean_w,
            "mean_abs_weight": mean_abs,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "p_value": pval, "significant": significant,
            "n_runs": n_runs
        }
    return edge_stats

# ---------------------------------------------------------------------
# ====== Data cleaning ================================================
# ---------------------------------------------------------------------
def clip_by_quantile(s: pd.Series, low=0.005, high=0.995):
    """
    Clip a Series by lower/upper quantiles to reduce the influence of extreme tails.

    Parameters
    ----------
    s : pd.Series
        Input data.
    low, high : float
        Quantile bounds (0..1).

    Returns
    -------
    pd.Series
        Clipped series.
    """
    lo, hi = s.quantile([low, high])
    return s.clip(lo, hi)


def load_and_clean(input_file: str) -> pd.DataFrame:
    """
    Load the CSV dataset and perform light cleaning:
      - Convert sentinel -9999.0 to NaN for selected columns.
      - Mild tail clipping for a few variables to limit extreme leverage.

    Parameters
    ----------
    input_file : str
        Path to CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = pd.read_csv(input_file).copy()
    # Convert sentinel -9999.0 to NaN (these columns contain sentinel values in your data)
    for c in ["oh_p50", "logMass_median", "ssfr_mean", "age_mean", "metallicity_mean"]:
        if c in df.columns:
            df.loc[df[c] == -9999.0, c] = np.nan
    # Light tail clipping to avoid domination by outliers
    for c in ["color_pc1", "v_disp"]:
        if c in df.columns:
            df[c] = clip_by_quantile(df[c], 0.005, 0.995)
    return df

# ---------------------------------------------------------------------
# ====== Gate implementation ==========================================
# ---------------------------------------------------------------------
def apply_gate(df: pd.DataFrame, gate: dict):
    """
    Apply a row filter ("gate") to the DataFrame.

    Supported operators
    -------------------
    '>=', '>', '<=', '<', 'between' (requires 'thr2').

    Behavior
    --------
    - If the gate column is missing and GATE_STRICT is True, raise KeyError.
      Otherwise, warn and skip.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with a progress print (kept rows, in-gate min/median/max).
    """
    col = gate.get("col")
    op  = gate.get("op", ">=")
    thr = gate.get("thr")
    if col not in df.columns:
        msg = f"[GATE] Column '{col}' not in DataFrame."
        if GATE_STRICT:
            raise KeyError(msg)
        print("[WARN]", msg, "Gate skipped.")
        return df
    before = len(df)
    if op == ">=":
        df2 = df[df[col] >= thr]
    elif op == ">":
        df2 = df[df[col] > thr]
    elif op == "<=":
        df2 = df[df[col] <= thr]
    elif op == "<":
        df2 = df[df[col] < thr]
    elif op == "between":
        thr2 = gate.get("thr2")
        if thr2 is None:
            raise ValueError("gate op='between' requires 'thr2'")
        lo, hi = min(thr, thr2), max(thr, thr2)
        df2 = df[(df[col] >= lo) & (df[col] <= hi)]
    else:
        raise ValueError(f"Unsupported gate op: {op}")

    after = len(df2)
    q = df2[col].quantile([0.0, 0.5, 1.0]) if after > 0 else pd.Series([np.nan, np.nan, np.nan], index=[0,0.5,1])
    print(f"[GATE] {col} {op} {thr} -> {after}/{before} rows kept; "
          f"min/med/max in-gate = {q.iloc[0]:.3f}/{q.iloc[1]:.3f}/{q.iloc[2]:.3f}")
    return df2

# ---------------------------------------------------------------------
# ====== partial R² (report only; not used for filtering) =============
# ---------------------------------------------------------------------
def _stable_parent_sets(edge_stats, min_freq=0.8, require_significant=True):
    """
    Extract, for each target, the "stable parent set" from edge_stats
    using only frequency and significance criteria (no effect-size thresholds).

    Returns
    -------
    dict[str, list[str]]
        Mapping target -> list of parents.
    """
    parents = {}
    for (src, tgt), s in edge_stats.items():
        if s["frequency"] >= min_freq and (not require_significant or s["significant"]):
            parents.setdefault(tgt, []).append(src)
    return parents


def _partial_r2_table(dfx: pd.DataFrame, parents_map: dict):
    """
    For each target, run an OLS with its stable parent set Pa(target) on standardized data,
    and compute per-parent partial R² via leave-one-parent-out reduction.

    Notes
    -----
    - Drops rows with any NaNs across {target} ∪ Pa.
    - Skips if sample size is too small for a reasonable regression.

    Returns
    -------
    pd.DataFrame
        Columns: source, target, partial_R2, n_used
    """
    rows = []
    for tgt, pa in parents_map.items():
        if len(pa) == 0:
            continue
        cols = [tgt] + pa
        sub = dfx[cols].dropna(how="any")
        if len(sub) < max(50, 2*len(pa)+5):
            continue  # skip very small samples to avoid overfitting
        # Standardize
        X = sub[pa].copy()
        y = sub[tgt].copy()
        X = (X - X.mean())/X.std(ddof=0)
        y = (y - y.mean())/y.std(ddof=0)
        Xc = sm.add_constant(X)
        full = sm.OLS(y, Xc).fit()
        rss_full = float(((y - full.predict(Xc))**2).sum())

        for src in pa:
            X_red = X.drop(columns=[src])
            Xrc = sm.add_constant(X_red)
            red = sm.OLS(y, Xrc).fit()
            rss_red = float(((y - red.predict(Xrc))**2).sum())
            pr2 = max(0.0, 1.0 - rss_full/rss_red)
            rows.append({
                "source": src, "target": tgt,
                "partial_R2": pr2,
                "n_used": len(sub)
            })
    return pd.DataFrame(rows)


def make_effects_report(dfx: pd.DataFrame, edge_stats: dict,
                        min_freq=0.8, require_significant=True):
    """
    Build the effects report by combining edge stability stats with partial R².
    No thresholds are applied to partial R² or effect sizes here.

    Steps
    -----
    1) Select edges that pass frequency (>= min_freq) and significance (if required).
    2) Compute partial R² based on the "stable parent sets".
    3) Merge and sort by (frequency, |mean_weight|).

    Returns
    -------
    pd.DataFrame
        Effects report ready for CSV export.
    """
    # 1) Use only frequency/significance-passing edges (to merge with partial R²)
    rows = []
    for (src, tgt), s in edge_stats.items():
        if s["frequency"] >= min_freq and (not require_significant or s["significant"]):
            rows.append({
                "source": src, "target": tgt,
                "frequency": s["frequency"],
                "mean_weight": s["mean_weight"],
                "abs_mean_weight": abs(s["mean_weight"]),
                "avg_weight": s["avg_weight"],
                "ci_low": s["ci_low"], "ci_high": s["ci_high"],
                "p_value": s["p_value"]
            })
    df_edges = pd.DataFrame(rows)
    if df_edges.empty:
        return df_edges

    # 2) partial R² via stable parents
    parents_map = _stable_parent_sets(edge_stats, min_freq=min_freq, require_significant=require_significant)
    df_pr2 = _partial_r2_table(dfx, parents_map)

    # 3) Merge (no "important" tag; no extra thresholding)
    out = df_edges.merge(df_pr2, on=["source","target"], how="left")
    out = out.sort_values(
        by=["frequency", "abs_mean_weight"],
        ascending=[False, False]
    )
    return out

# ---------------------------------------------------------------------
# ====== Scenario runner ==============================================
# ---------------------------------------------------------------------
def run_scenario(df: pd.DataFrame, scen: dict, base_outdir: Path):
    """
    Execute a single scenario:
      - Apply scenario filter and gate.
      - Build prior knowledge (layering, mutual exclusions, optional forbid/require).
      - Run DirectLiNGAM + bootstrap stability.
      - Export: ALL edges, stable edges, effects report.

    Returns
    -------
    dict
        Summary info for this scenario (paths to outputs, sample/var counts, etc.).
    """
    name     = scen["name"]
    category = scen["category"]
    vars_    = scen["vars"]
    filt_fn  = scen.get("filter", None)
    gate     = scen.get("gate", None)
    forbid_out = scen.get("forbid_outgoing", []) or []
    extra_forbid = scen.get("extra_forbid", []) or []
    extra_require = scen.get("extra_require", []) or []

    dfin = df.copy()
    if filt_fn is not None:
        dfin = filt_fn(dfin)
        print(f"[FILTER] {name}: rows after filter = {len(dfin)}")
    if gate is not None:
        dfin = apply_gate(dfin, gate)

    dfx = dfin[vars_].dropna(how="any")
    var_names = dfx.columns.tolist()

    outdir = (RESULTS_DIR / category)
    outdir.mkdir(parents=True, exist_ok=True)

    # === Output paths ===
    edges_stable_path = outdir / f"edges_{name}.csv"      # stable edges (filtered by frequency + significance)
    edges_all_path    = outdir / f"edges_all_{name}.csv"  # ALL edges (for ancestor graphs)
    effects_path      = outdir / f"effects_report_{name}.csv"

    print(f"\n=== Scenario [{name}] ({category}) ===")
    print(f"Samples: {len(dfx)} | Vars: {len(var_names)} -> {var_names}")

    # Hierarchy & mutual exclusivity (smaller layer index = more upstream)
    layer = {
        # Physical variables
        "z": 0,
        "logMass_median": 1, "v_disp": 1,
        "ssfr_mean": 2, "age_mean": 2, "metallicity_mean": 2, "oh_p50": 2,
        "color_pc1": 3,
        # Morphology
        "elliptical_prob": 4, "spiral_prob_eff": 4,
        "bar_prob": 5, "bulge_ev": 5, "odd_prob": 5,
        "arms_num_ev": 6, "arms_wind_ev": 6,
    }
    mutually_exclusive = [
        ("elliptical_prob", "spiral_prob_eff"),
        ("round_ev",        "spiral_prob_eff"),
        ("elliptical_prob", "bar_prob"),
        ("round_ev",        "bar_prob"),
        ("elliptical_prob", "bulge_ev"),
        ("round_ev",        "bulge_ev"),
        ("elliptical_prob", "odd_prob"),
        ("round_ev",        "odd_prob"),
    ]

    # forbid_outgoing: targets receive only incoming edges (affects prior only; not the "ALL edges" statistics)
    extra_forbid_local = list(extra_forbid)
    for node in forbid_out:
        if node in var_names:
            i = var_names.index(node)
            for j, tgt in enumerate(var_names):
                if j != i:
                    extra_forbid_local.append((node, tgt))

    P = build_prior_knowledge(var_names,
                              layer=layer,
                              mutually_exclusive=mutually_exclusive,
                              extra_forbid=extra_forbid_local,
                              extra_require=extra_require)

    # -- DirectLiNGAM + Bootstrap --
    edge_stats = bootstrap_lingam_stability(
        dfx, var_names,
        n_runs=N_RUNS, sample_frac=SAMPLE_FRAC, threshold=WEIGHT_THRESHOLD,
        prior_knowledge=P, alpha=ALPHA, n_boot_ci=N_BOOT_CI, random_state_base=RAND_BASE
    )

    # 0) Save ALL edges (no filtering), for ancestor graphs
    save_edges_to_csv(
        edge_stats, N_RUNS, edges_all_path,
        min_freq=0.0, rank_by="frequency",
        save_all=True, require_significant=False
    )

    # 1) Save "stable edges" (filtered by frequency + significance only)
    save_edges_to_csv(
        edge_stats, N_RUNS, edges_stable_path,
        min_freq=MIN_FREQ, rank_by=RANK_BY,
        save_all=False, require_significant=True
    )

    # 2) Build Effects report (no extra thresholds)
    effects_df = make_effects_report(
        dfx, edge_stats,
        min_freq=MIN_FREQ, require_significant=True
    )
    effects_df.to_csv(effects_path, index=False)
    print(f"[OK] Saved effects report -> {effects_path}")

    return {
        "category": category, "name": name,
        "n_samples": len(dfx), "n_vars": len(var_names),
        "edges_csv": str(edges_stable_path),
        "edges_all_csv": str(edges_all_path),
        "effects_csv": str(effects_path),
        "targets": forbid_out,   # used by ancestor-graph scripts
    }

# ---------------------------------------------------------------------
# ====== Main ==========================================================
# ---------------------------------------------------------------------
def main():
    """
    Entry point:
      - Prepare results directory.
      - Load & clean data.
      - Run all scenarios.
      - Write a summary CSV with per-scenario metadata.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(INPUT_FILE)
    summary_rows = []
    for scen in SCENARIOS:
        info = run_scenario(df, scen, base_outdir=RESULTS_DIR)
        summary_rows.append(info)
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "_summary.csv", index=False)
    print(f"\n[OK] Wrote summary -> {RESULTS_DIR / '_summary.csv'}")


if __name__ == "__main__":
    main()
