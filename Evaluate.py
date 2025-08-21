# -*- coding: utf-8 -*-
"""
evaluate.py
Evaluate the overall performance and falsifiability of DirectLiNGAM causal graphs
under each scenario.

Outputs:
  - results/<category>/eval_<SCENARIO>.csv        : CV-R² per target with baselines and parent sets
  - results/<category>/eval_<SCENARIO>_cvR2.png   : Bar plot of CV-R² per scenario (with baselines)
  - results/_evaluation_summary.csv               : Cross-scenario summary (weighted means, IQR, etc.)
  - Prints key summary info to the terminal for quick inspection

Dependencies:
  pip install scikit-learn statsmodels pandas numpy matplotlib
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.stattools import jarque_bera

# Reuse your scenarios and filter/gate definitions
from scenarios_with_control_var import SCENARIOS, TAU_SPIRAL  # noqa

# ---------------------------------------------------------------------
# ============ Paths & global parameters ===============================
# ---------------------------------------------------------------------
INPUT_FILE  = Path("data/main.csv")
RESULTS_DIR = Path("results")

K_FOLDS     = 5
RANDOM_SEED = 42

# Pretty labels (kept consistent with plotting)
NODE_LABEL_MAP = {
    "logMass_median": "Mass",
    "color_pc1": "color",
    "v_disp": "Vdisp",
    "oh_p50": "O/H",
    "ssfr_mean": "sSFR",
    "age_mean": "Age",
    "metallicity_mean": "Stellar Z",
    "elliptical_prob": "Elliptical",
    "spiral_prob_eff": "Spiral",
    "bar_prob": "Bar",
    "bulge_ev": "Bulge",
    "odd_prob": "Odd",
    "arms_num_ev": "Arms#",
    "arms_wind_ev": "Winding",
    "z": "z",
}
def pretty(x: str) -> str:
    """Map a raw variable name to a prettier label for display."""
    return NODE_LABEL_MAP.get(x, x)

# ---------------------------------------------------------------------
# ============ Data utilities: load / clean / gate ====================
# ---------------------------------------------------------------------
def clip_by_quantile(s: pd.Series, low=0.005, high=0.995):
    """Clip a Series by lower/upper quantiles to limit extreme tails."""
    lo, hi = s.quantile([low, high])
    return s.clip(lo, hi)

def load_and_clean(input_file: Path) -> pd.DataFrame:
    """Load CSV, convert sentinel -9999.0 to NaN for selected columns, light tail clipping."""
    df = pd.read_csv(input_file).copy()
    for c in ["oh_p50", "logMass_median", "ssfr_mean", "age_mean", "metallicity_mean"]:
        if c in df.columns:
            df.loc[df[c] == -9999.0, c] = np.nan
    for c in ["color_pc1", "v_disp"]:
        if c in df.columns:
            df[c] = clip_by_quantile(df[c], 0.005, 0.995)
    return df

def apply_gate(df: pd.DataFrame, gate: dict, strict: bool = True) -> pd.DataFrame:
    """
    Apply a row-wise gate (filter) specified by a dict:
      - keys: {'col', 'op', 'thr', (optional) 'thr2' if op == 'between'}
      - supported ops: '>=', '>', '<=', '<', 'between'
    If 'strict' and the column is missing, raise KeyError; otherwise warn and skip.
    """
    if not gate:
        return df
    col = gate.get("col")
    op  = gate.get("op", ">=")
    thr = gate.get("thr")
    if col not in df.columns:
        if strict:
            raise KeyError(f"[GATE] Column '{col}' not found.")
        print(f"[WARN] gate col {col} not found; skip.")
        return df
    if op == ">=":
        out = df[df[col] >= thr]
    elif op == ">":
        out = df[df[col] > thr]
    elif op == "<=":
        out = df[df[col] <= thr]
    elif op == "<":
        out = df[df[col] < thr]
    elif op == "between":
        thr2 = gate.get("thr2")
        lo, hi = min(thr, thr2), max(thr, thr2)
        out = df[(df[col] >= lo) & (df[col] <= hi)]
    else:
        raise ValueError(f"Unsupported gate op: {op}")
    return out

# ---------------------------------------------------------------------
# ============ Core evaluation functions ==============================
# ---------------------------------------------------------------------
def kfold_r2(X: np.ndarray, y: np.ndarray, k: int = K_FOLDS, seed: int = RANDOM_SEED) -> Tuple[float, float]:
    """Standardize + LinearRegression with K-fold CV; return (mean R², std)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(X):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        ytr = (y[tr] - y[tr].mean()) / (y[tr].std(ddof=0) + 1e-12)
        yte = (y[te] - y[te].mean()) / (y[te].std(ddof=0) + 1e-12)
        model = LinearRegression()
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        ss_res = ((yte - yhat) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        r2s.append(r2)
    return float(np.mean(r2s)), float(np.std(r2s))

def kfold_r2_lasso(X: np.ndarray, y: np.ndarray, k: int = K_FOLDS, seed: int = RANDOM_SEED) -> Tuple[float, float]:
    """Standardize + LassoCV within K-fold CV; return (mean R², std)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(X):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        ytr = (y[tr] - y[tr].mean()) / (y[tr].std(ddof=0) + 1e-12)
        yte = (y[te] - y[te].mean()) / (y[te].std(ddof=0) + 1e-12)
        lasso = LassoCV(cv=5, random_state=seed, n_jobs=None)
        lasso.fit(Xtr, ytr)
        yhat = lasso.predict(Xte)
        ss_res = ((yte - yhat) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        r2s.append(r2)
    return float(np.mean(r2s)), float(np.std(r2s))

def residual_checks(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit OLS once on the full sample (with standardization). Return:
      - 'jb_p': Jarque–Bera p-value of residuals (non-Gaussianity check)
      - 'resid_dep_maxabs': max absolute Pearson corr between residual and any parent
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    ys = (y - y.mean())/(y.std(ddof=0) + 1e-12)

    model = LinearRegression().fit(Xs, ys)
    resid = ys - model.predict(Xs)

    jb_stat, jb_p, _, _ = jarque_bera(resid)

    # Pearson correlation between residual and each parent
    corrs = []
    for j in range(Xs.shape[1]):
        xj = Xs[:, j]
        num = float(np.cov(xj, resid, ddof=0)[0, 1])
        den = float(np.std(xj, ddof=0) * np.std(resid, ddof=0) + 1e-12)
        corrs.append(num / den)
    max_abs_corr = float(np.max(np.abs(corrs))) if corrs else np.nan
    return {"jb_p": float(jb_p), "resid_dep_maxabs": max_abs_corr}

# ---------------------------------------------------------------------
# ============ Load parent sets from edges_<SCENARIO>.csv =============
# ---------------------------------------------------------------------
def load_parent_sets_for_scenario(category: str, scenario_name: str) -> Dict[str, List[str]]:
    """
    Return {target: [parents]} from results/<category>/edges_<scenario>.csv.
    If the file is missing, fall back to edges_all_<scenario>.csv (not preferred, but usable).
    """
    scen_dir = RESULTS_DIR / category
    f = scen_dir / f"edges_{scenario_name}.csv"
    if not f.exists():
        f = scen_dir / f"edges_all_{scenario_name}.csv"
        if not f.exists():
            return {}

    df = pd.read_csv(f)
    if df.empty:
        return {}

    parents_map: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        src, tgt = str(r["source"]), str(r["target"])
        parents_map.setdefault(tgt, []).append(src)

    # Deduplicate while preserving order
    for k, v in parents_map.items():
        seen, uniq = set(), []
        for x in v:
            if x not in seen:
                uniq.append(x); seen.add(x)
        parents_map[k] = uniq
    return parents_map

# ---------------------------------------------------------------------
# ============ Per-scenario evaluation ================================
# ---------------------------------------------------------------------
def evaluate_scenario(df_clean: pd.DataFrame, scen: dict) -> pd.DataFrame:
    """
    Evaluate one scenario:
      1) Load stable parent sets.
      2) Apply the scenario's filter and gate to create the subsample.
      3) For each target with non-empty parent set:
           - Compute CV-R² with LinearRegression.
           - Baseline 1: Mass-only (if available).
           - Baseline 2: Lasso (same parent candidates).
           - Diagnostics: JB p-value, max|corr(resid, parent)|.
      4) Save eval_<SCENARIO>.csv and eval_<SCENARIO>_cvR2.png.
    """
    name = scen["name"]; cat = scen["category"]
    print(f"\n[SCENARIO] {name} ({cat})")

    # 1) Load stable parent sets
    parents_map = load_parent_sets_for_scenario(cat, name)
    if not parents_map:
        print("  [WARN] no edges file found or empty; skip.")
        return pd.DataFrame()

    # 2) Apply scenario filter and gate, select sub-sample and variables
    filt_fn  = scen.get("filter", None)
    gate     = scen.get("gate", None)
    vars_    = scen["vars"]

    dfin = df_clean.copy()
    if callable(filt_fn):
        dfin = filt_fn(dfin)
    if gate:
        dfin = apply_gate(dfin, gate, strict=False)

    dfx = dfin[vars_].dropna(how="any")
    if len(dfx) < 100:
        print(f"  [WARN] only {len(dfx)} rows after filter/gate; results may be unstable.")

    out_rows = []
    for tgt, parents in parents_map.items():
        if tgt not in dfx.columns:
            continue
        # Only evaluate targets with non-empty parent sets
        parents = [p for p in parents if p in dfx.columns and p != tgt]
        if len(parents) == 0:
            continue

        sub = dfx[[tgt] + parents].dropna(how="any")
        if len(sub) < max(50, 2 * len(parents) + 5):
            print(f"  [INFO] target={tgt}: n={len(sub)} too small after dropna; skip CV.")
            continue

        X = sub[parents].values
        y = sub[tgt].values

        # Main model: stable parents
        r2_mean, r2_std = kfold_r2(X, y, k=K_FOLDS, seed=RANDOM_SEED)

        # Baseline 1: Mass-only (if available)
        if "logMass_median" in parents or "logMass_median" in sub.columns:
            Xb = sub[["logMass_median"]].values
            r2_mass_mean, r2_mass_std = kfold_r2(Xb, y, k=K_FOLDS, seed=RANDOM_SEED)
        else:
            r2_mass_mean = np.nan; r2_mass_std = np.nan

        # Baseline 2: Lasso (use the same candidate parents)
        r2_lasso_mean, r2_lasso_std = kfold_r2_lasso(X, y, k=K_FOLDS, seed=RANDOM_SEED)

        # Diagnostics: JB for non-Gaussian residuals; residual-parent max abs corr
        chk = residual_checks(X, y)

        out_rows.append({
            "scenario": name, "category": cat,
            "target": tgt, "target_nice": pretty(tgt),
            "n_samples_used": len(sub),
            "n_parents": len(parents),
            "parents": ",".join(parents),
            "parents_nice": ",".join(pretty(p) for p in parents),
            "cv_r2_mean": r2_mean, "cv_r2_std": r2_std,
            "baseline_mass_r2_mean": r2_mass_mean, "baseline_mass_r2_std": r2_mass_std,
            "baseline_lasso_r2_mean": r2_lasso_mean, "baseline_lasso_r2_std": r2_lasso_std,
            "jb_p": chk["jb_p"],
            "resid_dep_maxabs": chk["resid_dep_maxabs"],
        })

        print(f"  [OK] {pretty(tgt)}: CV-R²={r2_mean:.3f}±{r2_std:.3f} | "
              f"Mass={r2_mass_mean if not np.isnan(r2_mass_mean) else float('nan'):.3f} | "
              f"Lasso={r2_lasso_mean:.3f} | parents={len(parents)}")

    df_out = pd.DataFrame(out_rows)
    scen_dir = RESULTS_DIR / cat
    scen_dir.mkdir(parents=True, exist_ok=True)

    if not df_out.empty:
        df_out.to_csv(scen_dir / f"eval_{name}.csv", index=False)

        # ========== Plot: bars + lines, with numeric annotations ==========
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

        order = np.argsort(-df_out["cv_r2_mean"].values)
        labels = [df_out.iloc[i]["target_nice"] for i in order]
        vals   = np.array([df_out.iloc[i]["cv_r2_mean"] for i in order], dtype=float)
        massv  = np.array([df_out.iloc[i]["baseline_mass_r2_mean"] for i in order], dtype=float)
        lassov = np.array([df_out.iloc[i]["baseline_lasso_r2_mean"] for i in order], dtype=float)

        xpos = np.arange(len(vals))
        bars = ax.bar(xpos, vals, alpha=0.85, label="Stable-Parents CV-R²")

        # Line: Mass-only (may contain NaNs)
        if not np.all(np.isnan(massv)):
            valid_m = np.isfinite(massv)
            ax.plot(xpos[valid_m], massv[valid_m], marker="o", linestyle="--", label="Mass-only")

        # Line: Lasso
        valid_l = np.isfinite(lassov)
        ax.plot(xpos[valid_l], lassov[valid_l], marker="s", linestyle=":", label="Lasso (same parents)")

        # Leave headroom to avoid labels hitting the top
        ymax_candidates = [np.nanmax(vals)]
        if np.any(np.isfinite(massv)): ymax_candidates.append(np.nanmax(massv))
        if np.any(np.isfinite(lassov)): ymax_candidates.append(np.nanmax(lassov))
        ymax = float(np.nanmax(ymax_candidates))
        ax.set_ylim(0, ymax + 0.08 + 0.05 * ymax)

        # ====== Text annotations ======
        # 1) Annotate bar tops with CV-R² (kept commented; enable if desired)
        # for rect in bars:
        #     h = rect.get_height()
        #     ax.annotate(f"{h:.3f}",
        #                 xy=(rect.get_x() + rect.get_width() / 2, h),
        #                 xytext=(0, 4), textcoords="offset points",
        #                 ha="center", va="bottom", fontsize=6)

        # 2) Annotate line points (offset by 10/22 px to avoid overlap with bars)
        if not np.all(np.isnan(massv)):
            for x, y in zip(xpos[valid_m], massv[valid_m]):
                ax.annotate(f"{y:.3f}",
                            xy=(x, y),
                            xytext=(0, 12), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6)

        for x, y in zip(xpos[valid_l], lassov[valid_l]):
            ax.annotate(f"{y:.3f}",
                        xy=(x, y),
                        xytext=(0, 22), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6)

        # Axes & title
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("CV-R²")
        ax.set_title(f"CV-R² per target — {name}")
        ax.legend()
        fig.savefig(scen_dir / f"eval_{name}_cvR2.png", dpi=300)
        plt.close(fig)
        # ========== End plotting ==========

    return df_out

# ---------------------------------------------------------------------
# ============ Main ===================================================
# ---------------------------------------------------------------------
def main():
    """Run evaluation across all scenarios and write a cross-scenario summary CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(INPUT_FILE)

    all_rows = []
    for scen in SCENARIOS:
        df_s = evaluate_scenario(df, scen)
        if not df_s.empty:
            all_rows.append(df_s)

    if not all_rows:
        print("[WARN] No evaluation results produced.")
        return

    all_df = pd.concat(all_rows, ignore_index=True)

    # Cross-scenario summary helpers
    def iqr(x):
        """Interquartile range."""
        q1, q3 = np.nanpercentile(x, [25, 75])
        return q3 - q1

    summary = []
    for (cat, scen), sub in all_df.groupby(["category", "scenario"], sort=False):
        summary.append({
            "category": cat,
            "scenario": scen,
            "n_targets_eval": sub.shape[0],
            "cv_r2_mean_avg": float(np.nanmean(sub["cv_r2_mean"])),
            "cv_r2_mean_iqr": float(iqr(sub["cv_r2_mean"])),
            "baseline_mass_r2_mean_avg": float(np.nanmean(sub["baseline_mass_r2_mean"])),
            "baseline_lasso_r2_mean_avg": float(np.nanmean(sub["baseline_lasso_r2_mean"])),
            "jb_p_median": float(np.nanmedian(sub["jb_p"])),
            "resid_dep_maxabs_median": float(np.nanmedian(sub["resid_dep_maxabs"])),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / "_evaluation_summary.csv", index=False)

    print("\n======== EVALUATION SUMMARY ========")
    for _, r in summary_df.iterrows():
        print(f"[{r['scenario']}] targets={int(r['n_targets_eval'])} | "
              f"CV-R²(avg)={r['cv_r2_mean_avg']:.3f} (IQR {r['cv_r2_mean_iqr']:.3f}) | "
              f"Mass baseline={r['baseline_mass_r2_mean_avg']:.3f} | "
              f"Lasso={r['baseline_lasso_r2_mean_avg']:.3f} | "
              f"JB p~median={r['jb_p_median']:.3f} | "
              f"max|corr(resid, parent)|~median={r['resid_dep_maxabs_median']:.3f}")

if __name__ == "__main__":
    main()
