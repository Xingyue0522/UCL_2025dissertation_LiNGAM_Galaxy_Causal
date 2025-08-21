Here’s a **merged and concise README** that folds your preprocessing notes into the main project doc, using your actual file names/paths.

---

# Causal Inference in Galaxy Properties Based LiNGAM

**UCL Dissertation Project — Experimental Code**

This repository contains the experimental pipeline for **“Causal Inference in Galaxy Properties Based LiNGAM.”**
The workflow runs **DirectLiNGAM with bootstrap stability** across multiple scenario definitions, exports stable edges and effect reports, evaluates **predictive verifiability (CV-R²)** per scenario, visualizes **ancestor subgraphs**, and performs **path decomposition** (per-target and scenario-level **ALL PATHS** tables).

---

## Repository Layout

```
.
├── data/
│   ├── main.csv                      # Primary input for LiNGAM (after preprocessing)
│   ├── features_gz2.csv              # Galaxy Zoo 2 features (EV+gating)
│   ├── features_sdss.csv             # SDSS physical features
│   ├── sdss_labels.csv               # Labels derived during GZ2 processing
│   ├── sdss.npz                      # Optional SDSS intermediate
│   └── zoo2MainSpecz.csv.gz          # Raw GZ2/SDSS join source
├── preprocessing/
│   ├── preprocessing-GZ2.ipynb
│   ├── preprocessing-Merge.ipynb
│   ├── preprocessing-SDSS copy.ipynb
│   └── vif_report.csv                # VIF diagnostics written by Merge notebook
├── results/                          # Script outputs (auto-created)
│   └── ...
├── scenarios.py                      # Scenarios (no high-VIF controls)
├── scenarios_with_control_var.py     # Scenarios (with high-VIF controls)
├── Evaluate.py                       # Per-scenario CV-R² evaluation & summary
├── LiNGAM.py                         # DirectLiNGAM + bootstrap stability & exports
├── plot_ancestors_targets.py         # Ancestor subgraph plots (per scenario/target)
└── path_contrib_all_sources.py       # Path decomposition (per-target + ALL PATHS)
```

---
## Data acquisition address

**SDSS**: https://deepdip.iap.fr/#item/60ef1e05be2b8ebb048d951d 
**Galaxy Zoo 2**: https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/zoo2MainSpecz.csv.gz
> After downloading the data, please place it under "data" folder.

---

## Environment

* **Python** ≥ 3.9
* **Required**: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `networkx`, `matplotlib`, `causallearn`
* **Optional (for nicer graph layouts)**: `pygraphviz` (requires system `graphviz`)

```bash
pip install pandas numpy scikit-learn statsmodels networkx matplotlib causallearn
# Optional:
pip install pygraphviz
# System (if needed):
# sudo apt-get install graphviz graphviz-dev
```

---

## Preprocessing

This folder contains three notebooks that prepare inputs for training/evaluation and produce a single merged table.

### Inputs & Outputs

* **Inputs / intermediates**

  * `data/zoo2MainSpecz.csv.gz`, `data/sdss.npz` (raw/intermediate)
* **Per-source features**

  * `data/features_sdss.csv` — from `preprocessing-SDSS copy.ipynb`
  * `data/features_gz2.csv` — from `preprocessing-GZ2.ipynb`
  * `data/sdss_labels.csv` — from `preprocessing-GZ2.ipynb`
* **Merged table**

  * `data/main.csv` — from `preprocessing-Merge.ipynb`
* **Diagnostics**

  * `preprocessing/vif_report.csv` — Variance Inflation Factor (VIF) report from `preprocessing-Merge.ipynb`

### Notebooks & Key Steps

1. **`preprocessing-SDSS copy.ipynb`**

   * Load & clean SDSS (missing/outlier handling, scaling/derived features).
   * Build colors `[u−g, g−r, r−i, i−z]` from `dered_petro_{u,g,r,i,z}`, standardize, PCA → `color_pc1` (and optionally `color_pc2`).
   * Run **VIF** collinearity diagnostics; write `preprocessing/vif_report.csv` and drop high-VIF features using a configurable threshold (default 10; 5–20 reasonable).
   * Output: `data/features_sdss.csv` (and `data/sdss.npz` if needed).

2. **`preprocessing-GZ2.ipynb`**

   * Load GZ2 annotations;
   * Convert GZ2 probabilities to **expected-value (EV)** encodings under **decision-tree gating** (your EV gate). Threshold is configurable at the top.
   * Outputs: `data/features_gz2.csv`.

3. **`preprocessing-Merge.ipynb`**

   * Align & merge SDSS + GZ2 on shared keys (e.g., `objID/specObjID`).
   
   * Output: `data/main.csv`.



### Quick Start

Run in order:

1. `preprocessing-SDSS copy.ipynb`
2. `preprocessing-GZ2.ipynb`
3. `preprocessing-Merge.ipynb`

> If you change the **EV gate** or **VIF threshold**, re-run step 3 (and any upstream step whose outputs changed).

---

## Scenarios

* `scenarios.py`: scenarios **without** high-VIF controls (`ssfr_mean`, `age_mean`, `metallicity_mean`).
* `scenarios_with_control_var.py`: scenarios **with** those high-VIF controls.

---

## Run & Outputs

### 1) Structure learning + bootstrap stability

```bash
python LiNGAM.py
```

Outputs (under `results/<category>/`):

* `edges_all_<SCENARIO>.csv` — all edges (no filtering)
* `edges_<SCENARIO>.csv` — **stable** edges (freq threshold + significant CI)
* `effects_report_<SCENARIO>.csv` — frequency, mean/avg weights, CI, p-values, **partial R²**
* `results/_summary.csv` — scenario-level metadata

### 2) Per-scenario evaluation (CV-R²)

```bash
python Evaluate.py
```

Outputs:

* `results/<category>/eval_<SCENARIO>.csv`
* `results/<category>/eval_<SCENARIO>_cvR2.png`
* `results/_evaluation_summary.csv`

### 3) Ancestor subgraph visualization

```bash
python plot_ancestors_targets.py
```

Outputs:

* `results/<category>/ancestors_<SCENARIO>_<TARGET>.png`

### 4) Path decomposition (per-target + scenario-level ALL PATHS)

```bash
python path_contrib_all_sources.py --scenarios ALL --targets AUTO --maxlen 4
```

Outputs (under `results/<category>/path_tables_<SCENARIO>/`):

* `sources_summary_to_<target>.csv` — per-target source summary (TE, top ± paths, etc.)
* `all_paths_<SCENARIO>.csv` — **scenario-level ALL PATHS** consolidated table
* `node_flow_to_<target>.csv` — node-level net throughflow (when `--node-flow` is used)

---

## Key Parameters (high-level)

* **Bootstrap**: `N_RUNS=200`, `SAMPLE_FRAC=0.9`
* **Stability filter**: `MIN_FREQ=0.9`, `WEIGHT_THRESHOLD=0.05`, `RANK_BY= "avg_weight"`
* **Significance/CI**: `ALPHA=0.05`, `N_BOOT_CI=2000`
* **Structural priors**: `layer`, `mutually_exclusive`, `extra_forbid/require`
* **Gate behavior**: **EV gating** in preprocessing; threshold configurable
* **VIF**: threshold configurable (default 10), report at `preprocessing/vif_report.csv`
* **Evaluation**: `K_FOLDS`, `RANDOM_SEED`, and baselines (Mass-only, Lasso)

---

## Result Interpretation

* **edges\_\*.csv**: `frequency` (stability), `mean_weight/avg_weight` (effect size), `ci_low/high`, `p_value`.
* **effects\_report\_\*.csv**: edge stats + **partial R²** (parent-set explanatory power).
* **eval\_\*.csv / \_evaluation\_summary.csv**: CV-R² as verifiability; compare vs **Mass-only** and **Lasso** baselines.
* **ALL PATHS tables**: total effects decomposed into signed path products; identify dominant channels and sign cancellations.

---