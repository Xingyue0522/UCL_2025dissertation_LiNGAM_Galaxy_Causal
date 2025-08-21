# -*- coding: utf-8 -*-
"""
scenarios_with_control_var.py
Catalog of causal-discovery "scenarios" and categories. The hard gate
(spiral_prob_eff >= TAU_SPIRAL) is built into all class-B scenarios.

This version *includes* the high-VIF (>10) variables as a control set:
    PHYS2 = ["ssfr_mean", "age_mean", "metallicity_mean"]
They are used to study how these variables influence results.
These controlled variables are always used wgit remote -vhen investigating galaxy
formation / quenching / synthesis pathways.
"""

TAU_SPIRAL = 0.5   # Hard-gate threshold (you may clone 0.6/0.7 for robustness)

# ---------------------------------------------------------------------
# Optional filters (used by class A/C)
# ---------------------------------------------------------------------
def filt_all(df):
    """Return the input DataFrame unchanged (no filtering)."""
    return df

def filt_disc(df):
    """Approximate 'more disk-like' subsample; avoid feeding Smooth-branch variables back."""
    # 'More disk-like' approximate subsample (avoid feeding Smooth-branch definitions back)
    return df[df["elliptical_prob"] <= 0.2]

# ---------------------------------------------------------------------
# Unified lists of physical variables
# ---------------------------------------------------------------------
PHYS  = ["z", "logMass_median", "oh_p50", "v_disp", "color_pc1"]
PHYS2 = ["ssfr_mean", "age_mean", "metallicity_mean"]


# ---------------------------------------------------------------------
# ===================== Scenario registry =============================
# Each scenario is a dict:
# {
#   "name": scenario name,
#   "category": category (controls output subfolder),
#   "vars": list of variables to include,
#   "filter": subsample filter function or None,
#   "gate": {"col": <name>, "op": ">=", "thr": <value>}  # optional; class B has a built-in hard gate
#   "forbid_outgoing": [nodes allowed only incoming edges],
#   "extra_forbid": [(src, tgt), ...],   # optional, additional forbids
#   "extra_require": [(src, tgt), ...],  # optional, additional requirements
# }
# ---------------------------------------------------------------------
SCENARIOS = [
    # ---------- A: presence / gating on y ----------
    {
        "name": "ELLIPTICAL_VS_DISC",
        "category": "A_presence",
        "vars": PHYS + PHYS2 + ["elliptical_prob"],
        "filter": filt_all,
        "forbid_outgoing": ["elliptical_prob"],
    },
    {
        "name": "SPIRAL_PRESENCE",
        "category": "A_presence",
        "vars": PHYS + PHYS2 + ["bar_prob", "bulge_ev", "spiral_prob_eff"],
        "filter": filt_all,
        "forbid_outgoing": ["spiral_prob_eff"],
    },
    {
        "name": "BAR_PRESENCE_DISC",
        "category": "A_presence",
        "vars": PHYS + PHYS2 + ["bulge_ev", "bar_prob"],
        "filter": filt_disc,
        "forbid_outgoing": ["bar_prob"],
    },
    {
        "name": "BULGE_PROMINENCE_DISC",
        "category": "A_presence",
        "vars": PHYS + PHYS2 + ["bulge_ev"],
        "filter": filt_disc,
        "forbid_outgoing": ["bulge_ev"],
    },
    # {
    #     "name": "ODD_FEATURES_ALL",
    #     "category": "A_presence",
    #     "vars": PHYS + ["bar_prob", "bulge_ev", "spiral_prob_eff", "odd_prob"],
    #     "filter": filt_all,
    #     "forbid_outgoing": ["odd_prob"],
    # },

    # ---------- B: spiral geometry (hard gate) ----------
    {
        "name": "ARMS_GEOMETRY_BOTH",
        "category": "B_spiral_geometry",
        "vars": PHYS + PHYS2 + ["bar_prob", "bulge_ev", "arms_num_ev", "arms_wind_ev"],
        "filter": filt_all,
        "gate": {"col": "spiral_prob_eff", "op": ">=", "thr": TAU_SPIRAL},
        "forbid_outgoing": ["arms_wind_ev"],
    },

    # ---------- C: physical -> formation / quenching / synthesis pathways ----------
    # A) Formation/quenching mechanism: use sSFR as the outcome
    # Rationale: keep mass/dynamics/color/epoch; exclude age/stellar-Z to avoid amplifying the
    # classic color–metallicity degeneracy during mechanism analysis.
    {
        "name": "MORPH_QUENCH_DISC",
        "category": "C_physical_paths",
        "vars": [
            "ssfr_mean",        # Outcome: formation/quenching
            "bulge_ev",         # Morphology (effective after gating)
            "logMass_median",   # Mass baseline
            "v_disp",           # Dynamical scale
            "color_pc1",        # Observed color axis (allowed as covariate/path variable)
            "z"                 # Cosmic epoch
            # Optional: include "oh_p50" to consider chemical coupling
        ],
        "filter": filt_disc,
        "forbid_outgoing": ["ssfr_mean"],   # sSFR only receives edges (no outgoing)
    },

    # B) Color synthesis mechanism: use color_pc1 as the outcome
    # Rationale: do not colocate age_mean / stellar metallicity with color_pc1 to avoid the
    # classic age–metallicity–color degeneracy; use gas-phase metallicity (oh_p50) plus mass/epoch/dynamics.
    {
        "name": "COLOR_SYNTHESIS_CORE",
        "category": "C_physical_paths",
        "vars": [
            "color_pc1",        # Outcome: principal color axis
            "logMass_median",
            "oh_p50",           # Gas-phase metallicity (instantaneous chemical state)
            "v_disp",
            "z"
            # Optional: if dust indicators exist, add "EBV"
            # Note: do NOT include age_mean or metallicity_mean together with color_pc1
        ],
        "filter": filt_all,
        "forbid_outgoing": ["color_pc1"],   # color only receives edges (no outgoing)
    },

    # C) Dynamical support: use v_disp as the outcome
    # Rationale: drop age_mean to avoid degeneracy along the 'reddening axis' with color/metal/mass;
    # keep morphology, mass, epoch; optionally add color_pc1 for comparison.
    {
        "name": "DYNAMICAL_SUPPORT_DISC",
        "category": "C_physical_paths",
        "vars": [
            "v_disp",           # Outcome: velocity dispersion
            "logMass_median",
            "bulge_ev",
            "z"
            # Optional: add "color_pc1" for a control comparison
        ],
        "filter": filt_disc,
        "forbid_outgoing": ["v_disp"],      # v_disp only receives edges (no outgoing)
    },
]
