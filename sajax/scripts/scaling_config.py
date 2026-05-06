"""
Shared configuration, cost models, and metric functions for the scaling study.

Imported by scaling_run_sampler.py and scaling_plot.py.
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import OUTPUT_DIR, PARAM_NAMES, GROUND_TRUTH, PRIOR_DISTRIBUTIONS, LC_TRUE, compute_lc_from_constrained

SCRIPTS_DIR     = Path(__file__).parent
SCALING_OUT_DIR = OUTPUT_DIR / "scaling_compute"

# ---------------------------------------------------------------------------
# Budget levels and trial count
# ---------------------------------------------------------------------------

LOGP_BUDGETS = [100_000, 200_000, 500_000, 1_000_000]
NUM_TRIALS   = 5  # seeds 0 .. NUM_TRIALS-1

# ---------------------------------------------------------------------------
# Pilot estimates for unknowns in the cost models
# ---------------------------------------------------------------------------

SMC_EST_NUM_STEPS       = 91   # typical adaptive tempering steps
NS_EVALS_PER_LIVE_POINT = 3721  # total_likelihood_evals / NUM_LIVE_POINTS

# ---------------------------------------------------------------------------
# Fixed (non-effort) hyperparameters per algorithm
# ---------------------------------------------------------------------------

ALGO_FIXED = {
    "rwmh":   {"NUM_CHAINS": 4,   "NUM_BURNIN": 2000},

    "affinv": {"NUM_WALKERS": 52, "NUM_BURNIN": 2000},

    "smc":    {"NUM_MCMC_STEPS": 25, "TARGET_ESS": 0.75, "MAX_STEPS": 500,
               "SIGMA_FACTOR": 1.0},

    "ns":     {"NUM_POSTERIOR_DRAWS": 5000, "NUM_SLICES": 25,
               "DLOGZ_THRESHOLD": 5.0},

    "deo":    {"NUM_CHAINS": 30, "NUM_WARMUP": 500, "KERNEL": "rwmh",
               "ANNEALING_BASE": 1.4142135623730951,
               "STEP_SIZE_HOT_RWMH": 0.5, "STEP_SIZE_COLD_RWMH": 0.002},

    "seo":    {"NUM_CHAINS": 30, "NUM_WARMUP": 500, "KERNEL": "rwmh",
               "ANNEALING_BASE": 1.4142135623730951,
               "STEP_SIZE_HOT_RWMH": 0.5, "STEP_SIZE_COLD_RWMH": 0.002},
}

ALL_ALGORITHMS = list(ALGO_FIXED.keys())

ALGO_LABELS = {
    "rwmh":   "RWMH",
    "affinv": "Affine Invariant",
    "smc":    "Adaptive SMC",
    "ns":     "Nested Sampling",
    "deo":    "DEO-PT",
    "seo":    "SEO-PT",
}

ALGO_COLORS = {
    "rwmh":   "#1f77b4",
    "affinv": "#2ca02c",
    "smc":    "#d62728",
    "ns":     "#9467bd",
    "deo":    "#8c564b",
    "seo":    "#e377c2",
}

# Minimum values to avoid degenerate runs
MIN_SAMPLES     = 100
MIN_PARTICLES   = 10
MIN_LIVE_POINTS = 10

# ---------------------------------------------------------------------------
# Prior standard deviations for normalisation
# ---------------------------------------------------------------------------

def _prior_std(name):
    d = PRIOR_DISTRIBUTIONS[name]
    return float(np.sqrt(float(d.variance)))

PRIOR_STDS = {name: _prior_std(name) for name in PARAM_NAMES}

# ---------------------------------------------------------------------------
# Cost model: LDE budget -> native effort parameter
# ---------------------------------------------------------------------------

def back_compute(algo: str, budget: int) -> dict:
    """
    Given a log-density-equivalent budget, return a dict with keys:
      param  - name of the effort parameter
      value  - integer value to set
      fixed  - dict of fixed hyperparameters to set alongside it
    """
    f = ALGO_FIXED[algo]

    if algo == "rwmh":
        n = budget // f["NUM_CHAINS"] - f["NUM_BURNIN"]
        n = max(n, MIN_SAMPLES)
        return {"param": "NUM_SAMPLES", "value": n,
                "fixed": {"NUM_BURNIN": f["NUM_BURNIN"], "NUM_CHAINS": f["NUM_CHAINS"]}}

    if algo == "affinv":
        n = budget // f["NUM_WALKERS"]
        n = max(n, f["NUM_BURNIN"] + MIN_SAMPLES)
        return {"param": "NUM_SAMPLES", "value": n,
                "fixed": {"NUM_BURNIN": f["NUM_BURNIN"], "NUM_WALKERS": f["NUM_WALKERS"]}}

    if algo == "smc":
        cost_per_particle = SMC_EST_NUM_STEPS * (f["NUM_MCMC_STEPS"] + 2)
        p = budget // cost_per_particle
        p = max(p, MIN_PARTICLES)
        return {"param": "NUM_PARTICLES", "value": p,
                "fixed": {"NUM_MCMC_STEPS": f["NUM_MCMC_STEPS"],
                          "TARGET_ESS": f["TARGET_ESS"],
                          "MAX_STEPS": f["MAX_STEPS"],
                          "SIGMA_FACTOR": f["SIGMA_FACTOR"]}}

    if algo == "ns":
        nlp = budget // NS_EVALS_PER_LIVE_POINT
        nlp = max(nlp, MIN_LIVE_POINTS)
        return {"param": "NUM_LIVE_POINTS", "value": nlp,
                "fixed": {"MAX_SAMPLES": float(budget),
                          "NUM_POSTERIOR_DRAWS": f["NUM_POSTERIOR_DRAWS"],
                          "NUM_SLICES": f["NUM_SLICES"],
                          "DLOGZ_THRESHOLD": f["DLOGZ_THRESHOLD"]}}

    if algo in ("deo", "seo"):
        n = budget // f["NUM_CHAINS"] - f["NUM_WARMUP"]
        n = max(n, MIN_SAMPLES)
        fixed = {
            "NUM_WARMUP": f["NUM_WARMUP"],
            "NUM_CHAINS": f["NUM_CHAINS"],
            "KERNEL": f["KERNEL"],
            "ANNEALING_BASE": f["ANNEALING_BASE"],
            "STEP_SIZE_HOT_RWMH": f["STEP_SIZE_HOT_RWMH"],
            "STEP_SIZE_COLD_RWMH": f["STEP_SIZE_COLD_RWMH"],
        }
        return {"param": "NUM_SAMPLES", "value": n, "fixed": fixed}

    raise ValueError(f"Unknown algorithm: {algo}")


def extract_actual_oracle(algo: str, diag: dict):
    """Read actual oracle eval count from a diagnostics dict. Returns (count, type_string)."""
    if algo in ("rwmh", "affinv", "smc"):
        return diag.get("total_log_density_evals"), "logp"
    if algo == "ns":
        return diag.get("total_likelihood_evals"), "logp"
    if algo in ("deo", "seo"):
        for k in ("gradient_evals", "log_density_evals"):
            if k in diag:
                return diag[k], "grad" if "gradient" in k else "logp"
        return None, None
    return None, None

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_normalised_mae(posterior_means: dict, ground_truth: dict) -> float:
    errors = []
    for name in PARAM_NAMES:
        if name in posterior_means and name in ground_truth:
            err = abs(float(posterior_means[name]) - float(ground_truth[name]))
            std = PRIOR_STDS.get(name, 1.0)
            errors.append(err / std if std > 0 else err)
    return float(np.mean(errors)) if errors else float("nan")


def compute_raw_mae(posterior_means: dict, ground_truth: dict) -> float:
    errors = []
    for name in PARAM_NAMES:
        if name in posterior_means and name in ground_truth:
            errors.append(abs(float(posterior_means[name]) - float(ground_truth[name])))
    return float(np.mean(errors)) if errors else float("nan")


def compute_per_param_normalised_error(posterior_means: dict, ground_truth: dict) -> dict:
    out = {}
    for name in PARAM_NAMES:
        if name in posterior_means and name in ground_truth:
            err = abs(float(posterior_means[name]) - float(ground_truth[name]))
            std = PRIOR_STDS.get(name, 1.0)
            out[name] = float(err / std) if std > 0 else float(err)
    return out


def _sampled_to_constrained(posterior_means: dict) -> dict:
    """Convert sampled-parameter dict (PARAM_NAMES keys) to constrained dict for compute_lc_from_constrained."""
    sin_lat        = float(posterior_means.get("sin_lat", 0.0))
    ldc_q1         = float(posterior_means.get("ldc_q1", 0.0))
    ldc_q2         = float(posterior_means.get("ldc_q2", 0.0))
    ecc_h          = float(posterior_means.get("ecc_h", 0.0))
    ecc_k          = float(posterior_means.get("ecc_k", 0.0))
    semimajor_axis = float(posterior_means.get("semimajor_axis", 1.0))
    impact_param   = float(posterior_means.get("impact_param", 0.0))
    return {
        "spot_lat":      float(np.rad2deg(np.arcsin(np.clip(sin_lat, -1.0, 1.0)))),
        "spot_long":     float(posterior_means.get("spot_long", 0.0)),
        "spot_size":     float(posterior_means.get("spot_size", 11.0)),
        "spot_flux":     float(posterior_means.get("spot_flux", 0.7)),
        "p_rot":         float(posterior_means.get("p_rot", 0.5)),
        "P_orb":         float(posterior_means.get("P_orb", 1.0)),
        "planet_radius": float(posterior_means.get("planet_radius", 0.1)),
        "semimajor_axis": semimajor_axis,
        "inclination":   float(np.rad2deg(np.arccos(np.clip(impact_param / semimajor_axis, -1.0, 1.0)))),
        "eccentricity":  ecc_h ** 2 + ecc_k ** 2,
        "arg_periapsis": float(np.arctan2(ecc_k, ecc_h)),
        "ldc_u1":        float(2 * np.sqrt(max(ldc_q1, 0.0)) * ldc_q2),
        "ldc_u2":        float(np.sqrt(max(ldc_q1, 0.0)) * (1.0 - 2.0 * ldc_q2)),
    }


def compute_lc_rmse(posterior_means: dict) -> float:
    """RMSE of the posterior-mean light curve vs. the true noiseless light curve."""
    constrained = _sampled_to_constrained(posterior_means)
    lc_fit = np.array(compute_lc_from_constrained(constrained))
    return float(np.sqrt(np.mean((lc_fit - LC_TRUE) ** 2)))
