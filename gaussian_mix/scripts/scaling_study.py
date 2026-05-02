"""
Scaling study: mode weight MAE vs. wall-clock time across algorithms and effort levels.

Runs repeated trials of each inference algorithm over a range of effort levels.
Saves all raw per-trial results to JSON so plots can be regenerated without re-running.

Usage:
    python scaling_study.py                          # run trials if needed, then plot
    python scaling_study.py --force                  # re-run all trials
    python scaling_study.py --plot-only              # regenerate plots from saved JSON
    python scaling_study.py --algorithms rwmh nuts   # run/plot a subset of algorithms

Note on NUTS wall_time_core_s: warmup (window adaptation) is outside the core timer
because the sampling function cannot be AOT-compiled until warmup determines the kernel
parameters. NUTS core time measures sampling only; total time includes warmup + JIT.

Note on --algorithms and existing results: without --force, an existing results.json
is returned unchanged regardless of --algorithms. Use --force to re-run everything.

Total runtime estimate: ~175 trials x ~15 s/trial ~ 45 minutes.
"""

import argparse
import contextlib
import importlib
import io
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Suppress TF C++ backend noise before any tensorflow_probability import (used by ns.py)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import OUTPUT_DIR

SCRIPTS_DIR     = Path(__file__).parent
SCALING_OUT_DIR = OUTPUT_DIR / "scaling"
RESULTS_PATH    = SCALING_OUT_DIR / "results.json"

# ---------------------------------------------------------------------------
# Effort configurations
# ---------------------------------------------------------------------------
# param  : module-level constant to vary (monkey-patched before calling main())
# values : effort levels to sweep
# fixed  : module-level constants to override for the scaling study
#          (typically smaller warmup than the default benchmark settings)
#
# affinv note: NUM_SAMPLES is the *total* steps run; effective = NUM_SAMPLES - NUM_BURNIN.
# All affinv values must be > NUM_BURNIN (250).

EFFORT_CONFIGS: dict = {
    "rwmh": {
        "param":  "NUM_SAMPLES",
        "values": [250, 500, 1000, 2500, 5000, 10000],
        "fixed":  {"NUM_BURNIN": 100},
    },
    "nuts": {
        "param":  "NUM_SAMPLES",
        "values": [250, 500, 1000, 2500, 5000],
        "fixed":  {"NUM_WARMUP": 250},
    },
    "affinv": {
        "param":  "NUM_SAMPLES",
        "values": [500, 750, 1250, 2000, 3500, 5000],
        "fixed":  {"NUM_BURNIN": 250},
    },
    "smc": {
        "param":  "NUM_PARTICLES",
        "values": [100, 250, 500, 1000, 2000, 5000],
        "fixed":  {"NUM_MCMC_STEPS": 10},
    },
    "ns": {
        "param":  "NUM_LIVE_POINTS",
        "values": [100, 250, 500, 1000, 2000],
        "fixed":  {"MAX_SAMPLES": 1e5},
    },
    "deo": {
        "param":  "NUM_SAMPLES",
        "values": [250, 500, 1000, 2500, 5000],
        "fixed":  {"NUM_WARMUP": 100},
    },
    "seo": {
        "param":  "NUM_SAMPLES",
        "values": [250, 500, 1000, 2500, 5000],
        "fixed":  {"NUM_WARMUP": 100},
    },
}

NUM_TRIALS = 5  # seeds 0 .. NUM_TRIALS-1

ALGO_LABELS = {
    "rwmh":   "RWMH",
    "nuts":   "NUTS",
    "affinv": "Affine Invariant",
    "smc":    "Adaptive SMC",
    "ns":     "Nested Sampling",
    "deo":    "DEO-PT",
    "seo":    "SEO-PT",
}

# Guard: affinv effort values must all exceed NUM_BURNIN, or the post-burnin
# slice will be empty, producing silent garbage rather than a failed record.
_affinv_burnin = EFFORT_CONFIGS["affinv"]["fixed"]["NUM_BURNIN"]
assert all(v > _affinv_burnin for v in EFFORT_CONFIGS["affinv"]["values"]), (
    f"All affinv effort values must exceed NUM_BURNIN={_affinv_burnin}. "
    f"Got: {EFFORT_CONFIGS['affinv']['values']}"
)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(algo: str, effort_param: str, effort_value, fixed_params: dict, seed: int) -> dict:
    """
    Monkey-patch the algorithm module's effort constant, call main(save_outputs=False),
    and return a result record.

    Uses importlib.reload() per trial for a clean module state. This is necessary
    because the effort_value often changes the static scan length in jax.lax.scan,
    requiring a fresh JAX trace and AOT compilation at each new value.

    Returns a dict with None for numeric fields and an error string on failure.
    JSON does not support NaN, so None (serialized as null) is used for missing values.
    """
    base = {
        "algorithm":         algo,
        "effort_param":      effort_param,
        "effort_value":      effort_value,
        "trial":             seed,
        "seed":              seed,
        "wall_time_core_s":  None,
        "wall_time_total_s": None,
        "mode_weight_mae":   None,
        "mode_weights":      None,
        "true_mode_weights": None,
        "error":             None,
    }
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        mod = importlib.import_module(algo)
        importlib.reload(mod)

        setattr(mod, effort_param, effort_value)
        for k, v in fixed_params.items():
            setattr(mod, k, v)

        # Silence noisy third-party loggers during the trial.
        # "root" as a string names a child logger, not the actual root logger;
        # suppress logging.root explicitly to catch TFP/JAXNS/absl messages.
        noisy = ["arviz", "jaxns", "absl", "absl-py", "tensorflow", "tensorflow_probability"]
        _loggers   = [logging.getLogger(n) for n in noisy] + [logging.root]
        _prev_lvls = [lg.level for lg in _loggers]
        for lg in _loggers:
            lg.setLevel(logging.CRITICAL)

        sink = io.StringIO()
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(sink), \
             contextlib.redirect_stderr(sink):
            warnings.simplefilter("ignore")
            diag = mod.main(seed=seed, save_outputs=False)

        for lg, lv in zip(_loggers, _prev_lvls):
            lg.setLevel(lv)

        mw  = np.array(diag["mode_weights"])
        tw  = np.array(diag["true_mode_weights"])
        mae = float(np.mean(np.abs(mw - tw)))
        base.update({
            "wall_time_core_s":  float(diag["wall_time_core_s"]),
            "wall_time_total_s": float(diag["wall_time_total_s"]),
            "mode_weight_mae":   mae,
            "mode_weights":      mw.tolist(),
            "true_mode_weights": tw.tolist(),
        })
    except Exception as exc:
        base["error"] = str(exc)
    return base


# ---------------------------------------------------------------------------
# Batch runner with persistence
# ---------------------------------------------------------------------------

def run_all_trials(algorithms: list, force: bool = False) -> list:
    """
    Run all (algo, effort_value, seed) combinations and save to RESULTS_PATH.

    If results already exist and force=False, load and return without re-running.
    On force=True, runs everything and overwrites the file.
    """
    if RESULTS_PATH.exists() and not force:
        print(f"Results file found: {RESULTS_PATH}")
        print("Loading existing results. Use --force to re-run, --plot-only to just replot.")
        all_algos = list(EFFORT_CONFIGS.keys())
        if sorted(algorithms) != sorted(all_algos):
            print(
                f"WARNING: --algorithms filter {algorithms} is ignored when loading cached "
                f"results. The full results file will be used for plotting. "
                f"Use --force to re-run only the selected algorithms."
            )
        with open(RESULTS_PATH) as f:
            return json.load(f)

    total = sum(len(EFFORT_CONFIGS[a]["values"]) * NUM_TRIALS for a in algorithms)
    records = []
    n  = 0

    for algo in algorithms:
        cfg   = EFFORT_CONFIGS[algo]
        param = cfg["param"]
        fixed = cfg["fixed"]

        for effort_val in cfg["values"]:
            for trial in range(NUM_TRIALS):
                n += 1
                print(
                    f"[{n:3d}/{total}]  {algo.upper():<8}  "
                    f"{param}={effort_val:<6}  trial={trial}",
                    flush=True,
                )

                rec = run_trial(algo, param, effort_val, fixed, seed=trial)
                if rec["error"]:
                    print(f"  !! FAILED: {rec['error']}")
                records.append(rec)

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(records, f, indent=2)

    n_failed = sum(1 for r in records if r["error"])
    print(f"\nSaved {len(records)} records ({n_failed} failures) -> {RESULTS_PATH}")
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list) -> dict:
    """
    Aggregate the flat record list into:
      { algo: { effort_value: { core_times, total_times, maes } } }

    Skips failed records (error set or mode_weight_mae is None).
    """
    out = defaultdict(lambda: defaultdict(lambda: {
        "core_times":  [],
        "total_times": [],
        "maes":        [],
    }))
    for rec in records:
        if rec.get("error") or rec["mode_weight_mae"] is None:
            continue
        a  = rec["algorithm"]
        ev = rec["effort_value"]
        out[a][ev]["core_times"].append(rec["wall_time_core_s"])
        out[a][ev]["total_times"].append(rec["wall_time_total_s"])
        out[a][ev]["maes"].append(rec["mode_weight_mae"])
    return out


# ---------------------------------------------------------------------------
# Publication-quality plots
# ---------------------------------------------------------------------------

def make_plots(records: list) -> None:
    """
    Generate two publication-ready figures:
      mae_vs_core_time.png  -- ModeMAE vs wall_time_core_s  (JIT excluded)
      mae_vs_total_time.png -- ModeMAE vs wall_time_total_s (cold-start time)

    Each curve shows the median ModeMAE at the median wall time per effort level.
    The shaded band spans the 25th-75th percentile of ModeMAE across trials.
    Both axes are logarithmic.
    """
    _rc = {
        "font.family":         "serif",
        "font.size":           11,
        "axes.labelsize":      12,
        "axes.titlesize":      11,
        "axes.linewidth":      1.2,
        "legend.fontsize":     9,
        "legend.framealpha":   0.9,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.top":           True,
        "ytick.right":         True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }

    agg        = aggregate(records)
    algo_order = ["rwmh", "nuts", "affinv", "smc", "ns", "deo", "seo"]
    palette    = plt.cm.tab10(np.linspace(0, 0.85, len(algo_order)))
    color_map  = {a: palette[i] for i, a in enumerate(algo_order)}

    plot_configs: list = [
        (
            "core_times",
            "mae_vs_core_time.png",
            "Wall-clock time, core (s)",
            "Mode weight MAE vs. core sampling time",
        ),
        (
            "total_times",
            "mae_vs_total_time.png",
            "Wall-clock time, total (s)",
            "Mode weight MAE vs. total wall time (incl. JIT & init)",
        ),
    ]

    for time_key, fname, xlabel, title in plot_configs:
        with matplotlib.rc_context(_rc):
            fig, ax = plt.subplots(figsize=(7, 5))

            for algo in algo_order:
                if algo not in agg:
                    continue

                effort_data = agg[algo]
                med_time, med_mae, q25_mae, q75_mae = [], [], [], []

                for ev in sorted(effort_data.keys()):
                    d     = effort_data[ev]
                    times = np.array(d[time_key])
                    maes  = np.array(d["maes"])
                    if len(maes) == 0:
                        continue
                    med_time.append(float(np.median(times)))
                    med_mae.append(float(np.median(maes)))
                    q25_mae.append(float(np.percentile(maes, 25)))
                    q75_mae.append(float(np.percentile(maes, 75)))

                if not med_time:
                    continue

                # Sort by time so the connecting line is monotone on the x-axis.
                # (Increasing effort doesn't always map to increasing time, e.g. SMC
                # with more particles can converge in fewer temperature steps.)
                idx = np.argsort(med_time)
                mt  = np.array(med_time)[idx]
                mm  = np.array(med_mae)[idx]
                lo  = np.array(q25_mae)[idx]
                hi  = np.array(q75_mae)[idx]
                c   = color_map[algo]

                ax.plot(mt, mm, marker="o", markersize=5, lw=1.5,
                        color=c, label=ALGO_LABELS.get(algo, algo), zorder=3)
                ax.fill_between(mt, lo, hi, alpha=0.15, color=c, zorder=2)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mode weight MAE")
            ax.set_title(title, pad=8)
            ax.grid(True, which="major", linestyle="--", linewidth=0.6,
                    color="gray", alpha=0.4, zorder=1)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.4,
                    color="gray", alpha=0.2, zorder=1)
            ax.legend(loc="upper right")
            ax.annotate(
                f"Shaded band = IQR across {NUM_TRIALS} trials",
                xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8, color="gray",
            )
            fig.tight_layout()
            out_path = SCALING_OUT_DIR / fname
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scaling study: ModeMAE vs wall time for the Gaussian mixture benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all trials even if results.json already exists",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate plots from existing results.json without running any trials",
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        default=list(EFFORT_CONFIGS.keys()),
        choices=list(EFFORT_CONFIGS.keys()),
        metavar="ALGO",
        help=(
            "Subset of algorithms to run (default: all). "
            f"Choices: {list(EFFORT_CONFIGS.keys())}"
        ),
    )
    args = parser.parse_args()

    if args.plot_only and args.force:
        print("WARNING: --force is ignored when --plot-only is set.")

    if args.plot_only:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} not found. Run without --plot-only first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records from {RESULTS_PATH}")
    else:
        records = run_all_trials(args.algorithms, force=args.force)

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_plots(records)


if __name__ == "__main__":
    main()
