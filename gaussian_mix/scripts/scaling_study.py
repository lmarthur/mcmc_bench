"""
Scaling study: accuracy vs. wall-clock time across algorithms and effort levels.

Each sampler's main(seed, save_outputs=False) is called directly. No files are
written and no output is printed during the sweep. A single frontier plot is
saved at the end.

Effort parameters swept:
  rwmh, nuts, affinv, deo, seo : NUM_SAMPLES
  smc                           : NUM_PARTICLES
  ns                            : NUM_LIVE_POINTS
"""

import contextlib
import importlib
import io
import logging
import os
import sys
from pathlib import Path


@contextlib.contextmanager
def _suppress_fd2():
    """Redirect OS-level stderr (fd 2) to /dev/null for C++ library noise."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

# Suppress TF/absl C++ log messages before they initialize their logging system.
# Must be set before ns.py (which imports tensorflow_probability) is first loaded.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import OUTPUT_DIR

SCRIPTS_DIR     = Path(__file__).parent
SCALING_OUT_DIR = OUTPUT_DIR / "scaling_study"

NUM_SEEDS   = 20
NUM_EFFORTS = 5


def _logspace_int(lo, hi, n):
    """Return n log-spaced integers between lo and hi (inclusive)."""
    return sorted(set(int(round(v)) for v in np.geomspace(lo, hi, n)))

ALGO_CONFIG = {
    #            effort_param         lo      hi      n
    "rwmh":   {"effort_param": "NUM_SAMPLES",     "lo":  100, "hi": 30000, "n": NUM_EFFORTS},
    "nuts":   {"effort_param": "NUM_SAMPLES",     "lo":  100, "hi": 30000, "n": NUM_EFFORTS},
    "affinv": {"effort_param": "NUM_SAMPLES",     "lo":  100, "hi": 30000, "n": NUM_EFFORTS},
    "smc":    {"effort_param": "NUM_PARTICLES",   "lo":   50, "hi": 10000, "n": NUM_EFFORTS},
    "ns":     {"effort_param": "NUM_LIVE_POINTS", "lo":   50, "hi":  2000, "n": NUM_EFFORTS},
    "deo":    {"effort_param": "NUM_SAMPLES",     "lo":  100, "hi": 30000, "n": NUM_EFFORTS},
    "seo":    {"effort_param": "NUM_SAMPLES",     "lo":  100, "hi": 30000, "n": NUM_EFFORTS},
}

# Build effort grids from (lo, hi, n) specs
for _cfg in ALGO_CONFIG.values():
    _cfg["effort_grid"] = _logspace_int(_cfg["lo"], _cfg["hi"], _cfg["n"])

COLORS = {
    "rwmh": "#e6194b", "nuts": "#3cb44b", "affinv": "#4363d8",
    "smc":  "#f58231", "ns":   "#911eb4", "deo":    "#42d4f4",
    "seo":  "#f032e6",
}


def _load_mod(mod_name, effort_param, effort_val):
    """Reload module and set effort parameter. Called once per (algo, effort_val)."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)
    setattr(mod, effort_param, effort_val)
    return mod


def _run_one(mod, seed):
    """Run a single seed on an already-loaded module."""
    try:
        _silence = [logging.getLogger(n) for n in ("arviz", "jaxns", "absl")]
        _prev_levels = [lg.level for lg in _silence]
        for lg in _silence:
            lg.setLevel(logging.ERROR)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), _suppress_fd2():
            diag = mod.main(seed=seed, save_outputs=False)
        for lg, lvl in zip(_silence, _prev_levels):
            lg.setLevel(lvl)
    except Exception:
        return None
    if diag is None:
        return None
    mw = np.array(diag["mode_weights"])
    tw = np.array(diag["true_mode_weights"])
    return {
        "wall_time_s":     diag["wall_time_s"],
        "mode_weight_mae": float(np.mean(np.abs(mw - tw))),
    }


def run_sweep():
    results = {}
    for algo, cfg in ALGO_CONFIG.items():
        algo_results = []
        total = len(cfg["effort_grid"]) * NUM_SEEDS
        with tqdm(total=total, desc=f"{algo.upper():<8}", unit="run") as pbar:
            for effort_val in cfg["effort_grid"]:
                # Reload once per effort level so JAX only recompiles once,
                # not once per seed.
                mod = _load_mod(algo, cfg["effort_param"], effort_val)
                wall_times, maes = [], []
                for seed in range(NUM_SEEDS):
                    out = _run_one(mod, seed)
                    if out is not None:
                        wall_times.append(out["wall_time_s"])
                        maes.append(out["mode_weight_mae"])
                    pbar.set_postfix({cfg["effort_param"]: effort_val, "seed": seed})
                    pbar.update(1)
                algo_results.append({"effort_val": effort_val, "wall_times": wall_times, "maes": maes})
        results[algo] = algo_results

    # Print min/max summary across all effort levels and seeds
    print(f"\n{'Algorithm':<10} {'effort_param':<18} {'effort_min':>10} {'effort_max':>10} "
          f"{'mae_min':>10} {'mae_max':>10} {'time_min':>10} {'time_max':>10}")
    print("-" * 90)
    for algo, cfg in ALGO_CONFIG.items():
        all_maes, all_times = [], []
        for pt in results[algo]:
            all_maes.extend(pt["maes"])
            all_times.extend(pt["wall_times"])
        if not all_maes:
            continue
        print(f"{algo.upper():<10} {cfg['effort_param']:<18} "
              f"{cfg['effort_grid'][0]:>10} {cfg['effort_grid'][-1]:>10} "
              f"{min(all_maes):>10.4f} {max(all_maes):>10.4f} "
              f"{min(all_times):>10.2f} {max(all_times):>10.2f}")

    return results


def make_plot(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, algo_results in results.items():
        med_wall, med_mae, q25, q75 = [], [], [], []
        for pt in algo_results:
            if len(pt["maes"]) < 2:
                continue
            maes = np.array(pt["maes"])
            wall = np.array(pt["wall_times"])
            med_wall.append(np.median(wall))
            med_mae.append(np.median(maes))
            q25.append(np.percentile(maes, 25))
            q75.append(np.percentile(maes, 75))
        if not med_wall:
            continue
        order = np.argsort(med_wall)
        mw = np.array(med_wall)[order]
        mm = np.array(med_mae)[order]
        lo = np.array(q25)[order]
        hi = np.array(q75)[order]
        c = COLORS.get(algo, "gray")
        ax.plot(mw, mm, marker="o", markersize=5, lw=1.5, color=c, label=algo.upper())
        ax.fill_between(mw, lo, hi, alpha=0.15, color=c)

    ax.set_xlabel("Median wall-clock time (s)")
    ax.set_ylabel("Median mode weight MAE  [lower is better]")
    ax.set_title("Accuracy vs. compute — Pareto frontier\n"
                 "(shaded band = IQR over 20 seeds, burn-in=0)")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCALING_OUT_DIR / "pareto_frontier.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results = run_sweep()
    make_plot(results)
