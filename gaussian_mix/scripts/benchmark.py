"""
Benchmark runner for all Gaussian mixture inference algorithms.

Runs each algorithm's main() in sequence, collects diagnostics, and produces
a unified comparison table and plots. Metrics are chosen to be algorithm-agnostic:

  - wall_time_core_s     : sampling loop time only — JIT compilation and initialization
                           excluded via AOT compile (jax.jit.lower().compile()) before
                           the core timer starts. Use this for steady-state throughput
                           comparisons where one-time startup costs are amortized.
  - wall_time_total_s    : cold-start time — covers everything from kernel/model setup
                           through JIT compilation to the final host transfer. Use this
                           for short-run or one-shot comparisons.
  - mode_weight_mae      : mean |empirical_weight_k - true_weight_k| (accuracy)
  - modes_recovered      : number of modes with |error| < 0.05
  - ess_bulk_arviz       : sum of ArviZ bulk ESS over x1,x2 (computed uniformly
                           from idata.nc for ALL algorithms, including NS and SMC,
                           so the Kish/ArviZ mismatch is avoided)
  - ess_per_core_s       : ess_bulk_arviz / wall_time_core_s  (primary efficiency metric)
  - oracle_evals         : log-density or gradient evaluations (algorithm-specific
                           cost unit; labeled in the table — do not compare across
                           algorithms that use different oracle types)
  - ess_per_oracle_eval  : ess_bulk_arviz / oracle_evals  (within-type comparison only)

Note on NUTS: warmup (window adaptation) is outside the core timer because the sampling
function cannot be AOT-compiled until warmup determines the kernel parameters. NUTS core
time therefore measures sampling only; total time includes warmup + JIT + sampling.

ESS note: ArviZ bulk ESS on the resampled draws from NS/SMC is a conservative
(autocorrelation-adjusted) estimate. This makes the comparison fair versus MCMC,
but may under-report the true posterior approximation quality of NS/SMC.
"""

import importlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model import OUTPUT_DIR

BENCHMARK_OUTPUT_DIR = OUTPUT_DIR / "benchmark"

# (module_name, output_subdir, oracle_key, oracle_label)
#   oracle_key   : key in diagnostics.json holding oracle eval count
#   oracle_label : short label printed in the table
ALGORITHMS = [
    ("rwmh",   "rwmh",   "total_log_density_evals", "logp"),
    ("nuts",   "nuts",   "total_grad_evals",         "grad"),
    ("affinv", "affinv", "total_log_density_evals",  "logp"),
    ("smc",    "smc",    "total_log_density_evals",  "logp"),
    ("ns",     "ns",     "total_likelihood_evals",   "logp"),
    ("deo",    "deo",    None,                        "grad"),
    ("seo",    "seo",    None,                        "grad"),
]

# DEO/SEO store oracle counts under a dynamic key depending on kernel type.
# We resolve this at parse time by checking both possible keys.
_PT_ORACLE_KEYS = ("gradient_evals", "log_density_evals")

MODE_RECOVERY_THRESHOLD = 0.05  # |empirical - true| < this counts as "recovered"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_diagnostics(subdir: str) -> dict | None:
    path = OUTPUT_DIR / subdir / "diagnostics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_idata(subdir: str):
    path = OUTPUT_DIR / subdir / "idata.nc"
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return az.from_netcdf(str(path))


def _arviz_bulk_ess(idata) -> float:
    """Sum of ArviZ bulk ESS over all variables (x1, x2)."""
    if idata is None:
        return float("nan")
    summary = az.summary(idata, var_names=["x1", "x2"])
    return float(summary["ess_bulk"].sum())


def _oracle_evals(diag: dict, oracle_key: str | None, oracle_label: str) -> int | None:
    if oracle_key is not None:
        return diag.get(oracle_key)
    # PT algorithms: try both possible keys
    for k in _PT_ORACLE_KEYS:
        if k in diag:
            return diag[k]
    return None


def _mode_weight_mae(diag: dict) -> float:
    mw = np.array(diag["mode_weights"])
    tw = np.array(diag["true_mode_weights"])
    return float(np.mean(np.abs(mw - tw)))


def _modes_recovered(diag: dict, threshold: float = MODE_RECOVERY_THRESHOLD) -> int:
    mw = np.array(diag["mode_weights"])
    tw = np.array(diag["true_mode_weights"])
    return int(np.sum(np.abs(mw - tw) < threshold))


# ---------------------------------------------------------------------------
# Run all algorithms
# ---------------------------------------------------------------------------

def run_all() -> list[dict]:
    results = []
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))

    for mod_name, subdir, oracle_key, oracle_label in ALGORITHMS:
        print(f"\n{'='*60}")
        print(f"  Running {mod_name.upper()}")
        print(f"{'='*60}")

        try:
            mod = importlib.import_module(mod_name)
            # Re-import in case it was already imported in a previous run
            importlib.reload(mod)
            mod.main()
        except Exception as exc:
            print(f"  ERROR running {mod_name}: {exc}")
            results.append({"algorithm": mod_name, "error": str(exc)})
            continue

        diag = _load_diagnostics(subdir)
        idata = _load_idata(subdir)

        if diag is None:
            print(f"  WARNING: diagnostics.json not found for {mod_name}")
            results.append({"algorithm": mod_name, "error": "diagnostics.json missing"})
            continue

        ess = _arviz_bulk_ess(idata)
        wall = diag.get("wall_time_s", float("nan"))
        oracle = _oracle_evals(diag, oracle_key, oracle_label)
        mae = _mode_weight_mae(diag)
        recovered = _modes_recovered(diag)

        wall_core = diag.get("wall_time_core_s", wall)
        wall_total = diag.get("wall_time_total_s", wall)

        entry = {
            "algorithm": mod_name,
            "sampler": diag.get("sampler", mod_name),
            "wall_time_s": wall,
            "wall_time_core_s": wall_core,
            "wall_time_total_s": wall_total,
            "ess_bulk_arviz": ess,
            "ess_per_wall_s": ess / wall if wall > 0 else float("nan"),
            "ess_per_core_s": ess / wall_core if wall_core > 0 else float("nan"),
            "oracle_evals": oracle,
            "oracle_label": oracle_label,
            "ess_per_oracle_eval": (ess / oracle) if (oracle and oracle > 0) else float("nan"),
            "mode_weight_mae": mae,
            "modes_recovered": recovered,
            "num_modes": len(diag["true_mode_weights"]),
            "mode_weights": diag["mode_weights"],
            "true_mode_weights": diag["true_mode_weights"],
        }
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    header = (
        f"{'Algorithm':<10}  {'Core(s)':>8}  {'Total(s)':>9}  {'ESS':>8}  {'ESS/core-s':>10}  "
        f"{'Oracle':>10}  {'Type':>5}  {'ESS/oracle':>10}  "
        f"{'ModeMAE':>8}  {'Recovered':>9}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        if "error" in r:
            print(f"  {r['algorithm']:<10}  ERROR: {r['error']}")
            continue
        oracle_str = f"{r['oracle_evals']:>10,.0f}" if r["oracle_evals"] else f"{'N/A':>10}"
        ess_oracle_str = f"{r['ess_per_oracle_eval']:>10.4f}" if not np.isnan(r["ess_per_oracle_eval"]) else f"{'N/A':>10}"
        print(
            f"  {r['algorithm']:<10}  {r['wall_time_core_s']:>8.1f}  "
            f"{r['wall_time_total_s']:>9.1f}  "
            f"{r['ess_bulk_arviz']:>8.0f}  {r['ess_per_core_s']:>10.2f}  "
            f"{oracle_str}  {r['oracle_label']:>5}  {ess_oracle_str}  "
            f"{r['mode_weight_mae']:>8.4f}  {r['modes_recovered']:>4}/{r['num_modes']:<4}"
        )
    print(sep)
    print(
        "  Core(s): sampling loop only, JIT excluded.  "
        "Total(s): includes JIT compilation and all initialization overhead.\n"
        "  oracle type: 'logp' = log-density eval,  'grad' = gradient eval  "
        "(do not compare ESS/oracle across types)"
    )


def save_results(results: list[dict]) -> None:
    BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BENCHMARK_OUTPUT_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_path}")


def make_plots(results: list[dict]) -> None:
    good = [r for r in results if "error" not in r]
    if not good:
        print("No successful runs to plot.")
        return

    labels = [r["algorithm"].upper() for r in good]
    x = np.arange(len(labels))
    bar_w = 0.6

    BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: wall time (core vs total, grouped bars) ----
    bar_w2 = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    bars_core = ax.bar(x - bar_w2 / 2, [r["wall_time_core_s"] for r in good],
                       width=bar_w2, color="steelblue", alpha=0.85, label="Core (no JIT)")
    bars_total = ax.bar(x + bar_w2 / 2, [r["wall_time_total_s"] for r in good],
                        width=bar_w2, color="tomato", alpha=0.85, label="Total (with JIT)")
    ax.bar_label(bars_core, fmt="%.1f", padding=3, fontsize=7)
    ax.bar_label(bars_total, fmt="%.1f", padding=3, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Compute time per algorithm")
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(BENCHMARK_OUTPUT_DIR / "wall_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: mode weight MAE ----
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, [r["mode_weight_mae"] for r in good], width=bar_w, color="tomato", alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mode weight MAE")
    ax.set_title("Mode recovery accuracy (lower is better)")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(BENCHMARK_OUTPUT_DIR / "mode_weight_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: ESS per core-second ----
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, [r["ess_per_core_s"] for r in good], width=bar_w, color="seagreen", alpha=0.85)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ArviZ bulk ESS / core-second")
    ax.set_title("Sampling efficiency — core time only (higher is better)")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(BENCHMARK_OUTPUT_DIR / "ess_per_wall_second.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 4: Pareto scatter — core time vs. mode MAE (total time as secondary marker) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in good:
        ax.scatter(r["wall_time_core_s"], r["mode_weight_mae"], s=80, zorder=3, label=None)
        ax.scatter(r["wall_time_total_s"], r["mode_weight_mae"], s=40, marker="x",
                   zorder=3, alpha=0.6, label=None)
        ax.annotate(
            r["algorithm"].upper(),
            (r["wall_time_core_s"], r["mode_weight_mae"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="Core time"),
            Line2D([0], [0], marker="x", color="gray", markersize=8, label="Total time"),
        ],
        fontsize=8,
    )
    ax.set_xlabel("Wall-clock time (s)  [lower is better →]")
    ax.set_ylabel("Mode weight MAE  [lower is better ↓]")
    ax.set_title("Speed vs. accuracy (Pareto view; ● core, × total)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(BENCHMARK_OUTPUT_DIR / "pareto_time_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 5: mode weight recovery per algorithm (grouped bar) ----
    num_modes = good[0]["num_modes"]
    true_weights = np.array(good[0]["true_mode_weights"])
    n_algo = len(good)
    group_w = 0.8
    bar_per_mode = group_w / n_algo

    fig, ax = plt.subplots(figsize=(12, 5))
    mode_x = np.arange(num_modes)
    for i, r in enumerate(good):
        offsets = mode_x - group_w / 2 + (i + 0.5) * bar_per_mode
        ax.bar(offsets, r["mode_weights"], width=bar_per_mode * 0.9,
               label=r["algorithm"].upper(), alpha=0.8)
    ax.step(
        np.append(mode_x - 0.5, mode_x[-1] + 0.5),
        np.append(true_weights, true_weights[-1]),
        where="post", color="black", lw=1.5, linestyle="--", label="True weights",
    )
    ax.set_xticks(mode_x)
    ax.set_xticklabels([f"Mode {k}" for k in range(num_modes)])
    ax.set_ylabel("Weight")
    ax.set_title("Mode weight recovery (dashed = true weights)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(BENCHMARK_OUTPUT_DIR / "mode_weight_recovery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 5 benchmark plots to {BENCHMARK_OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_all()
    print_table(results)
    save_results(results)
    make_plots(results)
