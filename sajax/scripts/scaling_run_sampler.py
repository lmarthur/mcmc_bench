"""
Run scaling study trials for a single algorithm and save results to JSON.

Each algorithm writes its own results_{algo}.json so multiple jobs can run
concurrently without filesystem conflicts.  The plot script merges them.

Cost models (LDE budget -> native effort parameter):

  RWMH       :  LDE = NUM_CHAINS   x (NUM_BURNIN  + NUM_SAMPLES)
  Affine Inv :  LDE = NUM_WALKERS  x NUM_SAMPLES  (burn-in is internal)
  SMC        :  LDE = EST_SMC_STEPS x NUM_PARTICLES x (NUM_MCMC_STEPS + 2)
  NS         :  LDE ≈ NUM_LIVE_POINTS x evals_per_live_point
  DEO / SEO  :  LDE = NUM_CHAINS  x (NUM_WARMUP  + NUM_SAMPLES)

Usage:
    python scaling_run_sampler.py --algorithm rwmh
    python scaling_run_sampler.py --algorithm smc --budgets 100000 500000 --trials 3
    python scaling_run_sampler.py --algorithm ns --force
"""

import argparse
import contextlib
import importlib
import io
import json
import logging
import sys
import time
import warnings
from pathlib import Path

from scaling_config import (
    ALL_ALGORITHMS, GROUND_TRUTH, LOGP_BUDGETS, NUM_TRIALS,
    SCALING_OUT_DIR, SCRIPTS_DIR,
    back_compute, extract_actual_oracle,
    compute_normalised_mae, compute_raw_mae, compute_per_param_normalised_error,
)


def results_path(algo: str) -> Path:
    return SCALING_OUT_DIR / f"results_{algo}.json"


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(algo: str, budget: int, seed: int) -> dict:
    bc = back_compute(algo, budget)
    effort_param = bc["param"]
    effort_value = bc["value"]
    fixed_params = bc["fixed"]

    base = {
        "algorithm":           algo,
        "budget_lde":          budget,
        "effort_param":        effort_param,
        "effort_value":        effort_value,
        "trial":               seed,
        "seed":                seed,
        "wall_time_s":         None,
        "normalised_mae":      None,
        "raw_mae":             None,
        "per_param_error":     None,
        "posterior_means":     None,
        "ground_truth":        None,
        "total_bulk_ess":      None,
        "actual_oracle_evals": None,
        "actual_oracle_type":  None,
        "error":               None,
    }
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        mod = importlib.import_module(algo)
        importlib.reload(mod)

        setattr(mod, effort_param, effort_value)
        for k, v in fixed_params.items():
            setattr(mod, k, v)

        noisy = ["arviz", "jaxns", "absl", "absl-py", "tensorflow",
                 "tensorflow_probability", "jax", "blackjax"]
        _loggers   = [logging.getLogger(n) for n in noisy] + [logging.root]
        _prev_lvls = [lg.level for lg in _loggers]
        for lg in _loggers:
            lg.setLevel(logging.CRITICAL)

        sink = io.StringIO()
        t_start = time.perf_counter()
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(sink), \
             contextlib.redirect_stderr(sink):
            warnings.simplefilter("ignore")
            diag = mod.main(seed=seed, save_outputs=False)
        t_total = time.perf_counter() - t_start

        for lg, lv in zip(_loggers, _prev_lvls):
            lg.setLevel(lv)

        post_means = diag.get("posterior_means", {})
        gt         = diag.get("ground_truth", GROUND_TRUTH)

        norm_mae  = compute_normalised_mae(post_means, gt)
        raw_mae   = compute_raw_mae(post_means, gt)
        per_param = compute_per_param_normalised_error(post_means, gt)
        bulk_ess  = diag.get("total_bulk_ess")

        actual_evals, oracle_type = extract_actual_oracle(algo, diag)
        wall_time = diag.get("wall_time_s", t_total)

        base.update({
            "wall_time_s":         float(wall_time),
            "wall_time_total_s":   float(t_total),
            "normalised_mae":      norm_mae,
            "raw_mae":             raw_mae,
            "per_param_error":     per_param,
            "posterior_means":     {k: float(v) for k, v in post_means.items()},
            "ground_truth":        {k: float(v) for k, v in gt.items()},
            "total_bulk_ess":      float(bulk_ess) if bulk_ess is not None else None,
            "actual_oracle_evals": actual_evals,
            "actual_oracle_type":  oracle_type,
        })
    except Exception as exc:
        import traceback
        base["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return base


# ---------------------------------------------------------------------------
# Batch runner with incremental persistence
# ---------------------------------------------------------------------------

def run_algo(algo: str, budgets: list, num_trials: int, force: bool = False) -> list:
    out_path = results_path(algo)

    if out_path.exists() and not force:
        print(f"Results file found: {out_path}")
        print("Loading existing results. Use --force to re-run.")
        with open(out_path) as f:
            return json.load(f)

    _print_effort_table(algo, budgets)

    total   = len(budgets) * num_trials
    records = []
    n       = 0

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for budget in budgets:
        bc = back_compute(algo, budget)
        for trial in range(num_trials):
            n += 1
            print(
                f"[{n:3d}/{total}]  {algo.upper():<8}  "
                f"budget={budget:<10,}  {bc['param']}={bc['value']:<8}  "
                f"trial={trial}",
                flush=True,
            )
            rec = run_trial(algo, budget, seed=trial)
            if rec["error"]:
                print(f"  !! FAILED: {rec['error'][:200]}")
            else:
                print(
                    f"  -> norm_MAE={rec['normalised_mae']:.4f}  "
                    f"wall={rec['wall_time_s']:.1f}s  "
                    f"oracle={rec['actual_oracle_evals']}"
                )
            records.append(rec)

            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)

    n_failed = sum(1 for r in records if r["error"])
    print(f"\nCompleted {len(records)} records ({n_failed} failures) -> {out_path}")
    return records


def _print_effort_table(algo: str, budgets: list) -> None:
    bc0   = back_compute(algo, budgets[0])
    label = f"{algo.upper()}({bc0['param']})"
    print(f"\n{'='*60}")
    print(f"  Budget -> effort: {label}")
    print(f"{'='*60}")
    for budget in budgets:
        bc = back_compute(algo, budget)
        print(f"  {budget:>14,}  ->  {bc['value']:>10,}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run scaling study trials for a single algorithm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--algorithm", required=True, choices=ALL_ALGORITHMS,
        help="Algorithm to run",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if results_{algo}.json already exists",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=None,
        help="Override budget levels (space-separated integers)",
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override number of trials per budget",
    )
    args = parser.parse_args()

    budgets    = sorted(args.budgets) if args.budgets else LOGP_BUDGETS
    num_trials = args.trials if args.trials else NUM_TRIALS

    run_algo(args.algorithm, budgets, num_trials, force=args.force)


if __name__ == "__main__":
    main()
