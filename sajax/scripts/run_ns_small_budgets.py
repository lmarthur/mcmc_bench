"""
Run NS trials for the two small budget points (50_000 and 100_000 LDE)
that are missing from results_ns.json, and save to results_ns_small.json.

Results are written incrementally so the script is safe to interrupt and restart.

Usage:
    python run_ns_small_budgets.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scaling_config import SCALING_OUT_DIR, NUM_TRIALS
from scaling_run_sampler import run_trial

ALGO       = "ns"
NEW_BUDGETS = [50_000, 100_000]
OUT_PATH   = SCALING_OUT_DIR / f"results_{ALGO}_small.json"


def main():
    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else []
    done    = {(r["budget_lde"], r["trial"]) for r in records}

    pending = [
        (budget, trial)
        for budget in NEW_BUDGETS
        for trial in range(NUM_TRIALS)
        if (budget, trial) not in done
    ]

    if not pending:
        print(f"All trials already complete in {OUT_PATH}")
        return

    total = len(pending)
    for n, (budget, trial) in enumerate(pending, 1):
        print(
            f"[{n:3d}/{total}]  {ALGO.upper():<8}  "
            f"budget={budget:<10,}  trial={trial}",
            flush=True,
        )
        rec = run_trial(ALGO, budget, seed=trial)
        if rec["error"]:
            print(f"  !! FAILED: {rec['error'][:200]}")
        else:
            print(
                f"  -> norm_MAE={rec['normalised_mae']:.4f}  "
                f"wall={rec['wall_time_s']:.1f}s  "
                f"oracle={rec['actual_oracle_evals']}"
            )
        records.append(rec)
        OUT_PATH.write_text(json.dumps(records, indent=2))

    n_failed = sum(1 for r in records if r["error"])
    print(f"\nDone: {len(records)} records ({n_failed} failures) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
