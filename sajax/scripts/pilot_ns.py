"""
Pilot run: estimate likelihood evaluations per live point for JAXNS.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import ns

# Use moderate settings
ns.NUM_SLICES = 25
ns.DLOGZ_THRESHOLD = 5.0
ns.MAX_SAMPLES = 1e5

results = []
for num_live in [100, 250, 500, 1000]:
    ns.NUM_LIVE_POINTS = num_live
    for seed in range(5):
        diag = ns.main(seed=seed, save_outputs=False)
        total_evals = diag["total_likelihood_evals"]
        total_samples = diag["total_ns_samples"]
        nlp = 100
        evals_per_lp = total_evals / nlp
        evals_per_sample = diag["likelihood_evals_per_ns_sample"]
        print(f"  seed={seed}  total_evals={total_evals:>10,}  "
            f"ns_samples={total_samples:>6}  "
            f"evals/live_point={evals_per_lp:.0f}  "
            f"evals/sample={evals_per_sample:.1f}")
        results.append(evals_per_lp)

import numpy as np
print(f"\n  NS_EVALS_PER_LIVE_POINT recommendation:")
print(f"    mean   = {np.mean(results):.0f}")
print(f"    median = {np.median(results):.0f}")
print(f"    Use: {int(np.ceil(np.median(results)))}")