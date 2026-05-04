"""
Pilot run: estimate typical number of SMC tempering steps.
Run this a few times with different seeds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import smc

# Use moderate particle count — the number of tempering steps is roughly
# independent of NUM_PARTICLES (it depends on TARGET_ESS and the
# prior-to-posterior "distance" in KL sense).
smc.NUM_MCMC_STEPS = 25
smc.TARGET_ESS = 0.75

results = []
for num_partic in [200, 400, 600, 800, 1000, 2000]:
    smc.NUM_PARTICLES = num_partic
    for seed in range(5):
        diag = smc.main(seed=seed, save_outputs=False)
        n_steps = diag["num_smc_steps"]
        final_lam = diag["final_tempering_param"]
        total_evals = diag["total_log_density_evals"]
        print(f"  seed={seed}  steps={n_steps:>4}  final_λ={final_lam:.4f}  "
            f"evals={total_evals:>10,}")
        results.append(n_steps)

import numpy as np
print(f"\n  SMC_EST_NUM_STEPS recommendation:")
print(f"    mean   = {np.mean(results):.1f}")
print(f"    median = {np.median(results):.1f}")
print(f"    max    = {np.max(results)}")
print(f"    Use: {int(np.ceil(np.median(results)))}")