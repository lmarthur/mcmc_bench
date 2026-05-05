import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import time

from model import make_log_density, PRIOR_LOW, PRIOR_HIGH

log_density_fn = make_log_density()
grad_fn = jax.grad(log_density_fn)

# Compile both
x_test = jnp.array([0.0, 0.0])
_ = jax.jit(log_density_fn)(x_test)
_ = jax.jit(grad_fn)(x_test)

# Benchmark: many evaluations to amortize dispatch overhead
N = 100_000
keys = jax.random.split(jax.random.PRNGKey(0), N)
xs = jax.random.uniform(keys[0], shape=(N, 2), minval=PRIOR_LOW, maxval=PRIOR_HIGH)

# Vectorized logp timing
vmap_logp = jax.jit(jax.vmap(log_density_fn))
_ = vmap_logp(xs)

t0 = time.perf_counter()
for _ in range(10):
    result = vmap_logp(xs)
    jax.block_until_ready(result)
t_logp = (time.perf_counter() - t0) / 10

# Vectorized grad timing
vmap_grad = jax.jit(jax.vmap(grad_fn))
_ = vmap_grad(xs)

t0 = time.perf_counter()
for _ in range(10):
    result = vmap_grad(xs)
    jax.block_until_ready(result)
t_grad = (time.perf_counter() - t0) / 10

ratio = t_grad / t_logp

print(f"=== Grad-to-Logp Ratio ===")
print(f" logp time ({N} evals): {t_logp*1000:.2f} ms")
print(f"  grad time ({N} evals): {t_grad*1000:.2f} ms")
print(f"  ratio (grad/logp):    {ratio:.2f}")
print(f"")
print(f"  Set GRAD_TO_LOGP_RATIO = {ratio:.1f}")
