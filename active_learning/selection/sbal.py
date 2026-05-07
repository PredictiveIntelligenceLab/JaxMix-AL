import jax
import jax.numpy as jnp


def select_sbal(
    scores: jnp.ndarray,
    k: int,
    key: jax.Array,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Stochastic Batch Active Learning (SBAL) — Softmax variant.

    Implements the stochastic batch acquisition method from Kirsch et al.
    (2022, "Stochastic Batch Acquisition: A Simple Baseline for Deep Active
    Learning").  Uses the Gumbel-Top-K trick to sample k points without
    replacement from p(i) proportional to exp(score_i / temperature).

    Args:
        scores: Acquisition scores for pool, shape (pool_size,).
        k: Number of points to select.
        key: PRNG key for Gumbel noise.
        temperature: Inverse coldness (1/beta). Lower -> more exploitation
            (approaches top-k); higher -> more exploration (approaches uniform).

    Returns:
        Indices into the pool array, shape (k,).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = scores.shape[0]
    if k > n:
        raise ValueError(f"k={k} exceeds pool size {n}")

    # Shift scores for numerical stability before temperature scaling
    logits = (scores - scores.max()) / temperature
    # Use Gumbel-top-k trick for exact sampling without replacement
    gumbel_noise = jax.random.gumbel(key, shape=(n,))
    perturbed = logits + gumbel_noise
    return jnp.argsort(perturbed)[-k:][::-1]
