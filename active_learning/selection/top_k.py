import jax.numpy as jnp


def select_top_k(scores: jnp.ndarray, k: int) -> jnp.ndarray:
    """Returns indices of the top-k highest-scoring pool elements."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > scores.shape[0]:
        raise ValueError(f"k={k} exceeds pool size {scores.shape[0]}")
    return jnp.argsort(scores)[-k:][::-1]


