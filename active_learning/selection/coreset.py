"""k-Center-Greedy core-set selection (Sener & Savarese, ICLR 2018)."""
from __future__ import annotations

import jax.numpy as jnp


def select_kcenter_greedy(
    pool_features: jnp.ndarray,
    train_features: jnp.ndarray,
    k: int,
) -> jnp.ndarray:
    """k-Center-Greedy (Sener & Savarese 2018, Algorithm 1)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n_pool = pool_features.shape[0]
    if k > n_pool:
        raise ValueError(f"k={k} exceeds pool size {n_pool}")
    if pool_features.ndim != 2:
        raise ValueError(
            f"pool_features must be 2D (n_pool, dim), got shape {pool_features.shape}"
        )
    if train_features.shape[0] > 0 and train_features.shape[1] != pool_features.shape[1]:
        raise ValueError(
            f"train_features dim {train_features.shape[1]} != pool_features dim "
            f"{pool_features.shape[1]}"
        )

    min_dists = jnp.full((n_pool,), jnp.inf)

    if train_features.shape[0] > 0:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b can go slightly negative for
        # near-duplicate vectors due to fp cancellation; clamp to >= 0 so the
        # invariant "min_dists >= 0 for unselected" stays exact.
        d2_to_train = (
            jnp.sum(pool_features ** 2, axis=1, keepdims=True)
            + jnp.sum(train_features ** 2, axis=1)[None, :]
            - 2.0 * pool_features @ train_features.T
        )
        d2_to_train = jnp.maximum(d2_to_train, 0.0)
        min_dists = jnp.min(d2_to_train, axis=1)

    selected: list[int] = []

    if train_features.shape[0] == 0:
        # Cold start: paper assumes s^0 ≠ ∅; seed at the point farthest
        # from the pool centroid to start the recurrence deterministically.
        centroid = pool_features.mean(axis=0, keepdims=True)
        d2_to_centroid = jnp.maximum(
            jnp.sum((pool_features - centroid) ** 2, axis=1), 0.0
        )
        seed = int(jnp.argmax(d2_to_centroid))
        selected.append(seed)
        d2_to_seed = jnp.maximum(
            jnp.sum((pool_features - pool_features[seed]) ** 2, axis=1), 0.0
        )
        min_dists = jnp.minimum(min_dists, d2_to_seed)
        min_dists = min_dists.at[seed].set(-1.0)

    while len(selected) < k:
        next_idx = int(jnp.argmax(min_dists))
        selected.append(next_idx)
        d2 = jnp.maximum(
            jnp.sum((pool_features - pool_features[next_idx]) ** 2, axis=1), 0.0
        )
        min_dists = jnp.minimum(min_dists, d2)
        min_dists = min_dists.at[next_idx].set(-1.0)

    return jnp.asarray(selected, dtype=jnp.int32)
