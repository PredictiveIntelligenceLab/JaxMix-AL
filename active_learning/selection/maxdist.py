import jax.numpy as jnp

def select_maxdist(
    scores: jnp.ndarray,
    features: jnp.ndarray,
    k: int,
    train_features: jnp.ndarray = None,
    score_weight: float = 1.0,
) -> jnp.ndarray:
    """MaxDist: acquisition-weighted farthest-point sampling, training-point aware.

    Implements a score-weighted variant of the MaxDist / greedy core-set
    approach (LCMD-TP from Holzmuller et al., bmdal_reg):

    1. Initialize minimum distances using existing training points (TP-mode).
    2. At each step, select the point maximising:
           min_dist[i] * (1 + score_weight * norm_score[i])
       This balances spatial diversity with acquisition-function signal.
    3. Update minimum distances after each selection.

    When ``score_weight=0`` this reduces to pure farthest-point sampling.

    Args:
        scores: Acquisition scores for pool, shape (pool_size,).
        features: Feature vectors for the pool, shape (pool_size, feature_dim).
        k: Number of points to select.
        train_features: Feature vectors for existing training points,
            shape (train_size, feature_dim).
        score_weight: How strongly acquisition scores influence selection.
            Higher values bias selection towards high-scoring (uncertain) points.

    Returns:
        Indices into the pool array, shape (k,).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = features.shape[0]
    if k > n:
        raise ValueError(f"k={k} exceeds pool size {n}")

    # Normalize features so each dimension contributes comparably
    feat_std = jnp.maximum(features.std(axis=0), 1e-8)
    normed_features = features / feat_std[None, :]

    # Normalize acquisition scores to [0, 1] for stable weighting
    score_min = scores.min()
    score_range = jnp.maximum(scores.max() - score_min, 1e-8)
    norm_scores = (scores - score_min) / score_range  # [0, 1]

    # TP-mode: initialise min distances from existing training points
    min_dists = jnp.full(n, jnp.inf)
    if train_features is not None and train_features.shape[0] > 0:
        normed_train = train_features / feat_std[None, :]
        for i in range(normed_train.shape[0]):
            dists = jnp.sum((normed_features - normed_train[i][None, :]) ** 2, axis=-1)
            min_dists = jnp.minimum(min_dists, dists)

    # Seed: pick the point with the highest score among those far from training
    if train_features is not None and train_features.shape[0] > 0:
        # Among top-50% farthest points, pick the one with highest score
        dist_threshold = jnp.percentile(min_dists, 50.0)
        seed_scores = jnp.where(min_dists >= dist_threshold, scores, -jnp.inf)
        seed_idx = int(jnp.argmax(seed_scores))
    else:
        seed_idx = int(jnp.argmax(scores))

    selected = [seed_idx]

    # Update min_dists with seed point
    dists = jnp.sum((normed_features - normed_features[seed_idx][None, :]) ** 2, axis=-1)
    min_dists = jnp.minimum(min_dists, dists)
    min_dists = min_dists.at[seed_idx].set(-1.0)

    # Greedy acquisition-weighted farthest-point selection
    for _ in range(k - 1):
        # Weight distances by acquisition scores
        weighted = min_dists * (1.0 + score_weight * norm_scores)
        next_idx = int(jnp.argmax(weighted))
        selected.append(next_idx)

        dists = jnp.sum((normed_features - normed_features[next_idx][None, :]) ** 2, axis=-1)
        min_dists = jnp.minimum(min_dists, dists)
        min_dists = min_dists.at[jnp.array(selected)].set(-1.0)

    return jnp.array(selected)
