"""Pool-based active-learning scenario helpers."""

import jax
import jax.numpy as jnp
from jax import random

from active_learning.state import ActiveLearningState


def initialize_pool_state(
    key: jax.Array,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    initial_sample_count: int,
) -> ActiveLearningState:
    """Randomly splits a dataset into train/pool indices over a fixed dataset."""
    total_sample_count = inputs.shape[0]
    if initial_sample_count <= 0 or initial_sample_count >= total_sample_count:
        raise ValueError(
            f"initial_sample_count must be in [1, {total_sample_count - 1}], "
            f"got {initial_sample_count}"
        )

    permutation = random.permutation(key, total_sample_count)
    train_indices = permutation[:initial_sample_count]
    pool_indices = permutation[initial_sample_count:]

    return ActiveLearningState(
        inputs=inputs,
        targets=targets,
        train_indices=train_indices,
        pool_indices=pool_indices,
        key=key,
    )


def move_from_pool_to_train(
    state: ActiveLearningState,
    selected_indices: jnp.ndarray,
) -> ActiveLearningState:
    """Moves selected pool indices into the labeled training index set."""
    pool_size = state.pool_indices.shape[0]
    mask = jnp.ones((pool_size,), dtype=bool).at[selected_indices].set(False)
    selected_pool_indices = state.pool_indices[selected_indices]

    return ActiveLearningState(
        inputs=state.inputs,
        targets=state.targets,
        train_indices=jnp.concatenate([state.train_indices, selected_pool_indices], axis=0),
        pool_indices=state.pool_indices[mask],
        key=state.key,
        round_idx=state.round_idx + 1,
        history=list(state.history),
        synthetic_inputs=state.synthetic_inputs,
        synthetic_targets=state.synthetic_targets,
    )
