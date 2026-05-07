from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class ActiveLearningRoundSummary:
    """Stores a compact summary for a single active-learning round."""

    round_idx: int
    acquisition_name: str
    train_size: int
    pool_size: int
    test_loss: float
    mean_acquisition_score: float
    max_acquisition_score: float
    selected_indices: jnp.ndarray
    selected_scores: jnp.ndarray
    selected_inputs: jnp.ndarray
    final_round: bool = False
