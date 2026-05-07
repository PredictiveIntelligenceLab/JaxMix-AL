from dataclasses import dataclass, field
from typing import Any, List, Optional

import jax
import jax.numpy as jnp


@dataclass
class ActiveLearningState:
    """Tracks active-learning splits as indices into a fixed dataset.

    Supports both pool-based (train/pool indices) and membership-query synthesis
    (synthetic_inputs/synthetic_targets appended to the effective training set).
    """

    inputs: jnp.ndarray
    targets: jnp.ndarray
    train_indices: jnp.ndarray
    pool_indices: jnp.ndarray
    key: jax.Array
    round_idx: int = 0
    history: List[Any] = field(default_factory=list)
    # Membership query synthesis: these are appended to the labeled set for training.
    synthetic_inputs: Optional[jnp.ndarray] = None
    synthetic_targets: Optional[jnp.ndarray] = None

    @property
    def train_inputs(self) -> jnp.ndarray:
        from_pool = self.inputs[self.train_indices]
        if self.synthetic_inputs is not None and self.synthetic_inputs.size > 0:
            return jnp.concatenate([from_pool, self.synthetic_inputs], axis=0)
        return from_pool

    @property
    def train_targets(self) -> jnp.ndarray:
        from_pool = self.targets[self.train_indices]
        if self.synthetic_targets is not None and self.synthetic_targets.size > 0:
            return jnp.concatenate([from_pool, self.synthetic_targets], axis=0)
        return from_pool

    @property
    def pool_inputs(self) -> jnp.ndarray:
        return self.inputs[self.pool_indices]

    @property
    def pool_targets(self) -> jnp.ndarray:
        return self.targets[self.pool_indices]
