import warnings

import jax
import jax.numpy as jnp
from jax import random
from jax import jit

import torch.utils.data as torch_data
from functools import partial
from typing import Tuple, Optional, Any


class BatchedDataset(torch_data.Dataset):
  """A dataset loader that returns random batches of data.

  This dataset takes raw data (inputs, targets, weights) and returns random batches
  of specified size. If batch_size is None or larger than dataset size, returns full dataset.

  Args:
      raw_data (tuple): Tuple of (inputs, targets, weights) arrays
      key (jax.random.PRNGKey): Random key for batch sampling
      batch_size (int, optional): Size of batches to return. Defaults to None (full dataset).
  """
  def __init__(
      self,
      raw_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
      key: jax.Array,
      batch_size: Optional[int] = None
    ) -> None:
    super().__init__()
    self.inputs = raw_data[0]
    self.targets = raw_data[1]
    self.weights = raw_data[2]
    assert len(self.inputs) == len(self.targets), f'inputs and targets must have the same length, but got {len(self.inputs)} and {len(self.targets)}'
    assert len(self.inputs) == len(self.weights), f'inputs and weights must have the same length, but got {len(self.inputs)} and {len(self.weights)}'
    self.size = len(self.weights)
    self.key = key
    if batch_size is None: # Will use full batch
      self.batch_size = self.size
    else:
      if batch_size > self.size:
        warnings.warn(
          'batch_size is greater than the dataset size; using full batch instead.',
          stacklevel=2,
        )
        self.batch_size = self.size
      else:
        self.batch_size = batch_size
    
  def __len__(self) -> int:
    return self.size
  
  def __getitem__(self, idx: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Get a random batch of data.

    Args:
        idx: Unused, required by Dataset interface

    Returns:
        tuple: (batch_inputs, batch_targets, batched_weights) arrays
    """
    self.key, subkey = random.split(self.key)
    batch_inputs, batch_targets, batched_weights = self.__select_batch(subkey)
    return batch_inputs, batch_targets, batched_weights

  @partial(jit, static_argnums=(0,))
  def __select_batch(self, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select a random batch using the given key.

    Args:
        key (jax.random.PRNGKey): Random key for batch selection

    Returns:
        tuple: (batch_inputs, batch_targets, batched_weights) arrays
    """
    idx = random.choice(key, self.size, (self.batch_size,), replace=False)
    batch_inputs = self.inputs[idx]
    batch_targets = self.targets[idx]
    batched_weights = self.weights[idx]
    return batch_inputs, batch_targets, batched_weights