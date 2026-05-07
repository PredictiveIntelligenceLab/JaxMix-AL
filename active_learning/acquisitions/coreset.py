"""Backbone-feature extractor for Core-Set (Sener & Savarese 2018)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from active_learning.acquisitions.bait import _backbone_features


def coreset_features(
    model,
    inputs: jnp.ndarray,
    key: jax.Array,
    ensemble_member: int = 0,
) -> jnp.ndarray:
    """Return one ensemble member's backbone features, shape ``(B, h)``.

    Mirrors the paper's "activations of the final fully-connected layer"
    (§4.4): the shared backbone output before the MDN heads.
    """
    del key  # deterministic; signature kept consistent with other acquisitions
    feats = _backbone_features(model, inputs)  # (E, B, h)
    return feats[int(ensemble_member)]
