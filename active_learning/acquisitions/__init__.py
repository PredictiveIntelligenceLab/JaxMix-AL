from typing import Callable, Dict

from active_learning.acquisitions.bait import bait_fisher_embeddings
from active_learning.acquisitions.coreset import coreset_features
from active_learning.acquisitions.mi_lb import mi_lb_acquisition
from active_learning.acquisitions.mdn_variance import mdn_epistemic_variance_acquisition
from active_learning.acquisitions.random import random_acquisition

ACQUISITION_REGISTRY: Dict[str, Callable] = {
    "random": random_acquisition,
    "mdn_epistemic_variance": mdn_epistemic_variance_acquisition,
    "mi_lb": mi_lb_acquisition,
    # ``bait`` does not produce per-point scalar scores; it returns Fisher
    # embeddings and is consumed exclusively via ``selection_strategy="bait"``
    # in the active-learning loop.  We register the embedding function here
    # so callers can resolve it by name in the same registry.
    "bait": bait_fisher_embeddings,
    # ``coreset`` returns backbone features (B, h) for k-Center-Greedy
    # (Sener & Savarese 2018); used via ``selection_strategy="coreset"``.
    "coreset": coreset_features,
}


def get_acquisition_fn(name: str) -> Callable:
    """Looks up an acquisition scorer by name."""
    if name not in ACQUISITION_REGISTRY:
        raise ValueError(
            f"Unknown acquisition '{name}'. "
            f"Available acquisitions: {sorted(ACQUISITION_REGISTRY)}"
        )
    return ACQUISITION_REGISTRY[name]


__all__ = [
    "ACQUISITION_REGISTRY",
    "bait_fisher_embeddings",
    "coreset_features",
    "mi_lb_acquisition",
    "get_acquisition_fn",
    "mdn_epistemic_variance_acquisition",
    "random_acquisition",
]
