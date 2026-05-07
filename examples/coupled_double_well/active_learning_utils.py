"""Active learning utilities for the coupled double-well benchmark."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jax import random
import matplotlib.pyplot as plt
import numpy as np

_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT   = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR)

from examples.common_active_learning import (
    get_default_mdn_config,
    make_mdn_trainer_factory,
    plot_loss_comparison,
    run_pool_based_active_learning_experiment,
)

from coupled_double_well import generate_dataset, make_oracle


# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------

def create_shared_double_well_data(
    seed: int = 0,
    candidate_sample_count: int = 50_000,
    test_sample_count: int = 2_000,
    initial_sample_count: int = 100,
    P: int = 5,
    T: float = 5.0,
    dt: float = 0.005,
    n_snapshots: int = 0,
    sigma_range: Tuple[float, float] = (0.3, 2.0),
    kappa_range: Tuple[float, float] = (0.0, 3.0),
) -> Dict[str, Any]:
    """
    Build the shared benchmark dataset for the coupled double-well problem.

    Returns
    -------
    dict with keys:
      ``initial_labeled_inputs``  (n_init, P+2)
      ``initial_labeled_targets`` (n_init, P) or (n_init, n_snapshots*P)
      ``remaining_pool_inputs``   (n_pool, P+2)
      ``remaining_pool_targets``  (n_pool, ...)
      ``test_data``               tuple of (test_x, test_y)
      ``shared_state_key``        JAX PRNGKey
      ``oracle_fn``               callable (x, key) -> y
    """
    key = random.PRNGKey(seed)
    k_pool, k_test, k_perm, k_state = random.split(key, 4)

    print(f"Generating {candidate_sample_count} pool samples...")
    x_all, y_all = generate_dataset(
        k_pool, candidate_sample_count, P=P, T=T, dt=dt,
        n_snapshots=n_snapshots, sigma_range=sigma_range,
        kappa_range=kappa_range,
    )

    print(f"Generating {test_sample_count} test samples...")
    x_test, y_test = generate_dataset(
        k_test, test_sample_count, P=P, T=T, dt=dt,
        n_snapshots=n_snapshots, sigma_range=sigma_range,
        kappa_range=kappa_range,
    )

    # Random permutation for train/pool split
    perm = random.permutation(k_perm, candidate_sample_count)
    x_all = x_all[perm]
    y_all = y_all[perm]

    init_x = x_all[:initial_sample_count]
    init_y = y_all[:initial_sample_count]
    pool_x = x_all[initial_sample_count:]
    pool_y = y_all[initial_sample_count:]

    oracle_fn = make_oracle(P=P, T=T, dt=dt, n_snapshots=n_snapshots)

    return {
        "initial_labeled_inputs":  init_x,
        "initial_labeled_targets": init_y,
        "remaining_pool_inputs":   pool_x,
        "remaining_pool_targets":  pool_y,
        "test_data":               (x_test, y_test),
        "shared_state_key":        k_state,
        "oracle_fn":               oracle_fn,
    }


def plot_sigma_concentration(results: Dict[str, Dict], P: int = 5,
                             transition_zone: Tuple[float, float] = (0.5, 1.2)) -> plt.Figure:
    """Bar chart: fraction of acquired points in the transition zone."""
    fig, ax = plt.subplots(figsize=(6, 4))
    lo, hi = transition_zone
    names, fracs = [], []
    for label, r in results.items():
        state = r["state"]
        x_labeled = np.asarray(state.train_inputs)
        sigma_vals = x_labeled[:, P]  # sigma is the (P+1)-th column
        frac = float(np.mean((sigma_vals >= lo) & (sigma_vals <= hi)))
        names.append(label)
        fracs.append(frac)

    # Expected fraction under uniform sampling
    sigma_range = (0.3, 2.0)
    uniform_frac = (min(hi, sigma_range[1]) - max(lo, sigma_range[0])) / (sigma_range[1] - sigma_range[0])

    ax.bar(names, fracs, alpha=0.7, color="steelblue")
    ax.axhline(uniform_frac, color="red", ls="--", lw=1.5,
               label=f"Uniform ({uniform_frac:.1%})")
    ax.set_ylabel(f"Fraction in σ ∈ [{lo}, {hi}]")
    ax.set_title("Query concentration near phase boundary")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
    return fig


def make_single_run_figure(result: Dict, P: int = 5, title: str = "") -> plt.Figure:
    """NLL curve + (sigma, kappa) scatter + output histogram.

    Returns the figure without calling ``plt.show()`` so it can be used
    for headless logging (e.g. wandb via ``run_experiments.py``).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    state   = result["state"]
    history = result["history"]
    model   = result["model"]
    test_inputs, test_targets = result["test_data"]

    # Panel 1: NLL learning curve
    ax = axes[0]
    rounds = [h.round_idx for h in history]
    nlls   = [h.test_loss  for h in history]
    ax.plot(rounds, nlls, marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test NLL")
    if min(nlls) > 0:
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Learning curve")
    ax.grid(True, alpha=0.3)

    # Panel 2: labeled points in (sigma, kappa) plane
    ax = axes[1]
    x_labeled = np.asarray(state.train_inputs)
    ax.scatter(x_labeled[:, P], x_labeled[:, P + 1], s=4, alpha=0.5)
    ax.set_xlabel("$\\sigma$")
    ax.set_ylabel("$\\kappa$")
    ax.set_title(f"Labeled points (n={len(x_labeled)})")
    ax.grid(True, alpha=0.3)

    # Panel 3: histogram of test output q_0 at final snapshot — true vs MDN sample
    ax = axes[2]
    pred = model.apply(model.params, test_inputs)
    logit_weights, means, variances = pred
    samples = model.sample_from_mixture(
        random.PRNGKey(0), logit_weights, means, variances,
    )
    y_net0 = np.asarray(samples[0]).squeeze(1)  # (N, out_dim)
    # Last particle-0 index: for snapshots layout [snap0_p0..snap0_pP, snap1_p0..],
    # final snapshot particle 0 is at index (n_snaps-1)*P if snapshots, else 0
    out_dim = test_targets.shape[1]
    last_p0_idx = out_dim - P  # works for both snapshot and no-snapshot cases
    ax.hist(np.asarray(test_targets[:, last_p0_idx]), bins=60, density=True, alpha=0.5,
            color="steelblue", label="True $q_0(T)$")
    ax.hist(y_net0[:, last_p0_idx], bins=60, density=True, alpha=0.5,
            color="tomato", label="MDN $q_0(T)$")
    ax.set_xlabel("$q_0(T)$")
    ax.set_ylabel("Density")
    ax.set_title("Output marginal (particle 0, final snapshot)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_single_run(result: Dict, P: int = 5, title: str = "") -> None:
    """Show a single-run figure interactively."""
    make_single_run_figure(result, P=P, title=title)
    plt.show()
