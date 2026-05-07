"""
Active learning utilities for the multimodal-conditional synthetic benchmark.

What makes this example unique
-------------------------------
1. **Synthetic oracle** — data is drawn from a known distribution p*(y|x), so
   we can evaluate the true NLL = -E[log p*(y|x)] as an absolute reference.
   The gap between ensemble NLL and true NLL measures model miscalibration.

2. **High-dimensional I/O with manifold structure** — inputs x ∈ R^16 live on
   a 4-dimensional manifold; outputs y ∈ R^16.  There are no time series.

3. **Controllable multimodality** — the distribution has K=3 Gaussian
   components whose weights vary across the manifold.  Some regions of x-space
   are unimodal (one component dominates), others are genuinely multimodal.
   Active learning should preferentially query the multimodal regions.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jax import random
import matplotlib.pyplot as plt
import numpy as np

_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT    = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR)

from examples.common_active_learning import (
    evaluate_ensemble_nll,
    get_default_mdn_config,
    make_mdn_trainer_factory,
    plot_loss_comparison,
    run_pool_based_active_learning_experiment,
)

from multimodal_conditional import build_distribution, compute_true_nll, generate_dataset


# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------

def create_shared_multimodal_data(
    seed: int = 0,
    candidate_sample_count: int = 5_000,
    test_sample_count: int = 1_000,
    initial_sample_count: int = 100,
    d: int = 16,
    m: int = 16,
    L: int = 4,
    K: int = 3,
    p: int = 128,
    tau: float = 1.0,
    alpha: float = 0.0,
    c_scale: float = 3.0,
    dist_seed: int = 42,
    manifold_seed: int = 1,
    mixing_mode: str = "random",
    transition_sharpness: float = 8.0,
    transition_radius: float = 1.0,
    angular_sharpness: float = 5.0,
    logit_scale: float = 5.0,
) -> Dict[str, Any]:
    """
    Build the shared benchmark dataset for the multimodal-conditional problem.

    Separation of concerns
    ~~~~~~~~~~~~~~~~~~~~~~
    - ``dist_seed``     : fixes the distribution parameters (reproducible oracle)
    - ``manifold_seed`` : fixes the manifold geometry (all splits share the same surface)
    - ``seed``          : controls the random train / pool / test split

    All three should be fixed for fair multi-seed comparisons:  change only
    ``seed`` between seeds.

    Parameters
    ----------
    seed                  : Random seed for the data split
    candidate_sample_count: Total pool size (initial + remaining unlabelled)
    test_sample_count     : Number of held-out test points
    initial_sample_count  : Points labelled before AL round 0
    d, m, L, K, p         : Distribution dimensions (must match trainer config)
    tau, alpha, c_scale   : Distribution hyperparameters
    dist_seed             : Seed for p*(y|x) parameters (fixed across seeds)
    manifold_seed         : Seed for the manifold map   (fixed across seeds)
    mixing_mode           : ``"random"`` or ``"structured"``
    transition_sharpness  : (structured only) beta
    transition_radius     : (structured only) r0
    angular_sharpness     : (structured only) gamma
    logit_scale           : (structured only) logit magnitude

    Returns
    -------
    dict with keys:
      ``initial_labeled_inputs``  (n_init, d)
      ``initial_labeled_targets`` (n_init, m)
      ``remaining_pool_inputs``   (n_pool, d)
      ``remaining_pool_targets``  (n_pool, m)
      ``test_data``               tuple ((n_test, d), (n_test, m))
      ``shared_state_key``        JAX PRNGKey for the AL state
      ``generate_samples``        the oracle sampler
      ``log_prob``                the oracle log-density
      ``true_nll``                -E[log p*(y|x)] on the test set
    """
    generate_samples, log_prob = build_distribution(
        d=d, m=m, L=L, K=K, p=p,
        tau=tau, alpha=alpha, c_scale=c_scale,
        seed=dist_seed,
        mixing_mode=mixing_mode,
        transition_sharpness=transition_sharpness,
        transition_radius=transition_radius,
        angular_sharpness=angular_sharpness,
        logit_scale=logit_scale,
    )

    key = random.PRNGKey(seed)
    key, data_key, perm_key, state_key = random.split(key, 4)

    total   = candidate_sample_count + test_sample_count
    x_all, y_all = generate_dataset(
        total, generate_samples,
        key=data_key, d=d, L=L, manifold_seed=manifold_seed,
    )

    perm     = random.permutation(perm_key, total)
    test_idx = perm[:test_sample_count]
    cand_idx = perm[test_sample_count:]

    cand_x = x_all[cand_idx]
    cand_y = y_all[cand_idx]
    init_x = cand_x[:initial_sample_count]
    init_y = cand_y[:initial_sample_count]
    pool_x = cand_x[initial_sample_count:]
    pool_y = cand_y[initial_sample_count:]

    test_x = x_all[test_idx]
    test_y = y_all[test_idx]
    true_nll = compute_true_nll(log_prob, test_x, test_y)

    return {
        "initial_labeled_inputs":  init_x,
        "initial_labeled_targets": init_y,
        "remaining_pool_inputs":   pool_x,
        "remaining_pool_targets":  pool_y,
        "test_data":               (test_x, test_y),
        "shared_state_key":        state_key,
        "generate_samples":        generate_samples,
        "log_prob":                log_prob,
        "true_nll":                true_nll,
    }


def plot_nll_gap(results: Dict[str, Dict]) -> plt.Figure:
    """Build and return a figure of the calibration gap Δ = ensemble_NLL − true_NLL."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, r in results.items():
        true_nll = r.get("true_nll")
        if true_nll is None:
            continue
        history = r["history"]
        rounds  = [h.round_idx for h in history]
        gaps    = [h.test_loss - true_nll for h in history]
        ax.plot(rounds, gaps, marker="o", label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="Δ = 0")
    ax.set_xlabel("Round")
    ax.set_ylabel("NLL gap  (ensemble − true)")
    ax.set_title("Calibration gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig

def plot_radial_concentration(
    results: Dict[str, Dict],
    data: Dict[str, Any],
    L: int = 4,
    transition_radius: float = 1.3,
    transition_width: float = 0.3,
) -> plt.Figure:
    """Bar chart: fraction of acquired points near the radial phase boundary.

    Analog of the coupled-double-well ``plot_sigma_concentration`` for this
    example, where the unimodal-to-multimodal boundary sits at
    ||x[:L]|| ≈ *transition_radius*.

    Bars above the red line mean the method over-samples the boundary
    (the information frontier).  Random should sit near the line.
    """
    lo = transition_radius - transition_width
    hi = transition_radius + transition_width

    fig, ax = plt.subplots(figsize=(6, 4))
    names, fracs = [], []

    for label, r in results.items():
        state = r["state"]
        x_labeled = np.asarray(state.train_inputs)
        r_vals = np.linalg.norm(x_labeled[:, :L], axis=1)
        frac = float(np.mean((r_vals >= lo) & (r_vals <= hi)))
        names.append(label)
        fracs.append(frac)

    # Estimate uniform fraction from the full pool
    pool_x = np.asarray(data["remaining_pool_inputs"])
    init_x = np.asarray(data["initial_labeled_inputs"])
    all_x = np.concatenate([init_x, pool_x], axis=0)
    r_all = np.linalg.norm(all_x[:, :L], axis=1)
    uniform_frac = float(np.mean((r_all >= lo) & (r_all <= hi)))

    ax.bar(names, fracs, alpha=0.7, color="steelblue")
    ax.axhline(uniform_frac, color="red", ls="--", lw=1.5,
               label=f"Uniform ({uniform_frac:.1%})")
    ax.set_ylabel(f"Fraction in ||x[:L]|| ∈ [{lo:.1f}, {hi:.1f}]")
    ax.set_title("Query concentration near phase boundary")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    return fig


def plot_labeled_points_scatter(
    results: Dict[str, Dict],
    methods_to_show: Optional[List[str]] = None,
    dim_i: int = 0,
    dim_j: int = 1,
) -> None:
    """Plot scatter of queried input points (projected to two dims) for each method."""
    if methods_to_show is None:
        methods_to_show = list(results.keys())

    n     = len(methods_to_show)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, label in zip(axes_flat[:n], methods_to_show):
        state     = results[label]["state"]
        x_labeled = np.asarray(state.train_inputs)
        ax.scatter(x_labeled[:, dim_i], x_labeled[:, dim_j], s=4, alpha=0.5)
        ax.set_xlabel(f"$x_{{{dim_i}}}$")
        ax.set_ylabel(f"$x_{{{dim_j}}}$")
        ax.set_title(f"{label} (n={len(x_labeled)})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle("Labeled input distribution by method", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def make_single_run_figure(result: Dict, title: str = "") -> plt.Figure:
    """Like ``plot_single_run`` but returns the figure without calling plt.show()."""
    return _build_single_run_figure(result, title, show=False)


def plot_single_run(result: Dict, title: str = "") -> None:
    """Show NLL curve + labeled scatter + MDN output samples vs truth."""
    _build_single_run_figure(result, title, show=True)


def _build_single_run_figure(result: Dict, title: str = "", show: bool = True) -> plt.Figure:
    """Internal: build the 3-panel single-run figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    state    = result["state"]
    history  = result["history"]
    model    = result["model"]
    test_inputs, test_targets = result["test_data"]

    # Panel 1: NLL learning curve
    ax = axes[0]
    rounds = [h.round_idx for h in history]
    nlls   = [h.test_loss  for h in history]
    ax.plot(rounds, nlls, marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test NLL")
    ax.set_yscale("log") if min(nlls) > 0 else ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Learning curve")
    ax.grid(True, alpha=0.3)

    # Panel 2: labeled input scatter
    ax = axes[1]
    x_labeled = np.asarray(state.train_inputs)
    ax.scatter(x_labeled[:, 0], x_labeled[:, 1], s=4, alpha=0.5)
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.set_title(f"Labeled inputs (n={len(x_labeled)})")
    ax.grid(True, alpha=0.3)

    # Panel 3: MDN output samples (net 0) vs truth  in (y0, y1) space
    ax = axes[2]
    pred = model.apply(model.params, test_inputs)
    logit_weights, means, variances = pred
    samples = model.sample_from_mixture(
        random.PRNGKey(0), logit_weights, means, variances,
    )  # (E, N, 1, m) — singleton from take_along_axis; squeeze to (E, N, m)
    y_net0 = np.asarray(samples[0]).squeeze(1)   # (N, m)
    n_show = min(300, test_inputs.shape[0])
    ax.scatter(np.asarray(test_targets[:n_show, 0]), np.asarray(test_targets[:n_show, 1]),
               s=4, alpha=0.3, c="blue")
    ax.scatter(y_net0[:n_show, 0], y_net0[:n_show, 1],
               s=4, alpha=0.3, c="red")
    ax.plot([], [], c="blue", alpha=0.7, label="True")
    ax.plot([], [], c="red",  alpha=0.7, label="MDN sample")
    ax.set_xlabel("$y_0$")
    ax.set_ylabel("$y_1$")
    ax.set_title("Predictions (net 0)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    return fig
