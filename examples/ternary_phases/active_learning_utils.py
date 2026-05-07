"""Active-learning utilities for the ternary phase competition benchmark."""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR)

from examples.common_active_learning import (
    get_default_mdn_config,
    make_mdn_trainer_factory as _make_mdn_trainer_factory,
    plot_loss_comparison,
    run_pool_based_active_learning_experiment,
)

from ternary_phases import (
    create_shared_ternary_data,
    _simplex_grid,
    _to_cartesian,
    plot_acquired_on_simplex as _plot_acquired_on_simplex,
)
from ternary_phases import _components as _true_components


def make_mdn_trainer_factory(
    config: Dict[str, Any],
    n_iter: int = 20_000,
    batch_size: int = 128,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    adaptive_iters: bool = True,
    iter_per_sample: int = 80,
):
    return _make_mdn_trainer_factory(
        config,
        n_iter=n_iter,
        batch_size=batch_size,
        normalize_inputs=normalize_inputs,
        normalize_outputs=normalize_outputs,
        adaptive_iters=adaptive_iters,
        iter_per_sample=iter_per_sample,
        min_adaptive_iters=2_000,
        peak_lr=2e-4,
        weight_decay=5e-2,
    )


def plot_acquired_on_simplex(
    results: Dict[str, Dict],
    data: Dict[str, Any],
    methods: Optional[List[str]] = None,
    boundary_threshold: float = 0.7,
) -> plt.Figure:
    return _plot_acquired_on_simplex(
        data, results, methods=methods, boundary_threshold=boundary_threshold,
    )


def plot_epistemic_vs_aleatoric(
    model,
    data: Dict[str, Any],
    n_grid: int = 120,
) -> plt.Figure:
    system = data["system"]
    n_proc = system["n_proc_params"]

    x3 = _simplex_grid(n_grid)
    x_AB = x3[:, :2]
    proc = jnp.zeros((x_AB.shape[0], n_proc))
    x_grid = jnp.concatenate([x_AB, proc], axis=-1)

    logit_weights, means, variances = model.apply(model.params, x_grid)
    weights = jax.nn.softmax(logit_weights, axis=-2)
    cond_means = jnp.sum(weights * means, axis=-2)
    cond_var = jnp.sum(weights * (variances + means ** 2), axis=-2) - cond_means ** 2
    epistemic = jnp.var(cond_means, axis=0).mean(axis=-1)
    aleatoric = jnp.mean(cond_var, axis=0).mean(axis=-1)

    cart_x, cart_y = _to_cartesian(x3)
    import matplotlib.tri as tri
    triang = tri.Triangulation(np.asarray(cart_x), np.asarray(cart_y))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    verts = jnp.array([[0, 0], [1, 0], [0.5, float(jnp.sqrt(3.0)) / 2.0], [0, 0]])

    t1 = axes[0].tripcolor(triang, np.asarray(epistemic), cmap="magma")
    plt.colorbar(t1, ax=axes[0])
    axes[0].set_title("Epistemic variance (Var_Z E[Y|Z,x])")

    t2 = axes[1].tripcolor(triang, np.asarray(aleatoric), cmap="cividis")
    plt.colorbar(t2, ax=axes[1])
    axes[1].set_title("Aleatoric variance (E_Z Var[Y|Z,x])")

    for ax in axes:
        ax.plot(verts[:, 0], verts[:, 1], "k-", lw=1.5)
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.95)
        ax.axis("off")

    plt.suptitle("Model uncertainty decomposition on the simplex", fontsize=13)
    plt.tight_layout()
    plt.show()
    return fig


def simplex_grid_inputs(system: Dict[str, Any], n_grid: int = 120):
    """Return (x3, x_grid) — barycentric grid and model inputs with zero proc params."""
    n_proc = system["n_proc_params"]
    x3 = _simplex_grid(n_grid)
    x_AB = x3[:, :2]
    proc = jnp.zeros((x_AB.shape[0], n_proc))
    x_grid = jnp.concatenate([x_AB, proc], axis=-1)
    return x3, x_grid


def plot_mean_vs_truth_on_simplex(
    system: Dict[str, Any],
    pred_mean: jnp.ndarray,
    x3: jnp.ndarray,
    label: str = "Predicted",
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Three-panel simplex plot: true E*[Y|x], predicted, and residual."""
    import matplotlib.tri as tri

    n_proc = system["n_proc_params"]
    proc = jnp.zeros((x3.shape[0], n_proc))
    pis, mus, _ = jax.vmap(lambda xx, pp: _true_components(xx, pp, system))(x3, proc)
    true_mean = jnp.sum(pis * mus, axis=-1)

    residual = pred_mean - true_mean
    cart_x, cart_y = _to_cartesian(x3)
    triang = tri.Triangulation(np.asarray(cart_x), np.asarray(cart_y))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    verts = jnp.array([[0, 0], [1, 0], [0.5, float(jnp.sqrt(3.0)) / 2.0], [0, 0]])

    vmin = float(jnp.minimum(true_mean.min(), pred_mean.min()))
    vmax = float(jnp.maximum(true_mean.max(), pred_mean.max()))

    t1 = axes[0].tripcolor(triang, np.asarray(true_mean), cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(t1, ax=axes[0])
    axes[0].set_title("True E*[Y | x]")

    t2 = axes[1].tripcolor(triang, np.asarray(pred_mean), cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(t2, ax=axes[1])
    axes[1].set_title(f"{label} E[Y | x]")

    rmax = float(jnp.max(jnp.abs(residual)))
    t3 = axes[2].tripcolor(triang, np.asarray(residual), cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    plt.colorbar(t3, ax=axes[2])
    axes[2].set_title(f"Residual ({label} − true)")

    for ax in axes:
        ax.plot(verts[:, 0], verts[:, 1], "k-", lw=1.5)
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.95)
        ax.axis("off")

    plt.suptitle(suptitle or "Conditional mean on the simplex (proc params = 0)",
                 fontsize=13)
    plt.tight_layout()
    plt.show()
    return fig


def plot_prediction_vs_truth(
    model,
    data: Dict[str, Any],
    n_grid: int = 120,
) -> plt.Figure:
    system = data["system"]
    x3, x_grid = simplex_grid_inputs(system, n_grid=n_grid)

    logit_weights, means, _vars = model.apply(model.params, x_grid)
    weights = jax.nn.softmax(logit_weights, axis=-2)
    per_member_mean = jnp.sum(weights * means, axis=-2)
    pred_mean = jnp.mean(per_member_mean, axis=0)[..., 0]

    return plot_mean_vs_truth_on_simplex(system, pred_mean, x3, label="Predicted")


def make_single_run_figure(result: Dict, title: str = "") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    history = result["history"]
    rounds = [h.round_idx for h in history]
    nlls = [h.test_loss for h in history]
    axes[0].plot(rounds, nlls, marker="o")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Test NLL")
    axes[0].set_title("Learning curve")
    axes[0].grid(True, alpha=0.3)
    if "true_nll" in result and result["true_nll"] is not None:
        axes[0].axhline(result["true_nll"], color="gray", linestyle="--", lw=1.0,
                        label="True NLL")
        axes[0].legend()

    state = result["state"]
    x_labeled = np.asarray(state.train_inputs)
    x_C = 1.0 - x_labeled[:, 0] - x_labeled[:, 1]
    cart_x = x_labeled[:, 1] + 0.5 * x_C
    cart_y = (math.sqrt(3.0) / 2.0) * x_C
    verts = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3.0) / 2.0], [0, 0]])
    axes[1].plot(verts[:, 0], verts[:, 1], "k-", lw=1.5)
    axes[1].scatter(cart_x, cart_y, s=6, alpha=0.5, c="steelblue", edgecolors="none")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    axes[1].set_title(f"Labeled compositions (n={len(x_labeled)})")

    plt.tight_layout()
    return fig
