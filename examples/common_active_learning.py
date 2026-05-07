"""Shared active-learning helpers used by the example benchmarks."""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import optax

from active_learning import run_pool_based_active_learning
from active_learning.state import ActiveLearningState
from active_learning.wandb_logger import WandbLogger
from jaxmix.archs import Ensemble, MDN, MLP
from jaxmix.data_loaders import BatchedDataset
from jaxmix.trainers import MDNTrainer, mdn_loss_func


def get_default_mdn_config(
    out_dim: int,
    hidden_features: int = 128,
    depth: int = 3,
    num_mixtures: int = 8,
    ensemble_size: int = 8,
) -> Dict[str, Any]:
    base_arch = MLP(features=[hidden_features] * depth)
    backbone_arch = Ensemble(base_arch, ensemble_size)
    mdn_arch = MDN(
        num_mixtures=num_mixtures,
        num_output_dims=out_dim,
        backbone=backbone_arch,
        ensemble_size=ensemble_size,
    )
    return {"arch": mdn_arch, "ensemble_size": ensemble_size}


def make_optimizer(n_iter: int, peak_lr: float = 5e-4, weight_decay: float = 1e-2):
    warmup_steps = min(500, n_iter // 5)
    lr = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps,
            ),
            optax.exponential_decay(
                init_value=peak_lr, transition_steps=2_000, decay_rate=0.9,
            ),
        ],
        boundaries=[warmup_steps],
    )
    return optax.chain(
        optax.adaptive_grad_clip(0.1),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay),
    )


def evaluate_ensemble_nll(
    model: Any,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> float:
    logit_weights, means, variances = model.apply(model.params, inputs)
    per_element_nll = mdn_loss_func(logit_weights, means, variances, targets)
    ensemble_size = per_element_nll.shape[0]
    log_likelihoods = -per_element_nll
    ensemble_nll = -(logsumexp(log_likelihoods, axis=0) - jnp.log(ensemble_size))
    return float(ensemble_nll.mean())


def make_mdn_trainer_factory(
    config: Dict[str, Any],
    n_iter: int = 30_000,
    batch_size: int = 128,
    normalize_inputs: bool = True,
    normalize_outputs: bool = True,
    adaptive_iters: bool = False,
    iter_per_sample: int = 100,
    min_adaptive_iters: int = 0,
    peak_lr: float = 5e-4,
    weight_decay: float = 1e-2,
):
    arch = config["arch"]

    def trainer_factory(
        train_data: Tuple[jnp.ndarray, jnp.ndarray],
        key: jax.Array,
    ) -> MDNTrainer:
        train_inputs, train_targets = train_data
        train_weights = jnp.ones((train_inputs.shape[0], 1))
        init_batch = (train_inputs, train_targets, train_weights)

        if adaptive_iters:
            effective_n_iter = min(
                n_iter,
                max(iter_per_sample * train_inputs.shape[0], min_adaptive_iters),
            )
        else:
            effective_n_iter = n_iter

        optimizer = make_optimizer(
            effective_n_iter, peak_lr=peak_lr, weight_decay=weight_decay,
        )

        key, init_key = random.split(key)
        model = MDNTrainer(
            arch=arch,
            init_batch=init_batch,
            key=init_key,
            optimizer=optimizer,
            normalize_inputs=normalize_inputs,
            normalize_outputs=normalize_outputs,
        )
        key, loader_key = random.split(key)
        loader = BatchedDataset(init_batch, loader_key, batch_size=batch_size)
        model.train(loader, nIter=effective_n_iter)
        return model

    return trainer_factory


def create_pool_state_from_shared_benchmark(
    shared_benchmark: Dict[str, Any],
) -> ActiveLearningState:
    init_x = shared_benchmark["initial_labeled_inputs"]
    init_y = shared_benchmark["initial_labeled_targets"]
    pool_x = shared_benchmark["remaining_pool_inputs"]
    pool_y = shared_benchmark["remaining_pool_targets"]

    combined_x = jnp.concatenate([init_x, pool_x], axis=0)
    combined_y = jnp.concatenate([init_y, pool_y], axis=0)
    n_init = init_x.shape[0]
    n_total = combined_x.shape[0]

    return ActiveLearningState(
        inputs=combined_x,
        targets=combined_y,
        train_indices=jnp.arange(n_init, dtype=jnp.int32),
        pool_indices=jnp.arange(n_init, n_total, dtype=jnp.int32),
        key=shared_benchmark["shared_state_key"],
    )


def resolve_acquisition(acquisition_name: str, selection_strategy: str):
    actual_acq = acquisition_name
    if acquisition_name.startswith("sbal_"):
        actual_acq = acquisition_name[len("sbal_"):]
        selection_strategy = "sbal"
    elif acquisition_name.startswith("maxdist_"):
        actual_acq = acquisition_name[len("maxdist_"):]
        selection_strategy = "maxdist"
    elif acquisition_name == "bait":
        selection_strategy = "bait"
    elif acquisition_name == "coreset":
        selection_strategy = "coreset"
    return actual_acq, selection_strategy


def run_pool_based_active_learning_experiment(
    trainer_factory,
    shared_benchmark: Dict[str, Any],
    acquisition_name: str = "mdn_epistemic_variance",
    al_iters: int = 15,
    query_batch_size: int = 20,
    acquisition_batch_size: Optional[int] = 256,
    selection_strategy: str = "top_k",
    sbal_temperature: float = 1.0,
    maxdist_score_weight: float = 1.0,
    coreset_ensemble_member: int = 0,
    coreset_pool_subsample: Optional[int] = None,
    logger: Optional[WandbLogger] = None,
) -> Dict:
    initial_state = create_pool_state_from_shared_benchmark(shared_benchmark)
    test_data = shared_benchmark["test_data"]
    true_nll = shared_benchmark.get("true_nll")
    actual_acq, selection_strategy = resolve_acquisition(
        acquisition_name, selection_strategy,
    )

    def on_round_end(summary, trainer=None):
        if logger is not None:
            logger.log_round(summary, trainer=trainer)

    final_state, history, model, final_metric = run_pool_based_active_learning(
        initial_state=initial_state,
        trainer_factory=trainer_factory,
        test_data=test_data,
        num_rounds=al_iters,
        query_batch_size=query_batch_size,
        acquisition_name=actual_acq,
        evaluate_fn=evaluate_ensemble_nll,
        acquisition_batch_size=acquisition_batch_size,
        selection_strategy=selection_strategy,
        sbal_temperature=sbal_temperature,
        maxdist_score_weight=maxdist_score_weight,
        coreset_ensemble_member=coreset_ensemble_member,
        coreset_pool_subsample=coreset_pool_subsample,
        on_round_end=on_round_end,
    )

    if logger is not None:
        logger.log_final(final_metric, final_state, final_trainer=model,
                         final_round_idx=al_iters)

    result = {
        "state": final_state,
        "history": history,
        "model": model,
        "test_data": test_data,
        "final_test_nll": final_metric,
    }
    if true_nll is not None:
        result["true_nll"] = true_nll
    return result


def plot_loss_comparison(
    results: Dict[str, Dict],
    true_nll: Optional[float] = None,
    title: str = "Active learning curves",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    all_nlls = []
    for label, result in results.items():
        history = result["history"]
        rounds = [h.round_idx for h in history]
        nlls = [h.test_loss for h in history]
        all_nlls.extend(nlls)
        ax.plot(rounds, nlls, marker="o", label=label)

    ref = true_nll
    if ref is None:
        for result in results.values():
            if result.get("true_nll") is not None:
                ref = float(result["true_nll"])
                break
    if ref is not None:
        ax.axhline(ref, color="gray", linestyle="--", linewidth=1.0, label="True NLL")

    ax.set_xlabel("Round")
    ax.set_ylabel("Test NLL")
    if all_nlls and min(all_nlls) > 0:
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()
    return fig
