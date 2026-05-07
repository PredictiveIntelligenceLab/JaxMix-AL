from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from active_learning.acquisitions import (
    bait_fisher_embeddings,
    coreset_features,
    get_acquisition_fn,
)
from active_learning.history import ActiveLearningRoundSummary
from active_learning.query_scenarios import move_from_pool_to_train
from active_learning.selection import (
    select_bait,
    select_kcenter_greedy,
    select_maxdist,
    select_sbal,
    select_top_k,
)
from active_learning.state import ActiveLearningState


def _batched_acquisition(
    acquisition_fn: Callable,
    model: Any,
    inputs: jnp.ndarray,
    key: jax.Array,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Evaluates an acquisition function in chunks to limit memory usage."""
    n = inputs.shape[0]
    if batch_size is None or batch_size >= n:
        return acquisition_fn(model, inputs, key)

    scores_list = []
    for start in range(0, n, batch_size):
        chunk = inputs[start : start + batch_size]
        key, chunk_key = jax.random.split(key)
        scores_list.append(acquisition_fn(model, chunk, chunk_key))
    return jnp.concatenate(scores_list, axis=0)


def _batched_bait_embeddings(
    model: Any,
    inputs: jnp.ndarray,
    key: jax.Array,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Compute BAIT Fisher embeddings in chunks to bound peak memory.

    Returns a single (N, rank, dim) tensor concatenated across chunks.
    """
    n = inputs.shape[0]
    if n == 0:
        # Probe with a single input so the chunk loop sees the correct rank/dim.
        probe = bait_fisher_embeddings(
            model, inputs[:1] if inputs.ndim > 1 else inputs.reshape(1, -1), key,
        )
        return np.zeros((0, *probe.shape[1:]), dtype=np.asarray(probe).dtype)
    if batch_size is None or batch_size >= n:
        return np.asarray(bait_fisher_embeddings(model, inputs, key))

    pieces = []
    for start in range(0, n, batch_size):
        chunk = inputs[start : start + batch_size]
        key, chunk_key = jax.random.split(key)
        pieces.append(np.asarray(bait_fisher_embeddings(model, chunk, chunk_key)))
    return np.concatenate(pieces, axis=0)


def _batched_coreset_features(
    model: Any,
    inputs: jnp.ndarray,
    key: jax.Array,
    batch_size: Optional[int] = None,
    ensemble_member: int = 0,
) -> jnp.ndarray:
    """Compute Core-Set backbone features in chunks; returns (N, h)."""
    n = inputs.shape[0]
    if n == 0:
        probe = coreset_features(
            model,
            inputs[:1] if inputs.ndim > 1 else inputs.reshape(1, -1),
            key,
            ensemble_member=ensemble_member,
        )
        return jnp.zeros((0, probe.shape[-1]))
    if batch_size is None or batch_size >= n:
        return coreset_features(model, inputs, key, ensemble_member=ensemble_member)

    pieces = []
    for start in range(0, n, batch_size):
        chunk = inputs[start : start + batch_size]
        key, chunk_key = jax.random.split(key)
        pieces.append(
            coreset_features(model, chunk, chunk_key, ensemble_member=ensemble_member)
        )
    return jnp.concatenate(pieces, axis=0)


def _get_mdn_features(model, inputs):
    """Extract per-ensemble conditional means as feature vectors for SBAL.

    Returns shape (batch, E*D) where E is ensemble size and D is output dim.
    Falls back to raw inputs if the model doesn't return MDN tuples.
    """
    try:
        pred = model.apply(model.params, inputs)
        logit_weights, means, _vars = pred  # (E, B, K, ...)
        weights = jax.nn.softmax(logit_weights, axis=-2)
        cond_means = (weights * means).sum(axis=-2)  # (E, B, D)
        B = cond_means.shape[1]
        return cond_means.transpose(1, 0, 2).reshape(B, -1)  # (B, E*D)
    except (ValueError, TypeError):
        return inputs


def run_pool_based_active_learning(
    initial_state,
    trainer_factory: Callable[[Tuple[jnp.ndarray, jnp.ndarray], jax.Array], Any],
    test_data: Tuple[jnp.ndarray, jnp.ndarray],
    num_rounds: int,
    query_batch_size: int,
    evaluate_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], float],
    acquisition_name: str = "mdn_epistemic_variance",
    acquisition_batch_size: Optional[int] = None,
    selection_strategy: str = "top_k",
    sbal_temperature: float = 1.0,
    maxdist_score_weight: float = 1.0,
    coreset_ensemble_member: int = 0,
    coreset_pool_subsample: Optional[int] = None,
    on_round_end: Optional[Callable] = None,
):
    """Runs a minimal pool-based active-learning loop.

    Args:
        selection_strategy: "top_k" for greedy top-k, "sbal" for Stochastic
            Batch Active Learning (Kirsch et al., 2022) using Gumbel-top-k
            softmax sampling from acquisition scores, "maxdist" for
            farthest-point sampling in model prediction space with
            training-point-aware initialisation (al4pde / bmdal_reg), or
            "bait" for BAIT batch selection (Ash et al., NeurIPS 2021,
            "Gone Fishing").  When ``bait`` is selected, ``acquisition_name``
            must be ``"bait"`` (Fisher embeddings, not scalar scores).
        BAIT uses the fixed paper-style repo setting: ensemble member 0,
            mean-head Fisher, one MC sample, no pool subsampling, and
            ``select_bait``'s default ridge regularization.
    """
    if num_rounds <= 0:
        raise ValueError(f"num_rounds must be positive, got {num_rounds}")
    if query_batch_size <= 0:
        raise ValueError(f"query_batch_size must be positive, got {query_batch_size}")

    is_bait = selection_strategy == "bait"
    if is_bait and acquisition_name != "bait":
        raise ValueError(
            "selection_strategy='bait' requires acquisition_name='bait' "
            f"(got '{acquisition_name}')."
        )
    is_coreset = selection_strategy == "coreset"
    if is_coreset and acquisition_name != "coreset":
        raise ValueError(
            "selection_strategy='coreset' requires acquisition_name='coreset' "
            f"(got '{acquisition_name}')."
        )

    acquisition_fn = get_acquisition_fn(acquisition_name)

    state = ActiveLearningState(
        inputs=initial_state.inputs,
        targets=initial_state.targets,
        train_indices=initial_state.train_indices,
        pool_indices=initial_state.pool_indices,
        key=initial_state.key,
        round_idx=initial_state.round_idx,
        history=list(initial_state.history),
        synthetic_inputs=initial_state.synthetic_inputs,
        synthetic_targets=initial_state.synthetic_targets,
    )
    test_inputs, test_targets = test_data

    for round_idx in range(num_rounds):
        if state.pool_inputs.shape[0] == 0:
            break

        train_data = (state.train_inputs, state.train_targets)
        state.key, trainer_key, acquisition_key, select_key = jax.random.split(state.key, 4)
        model = trainer_factory(train_data, trainer_key)

        pool_size = state.pool_inputs.shape[0]
        k = min(query_batch_size, pool_size)

        if is_bait:
            # BAIT branch: paper-style Fisher embeddings on the full pool,
            # followed by greedy forward+backward selection.
            pool_emb = _batched_bait_embeddings(
                model, state.pool_inputs, acquisition_key,
                batch_size=acquisition_batch_size,
            )
            state.key, train_emb_key = jax.random.split(state.key)
            train_emb = _batched_bait_embeddings(
                model, state.train_inputs, train_emb_key,
                batch_size=acquisition_batch_size,
            )
            # ``select_bait`` accepts JAX arrays directly so the heavy linalg
            # stays on GPU; only the small Python list of selected indices
            # lives on the host.
            selected_local = select_bait(pool_emb, train_emb, k)
            selected_local_jnp = jnp.asarray(selected_local, dtype=jnp.int32)
            selected_indices = selected_local_jnp
            # No per-point scalar scores for BAIT; use embedding row norms as
            # a logging proxy so summary statistics remain meaningful.
            acquisition_scores = jnp.asarray(np.linalg.norm(pool_emb, axis=(1, 2)))
        elif is_coreset:
            # Core-Set branch (Sener & Savarese 2018, k-Center-Greedy):
            # geometric coverage in backbone-feature space, no scores.
            pool_inputs_full = state.pool_inputs
            if (
                coreset_pool_subsample is not None
                and coreset_pool_subsample < pool_size
            ):
                state.key, sub_key = jax.random.split(state.key)
                sub_indices = jax.random.choice(
                    sub_key, pool_size, shape=(coreset_pool_subsample,),
                    replace=False,
                )
                pool_inputs_used = pool_inputs_full[sub_indices]
            else:
                sub_indices = None
                pool_inputs_used = pool_inputs_full

            pool_feats = _batched_coreset_features(
                model, pool_inputs_used, acquisition_key,
                batch_size=acquisition_batch_size,
                ensemble_member=coreset_ensemble_member,
            )
            state.key, train_feat_key = jax.random.split(state.key)
            train_feats = _batched_coreset_features(
                model, state.train_inputs, train_feat_key,
                batch_size=acquisition_batch_size,
                ensemble_member=coreset_ensemble_member,
            )
            selected_local = select_kcenter_greedy(pool_feats, train_feats, k)
            selected_local_jnp = jnp.asarray(selected_local, dtype=jnp.int32)
            if sub_indices is not None:
                selected_indices = sub_indices[selected_local_jnp].astype(jnp.int32)
            else:
                selected_indices = selected_local_jnp
            # For logging: distance from each candidate to its nearest
            # labelled point — the quantity k-Center-Greedy maximises.
            if train_feats.shape[0] > 0:
                d2 = (
                    jnp.sum(pool_feats ** 2, axis=1, keepdims=True)
                    + jnp.sum(train_feats ** 2, axis=1)[None, :]
                    - 2.0 * pool_feats @ train_feats.T
                )
                local_min_d = jnp.sqrt(jnp.maximum(jnp.min(d2, axis=1), 0.0))
            else:
                local_min_d = jnp.zeros((pool_feats.shape[0],))
            if sub_indices is not None:
                acquisition_scores = jnp.full((pool_size,), jnp.nan)
                acquisition_scores = acquisition_scores.at[sub_indices].set(local_min_d)
            else:
                acquisition_scores = local_min_d
        else:
            acquisition_scores = _batched_acquisition(
                acquisition_fn, model, state.pool_inputs, acquisition_key,
                batch_size=acquisition_batch_size,
            )

            if selection_strategy == "sbal":
                selected_indices = select_sbal(
                    acquisition_scores, k, select_key,
                    temperature=sbal_temperature,
                )
            elif selection_strategy == "maxdist":
                pool_features = _get_mdn_features(model, state.pool_inputs)
                train_features = _get_mdn_features(model, state.train_inputs)
                selected_indices = select_maxdist(
                    acquisition_scores, pool_features, k,
                    train_features=train_features,
                    score_weight=maxdist_score_weight,
                )
            else:
                selected_indices = select_top_k(acquisition_scores, k)

        selected_scores = acquisition_scores[selected_indices]
        test_loss = evaluate_fn(model, test_inputs, test_targets)

        summary = ActiveLearningRoundSummary(
            round_idx=round_idx,
            acquisition_name=acquisition_name,
            train_size=int(state.train_indices.shape[0]),
            pool_size=int(state.pool_indices.shape[0]),
            test_loss=test_loss,
            mean_acquisition_score=float(jnp.nanmean(acquisition_scores)),
            max_acquisition_score=float(jnp.nanmax(acquisition_scores)),
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            selected_inputs=state.pool_inputs[selected_indices],
            final_round=(round_idx == num_rounds - 1) or (k == state.pool_inputs.shape[0]),
        )

        state.history.append(summary)
        if on_round_end is not None:
            on_round_end(summary, model)
        state = move_from_pool_to_train(state, selected_indices)

    train_data = (state.train_inputs, state.train_targets)
    state.key, final_key = jax.random.split(state.key)
    final_model = trainer_factory(train_data, final_key)
    final_test_loss = evaluate_fn(final_model, test_inputs, test_targets)

    return state, state.history, final_model, final_test_loss
