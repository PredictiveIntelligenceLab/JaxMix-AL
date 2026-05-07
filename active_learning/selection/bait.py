"""BAIT batch selection (Ash et al., NeurIPS 2021, "Gone Fishing").

Given per-input low-rank Fisher embeddings :math:`G(x) \\in \\mathbb{R}^{r\\times d}`
such that the per-input Fisher information is :math:`F(x) = G(x)^\\top G(x)`,
BAIT picks a batch :math:`S \\subseteq U` of size ``k`` minimising the trace
of :math:`M_S^{-1} M_U` where

    M_U = (1/|U|) Σ_{x ∈ U} F(x)                  (full Fisher of all candidates)
    M_S = λ I + Σ_{x ∈ L} F(x) + Σ_{x ∈ S} F(x)   (selected + already-labelled)

via greedy forward selection (over-sampling by 2×) followed by greedy
backward pruning, both implemented with the Woodbury / Sherman-Morrison
identity for low-rank updates.

This is a port of the reference implementation in
https://github.com/JordanAsh/badge/blob/master/query_strategies/bait_sampling.py
written in JAX so the per-step linear-algebra runs on the GPU.  The greedy
outer loop is intrinsically sequential — each step depends on the rank-r
Woodbury update of ``currentInv`` from the previous step — so we keep a
Python loop and call a single jit'd ``_bait_step`` function per iteration.
That function does the (N, rank, dim) × (dim, dim) matmuls, the batched
(rank, rank) inverse, the trace einsum, and the Woodbury update in a single
fused kernel.
"""
from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _fisher_from_embeddings(X: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Σ_x G(x)^T G(x) · weight   →   (dim, dim).

    ``X`` has shape (N, rank, dim).  Returns a single (dim, dim) matrix.
    """
    N, rank, dim = X.shape
    if N == 0:
        return jnp.zeros((dim, dim), dtype=X.dtype)
    Xf = X.reshape(-1, dim)
    return weight * (Xf.T @ Xf)


@partial(jax.jit, static_argnames=("sign",))
def _bait_step(
    currentInv: jnp.ndarray,
    Xs: jnp.ndarray,
    fisher_U: jnp.ndarray,
    eligible_mask: jnp.ndarray,
    sign: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One greedy step (jit'd, runs on GPU when available).

    Forward (sign=+1): we ADD the chosen point to S; rank-r Woodbury update.
    Backward (sign=-1): we REMOVE the chosen point from S; rank-r downdate.

    Args:
        currentInv: (dim, dim) current M^{-1}.
        Xs:         (N, rank, dim) (rescaled) Fisher embeddings.
        fisher_U:   (dim, dim) full pool Fisher M_U.
        eligible_mask: (N,) bool mask — only these indices may be chosen.
        sign:       +1.0 for forward selection, -1.0 for backward removal.

    Returns:
        new_currentInv, chosen index (int32 scalar), trace value at chosen.
    """
    rank = Xs.shape[1]
    eye_r = jnp.eye(rank, dtype=Xs.dtype)

    # Per-candidate trace estimate (Eq. used by BADGE/BAIT reference):
    #   trace_n  =  tr(  G_n M^{-1} F_U M^{-1} G_n^T  ·  (sign·I + G_n M^{-1} G_n^T)^{-1}  )
    gM       = Xs @ currentInv                                # (N, rank, dim)
    gMg      = jnp.einsum("nrd,nsd->nrs", gM, Xs)             # (N, rank, rank)
    innerInv = jnp.linalg.inv(sign * eye_r + gMg)             # (N, rank, rank)
    gMF      = gM @ fisher_U                                   # (N, rank, dim)
    # gMFgM_n = gM_n F_U gM_n^T  =  G_n M^{-1} F_U M^{-1} G_n^T
    gMFgM    = jnp.einsum("nrd,nsd->nrs", gMF, gM)            # (N, rank, rank)
    traces   = jnp.einsum("nrs,nsr->n", gMFgM, innerInv)      # (N,)

    masked   = jnp.where(eligible_mask, traces, -jnp.inf)
    chosen   = jnp.argmax(masked).astype(jnp.int32)

    # Rank-r Woodbury (un)update of  currentInv:
    #   (M ± G^T G)^{-1} = M^{-1}  ∓  M^{-1} G^T (sign·I + G M^{-1} G^T)^{-1} G M^{-1}
    g            = Xs[chosen]                                 # (rank, dim)
    gM_c         = g @ currentInv                             # (rank, dim)
    inner_i      = jnp.linalg.inv(sign * eye_r + gM_c @ g.T)  # (rank, rank)
    currentInv_n = currentInv - gM_c.T @ inner_i @ gM_c

    return currentInv_n, chosen, traces[chosen]


def select_bait(
    pool_embeddings,
    train_embeddings,
    k: int,
    over_sample: int = 2,
    lamb: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """Forward+backward greedy BAIT selection.

    All linear algebra runs on the JAX default backend (GPU if available);
    only the small Python list of selected indices lives on the host.

    Args:
        pool_embeddings: (N_pool, rank, dim) Fisher embeddings (numpy or JAX).
        train_embeddings: (N_train, rank, dim) embeddings for already-labelled
            points, or ``None``/empty for the cold-start round.
        k: Batch size to select.
        over_sample: Forward over-sampling factor (paper uses 2). Must be >= 1
            so that ``int(over_sample * k) >= k`` and the backward prune has
            a non-empty oversample to operate on.
        lamb: Ridge regularization on the Fisher: ``M_S = lamb * I + Σ F``.
        verbose: If True print per-step trace estimates.

    Returns:
        Selected pool indices as a numpy int64 array of shape (k,).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if over_sample < 1:
        raise ValueError(
            f"over_sample must be >= 1 (paper uses 2); got {over_sample}. "
            "Smaller values would forward-select fewer than k points and the "
            "backward prune would silently return |S| < k."
        )
    if lamb <= 0:
        raise ValueError(f"lamb must be positive, got {lamb}")

    cpu = jax.devices("cpu")[0]
    Xpool = jax.device_put(jnp.asarray(pool_embeddings), cpu)
    N_pool, rank, dim = Xpool.shape
    if k > N_pool:
        raise ValueError(f"k={k} exceeds pool size {N_pool}")

    n_labeled = 0 if train_embeddings is None else int(train_embeddings.shape[0])

    with jax.default_device(cpu):
        fisher_U = _fisher_from_embeddings(Xpool, weight=1.0 / N_pool)

        if n_labeled > 0:
            # Unweighted sum Σ_L G_l^T G_l so the labelled term in M_S below
            # matches the docstring's ``Σ_{x ∈ L} F(x)``.  The earlier
            # ``weight=1/n_labeled`` form under-weighted L by a factor of
            # n_labeled, making BAIT effectively forget already-labelled points.
            train_arr = jax.device_put(jnp.asarray(train_embeddings), cpu)
            fisher_L = _fisher_from_embeddings(train_arr, weight=1.0)
        else:
            fisher_L = jnp.zeros((dim, dim), dtype=Xpool.dtype)

        # ---- Initial inverse and pool rescaling (matches reference BADGE/BAIT) --
        # M0 = lamb*I + (n_L/(n_L+k)) Σ_L F.  The (n_L/(n_L+k)) factor is the
        # standard BADGE/BAIT rescaling so that L and the eventual S = k items
        # contribute on the same overall scale.
        scale_L = n_labeled / (n_labeled + k) if (n_labeled + k) > 0 else 0.0
        M0 = lamb * jnp.eye(dim, dtype=Xpool.dtype) + fisher_L * scale_L
        currentInv = jnp.linalg.inv(M0)

        if (n_labeled + k) > 0:
            scale_X = float(np.sqrt(k / (n_labeled + k)))
        else:
            scale_X = 1.0
        Xs = Xpool * scale_X

        # -------------------------------- Forward greedy ------------------------
        selected_mask = jnp.zeros(N_pool, dtype=bool)
        selected = []
        n_forward = min(int(over_sample * k), N_pool)
        for it in range(n_forward):
            currentInv, chosen, trace_val = _bait_step(
                currentInv, Xs, fisher_U,
                eligible_mask=~selected_mask,
                sign=+1.0,
            )
            idx = int(chosen)
            selected.append(idx)
            selected_mask = selected_mask.at[idx].set(True)
            if verbose:
                print(f"[BAIT/forward] {it:4d}  idx={idx:6d}  trace={float(trace_val):.4e}")

        # -------------------------------- Backward prune ------------------------
        n_remove = n_forward - k
        for it in range(n_remove):
            currentInv, chosen, trace_val = _bait_step(
                currentInv, Xs, fisher_U,
                eligible_mask=selected_mask,
                sign=-1.0,
            )
            idx = int(chosen)
            selected_mask = selected_mask.at[idx].set(False)
            selected.remove(idx)
            if verbose:
                print(f"[BAIT/back]    {it:4d}  rm={idx:6d}  trace={float(trace_val):.4e}")

    return np.asarray(selected, dtype=np.int64)
