"""Fisher embeddings for the BAIT acquisition (Ash et al., NeurIPS 2021).

BAIT does not produce a scalar acquisition score — it returns a per-input
Fisher *embedding* matrix G(x) of shape (rank, dim) such that the rank-(rank)
Fisher information matrix for x is F(x) = G(x)^T G(x).  These are then fed to
``select_bait`` which performs the forward+backward greedy trace minimisation
described in the paper (Algorithm 1).

For an MDN regression model we follow the paper's last-layer Fisher recipe:

    g_s(x) = ∇_W log p(y_s | x; θ)        with   y_s ~ p(· | x; θ)

where W are the **mean-head weights** of the first ensemble member.  This keeps
the implementation close to the paper's single-network, last-layer Fisher
setting.

Closed-form gradients
---------------------
With backbone features ``z(x) ∈ R^h``, mixture weights ``w_k(x)``,
component means ``μ_k(x)``, component variances ``σ²_k(x)`` (parameterised
through softplus of pre-activation ``v``), and posterior responsibilities
``γ_k(y, x) = w_k N(y; μ_k, σ²_k) / Σ_j w_j N(y; μ_j, σ²_j)``:

* mean head:   ∂log p / ∂W_μ[j,k,d] = γ_k · (y_d − μ_{k,d}) / σ²_{k,d} · z_j

The embedding is the outer product of the mean-head coefficient with the
backbone features ``z(x)``; we flatten it to one rank-1 row.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Per-member, jit'd embedding.
# ---------------------------------------------------------------------------


@jax.jit
def _embedding_one_member(
    logits_z: jnp.ndarray,    # (B, K)
    mu_z: jnp.ndarray,        # (B, K, D)
    var_z: jnp.ndarray,       # (B, K, D)
    bb_z: jnp.ndarray,        # (B, h)
    sig_y: jnp.ndarray,       # (1, 1, D) — output-norm std (1.0 if disabled)
    key: jax.Array,
) -> jnp.ndarray:
    """Compute per-input Fisher embeddings for one ensemble member.

    Returns a tensor of shape (B, 1, h*K*D) for the mean-head Fisher row.

    ``sig_y`` is the per-output-dim chain-rule factor for output
    normalisation: the network parameter ``W_μ`` produces ``μ_norm`` and the
    trainer applies ``μ = (sig_y + eps) μ_norm + mu_y``, so
    ``∂log p / ∂W_μ = γ (y − μ) / σ² · sig_y · z``.  Pass an all-ones array
    when output normalisation is disabled.
    """
    eps_var = 1e-8
    var_z = jnp.maximum(var_z, eps_var)

    log_w = jax.nn.log_softmax(logits_z, axis=-1)        # (B, K)

    B, K, D = mu_z.shape
    h = bb_z.shape[-1]

    # ------------------------------------------------------------------
    # 1) Sample y ~ p(y | x; θ) from the per-member mixture.
    # ------------------------------------------------------------------
    cat_key, gauss_key = jax.random.split(key)
    comp_idx = jax.random.categorical(cat_key, logits_z, axis=-1)             # (B,)
    mu_pick = jnp.take_along_axis(mu_z, comp_idx[:, None, None], axis=1)[:, 0, :]
    var_pick = jnp.take_along_axis(var_z, comp_idx[:, None, None], axis=1)[:, 0, :]
    eps = jax.random.normal(gauss_key, shape=(B, D))
    y_sample = mu_pick + jnp.sqrt(var_pick) * eps

    # ------------------------------------------------------------------
    # 2) Per-point responsibilities γ_k(y, x) and residuals (y − μ_k).
    # ------------------------------------------------------------------
    diff = y_sample[:, None, :] - mu_z                              # (B, K, D)
    log_n = -0.5 * (
        jnp.log(2.0 * jnp.pi) + jnp.log(var_z) + diff ** 2 / var_z
    ).sum(axis=-1)                                                   # (B, K)
    log_resp = log_w + log_n                                         # (B, K)
    gammas = jax.nn.softmax(log_resp, axis=-1)                       # (B, K)

    # ------------------------------------------------------------------
    # 3) Closed-form mean-head coefficients.
    # The ``* sig_y`` chains the gradient through output denormalisation
    # so we measure the Fisher of the actual network parameter (not of the
    # denormalised mean).  BAIT's λI ridge is NOT invariant under per-coord
    # rescaling of the embedding, so this factor matters when
    # ``normalize_outputs=True``.
    # ------------------------------------------------------------------
    coef_mu = gammas[..., None] * diff / var_z * sig_y               # (B, K, D)
    z_b = bb_z[:, :, None, None]                                     # (B, h, 1, 1)
    embed_mu = (coef_mu[:, None, :, :] * z_b).reshape(B, h * K * D)
    return embed_mu[:, None, :]                                      # (B, 1, dim)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def bait_fisher_embeddings(
    model,
    inputs: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Compute per-input Fisher embeddings for BAIT.

    Args:
        model: Trained MDNTrainer with ``apply``/``params`` and an MDN arch.
        inputs: (B, d) input batch.
        key: PRNG key for the single MC sample y ~ p(y|x).

    Returns:
        Tensor of shape ``(B, 1, h*K*D)`` from ensemble member 0's mean head.
    """
    logit_weights, means, variances = model.apply(model.params, inputs)
    backbone_z = _backbone_features(model, inputs)
    # logit_weights: (E, B, K, 1) → (E, B, K)
    logits = logit_weights[..., 0]
    D = means.shape[-1]
    sig_y = _output_norm_sigma(model, D, dtype=means.dtype)              # (1, 1, D)
    return _embedding_one_member(
        logits[0], means[0], variances[0], backbone_z[0], sig_y, key=key,
    )


def _maybe_normalize_inputs(model, inputs: jnp.ndarray) -> jnp.ndarray:
    """Apply the same input normalisation that ``MDNTrainer.apply`` wraps.

    Centralised here so ``_backbone_features`` and any future direct-arch
    callers cannot drift from the trainer's convention.  If
    ``MDNTrainer.__set_apply_function`` ever changes (e.g. switches to
    layer-norm) this helper must be updated in lockstep.
    """
    stats = getattr(model, "input_norm_stats", None)
    if stats is None:
        return inputs
    mu_x, sig_x = stats
    return (inputs - mu_x) / (sig_x + 1e-6)


def _output_norm_sigma(model, D: int, dtype) -> jnp.ndarray:
    """Return per-output-dim ``sig_y + 1e-6`` for chain-rule scaling.

    Matches ``MDNTrainer._apply_mdn_output_normalization``: the trainer
    denormalises ``μ = (sig_y + 1e-6) μ_norm + mu_y`` so the gradient w.r.t.
    the network parameter picks up exactly that factor.  Returns shape
    ``(1, 1, D)`` for broadcasting against ``(B, K, D)``; returns all ones
    when output normalisation is disabled.
    """
    stats = getattr(model, "output_norm_stats", None)
    if stats is None:
        return jnp.ones((1, 1, D), dtype=dtype)
    _, sig_y = stats
    return (sig_y + 1e-6).astype(dtype).reshape((1, 1, D))


def _backbone_features(model, inputs: jnp.ndarray) -> jnp.ndarray:
    """Run only the MDN's backbone (everything except the final ``DenseGeneral``).

    Returns features of shape (E, B, h) for ensembled MDNs.
    """
    backbone = model.arch.backbone
    backbone_params = model.params["params"]["backbone"]
    x = _maybe_normalize_inputs(model, inputs)
    return backbone.apply({"params": backbone_params}, x)
