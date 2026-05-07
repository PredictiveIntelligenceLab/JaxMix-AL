import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial


@partial(jax.jit, static_argnames=())
def _mi_lb_score(logit_weights, means, variances):
    """JIT-compiled core of the MI-LB acquisition (no Python-level model call)."""
    variances = jnp.maximum(variances, 1e-8)
    # logit_weights: (E, B, K, 1)
    # means:         (E, B, K, D)
    # variances:     (E, B, K, D)  -- diagonal covariances
    E, B, K, D = means.shape

    # -- Per-member mixture weights alpha_i^{(z)} --
    log_alpha = jax.nn.log_softmax(logit_weights, axis=-2)  # (E, B, K, 1)
    alpha = jnp.exp(log_alpha)                               # (E, B, K, 1)

    # -- Uniform ensemble weights w_z = 1/E --
    log_w = -jnp.log(E)

    # -- Marginal mixture weights beta_{z,i} = w_z * alpha_i^{(z)} --
    # Flatten (E, K) into M = E*K components
    log_beta = log_w + log_alpha[..., 0]                      # (E, B, K)
    log_beta_flat = jnp.transpose(log_beta, (1, 0, 2)).reshape(B, E * K)

    means_flat = jnp.transpose(means, (1, 0, 2, 3)).reshape(B, E * K, D)
    vars_flat = jnp.transpose(variances, (1, 0, 2, 3)).reshape(B, E * K, D)

    # ================================================================
    # H_lower
    # Lower bound on marginal entropy H(Y | x)
    # ================================================================
    # Pairwise log N(mu_i; mu_j, diag(var_i + var_j))
    mu_i = means_flat[:, :, None, :]    # (B, M, 1, D)
    mu_j = means_flat[:, None, :, :]    # (B, 1, M, D)
    var_i = vars_flat[:, :, None, :]    # (B, M, 1, D)
    var_j = vars_flat[:, None, :, :]    # (B, 1, M, D)

    sum_var = var_i + var_j              # (B, M, M, D)
    diff = mu_i - mu_j                   # (B, M, M, D)

    log_gauss_pairwise = jnp.sum(
        -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * jnp.log(sum_var)
        - 0.5 * diff ** 2 / sum_var,
        axis=-1,
    )  # (B, M, M)

    # H_lower = -sum_i beta_i log(sum_j beta_j N(mu_i; mu_j, ...))
    log_inner = log_beta_flat[:, None, :] + log_gauss_pairwise  # (B, M, M)
    log_sum_j = logsumexp(log_inner, axis=-1)                    # (B, M)

    beta_flat = jnp.exp(log_beta_flat)
    h_lower = -jnp.sum(beta_flat * log_sum_j, axis=-1)          # (B,)

    # ================================================================
    # H_upper 
    # Upper bound on conditional entropy per ensemble member z
    # ================================================================
    # H_upper(z) = sum_i alpha_i(-log alpha_i + 0.5 D log(2 pi e) + 0.5 sum_d log var_{i,d})
    log_det = jnp.sum(jnp.log(variances), axis=-1)              # (E, B, K)
    component_entropy = (
        0.5 * D * jnp.log(2.0 * jnp.pi * jnp.e) + 0.5 * log_det
    )                                                            # (E, B, K)

    alpha_k = alpha[..., 0]                                      # (E, B, K)
    log_alpha_k = log_alpha[..., 0]                              # (E, B, K)

    h_upper_per_z = jnp.sum(
        alpha_k * (-log_alpha_k + component_entropy),
        axis=-1,
    )                                                            # (E, B)

    # Average over ensemble (uniform weights)
    h_upper = jnp.mean(h_upper_per_z, axis=0)                   # (B,)

    # ================================================================
    # MI-LB = H_lower - H_upper  (lower bound on I(Y; Z | x))
    # ================================================================
    return h_lower - h_upper                                     # (B,)


def mi_lb_acquisition(
    model,
    inputs: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Scores inputs using the Mutual Information Lower Bound (MI-LB) acquisition.

    MI-LB is a certified lower bound on the mutual information I(Y; Z | x),
    quantifying epistemic uncertainty for MDN ensemble models.

        MI-LB(x) = H_lower(Y | x) - sum_z w_z H_upper(Y | x, Z=z)

    where H_lower is the Huber et al. (2008) lower bound (Eq. 12) applied to
    the full n_ens*K-component marginal mixture, and H_upper is the upper bound
    (Eq. 13) applied to each per-member K-component conditional mixture.

    Assumes diagonal covariances throughout (standard MDN parameterization).
    Fully differentiable w.r.t. inputs (no sampling), suitable for
    gradient-based membership-query synthesis.
    """
    del key
    logit_weights, means, variances = model.apply(model.params, inputs)
    return _mi_lb_score(logit_weights, means, variances)
