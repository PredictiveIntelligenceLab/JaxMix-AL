import jax
import jax.numpy as jnp


def mdn_epistemic_variance_acquisition(
    model,
    inputs: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Scores inputs by analytic epistemic variance for MDN ensemble models.

    Implements Var_Z(E_eps[Y | Z, x]) from the law-of-total-variance
    decomposition (Eq. 6 of the paper).  For each ensemble member z the
    conditional mean is the mixture-weight-averaged mean; the epistemic
    variance is the variance of those conditional means across ensemble
    members.

    Fully differentiable w.r.t. inputs (no sampling), making it suitable
    for gradient-based membership-query synthesis.
    """
    del key
    logit_weights, means, _variances = model.apply(model.params, inputs)
    # logit_weights: (ensemble, batch, K, 1)
    # means:         (ensemble, batch, K, output_dim)
    weights = jax.nn.softmax(logit_weights, axis=-2)        # (ensemble, batch, K, 1)
    cond_means = (weights * means).sum(axis=-2)              # (ensemble, batch, output_dim)
    epistemic_var = jnp.var(cond_means, axis=0)              # (batch, output_dim)
    return jnp.mean(epistemic_var, axis=-1)                  # (batch,)
