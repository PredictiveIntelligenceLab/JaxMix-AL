"""
Multimodal conditional distribution p*(y|x) implemented as a mixture of Gaussians
with diagonal covariances, following the specification in multimodal_conditional_spec.md.

Usage
-----
    generate_samples, log_prob = make_multimodal_conditional(
        d=16, m=16, L=4, K=3, p=128,
        tau=1.0, alpha=0.0, c_scale=1.0,
        key=jax.random.PRNGKey(0),
    )

    y  = generate_samples(x, key)  # x: (N, d) -> y: (N, m)
    lp = log_prob(y, x)            # -> (N,)
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers (single-sample, vectorised over batch via vmap)
# ---------------------------------------------------------------------------

def _rff(x, Omega, phi):
    """Random Fourier Features: h(x) = cos(Omega x + phi).  Shape: (p,)."""
    return jnp.cos(Omega @ x + phi)


def _mixing_weights(x, W, tau):
    """Softmax mixing weights pi_k(x).  Shape: (K,)."""
    logits = W @ x / tau          # (K,)
    return jax.nn.softmax(logits)


def _mixing_weights_structured(x, L, V, beta, r0, gamma, scale):
    """
    Structured mixing weights with a sharp unimodal-to-multimodal transition.

    Component 0 dominates inside radius r0 (unimodal region).
    Components 1..K-1 activate outside r0 (multimodal region),
    split into angular sectors by projection directions V.

    Parameters
    ----------
    x     : (d,) input vector
    L     : int, number of active-subspace dimensions (first L dims of x)
    V     : (K-1, L) angular direction matrix (unit-row not required)
    beta  : float, transition sharpness (higher = sharper boundary)
    r0    : float, transition radius in ||x[:L]|| space
    gamma : float, angular sharpness for exterior sector splitting
    scale : float, logit magnitude (higher = more one-hot-like weights)

    Returns
    -------
    pi : (K,) mixing weights summing to 1
    """
    # Radial distance in active subspace
    r = jnp.linalg.norm(x[:L])

    # Sharp radial gate: ~0 inside r0, ~1 outside r0
    radial_gate = 0.5 * (1 + jnp.tanh(beta * (r - r0)))

    # Angular allocation among K-1 exterior components
    angular_scores = jax.nn.softmax(gamma * V @ x[:L])  # (K-1,)

    # Interior gets (1-gate)*scale, each exterior component gets gate*score*scale
    interior_logit = scale * (1 - radial_gate)
    exterior_logits = scale * radial_gate * angular_scores

    logits = jnp.concatenate([interior_logit[None], exterior_logits])
    return jax.nn.softmax(logits)


def _means_and_log_vars(hx, pi, B, c, C, b_var, alpha):
    """
    Compute centered means mu_k and log-variances for all K components.

    Parameters
    ----------
    c : (K, m)  — component-specific offset constants added before centering

    Returns
    -------
    mu     : (K, m)  — centered means
    log_var: (K, m)  — log diagonal variances
    """
    # Free means including per-component offsets: (K, m)
    mu_tilde = jnp.einsum("kmp,p->km", B, hx) + c

    # Centering: weighted average across components, shape (m,)
    # Adding c before centering preserves E[y|x] = 0 exactly.
    mu_bar = jnp.einsum("k,km->m", pi, mu_tilde)

    # Centered means: mu_k = mu_tilde_k - sum_j pi_j mu_tilde_j
    mu = mu_tilde - mu_bar[None, :]       # (K, m)

    # Log-variances: (K, m)
    log_var = jnp.einsum("kmp,p->km", C, hx) + b_var  # (K, m)
    if alpha != 0.0:
        log_var = log_var - alpha * jnp.log(pi + 1e-30)[:, None]

    return mu, log_var


def _per_sample_components(x, W, tau, Omega, phi, B, c, C, b_var, alpha, pi_fn=None):
    """Return pi (K,), mu (K, m), log_var (K, m) for a single x.

    If *pi_fn* is provided it is called as ``pi_fn(x)`` to obtain the mixing
    weights, bypassing the default ``_mixing_weights(x, W, tau)``.
    """
    hx = _rff(x, Omega, phi)
    if pi_fn is not None:
        pi = pi_fn(x)
    else:
        pi = _mixing_weights(x, W, tau)
    mu, log_var = _means_and_log_vars(hx, pi, B, c, C, b_var, alpha)
    return pi, mu, log_var


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_multimodal_conditional(
    *,
    d: int = 16,
    m: int = 16,
    L: int = 4,
    K: int = 3,
    p: int = 128,
    tau: float = 1.0,
    alpha: float = 0.0,
    c_scale: float = 1.0,
    key: jax.Array,
    mixing_mode: str = "random",
    transition_sharpness: float = 8.0,
    transition_radius: float = 1.0,
    angular_sharpness: float = 5.0,
    logit_scale: float = 5.0,
):
    """
    Build a fixed-parameter multimodal conditional distribution p*(y|x).

    Parameters
    ----------
    d       : Input dimension  (x ∈ R^d, must be >= 10)
    m       : Output dimension (y ∈ R^m, must be >= 10)
    L       : Latent manifold dimension (3, 4, or 5)
    K       : Number of mixture components
    p       : Number of random Fourier features
    tau     : Softmax temperature for mixing weights
    alpha   : Variance-coupling coefficient (0 = disabled)
    c_scale : Std-dev of per-component offset constants c_k ∈ R^m.
              Larger values push modes further apart in output space.
              The zero conditional mean E[y|x]=0 is preserved for any value.
    key     : JAX PRNGKey used to draw all fixed parameters
    mixing_mode : ``"random"`` (original softmax-linear) or ``"structured"``
        (sharp radial boundary between a unimodal interior and multimodal
        exterior — designed to give active learning an exploitable landscape).
    transition_sharpness : (structured only) beta — higher = sharper boundary
    transition_radius    : (structured only) r0 — boundary in ||x[:L]|| space
    angular_sharpness    : (structured only) gamma — sharpness of angular sectors
    logit_scale          : (structured only) scale — magnitude of logits

    Returns
    -------
    generate_samples : callable (x, key) -> y
        x   : (N, d) array of inputs
        key : JAX PRNGKey for sampling
        y   : (N, m) samples from p*(y|x)

    log_prob : callable (y, x) -> log_p
        y     : (N, m)
        x     : (N, d)
        log_p : (N,)  log p*(y|x)
    """
    if mixing_mode not in ("random", "structured"):
        raise ValueError(f"mixing_mode must be 'random' or 'structured', got {mixing_mode!r}")

    # ---- draw all fixed parameters ----------------------------------------
    keys = jax.random.split(key, 10)  # one extra key for structured mixing

    # Manifold map l -> x(l) = tanh(A l + b_m)
    A        = jax.random.normal(keys[0], (d, L)) / jnp.sqrt(L)
    b_m      = jax.random.normal(keys[1], (d,))

    # RFF kernel
    Omega    = jax.random.normal(keys[2], (p, d)) / jnp.sqrt(d)
    phi      = jax.random.uniform(keys[3], (p,), minval=0.0, maxval=2 * jnp.pi)

    # Mixing weight vectors (used only when mixing_mode="random")
    W        = jax.random.normal(keys[4], (K, d)) / jnp.sqrt(d)

    # Mean maps B_k, log-variance maps C_k
    B        = jax.random.normal(keys[5], (K, m, p)) / jnp.sqrt(p)
    C        = jax.random.normal(keys[6], (K, m, p)) / jnp.sqrt(p)
    b_var    = jnp.zeros((K, m))   # baseline log-variance per component

    # Per-component offset constants c_k ∈ R^m — shift uncentered means so
    # modes are more distinct; centering in _means_and_log_vars still ensures
    # E[y|x] = 0 exactly.
    c        = jax.random.normal(keys[7], (K, m)) * c_scale

    # ---- mixing weight function -------------------------------------------
    pi_fn = None  # default: use _mixing_weights(x, W, tau)

    if mixing_mode == "structured":
        assert K >= 2, "structured mixing requires K >= 2"
        # Draw K-1 random directions in R^L for angular sector splitting
        V = jax.random.normal(keys[9], (K - 1, L))
        V = V / jnp.linalg.norm(V, axis=-1, keepdims=True)  # unit rows

        # Create closure that captures all structured params
        _L = L
        _V, _beta, _r0 = V, transition_sharpness, transition_radius
        _gamma, _scale = angular_sharpness, logit_scale

        def pi_fn(x):
            return _mixing_weights_structured(x, _L, _V, _beta, _r0, _gamma, _scale)

    # ---- close over fixed params ------------------------------------------
    _components = partial(
        _per_sample_components,
        W=W, tau=tau, Omega=Omega, phi=phi, B=B, c=c, C=C, b_var=b_var, alpha=alpha,
        pi_fn=pi_fn,
    )
    _components_batch = jax.vmap(_components)  # (N, d) -> (N, K), (N, K, m), (N, K, m)

    # ---- generate_samples --------------------------------------------------
    @jax.jit
    def generate_samples(x, key):
        """
        Sample y ~ p*(y|x) for each row of x.

        Parameters
        ----------
        x   : (N, d) array of conditioning inputs
        key : JAX PRNGKey

        Returns
        -------
        y : (N, m)
        """
        pi, mu, log_var = _components_batch(x)  # (N,K), (N,K,m), (N,K,m)

        key_cat, key_norm = jax.random.split(key)

        # Sample component index k* for each of the N inputs
        k_star = jax.random.categorical(key_cat, jnp.log(pi + 1e-30))  # (N,)

        # Gather mu and log_var for the selected component
        # mu[n, k_star[n], :]  ->  (N, m)
        mu_sel      = mu[jnp.arange(x.shape[0]), k_star, :]       # (N, m)
        log_var_sel = log_var[jnp.arange(x.shape[0]), k_star, :]  # (N, m)

        # Reparameterised sample: y = mu + std * eps
        eps = jax.random.normal(key_norm, mu_sel.shape)            # (N, m)
        y   = mu_sel + jnp.exp(0.5 * log_var_sel) * eps           # (N, m)
        return y

    # ---- log_prob ----------------------------------------------------------
    @jax.jit
    def log_prob(y, x):
        """
        Evaluate log p*(y|x) = log sum_k pi_k(x) N(y; mu_k(x), Sigma_k(x)).

        Parameters
        ----------
        y : (N, m)
        x : (N, d)

        Returns
        -------
        log_p : (N,)
        """
        pi, mu, log_var = _components_batch(x)  # (N,K), (N,K,m), (N,K,m)

        # Per-component log-likelihood: log N(y; mu_k, diag(exp(log_var_k)))
        # y[:, None, :] - mu  shape: (N, K, m)
        diff     = y[:, None, :] - mu                                # (N, K, m)
        log_det  = jnp.sum(log_var, axis=-1)                         # (N, K)
        quad     = jnp.sum(diff ** 2 / jnp.exp(log_var), axis=-1)   # (N, K)
        log_gauss = -0.5 * (m * jnp.log(2 * jnp.pi) + log_det + quad)  # (N, K)

        # log sum_k pi_k * N(y; mu_k, Sigma_k)  via logsumexp
        log_p = jax.scipy.special.logsumexp(
            jnp.log(pi + 1e-30) + log_gauss, axis=-1
        )  # (N,)
        return log_p

    # ---- optional: return fixed params for reproducibility ----------------
    def get_params():
        params = dict(A=A, b_m=b_m, Omega=Omega, phi=phi, W=W, B=B, c=c, C=C, b_var=b_var,
                      mixing_mode=mixing_mode)
        if mixing_mode == "structured":
            params.update(V=V, transition_sharpness=transition_sharpness,
                          transition_radius=transition_radius,
                          angular_sharpness=angular_sharpness,
                          logit_scale=logit_scale)
        return params

    generate_samples.log_prob  = log_prob
    generate_samples.get_params = get_params
    generate_samples._components_batch = _components_batch

    return generate_samples, log_prob


# ---------------------------------------------------------------------------
# Manifold sampler (convenience)
# ---------------------------------------------------------------------------

def make_manifold_sampler(*, d: int = 16, L: int = 4, key: jax.Array):
    """
    Return a function that maps latent l ~ N(0, I_L) to x = tanh(A l + b).

    Parameters
    ----------
    d   : Output (input-space) dimension
    L   : Latent dimension
    key : JAX PRNGKey

    Returns
    -------
    sample_x : callable (n, key) -> x of shape (n, d)
    """
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (d, L)) / jnp.sqrt(L)
    b = jax.random.normal(k2, (d,))

    @partial(jax.jit, static_argnums=(0,))
    def sample_x(n: int, key: jax.Array):
        l = jax.random.normal(key, (n, L))
        return jnp.tanh(l @ A.T + b)

    return sample_x


def build_distribution(
    *,
    d: int = 16,
    m: int = 16,
    L: int = 4,
    K: int = 3,
    p: int = 128,
    tau: float = 1.0,
    alpha: float = 0.0,
    c_scale: float = 3.0,
    seed: int = 42,
    mixing_mode: str = "random",
    transition_sharpness: float = 8.0,
    transition_radius: float = 1.0,
    angular_sharpness: float = 5.0,
    logit_scale: float = 5.0,
):
    """Build the fixed-parameter distribution p*(y|x)."""
    key = jax.random.PRNGKey(seed)
    return make_multimodal_conditional(
        d=d, m=m, L=L, K=K, p=p,
        tau=tau, alpha=alpha, c_scale=c_scale,
        key=key,
        mixing_mode=mixing_mode,
        transition_sharpness=transition_sharpness,
        transition_radius=transition_radius,
        angular_sharpness=angular_sharpness,
        logit_scale=logit_scale,
    )


def generate_manifold_inputs(
    n: int,
    *,
    key: jax.Array,
    d: int = 16,
    L: int = 4,
    manifold_seed: int = 1,
) -> jnp.ndarray:
    """Sample ``n`` input points from the fixed latent manifold."""
    manifold_key = jax.random.PRNGKey(manifold_seed)
    sample_x = make_manifold_sampler(d=d, L=L, key=manifold_key)
    return sample_x(n, key)


def generate_dataset(
    n: int,
    generate_samples,
    *,
    key: jax.Array,
    d: int = 16,
    L: int = 4,
    manifold_seed: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Draw ``n`` (x, y) pairs where x is on the manifold and y ~ p*(y|x)."""
    _, x_key, y_key = random.split(key, 3)
    x = generate_manifold_inputs(n, key=x_key, d=d, L=L, manifold_seed=manifold_seed)
    y = generate_samples(x, y_key)
    return x, y


def compute_true_nll(log_prob, inputs: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute -E[log p*(y|x)] on a dataset."""
    lp = log_prob(targets, inputs)
    return float(-lp.mean())


def plot_output_marginals(
    y: jnp.ndarray,
    n_dims: int = 4,
    title: str = "Output marginals  (first k dimensions)",
) -> None:
    """Plot histograms of the first ``n_dims`` output dimensions."""
    y_np = np.asarray(y)
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 3.5))
    for i, ax in enumerate(axes):
        ax.hist(y_np[:, i], bins=60, edgecolor="none", alpha=0.75)
        ax.axvline(0, color="red", linewidth=1.5, linestyle="--", label="mean = 0")
        ax.set_title(f"$y_{{{i}}}$")
        ax.set_xlabel(f"$y_{{{i}}}$")
        ax.set_ylabel("Count")
    axes[0].legend()
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_output_scatter_by_component(
    x: jnp.ndarray,
    y: jnp.ndarray,
    generate_samples,
    K: int = 3,
    tau: float = 1.0,
    alpha: float = 0.0,
    dim_i: int = 0,
    dim_j: int = 1,
    title: str = "Samples coloured by dominant mixture component",
) -> None:
    """Plot output samples coloured by the dominant mixture component at each input."""
    pi_batch, _, _ = generate_samples._components_batch(x)
    k_star = np.asarray(jnp.argmax(pi_batch, axis=-1))
    y_np = np.asarray(y)
    colors = ["steelblue", "tomato", "seagreen", "orange", "purple"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for k in range(K):
        mask = k_star == k
        ax.scatter(
            y_np[mask, dim_i], y_np[mask, dim_j],
            s=5, alpha=0.5, color=colors[k % len(colors)], label=f"dominant k={k}",
        )
    ax.set_xlabel(f"$y_{{{dim_i}}}$")
    ax.set_ylabel(f"$y_{{{dim_j}}}$")
    ax.set_title(title)
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.show()


def plot_log_prob_distribution(
    log_prob,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    title: str = "True log-likelihood distribution",
) -> None:
    """Plot the distribution of true log likelihoods across a dataset."""
    lp = np.asarray(log_prob(targets, inputs))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lp, bins=60, edgecolor="none", alpha=0.75)
    ax.axvline(lp.mean(), color="red", linewidth=1.5, linestyle="--",
               label=f"mean = {lp.mean():.2f}  (= -true NLL)")
    ax.set_xlabel(r"$\log\,p_\star(y\,|\,x)$")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_mixing_weights_2d(
    x: jnp.ndarray,
    generate_samples,
    K: int = 3,
    tau: float = 1.0,
    alpha: float = 0.0,
    dim_i: int = 0,
    dim_j: int = 1,
) -> None:
    """Plot mixture weights over a two-dimensional projection of the inputs."""
    pi_batch, _, _ = generate_samples._components_batch(x)
    pi_np = np.asarray(pi_batch)
    x_np = np.asarray(x)

    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4))
    for k, ax in enumerate(axes):
        sc = ax.scatter(
            x_np[:, dim_i], x_np[:, dim_j],
            c=pi_np[:, k], s=5, alpha=0.6, cmap="RdYlBu_r", vmin=0.0, vmax=1.0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(f"$x_{{{dim_i}}}$")
        ax.set_ylabel(f"$x_{{{dim_j}}}$")
        ax.set_title(f"$\\pi_{{{k}}}(x)$")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Mixing weights in 2-D input projection  ($x_{{{dim_i}}}$, $x_{{{dim_j}}}$)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()
