"""Ternary phase competition benchmark for active learning."""

import jax
import jax.numpy as jnp
import jax.random as jr


def make_phase_system(seed=0, n_phases=4, tau_G=0.15, n_proc_params=2,
                      c_mu_scale=3.0):
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 10)

    raw = jr.normal(keys[0], (n_phases, 3, 3)) * 0.5
    H = jnp.einsum("pij,pik->pjk", raw, raw) + 0.3 * jnp.eye(3)[None, :, :]

    b = jr.normal(keys[1], (n_phases, 3)) * 2.0

    c_mu = jr.normal(keys[2], (n_phases, 3)) * c_mu_scale
    d_mu = jr.normal(keys[3], (n_phases,)) * 2.0
    omega_mu = jr.normal(keys[4], (n_phases, 3)) * 2.0

    e_sigma = jr.normal(keys[5], (n_phases, 3)) * 0.5
    f_sigma = jr.normal(keys[6], (n_phases,)) * 0.3 - 1.0

    if n_proc_params > 0:
        W_proc = jr.normal(keys[7], (n_phases, n_proc_params)) * 1.0
    else:
        W_proc = None

    return dict(
        n_phases=n_phases, tau_G=tau_G, n_proc_params=n_proc_params,
        H=H, b=b, c_mu=c_mu, d_mu=d_mu, omega_mu=omega_mu,
        e_sigma=e_sigma, f_sigma=f_sigma, W_proc=W_proc,
    )


def _free_energies(x3, system):
    H, b = system["H"], system["b"]
    quad = jnp.einsum("i,pij,j->p", x3, H, x3)
    lin = jnp.einsum("pi,i->p", b, x3)
    return quad + lin


def _phase_probs(x3, system):
    G = _free_energies(x3, system)
    return jax.nn.softmax(-G / system["tau_G"])


def _means(x3, proc, system):
    c, d, omega = system["c_mu"], system["d_mu"], system["omega_mu"]
    mu = jnp.einsum("pi,i->p", c, x3) + d
    mu = mu + 0.5 * jnp.sin(jnp.einsum("pi,i->p", omega, x3))
    if proc is not None and system["W_proc"] is not None and proc.shape[0] > 0:
        mu = mu + jnp.einsum("pj,j->p", system["W_proc"], proc)
    return mu


def _log_variances(x3, system):
    e, f = system["e_sigma"], system["f_sigma"]
    return jnp.einsum("pi,i->p", e, x3) + f


def _components(x3, proc, system):
    pi = _phase_probs(x3, system)
    mu = _means(x3, proc, system)
    log_var = _log_variances(x3, system)
    return pi, mu, jnp.exp(log_var)


def _sample_one(x3, proc, system, key):
    pi, mu, sigma2 = _components(x3, proc, system)
    k1, k2 = jr.split(key)
    phi = jr.categorical(k1, jnp.log(pi + 1e-30))
    y = mu[phi] + jnp.sqrt(sigma2[phi]) * jr.normal(k2)
    return y


def _log_prob_one(y, x3, proc, system):
    pi, mu, sigma2 = _components(x3, proc, system)
    log_components = (
        jnp.log(pi + 1e-30)
        - 0.5 * jnp.log(2 * jnp.pi * sigma2)
        - 0.5 * (y - mu) ** 2 / sigma2
    )
    return jax.scipy.special.logsumexp(log_components)


def _split_x(x, n_proc):
    x_AB = x[:, :2]
    x_C = 1.0 - x_AB[:, 0:1] - x_AB[:, 1:2]
    x3 = jnp.concatenate([x_AB, x_C], axis=-1)
    proc = x[:, 2:] if n_proc > 0 else jnp.zeros((x.shape[0], 0))
    return x3, proc


def generate_dataset(system, key, n, proc_range=(-1.0, 1.0)):
    n_proc = system["n_proc_params"]
    k1, k2, k3 = jr.split(key, 3)

    x3 = jr.dirichlet(k1, jnp.ones(3), shape=(n,))
    x_AB = x3[:, :2]

    if n_proc > 0:
        proc = jr.uniform(k2, (n, n_proc),
                          minval=proc_range[0], maxval=proc_range[1])
    else:
        proc = jnp.zeros((n, 0))

    sample_keys = jr.split(k3, n)

    @jax.vmap
    def _sample_batch(x3_i, proc_i, key_i):
        return _sample_one(x3_i, proc_i, system, key_i)

    y = _sample_batch(x3, proc, sample_keys)

    x = jnp.concatenate([x_AB, proc], axis=-1)
    return x, y[:, None]


def log_prob(system, y, x):
    n_proc = system["n_proc_params"]
    x3, proc = _split_x(x, n_proc)

    @jax.vmap
    def _lp_batch(y_i, x3_i, proc_i):
        return _log_prob_one(y_i.squeeze(), x3_i, proc_i, system)

    return _lp_batch(y, x3, proc)


def create_shared_ternary_data(
    seed=0,
    system_seed=0,
    candidate_sample_count=10_000,
    test_sample_count=2_000,
    initial_sample_count=50,
    n_phases=4,
    tau_G=0.15,
    n_proc_params=2,
    c_mu_scale=3.0,
):
    system = make_phase_system(
        seed=system_seed, n_phases=n_phases,
        tau_G=tau_G, n_proc_params=n_proc_params,
        c_mu_scale=c_mu_scale,
    )

    key = jr.PRNGKey(seed)
    k_pool, k_test, k_state = jr.split(key, 3)

    x_all, y_all = generate_dataset(system, k_pool, candidate_sample_count)
    x_test, y_test = generate_dataset(system, k_test, test_sample_count)

    true_nll = -jnp.mean(log_prob(system, y_test, x_test))

    data = {
        "initial_labeled_inputs":  x_all[:initial_sample_count],
        "initial_labeled_targets": y_all[:initial_sample_count],
        "remaining_pool_inputs":   x_all[initial_sample_count:],
        "remaining_pool_targets":  y_all[initial_sample_count:],
        "test_data":               (x_test, y_test),
        "shared_state_key":        k_state,
        "true_nll":                float(true_nll),
        "system":                  system,
    }
    return data


def _simplex_grid(n_grid):
    points = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            k = n_grid - i - j
            points.append([i / n_grid, j / n_grid, k / n_grid])
    return jnp.array(points)


def _to_cartesian(x3):
    cart_x = x3[:, 1] + 0.5 * x3[:, 2]
    cart_y = (float(jnp.sqrt(3.0)) / 2.0) * x3[:, 2]
    return cart_x, cart_y


def verify_phase_diagram(system, n_grid=200, boundary_threshold=0.7):
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    x3 = _simplex_grid(n_grid)

    probs_fn = jax.vmap(lambda x: _phase_probs(x, system))
    probs = probs_fn(x3)
    dominant = jnp.argmax(probs, axis=-1)
    max_prob = jnp.max(probs, axis=-1)
    boundary = (max_prob < boundary_threshold).astype(float)

    means_fn = jax.vmap(lambda x: _means(x, jnp.zeros(system["n_proc_params"]), system))
    all_means = means_fn(x3)
    dominant_mean = jnp.take_along_axis(all_means, dominant[:, None], axis=1).squeeze()

    cart_x, cart_y = _to_cartesian(x3)
    triang = tri.Triangulation(cart_x, cart_y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].tripcolor(triang, dominant, cmap="Set2", alpha=0.8)
    axes[0].set_title("Dominant phase")

    axes[1].tripcolor(triang, boundary, cmap="Reds", alpha=0.8)
    axes[1].set_title(f"Phase boundary (max π < {boundary_threshold})")

    tcf = axes[2].tripcolor(triang, dominant_mean, cmap="viridis", alpha=0.8)
    plt.colorbar(tcf, ax=axes[2])
    axes[2].set_title("Response μ(x) of dominant phase")

    verts = jnp.array([[0, 0], [1, 0], [0.5, float(jnp.sqrt(3.0)) / 2.0], [0, 0]])
    for ax in axes:
        ax.plot(verts[:, 0], verts[:, 1], "k-", lw=1.5)
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.95)
        ax.axis("off")

    plt.suptitle("Ternary phase competition — ground truth", fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


def plot_acquired_on_simplex(data, results, methods=None,
                             n_grid=150, boundary_threshold=0.7):
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    system = data["system"]
    if methods is None:
        methods = list(results.keys())

    x3_grid = _simplex_grid(n_grid)
    probs_fn = jax.vmap(lambda x: _phase_probs(x, system))
    max_prob = jnp.max(probs_fn(x3_grid), axis=-1)
    boundary = (max_prob < boundary_threshold).astype(float)
    cart_gx, cart_gy = _to_cartesian(x3_grid)
    triang = tri.Triangulation(cart_gx, cart_gy)

    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    verts = jnp.array([[0, 0], [1, 0], [0.5, float(jnp.sqrt(3.0)) / 2.0], [0, 0]])

    for ax, name in zip(axes, methods):
        ax.tripcolor(triang, boundary, cmap="Reds", alpha=0.3)

        state = results[name]["state"]
        x_labeled = state.train_inputs
        x_AB = x_labeled[:, :2]
        x_C = 1.0 - x_AB[:, 0] - x_AB[:, 1]
        cart_x = x_AB[:, 1] + 0.5 * x_C
        cart_y = (float(jnp.sqrt(3.0)) / 2.0) * x_C
        ax.scatter(cart_x, cart_y, s=8, alpha=0.5, c="steelblue", edgecolors="none")

        ax.plot(verts[:, 0], verts[:, 1], "k-", lw=1.5)
        ax.set_title(f"{name} (n={x_labeled.shape[0]})")
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.95)
        ax.axis("off")

    plt.suptitle("Acquired compositions overlaid on phase boundaries", fontsize=13)
    plt.tight_layout()
    plt.show()
    return fig
