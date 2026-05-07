"""
Coupled double-well benchmark for active learning.

P particles in coupled 1D double-well potentials evolved with overdamped
Langevin dynamics.  The noise-to-barrier ratio sigma^2/a controls a sharp
unimodal-to-bimodal phase transition in the output distribution p(y|x).

Usage
-----
    from coupled_double_well import generate_dataset, make_oracle

    x_pool, y_pool = generate_dataset(jax.random.PRNGKey(0), n=50_000)
    x_test, y_test = generate_dataset(jax.random.PRNGKey(1), n=2_000)

    oracle = make_oracle()
    y_new = oracle(x_query, jax.random.PRNGKey(42))
"""

import jax
import jax.numpy as jnp
import jax.random as jr


# ── Simulator ────────────────────────────────────────────────────────────

def _step(q, noise, a, kappa, sigma, dt):
    """Single Euler-Maruyama step."""
    P = q.shape[0]
    # Double-well force: -dV/dq_i = a_i * (q_i - q_i^3)
    force = a * (q - q ** 3)
    # Nearest-neighbor coupling (open boundary)
    if P > 1:
        left  = jnp.concatenate([jnp.zeros(1), q[:-1]])
        right = jnp.concatenate([q[1:], jnp.zeros(1)])
        n_left  = jnp.concatenate([jnp.zeros(1), jnp.ones(P - 1)])
        n_right = jnp.concatenate([jnp.ones(P - 1), jnp.zeros(1)])
        coupling = kappa * (n_left * (left - q) + n_right * (right - q))
    else:
        coupling = 0.0
    q_new = q + (force + coupling) * dt + sigma * jnp.sqrt(dt) * noise
    return q_new


def simulate(q0, sigma, kappa, key, a=None, T=5.0, dt=0.005, n_snapshots=0):
    """
    Integrate one trajectory.

    Parameters
    ----------
    q0    : (P,) initial positions
    sigma : scalar noise intensity
    kappa : scalar coupling strength
    key   : PRNGKey
    a     : (P,) barrier heights (default: all ones)
    T     : integration time
    dt    : time step
    n_snapshots : if >0, return (n_snapshots, P) array of intermediate + final
                  positions evenly spaced in [T/n_snapshots, T].
                  if 0, return (P,) final positions only.

    Returns
    -------
    q_final : (P,) if n_snapshots == 0, else (n_snapshots, P)
    """
    P = q0.shape[0]
    if a is None:
        a = jnp.ones(P)
    n_steps = int(T / dt)

    noise = jr.normal(key, (n_steps, P))

    if n_snapshots > 0:
        # Need to save the full trajectory for snapshot extraction
        def scan_fn(q, noise_i):
            q = _step(q, noise_i, a, kappa, sigma, dt)
            return q, q

        q_final, q_traj = jax.lax.scan(scan_fn, q0, noise)
        indices = jnp.linspace(n_steps // n_snapshots - 1, n_steps - 1,
                               n_snapshots).astype(int)
        return q_traj[indices]  # (n_snapshots, P)
    else:
        # Only need final state — don't save trajectory (saves memory)
        def scan_fn(q, noise_i):
            q = _step(q, noise_i, a, kappa, sigma, dt)
            return q, None

        q_final, _ = jax.lax.scan(scan_fn, q0, noise)
        return q_final  # (P,)


# Batched version
_simulate_batch = jax.vmap(
    simulate, in_axes=(0, 0, 0, 0, None, None, None, None),
)


# ── Dataset generation ───────────────────────────────────────────────────

def generate_dataset(key, n, P=5, T=5.0, dt=0.005, n_snapshots=0,
                     sigma_range=(0.3, 2.0), kappa_range=(0.0, 3.0),
                     chunk_size=5000):
    """
    Generate (x, y) pairs.

    Parameters
    ----------
    key   : PRNGKey
    n     : number of samples
    chunk_size : generate in chunks to avoid OOM from vmap over large N

    Returns
    -------
    x : (n, P+2) — [q0, sigma, kappa]
    y : (n, P) if n_snapshots==0, else (n, n_snapshots*P)
    """
    k1, k2, k3, k4 = jr.split(key, 4)

    q0 = jr.uniform(k1, (n, P), minval=-1.5, maxval=1.5)
    sigma = jr.uniform(k2, (n,), minval=sigma_range[0], maxval=sigma_range[1])
    kappa = jr.uniform(k3, (n,), minval=kappa_range[0], maxval=kappa_range[1])
    sim_keys = jr.split(k4, n)

    # Generate in chunks to limit peak memory from vmap
    y_chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        y_chunk = _simulate_batch(
            q0[start:end], sigma[start:end], kappa[start:end],
            sim_keys[start:end], None, T, dt, n_snapshots,
        )
        y_chunks.append(y_chunk)
    y = jnp.concatenate(y_chunks, axis=0)

    if n_snapshots > 0:
        y = y.reshape(n, -1)  # (n, n_snapshots * P)

    x = jnp.concatenate([q0, sigma[:, None], kappa[:, None]], axis=-1)
    return x, y


# ── Oracle for active learning ───────────────────────────────────────────

def make_oracle(P=5, T=5.0, dt=0.005, n_snapshots=0, chunk_size=5000):
    """
    Returns a function oracle(x_query, key) -> y that labels query points.
    x_query : (n, P+2)
    """
    def oracle(x_query, key):
        n = x_query.shape[0]
        q0 = x_query[:, :P]
        sigma = x_query[:, P]
        kappa = x_query[:, P + 1]
        sim_keys = jr.split(key, n)

        y_chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            y_chunk = _simulate_batch(
                q0[start:end], sigma[start:end], kappa[start:end],
                sim_keys[start:end], None, T, dt, n_snapshots,
            )
            y_chunks.append(y_chunk)
        y = jnp.concatenate(y_chunks, axis=0)

        if n_snapshots > 0:
            y = y.reshape(n, -1)
        return y
    return oracle


# ── Diagnostic: verify phase diagram ─────────────────────────────────────

def verify_phase_diagram():
    """
    Quick sanity check: plot histograms of q_1(T) for different sigma values,
    starting from q_1(0) = -0.5.  Should transition from unimodal to bimodal.
    """
    import matplotlib.pyplot as plt

    P = 1  # single particle for clarity
    n_replicates = 2000
    sigmas = [0.3, 0.7, 1.0, 1.5]

    fig, axes = plt.subplots(1, len(sigmas), figsize=(3.5 * len(sigmas), 3),
                             sharex=True, sharey=True)
    for ax, sigma_val in zip(axes, sigmas):
        key = jr.PRNGKey(0)
        q0 = jnp.full((n_replicates, P), -0.5)
        sigma = jnp.full(n_replicates, sigma_val)
        kappa = jnp.zeros(n_replicates)
        sim_keys = jr.split(key, n_replicates)
        y = _simulate_batch(q0, sigma, kappa, sim_keys, None, 5.0, 0.005, 0)
        ax.hist(y[:, 0], bins=60, density=True, alpha=0.7, color="steelblue")
        ax.set_title(f"$\\sigma$ = {sigma_val}")
        ax.set_xlabel("$q(T)$")
        ax.axvline(-1, color="k", ls="--", lw=0.8)
        ax.axvline(+1, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("density")
    plt.suptitle("Phase transition: unimodal → bimodal", fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Generating 1000 samples...")
    x, y = generate_dataset(jr.PRNGKey(0), 1000, P=5)
    print(f"  x: {x.shape}, y: {y.shape}")
    print(f"  x range: [{float(x.min()):.2f}, {float(x.max()):.2f}]")
    print(f"  y range: [{float(y.min()):.2f}, {float(y.max()):.2f}]")

    print("\nVerifying phase diagram...")
    verify_phase_diagram()
