import jax
import jax.numpy as jnp


def random_acquisition(
    model,
    inputs: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Assigns random scores to inputs as a baseline."""
    del model
    return jax.random.uniform(key, shape=(inputs.shape[0],))
