"""When is the memory's inner loop stable?

Writes 256 key-value pairs (values a fixed nonlinear function of the keys) into a fresh
memory with constant theta and eta, then reports held-out loss after / before. Below 1
means the memory learned; nan means it diverged.
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp

import titans_mac as tm

D, H, T = 64, 128, 256


def make_data(seed=0):
    key = tm.KeyGen(seed)
    weights = [tm.dense(key(), D, H), tm.dense(key(), H, D)]
    G = jax.random.normal(key(), (D, D))

    def pairs(n):
        k = tm.l2_normalize(jax.nn.silu(jax.random.normal(key(), (n, D))))
        return k, jax.nn.silu(k @ G)

    return weights, pairs(T), pairs(64)


@functools.partial(jax.jit, static_argnums=2)
def heldout_ratio(theta, eta, normalize_values, weights, train, test):
    (k, v), (k_test, v_test) = train, test
    if normalize_values:
        v, v_test = tm.l2_normalize(v), tm.l2_normalize(v_test)

    def loss(w):
        return jnp.mean(jax.vmap(lambda a, b: tm.memory_loss(w, a, b))(k_test, v_test))

    full = lambda x: jnp.full(T, x)
    state = (weights, jax.tree.map(jnp.zeros_like, weights))
    (after, _), _ = tm.memory_write(state, k, v, k, full(theta), full(eta), full(0.0025))
    return loss(after) / loss(weights)


if __name__ == "__main__":
    weights, train, test = make_data()
    thetas, etas = [0.05, 0.1, 0.3, 0.5, 1.0], [0.0, 0.5, 0.7, 0.9]
    for normalize in (False, True):
        print(f"\nvalues {'L2-normalized' if normalize else 'raw'}:")
        print("          " + "".join(f"theta={t:<6}" for t in thetas))
        for eta in etas:
            ratios = [float(heldout_ratio(t, eta, normalize, weights, train, test)) for t in thetas]
            print(f"eta={eta:<4}  " + "".join(f"{r:<12.3f}" for r in ratios))
    print("\ntitans_mac.py uses theta <= 0.3, eta <= 0.7, normalized values.")
