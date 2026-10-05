"""Run with `python test_titans_mac.py` or `pytest test_titans_mac.py`."""
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

import titans_mac as tm

SMALL = tm.Config(seq_len=64, batch_size=2, d_model=32, n_layers=2, n_heads=2,
                  segment_len=16, n_persistent=2, mem_hidden=32, steps=3)


def test_one_shot_write_is_the_delta_rule():
    """With a linear memory, a unit-norm key, theta=1 and eta=alpha=0, one write stores v
    exactly (the delta rule, cf. Appendix C)."""
    d = 8
    W = [tm.dense(jax.random.key(0), d, d)]
    k = tm.l2_normalize(jax.random.normal(jax.random.key(1), (d,)))
    v = jax.random.normal(jax.random.key(2), (d,))
    one = jnp.ones(1)
    (W_new, _), _ = tm.memory_write((W, [jnp.zeros((d, d))]),
                                    k[None], v[None], k[None], one, 0 * one, 0 * one)
    assert float(jnp.abs(tm.memory_mlp(W, k) - v).max()) > 0.1
    assert float(jnp.abs(tm.memory_mlp(W_new, k) - v).max()) < 1e-4


def test_memory_learns_and_stays_stable():
    """At the largest theta and eta the gates allow, repeated writes reduce the loss
    without diverging."""
    d, n = 16, 8
    cfg = tm.Config()
    key = tm.KeyGen(0)
    W = [tm.dense(key(), d, 32), tm.dense(key(), 32, d)]
    k = tm.l2_normalize(jax.nn.silu(jax.random.normal(key(), (n, d))))
    v = tm.l2_normalize(jax.nn.silu(jax.random.normal(key(), (n, d))))

    def avg_loss(weights):
        return float(jnp.mean(jax.vmap(lambda a, b: tm.memory_loss(weights, a, b))(k, v)))

    state = (W, jax.tree.map(jnp.zeros_like, W))
    before = avg_loss(W)
    for _ in range(5):
        state, _ = tm.memory_write(state, k, v, k, jnp.full(n, cfg.max_inner_lr),
                                   jnp.full(n, cfg.max_momentum), jnp.full(n, 0.0))
    after = avg_loss(state[0])
    assert np.isfinite(after) and after < 0.3 * before, (before, after)


def test_causality():
    """Changing token j must not change predictions before j, through attention or memory."""
    cfg = SMALL
    params = tm.init_params(cfg)
    forward = jax.jit(lambda p, t: tm.model_apply(p, t, cfg))
    tokens = np.random.default_rng(0).integers(0, 256, cfg.seq_len)
    base = forward(params, jnp.asarray(tokens, dtype=jnp.int32))
    C = cfg.segment_len
    for j in [3, C, C + 5, cfg.seq_len - 1]:
        changed = tokens.copy()
        changed[j] = (changed[j] + 1) % 256
        out = forward(params, jnp.asarray(changed, dtype=jnp.int32))
        assert float(jnp.abs(out[:j] - base[:j]).max()) < 1e-5, f"future leaked into positions < {j}"
        assert float(jnp.abs(out[j] - base[j]).max()) > 1e-4, "token j had no effect"


def test_memory_is_the_only_bridge_between_segments():
    """Changing segment 0 changes segment 1's predictions only when the memory is on."""
    C = SMALL.segment_len
    tokens = np.random.default_rng(1).integers(0, 256, SMALL.seq_len)
    changed = tokens.copy()
    changed[:C] = (changed[:C] + 1) % 256
    for use_memory in (True, False):
        cfg = dataclasses.replace(SMALL, use_memory=use_memory)
        params = tm.init_params(cfg)
        forward = jax.jit(lambda p, t: tm.model_apply(p, t, cfg))
        a = forward(params, jnp.asarray(tokens, dtype=jnp.int32))[C:2 * C]
        b = forward(params, jnp.asarray(changed, dtype=jnp.int32))[C:2 * C]
        diff = float(jnp.abs(a - b).max())
        if use_memory:
            assert diff > 1e-4
        else:
            assert diff < 1e-5


def test_training_step_and_meta_gradient():
    """W_K is used only to write the memory, so a nonzero gradient on it means the outer
    loop is learning through the inner-loop updates."""
    cfg = SMALL
    params = tm.init_params(cfg)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.integers(0, 256, (cfg.batch_size, cfg.seq_len)), dtype=jnp.int32)
    y = jnp.asarray(rng.integers(0, 256, (cfg.batch_size, cfg.seq_len)), dtype=jnp.int32)
    loss, grads = jax.value_and_grad(lambda p: tm.loss_fn(p, x, y, cfg))(params)
    assert np.isfinite(float(loss))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))
    assert float(jnp.abs(grads["layers"][0]["memory"]["wk"]).max()) > 0
    params, opt, metrics = tm.make_train_step(cfg)(params, tm.adamw_init(params), x, y)
    assert np.isfinite(float(metrics["loss"]))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS  {name}")
