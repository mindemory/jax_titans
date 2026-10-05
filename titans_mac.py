"""Titans, Memory as a Context (MAC) variant, in pure JAX.

Behrouz, Zhong & Mirrokni (2024), "Titans: Learning to Memorize at Test Time".
Equation numbers refer to the paper.

Each MAC layer splits the sequence into segments of C tokens and, for each segment:
    1. recall    h = M_{t-1}(q)                          Eq. 21
    2. attend    y = Attn([persistent | h | segment])    Eqs. 22-23
    3. write     M_t = M_{t-1} updated with y            Eq. 24
    4. gate      o = y * gate(M_t(y))                    Eq. 25

The memory M is a small MLP whose weights are updated by gradient descent during the
forward pass (the inner loop). The outer loop trains everything else, including M's
initial weights, by backpropagating through those updates.
"""
import argparse
import dataclasses
import os
import time
import urllib.request

import numpy as np
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Config:
    vocab_size: int = 256        # raw bytes
    seq_len: int = 256
    batch_size: int = 8
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    segment_len: int = 32
    n_persistent: int = 4
    mem_depth: int = 2           # 1 = linear memory
    mem_hidden: int = 128
    max_inner_lr: float = 0.3    # upper bounds on theta and eta, see stability_sweep.py
    max_momentum: float = 0.7
    use_memory: bool = True
    steps: int = 600
    lr: float = 3e-3
    warmup_steps: int = 50
    weight_decay: float = 0.1
    clip_norm: float = 1.0
    seed: int = 0
    log_every: int = 50

    def __post_init__(self):
        assert self.seq_len % self.segment_len == 0
        assert self.d_model % self.n_heads == 0


TINY_SHAKESPEARE = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
                    "data/tinyshakespeare/input.txt")


def load_bytes(path):
    if not os.path.exists(path):
        print(f"Downloading Tiny Shakespeare to {path} ...")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE, path)
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    split = int(0.9 * len(data))
    return data[:split], data[split:]


def batches(data, cfg, rng):
    while True:
        starts = rng.integers(0, len(data) - cfg.seq_len - 1, size=cfg.batch_size)
        x = np.stack([data[s: s + cfg.seq_len] for s in starts])
        y = np.stack([data[s + 1: s + cfg.seq_len + 1] for s in starts])
        yield jnp.asarray(x, dtype=jnp.int32), jnp.asarray(y, dtype=jnp.int32)


class KeyGen:
    def __init__(self, seed):
        self.key = jax.random.key(seed)

    def __call__(self):
        self.key, sub = jax.random.split(self.key)
        return sub


def dense(key, n_in, n_out):
    return jax.random.normal(key, (n_in, n_out)) / n_in ** 0.5


def init_params(cfg):
    key = KeyGen(cfg.seed)
    d = cfg.d_model
    mem_dims = [d] + [cfg.mem_hidden] * (cfg.mem_depth - 1) + [d]

    def layer():
        p = {
            "norm_attn": jnp.ones(d),
            "norm_ffn": jnp.ones(d),
            "attn": {name: dense(key(), d, d) for name in ("wq", "wk", "wv", "wo")},
            "persistent": jax.random.normal(key(), (cfg.n_persistent, d)),
            "memory": {
                "wq": dense(key(), d, d),
                "wk": dense(key(), d, d),
                "wv": dense(key(), d, d),
                "init": [dense(key(), a, b) for a, b in zip(mem_dims[:-1], mem_dims[1:])],  # M_0
                # theta, eta, alpha start at half their maxima, half, and sigmoid(-6)
                "w_gates": jnp.zeros((d, 3)),
                "b_gates": jnp.array([0.0, 0.0, -6.0]),
                "norm_gate": jnp.ones(d),
            },
            "ffn": {"w1": dense(key(), d, 4 * d), "w2": dense(key(), 4 * d, d)},
        }
        if not cfg.use_memory:
            del p["memory"]   # created anyway so both variants draw the same random numbers
        return p

    return {
        "embed": jax.random.normal(key(), (cfg.vocab_size, d)),
        "layers": [layer() for _ in range(cfg.n_layers)],
        "norm_out": jnp.ones(d),
        "unembed": dense(key(), d, cfg.vocab_size),
    }


def rms_norm(x, scale, eps=1e-6):
    return x / jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * scale


def l2_normalize(x, eps=1e-6):
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def sinusoids(length, d):
    pos = np.arange(length)[:, None]
    freq = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
    pe = np.zeros((length, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    return jnp.asarray(pe)


def attention(p, x_q, x_kv, mask, n_heads):
    Tq, d = x_q.shape
    hd = d // n_heads
    q = (x_q @ p["wq"]).reshape(Tq, n_heads, hd)
    k = (x_kv @ p["wk"]).reshape(-1, n_heads, hd)
    v = (x_kv @ p["wv"]).reshape(-1, n_heads, hd)
    scores = jnp.einsum("qhe,khe->hqk", q, k) / hd ** 0.5
    scores = jnp.where(mask[None], scores, -1e30)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hqk,khe->qhe", weights, v).reshape(Tq, d) @ p["wo"]


def feed_forward(p, x):
    return jax.nn.gelu(x @ p["w1"]) @ p["w2"]


# Long-term memory (Section 3.1)

def memory_mlp(weights, x):
    for i, w in enumerate(weights):
        x = x @ w
        if i < len(weights) - 1:
            x = jax.nn.silu(x)
    return x


def memory_loss(weights, k, v):
    # Eq. 12. The 1/2 is not in the paper; it only rescales theta.
    return 0.5 * jnp.sum((memory_mlp(weights, k) - v) ** 2)


def memory_qkv(p, x):
    # The paper normalizes q and k (Sec. 4.4). Normalizing v as well keeps the inner loop
    # stable for a nonlinear memory.
    q = l2_normalize(jax.nn.silu(x @ p["wq"]))
    k = l2_normalize(jax.nn.silu(x @ p["wk"]))
    v = l2_normalize(jax.nn.silu(x @ p["wv"]))
    return q, k, v


def memory_gates(p, x, cfg):
    g = x @ p["w_gates"] + p["b_gates"]
    theta = cfg.max_inner_lr * jax.nn.sigmoid(g[:, 0])
    eta = cfg.max_momentum * jax.nn.sigmoid(g[:, 1])
    alpha = jax.nn.sigmoid(g[:, 2])
    return theta, eta, alpha


def memory_write(state, k, v, q, theta, eta, alpha):
    """Write tokens one at a time (Eqs. 13-14). Also returns M(q_i) read right after
    writing token i, so that position i depends only on tokens <= i."""

    def step(state, token):
        weights, momentum = state
        k_i, v_i, q_i, theta_i, eta_i, alpha_i = token
        surprise = jax.grad(memory_loss)(weights, k_i, v_i)
        momentum = jax.tree.map(lambda s, g: eta_i * s - theta_i * g, momentum, surprise)
        weights = jax.tree.map(lambda w, s: (1 - alpha_i) * w + s, weights, momentum)
        return (weights, momentum), memory_mlp(weights, q_i)

    return jax.lax.scan(step, state, (k, v, q, theta, eta, alpha))


# Memory as a Context (Section 4.1)

def mac_mask(C, n_persistent, with_memory):
    # h[j] is recalled with token j as the cue, so the h block must be causal too.
    causal = jnp.tril(jnp.ones((C, C), dtype=bool))
    blocks = [jnp.ones((C, n_persistent), dtype=bool)]
    blocks += [causal, causal] if with_memory else [causal]
    return jnp.concatenate(blocks, axis=1)


def mac_layer(p, x, cfg):
    T, d = x.shape
    C = cfg.segment_len
    mask = mac_mask(C, cfg.n_persistent, cfg.use_memory)
    segments = x.reshape(T // C, C, d)

    if not cfg.use_memory:
        def segment_step(_, segment):
            context = jnp.concatenate([p["persistent"], segment])
            return None, attention(p["attn"], segment, context, mask, cfg.n_heads)

        _, out = jax.lax.scan(segment_step, None, segments)
        return out.reshape(T, d)

    mem = p["memory"]

    def segment_step(state, segment):
        q, _, _ = memory_qkv(mem, segment)
        h = memory_mlp(state[0], q)
        context = jnp.concatenate([p["persistent"], h, segment])
        y = attention(p["attn"], segment, context, mask, cfg.n_heads)
        q_y, k_y, v_y = memory_qkv(mem, y)
        theta, eta, alpha = memory_gates(mem, y, cfg)
        state, readout = memory_write(state, k_y, v_y, q_y, theta, eta, alpha)
        return state, y * jax.nn.sigmoid(rms_norm(readout, mem["norm_gate"]))

    # the memory restarts from the learned M_0 for every sequence
    state0 = (mem["init"], jax.tree.map(jnp.zeros_like, mem["init"]))
    _, out = jax.lax.scan(segment_step, state0, segments)
    return out.reshape(T, d)


def model_apply(params, tokens, cfg):
    T = tokens.shape[0]
    # positions restart every segment, since attention never crosses one
    pos = jnp.tile(sinusoids(cfg.segment_len, cfg.d_model), (T // cfg.segment_len, 1))
    x = params["embed"][tokens] + pos
    for layer in params["layers"]:
        x = x + mac_layer(layer, rms_norm(x, layer["norm_attn"]), cfg)
        x = x + feed_forward(layer["ffn"], rms_norm(x, layer["norm_ffn"]))
    return rms_norm(x, params["norm_out"]) @ params["unembed"]


def nll_by_slot(params, x, y, cfg):
    """Per-byte loss shaped (batch, segment, position in segment)."""
    logits = jax.vmap(lambda tokens: model_apply(params, tokens, cfg))(x)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    return nll.reshape(x.shape[0], -1, cfg.segment_len)


def loss_fn(params, x, y, cfg):
    return nll_by_slot(params, x, y, cfg).mean()


# Training

def adamw_init(params):
    return {"m": jax.tree.map(jnp.zeros_like, params),
            "v": jax.tree.map(jnp.zeros_like, params),
            "step": jnp.zeros((), jnp.int32)}


def learning_rate(step, cfg):
    """Linear warmup, then cosine decay to 10% of the peak."""
    warm = jnp.minimum(1.0, (step + 1) / cfg.warmup_steps)
    progress = jnp.clip((step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps), 0.0, 1.0)
    return cfg.lr * warm * (0.1 + 0.9 * 0.5 * (1 + jnp.cos(jnp.pi * progress)))


def adamw_update(params, grads, opt, lr, cfg, b1=0.9, b2=0.95, eps=1e-8):
    t = opt["step"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)

    def update(p, m, v):
        direction = (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        decay = cfg.weight_decay * p if p.ndim >= 2 else 0.0
        return p - lr * (direction + decay)

    return jax.tree.map(update, params, m, v), {"m": m, "v": v, "step": t}


def clip_by_global_norm(grads, max_norm):
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads), norm


def make_train_step(cfg):
    @jax.jit
    def train_step(params, opt, x, y):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, x, y, cfg))(params)
        grads, grad_norm = clip_by_global_norm(grads, cfg.clip_norm)
        lr = learning_rate(opt["step"], cfg)
        params, opt = adamw_update(params, grads, opt, lr, cfg)
        return params, opt, {"loss": loss, "grad_norm": grad_norm, "lr": lr}

    return train_step


# Evaluation and sampling

def report_by_position(nll):
    """Mean +- s.e.m. loss at each position in a segment, first segment vs later ones.
    In later segments, position 0 can only see earlier text through the memory."""
    first, later = nll[:, 0, :], nll[:, 1:, :].mean(axis=1)
    cell = lambda a: f"{a.mean():.2f}+-{a.std() / np.sqrt(len(a)):.2f}"
    print("\nloss by position inside a segment (validation, nats/byte):")
    print(" " * 18 + "".join(f"pos {i:<10}" for i in range(4)) + "pos 4+")
    for name, rows in [("first segment ", first), ("later segments", later)]:
        print(f"  {name}  " + "".join(f"{cell(rows[:, i]):<14}" for i in range(4))
              + cell(rows[:, 4:].mean(axis=1)))


def make_sampler(cfg):
    @jax.jit
    def last_logits(params, window):
        return model_apply(params, window, cfg)[-1]

    def sample(params, prompt, n_bytes, temperature=0.8, seed=0):
        rng = np.random.default_rng(seed)
        out = list(prompt.encode())
        for _ in range(n_bytes):
            window = ([ord(" ")] * cfg.seq_len + out)[-cfg.seq_len:]
            logits = np.asarray(last_logits(params, jnp.asarray(window, dtype=jnp.int32)),
                                dtype=np.float64)
            probs = np.exp((logits - logits.max()) / temperature)
            out.append(int(rng.choice(cfg.vocab_size, p=probs / probs.sum())))
        return bytes(out).decode("utf-8", errors="replace")

    return sample


def main():
    parser = argparse.ArgumentParser(description="Train a small Titans (MAC) byte-level LM.")
    parser.add_argument("--data", default="data/tinyshakespeare.txt")
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--no_memory", "--no-memory", action="store_true")
    parser.add_argument("--max_inner_lr", "--max-inner-lr", type=float, default=Config.max_inner_lr)
    parser.add_argument("--segment_len", "--segment-len", type=int, default=Config.segment_len,
                        help="must divide seq_len (256)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config(steps=args.steps, use_memory=not args.no_memory, seed=args.seed,
                 max_inner_lr=args.max_inner_lr, segment_len=args.segment_len)
    train_data, val_data = load_bytes(args.data)
    params = init_params(cfg)
    opt = adamw_init(params)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"{n_params:,} parameters | memory {'on' if cfg.use_memory else 'off'} | "
          f"devices: {jax.devices()}")

    train_step = make_train_step(cfg)
    stream = batches(train_data, cfg, np.random.default_rng(cfg.seed))
    start = time.time()
    for step in range(cfg.steps):
        x, y = next(stream)
        params, opt, metrics = train_step(params, opt, x, y)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            loss = float(metrics["loss"])
            print(f"step {step:4d} | loss {loss:.3f} | "
                  f"grad norm {float(metrics['grad_norm']):.2f} | lr {float(metrics['lr']):.1e} | "
                  f"{time.time() - start:.0f}s")
            if not np.isfinite(loss):
                raise SystemExit("Loss is not finite. Try a smaller --max_inner_lr (e.g. 0.1).")

    evaluate = jax.jit(lambda p, x, y: nll_by_slot(p, x, y, cfg))
    val_stream = batches(val_data, cfg, np.random.default_rng(1234))
    nll = np.concatenate([np.asarray(evaluate(params, *next(val_stream))) for _ in range(100)])
    print(f"\nvalidation loss: {nll.mean():.3f}   (uniform = ln 256 = {np.log(256):.3f})")
    report_by_position(nll)
    print("\nsample:\n" + make_sampler(cfg)(params, "ROMEO:\n", 300))


if __name__ == "__main__":
    main()
