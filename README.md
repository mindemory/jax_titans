# Titans (Memory as a Context) in JAX

An implementation of the MAC variant of [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (Behrouz, Zhong & Mirrokni, 2024) using only `jax` and `numpy`. It trains a small byte-level language model on Tiny Shakespeare.

```bash
pip install -U jax numpy
python test_titans_mac.py         # unit tests
python titans_mac.py              # train (downloads Tiny Shakespeare on first run)
python titans_mac.py --no_memory  # same model without the long-term memory
python stability_sweep.py         # inner-loop stability for different theta, eta
```

Other options: `--data`, `--steps`, `--segment_len`, `--max_inner_lr`, `--seed`. Tested with Python 3.14 and JAX 0.11.2 on an Apple M4 CPU.

## Implementation

The core is `memory_write` (Eqs. 13–14) and `mac_layer` (Eqs. 21–25). The memory is a 2-layer MLP whose weights are updated by gradient descent on ‖M(k) − v‖² as tokens stream in, starting from learned weights M₀ for every sequence. The outer loss is backpropagated through these updates.

Choices where the paper is ambiguous or where I deviated:

- The recalled tokens h are masked causally, since h[j] is recalled with token j as the cue. Likewise, the output gate (Eq. 25) reads the memory right after each token's own write rather than at the end of the segment. Without either fix, future tokens leak into earlier positions; `test_causality` checks this.
- Memory updates are exact and token by token, not the chunked parallel version (§3.2). This is simpler but about 19× slower to train.
- θ ≤ 0.3, η ≤ 0.7, and memory values are ℓ2-normalized along with queries and keys. With unnormalized values the inner loop diverges at much smaller step sizes (see `stability_sweep.py`).
- The gate is `y * sigmoid(RMSNorm(M_t(q)))`. Positional encodings restart in every segment, so the memory is the only path to earlier segments.

Not implemented: the depthwise convolutions, multi-head memory, chunked training, and the MAG/MAL variants.

Tests cover the delta-rule special case (Appendix C), inner-loop stability at the maximum gates, causality, that the memory is the only path between segments, and that the memory's key projection gets a gradient through the inner loop.

## Results

600 steps, batch 8 × 256 bytes, seed 0. Validation loss in nats per byte (uniform = 5.545):

| segment length | with memory | without memory |
|---|---|---|
| 32 | 2.035 | 2.035 |
| 8 | 2.007 | 2.038 |

With 32-byte segments the memory makes no difference. Breaking the loss down by position shows the model only uses about 4 bytes of context, so there is little for the memory to add. With 8-byte segments it helps a little, and in the right place. At the first position of later segments, where the memory is the only access to earlier text, loss is 2.42 with memory vs 2.48 without. In the first segment, where there is nothing to remember, the two models match. This is a single seed, and the effect is small.

Run `python titans_mac.py` to see the per-position breakdown.

The code was written with Claude Opus 5.5, then tested and debugged on an Apple M4 Mac.
