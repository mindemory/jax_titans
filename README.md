# Titans (Memory as a Context) in pure JAX

A small implementation of the MAC variant of *Titans: Learning to Memorize at Test Time* (Behrouz, Zhong & Mirrokni, 2024), using only `jax` and `numpy` (no Flax or Optax). It trains a byte-level language model on Tiny Shakespeare.

## Run

```bash
pip install -U jax numpy
python test_titans_mac.py         # unit tests (~10 s)
python titans_mac.py              # train with memory (downloads Tiny Shakespeare on first run)
python titans_mac.py --no_memory  # ablation: same model without the long-term memory
python stability_sweep.py         # inner-loop stability vs. theta and eta
```

`--data path/to/file.txt` trains on any text file, and `--steps 100` gives a quick check. Tested with Python 3.14 and JAX 0.11.2 on CPU (Apple M4).

## Mapping to the paper

| Paper | Code |
|---|---|
| Associative loss (Eq. 12); its gradient is the surprise (Eq. 8) | `memory_loss` |
| Momentum and forgetting (Eqs. 13–14) | `memory_write` |
| Data-dependent θ, η, α | `memory_gates` |
| Deep memory (§3.1) and retrieval (Eq. 15) | `memory_mlp` |
| SiLU and ℓ2-normalized q, k (§4.4) | `memory_qkv` |
| Persistent memory (§3.3) | `persistent` parameters |
| MAC block (Eqs. 21–25, Fig. 2) and mask (Fig. 3a) | `mac_layer`, `mac_mask` |
| AdamW, weight decay 0.1 (§5.1) | `adamw_update` |

The memory MLP's weights and momentum are inner-loop state: they are updated during the forward pass and reset to the learned initial weights M₀ for each sequence. Everything else is trained by the outer loop, which backpropagates through the inner updates.

## Design choices

1. **Causal mask over recalled tokens.** h[j] is recalled using token j as the cue, so a plain causal mask over [persistent ‖ h ‖ segment] would let token i see h[j] for j > i. `mac_mask` makes the h block lower-triangular.
2. **Causal output gate.** Read literally, Eq. 25 uses the memory after the whole segment has been written, which leaks later tokens into earlier positions. Each position instead reads the memory right after its own write. `test_causality` fails if either this fix or the previous one is removed.
3. **Token-by-token inner updates.** Eqs. 13–14 are applied one token at a time instead of in the chunked parallel form of §3.2. This is exact and simple but slow, since backprop stores the memory weights at every token.
4. **Bounded gates and normalized values.** θ ≤ 0.3, η ≤ 0.7, and values are ℓ2-normalized like queries and keys. For a nonlinear memory the scale of the values affects stability. In `stability_sweep.py`, raw values stop learning once the effective step θ/(1−η) reaches about 0.5 and diverge by 0.6, while normalized values stay stable up to about 2 and everywhere inside the bounds.
5. **Gating.** The paper leaves ⊗ open; here it is `y * sigmoid(RMSNorm(M_t(y)))`.
6. **Inputs.** The recall query comes from the segment input (Eq. 21). Gates, keys and values for the write come from the attention output y (Eq. 24).
7. **Positions restart each segment.** Attention never spans more than one segment, so sinusoidal positions are reused per segment, and the memory is the only path to older text.

Not implemented: the depthwise convolutions (§4.4), multi-head memory, chunked parallel training (§3.2), and the MAG/MAL variants.

## Tests

- **Delta rule.** With a linear memory, a unit-norm key, θ = 1 and η = α = 0, one write stores v exactly (Appendix C).
- **Stability.** At the largest allowed θ and η, repeated writes reduce the memory loss without diverging.
- **Causality.** Changing token j changes no prediction before j.
- **Segment bridge.** Changing segment 0 changes segment 1's predictions with memory, and nothing without it.
- **Meta-gradient.** The memory key projection W_K is used only for writing, yet receives a nonzero gradient.

## Results

600 steps, batch 8 × 256 bytes (about 1.2 epochs), seed 0, CPU. Runs are deterministic.

| | parameters | validation loss (nats/byte) | training time |
|---|---|---|---|
| uniform | | 5.545 | |
| MAC with memory | 189,766 | 2.035 | 171 s |
| `--no_memory` | 131,904 | 2.035 | 9 s |

For reference, a character-level bigram model gets about 2.5 on this data, and a well-trained ~10M-parameter transformer about 1.5.

Loss by position within a 32-byte segment (validation, mean ± s.e.m.):

| | pos 0 | pos 1 | pos 2 | pos 3 | pos 4+ |
|---|---|---|---|---|---|
| memory, first segment | 2.48 ± 0.05 | 2.26 ± 0.05 | 1.98 ± 0.05 | 2.12 ± 0.05 | 2.00 ± 0.01 |
| memory, later segments | 2.49 ± 0.02 | 2.15 ± 0.02 | 2.06 ± 0.02 | 2.05 ± 0.02 | 2.02 ± 0.01 |
| no memory, first segment | 2.49 ± 0.05 | 2.24 ± 0.05 | 1.98 ± 0.05 | 2.10 ± 0.05 | 1.99 ± 0.01 |
| no memory, later segments | 2.52 ± 0.02 | 2.18 ± 0.02 | 2.07 ± 0.02 | 2.05 ± 0.02 | 2.01 ± 0.00 |

The validation windows are sampled at random and can overlap, so the standard errors are somewhat optimistic.

The memory makes no measurable difference here. Position 0 sees only one byte and behaves like a bigram model (about 2.5). Most of the benefit of context arrives within 3–4 bytes, and the remaining positions sit near 2.0. So even perfect recall of the previous segment would improve the overall loss by only about 0.02 nats. The observed differences (0.03 better at positions 0–1 of later segments, 0.01 worse at 4+) are within about one standard error. The memory is wired correctly, as the tests show, but this model and task leave little room for long-range context, and the token-by-token updates make training about 19× slower.

### Shorter window: `--segment_len 8`

With 8-byte segments, half of all positions see fewer than the ~4 bytes of context the model uses, which gives the memory more room to matter. Same settings otherwise.

| | validation loss | training time |
|---|---|---|
| MAC with memory | 2.007 | 176 s |
| `--no_memory` | 2.038 | 8 s |

| | pos 0 | pos 1 | pos 2 | pos 3 | pos 4–7 |
|---|---|---|---|---|---|
| memory, first segment | 2.47 ± 0.04 | 2.19 ± 0.05 | 1.88 ± 0.05 | 2.03 ± 0.05 | 1.92 ± 0.03 |
| memory, later segments | 2.42 ± 0.01 | 2.10 ± 0.01 | 1.97 ± 0.01 | 1.93 ± 0.01 | 1.91 ± 0.01 |
| no memory, first segment | 2.47 ± 0.05 | 2.18 ± 0.05 | 1.92 ± 0.05 | 2.03 ± 0.05 | 1.93 ± 0.03 |
| no memory, later segments | 2.48 ± 0.01 | 2.12 ± 0.01 | 2.00 ± 0.01 | 1.96 ± 0.01 | 1.93 ± 0.01 |

Here the memory helps by 0.03 nats overall, and the gain is where it should be. In the first segment, where there is nothing earlier to remember, the two models match. In later segments the memory model is better at every position, most of all at position 0 (2.42 vs 2.48), where the memory is the only access to preceding text. Within the memory model, position 0 is also better in later segments than in the first (2.42 vs 2.47); without memory it is not (2.48 vs 2.47). This within-model comparison doesn't depend on differences between training runs.

Caveats: this is one seed, so run-to-run variance between the two trainings is not measured, and the standard errors are optimistic as noted above. Shrinking the window from 32 to 8 barely hurt the no-memory model (2.035 → 2.038), which confirms it relies on very short context.

Possible reasons the memory contributes little overall: recall is cued by the content of the current tokens, which doesn't directly encode "what came just before", and the bounded step size with 600 training steps may be too little. Next steps would be more seeds, longer training, and a synthetic task that needs recall across segments.

## Notes on the paper

- The text describes surprise as a gradient with respect to the input, but Eq. 8 uses the gradient with respect to the memory's parameters.
- Eq. 17 drops the key/value projections; it should read (W₀k − v)kᵀ.
- Eq. 32 has M_t on both sides; the right-hand side should be M_{t−1}.

## Acknowledgment

The code was generated with Claude Opus 5.5, then tested and debugged on an Apple M4 Mac.
