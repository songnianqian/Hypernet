# HyperNet-GPT2

A GPT-2 transformer augmented with a **shared clustered hypernetwork controller**, evaluated on two tasks: **WikiText-103** language modeling and **Dyck-k** bracket matching. The hypernet routes mid-network hidden states through a small set of learned centroids and injects a structured "reasoning signal" into the feed-forward block. On Dyck, a per-configuration **addressable memory bank** is added to test recursive, depth-generalizing behavior.

## Table of contents

- [Motivation](#motivation)
- [Shared architecture: the HyperNet controller](#shared-architecture-the-hypernet-controller)
  - [Base stack](#base-stack)
  - [The clustered hypernetwork](#the-clustered-hypernetwork)
  - [Injection into the MLP block](#injection-into-the-mlp-block)
  - [Why "shared": parameter efficiency](#why-shared-parameter-efficiency)
- [Configuration A — WikiText-103](#configuration-a--wikitext-103)
- [Configuration B — Dyck-k](#configuration-b--dyck-k)
  - [Addressable memory bank](#addressable-memory-bank)
  - [Depth-extrapolation results](#depth-extrapolation-results)
- [Planned extension: soft AND/OR routing](#planned-extension-soft-andor-routing)
- [Installation](#installation)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Language-model vocabularies are large (GPT-2 has ~50k tokens), but the *semantic* space is much smaller — many tokens carry near-identical meaning, so the number of distinct "concepts" the model needs is far lower than the token count. This project asks whether a compact, **shared** set of learned centroids can act as that smaller abstraction layer: tokens' contextual hidden states are routed to a handful of centroids, and the resulting routing signal is fed back into the network to shape the next computation.

The second, harder question is computational: standard transformers without recurrence or external state are limited in the kinds of recursive, unbounded-depth structure they can track. By adding a controller (the hypernet) plus persistent state (the Dyck memory bank), we test whether the system can handle **recursive** tasks — nested bracket structures — at depths beyond those seen in training. Dyck-k is the canonical minimal test for recursive state, which is why it is used here as the witness task.

We are explicit that this is **evidence, not proof**. The results below are consistent with recursive computation; they do not establish formal Turing-completeness. That framing is deliberate (see [Limitations](#limitations)).

---

## Shared architecture: the HyperNet controller

Both configurations share the same controller. The only difference is that the Dyck configuration adds an addressable memory bank (described in [Configuration B](#addressable-memory-bank)).

### Base stack

The backbone is a GPT-2 stack of `num_layers` transformer blocks. The first `hypernet_start_layer` blocks are **standard** GPT-2 blocks; the remaining blocks are **hypernet-augmented**. Pretrained GPT-2 weights are loaded into all blocks at initialization, so the augmented model starts from the pretrained baseline and the hypernet contribution begins near zero (small learned gate `α`, initialized to 0.1).

- WikiText config (this repo's `hypernet_model.py`): `num_layers = 20`, `hypernet_start_layer = 10` → layers 0–9 standard, 10–19 augmented.
- Dyck config: `num_layers = 12`, `hypernet_start_layer = 6` → layers 0–5 standard, 6–11 augmented.
- Setting `hypernet_start_layer = num_layers` disables the hypernet entirely — this is the **"OFF"** ablation used throughout.

> **What "OFF" means, precisely.** OFF is the *same architecture* with the hypernet path removed (no augmented layers, no centroid routing, no injection). On Dyck, OFF additionally means the memory bank is not consulted. It is an ablation of our own model, not a separate baseline model.

### The clustered hypernetwork

A **single** `ClusteredHypernetwork` instance is shared across all augmented layers. It holds `M = 128` learnable centroids $C = \{c_1, \dots, c_M\}$, each a vector in $\mathbb{R}^{d}$ where $d$ is the hidden size.

**Step 1 — RoPE feature and layer-window aggregation.** At an augmented layer, the controller receives the recent history of layer inputs (the last `memory_window` hidden states). Each hidden state $X \in \mathbb{R}^{B\times S\times d}$ is transformed into a rotary-encoded (RoPE) feature and split into interleaved sin/cos components. Writing $X_{\text{rot}}$ for the rotary-rotated hidden state,

$$
s_X = \sin(X_{\text{rot}}), \qquad c_X = \cos(X_{\text{rot}}),
$$

and the interleaved feature $Y$ takes its even indices from $s_X$ and its odd indices from $c_X$. These are averaged over the window of recent layers and L2-normalized:

$$
\hat{Y} = \frac{Y}{\lVert Y \rVert_2}.
$$

Routing on $\hat{Y}$ (rather than on raw token embeddings) means the centroids cluster **contextual mid-network states**, not vocabulary items.

**Step 2 — soft centroid assignment.** With centroids also L2-normalized ($\hat{c}_i$), assignment is a cosine-similarity softmax — a full distribution over centroids, not a hard pick:

$$
p_i \;=\; \operatorname{softmax}_i\!\big(\langle \hat{Y}, \hat{c}_i\rangle\big), \qquad p \in \mathbb{R}^{M}.
$$

(An optional reinforcement-learning path can bias this distribution and sample a discrete centroid; it is **off by default** and not used in the reported results.)

**Step 3 — the reasoning vector (three channels).** The controller emits a `3M = 384`-dimensional reasoning vector per token, concatenating three centroid-space signals:

$$
r \;=\; \big[\; p^{\text{router}} \;\big\Vert\; p^{\text{dsin}} \;\big\Vert\; p^{\text{dcos}} \;\big] \;\in\; \mathbb{R}^{3M}.
$$

- $p^{\text{router}}$: a small shared perceptron applied to the routing distribution $p$.
- $p^{\text{dsin}}$, $p^{\text{dcos}}$: softmax cosine-similarities to the same centroids, but computed from the **derivatives** of the sin/cos features, $\tfrac{d}{dx}\sin = \cos$ and $\tfrac{d}{dx}\cos = -\sin$. Intuitively these let the router see the **rate of change** of the features, not just their value.

All three channels are matched against the *same* shared centroid set.

### Injection into the MLP block

The reasoning vector is injected into the **feed-forward (MLP) sublayer only** — attention is left unmodified. Inside an augmented block, let $x$ be the post-attention residual stream and $\text{LN}_2$ the second layer-norm. The up-projected FFN hidden state is

$$
h_{\text{ffn}} = W_{fc}\,\text{LN}_2(x),
$$

and the hypernet adds a gated, learned projection of the reasoning vector **before** the GELU activation:

$$
h_{\text{ffn}} \;\leftarrow\; h_{\text{ffn}} \;+\; \alpha \cdot W_{\text{inj}}\, r, \qquad W_{\text{inj}} : \mathbb{R}^{3M} \to \mathbb{R}^{4d},
$$

where $\alpha$ (`alpha_hypernet`, init 0.1) is a learned scalar gate. Because $\alpha$ starts small and the backbone is pretrained, the augmented model begins close to baseline GPT-2 behavior and learns how much hypernet signal to admit.

```
x ──► LN_2 ──► W_fc ──► h_ffn ─────────────┐
                                           (+)──► GELU ──► W_proj ──► + residual
hypernet(layer history) ──► r ∈ R^384 ──► α · W_inj ──┘
```

> **Note.** In the current code the hypernet touches the MLP only. Injecting into attention is a possible extension but is **not** implemented or evaluated here.

### Why "shared": parameter efficiency

The centroids and perceptrons live in **one** hypernetwork instance reused by every augmented layer, rather than a separate routing module per layer. Concretely, the 128 shared centroids of size $d$ plus the shared perceptrons are added **once**, not ~10 times. The hypernet's total parameter overhead relative to the backbone is printed at model-build time:

```
📊 Model Statistics:
   Total parameters:   [TODO: fill from your build log]
   Hypernet overhead:  [TODO] ([TODO]%)
```

> **TODO:** paste the real numbers from your `create_hypernet_gpt2(...)` print-out for each config, and state the comparison baseline explicitly (e.g. "vs. a per-layer routing module, sharing reduces the routing parameters by ~N×").

---

## Configuration A — WikiText-103

Standard autoregressive language modeling on WikiText-103. This configuration uses the shared hypernet controller **without** the memory bank; "memory" here is only the `memory_window` averaging of recent layer states.

**Hypothesis.** If many tokens map to a small number of shared centroids, the controller acts as a learned abstraction layer that improves next-token prediction.

### Results

| Model | Config | Perplexity ↓ |
|---|---|---|
| GPT-2 backbone, hypernet **OFF** | 20 layers, hypernet_start=20 | [TODO] |
| HyperNet **ON** | 20 layers, hypernet_start=10 | [TODO] |

**Abstraction evidence (recommended plot).** Because the controller's whole premise is that a large vocabulary collapses onto few effective clusters, the most direct evidence is a **centroid-usage histogram / usage entropy** measured over the WikiText eval set: how many of the 128 centroids are actually used, and how peaked the routing is. If routing concentrates on an effective handful of centroids, that directly supports the abstraction claim.


## Configuration B — Dyck-k

Dyck-k is the language of balanced brackets over `k` bracket types. It is the canonical minimal test for recursive state: tracking nesting depth requires stack-like memory, and a purely finite-state model cannot solve it at unbounded depth. We train at a **shallow** maximum depth and evaluate at **much deeper** depths to test extrapolation.

Sequences are generated by `dyck_dataset.py` (balanced-bracket sampler with configurable `max_depth`, `min_depth`, and stop probability). The evaluation metric is **close-bracket accuracy**: whether the model predicts the correct closing bracket at close positions.

### Addressable memory bank

The Dyck configuration adds a **`DynamicMemoryBank`** on top of the shared controller: **8 slots × 64-d**, read out through a projection to **32-d**. Memory access is gated by explicit per-token masks:

- `store_mask` / `write_mask` — which tokens may **write** to memory.
- `query_mask` (= read mask) — which tokens **read** from memory.

**Verified write-gating.** A strict test measures the change in memory state (`|Δ memory|`) under two conditions:

| Condition | `write_mask` | Observed `|Δ memory|` |
|---|---|---|
| Store tokens allowed to write | store positions = True | **420.08** |
| No tokens allowed to write | all False | **0.00** |

A large delta when writes are permitted and **exactly zero** when they are not confirms that (a) query/read tokens cannot write, (b) mask gating is correct, and (c) there is no silent memory leakage.

### Depth-extrapolation results

Trained on shallow depth (max depth 10 or 20), evaluated on depths 200–400. **ON** = hypernet + memory enabled; **OFF** = same architecture with the hypernet path ablated. Values are approximate.

| Train depth (TD) | Eval depth (ED) | OFF | ON |
|---|---|---|---|
| 10 | 200 | ~47% | **~65%** |
| 10 | 300 | ~50% | **~70%** |
| 10 | 400 | ~50% | **~61%** |
| 20 | 300 | ~55% | **~71%** |

**Findings:**

1. **Consistent ablation gap.** At every train/eval setting, turning the hypernet + memory ON improves close-bracket accuracy by roughly **15–20 percentage points** over the ablated OFF model (e.g. TD 10 / ED 300: ~50% → ~70%). The controller and memory are doing the work, not the backbone alone.
2. **Depth extrapolation well beyond training.** Trained at max depth 10, the ON model still closes ~61–70% of brackets at eval depths 300–400 — far beyond its training depth — while the OFF model stays near chance-like ~50%.
3. **Training depth has limited effect.** TD 10 / ED 300 (~70% ON) and TD 20 / ED 300 (~71% ON) are essentially the same, suggesting the mechanism generalizes rather than memorizing a specific depth range. The OFF baseline improves modestly with deeper training (~50% → ~55%), but remains far below ON.

> **Honest caveat on non-monotonicity.** Accuracy is **not** strictly monotonic in eval depth — for TD 10, ON is higher at ED 300 (~70%) than at ED 200 (~65%), then drops at ED 400 (~61%). The eval set is regenerated per depth and the sampler's `min_depth` / stop settings mean the depth-200, -300, -400 sets differ in more than depth alone, so small differences between adjacent depths are **not** meaningful. Read these as extrapolation evidence, not a smooth scaling law. The one robust, repeated signal is the ON-vs-OFF gap at matched settings.

## Planned extension: soft AND/OR routing

**Not implemented — design direction only.** The current model's two derivative channels ($p^{\text{dsin}}$, $p^{\text{dcos}}$) are planned to be replaced by two **soft-logic** channels over the centroids, so the reasoning vector expresses agreement and coverage between the two views.

Let $s^{\sin}_i$ and $s^{\cos}_i$ be the centroid similarities of the sin- and cos-derivative features. Squash each to a fuzzy membership in $[0,1]$ (sigmoid with temperature $\tau$):

$$
a_i = \sigma(\tau\, s^{\sin}_i), \qquad b_i = \sigma(\tau\, s^{\cos}_i).
$$

Combine per centroid with the product t-norm (AND) and its dual probabilistic-sum t-conorm (OR):

$$
\text{AND}_i = a_i\, b_i, \qquad \text{OR}_i = a_i + b_i - a_i b_i,
$$

giving

$$
r = \big[\, p^{\text{router}} \;\Vert\; \text{AND} \;\Vert\; \text{OR}\,\big] \in \mathbb{R}^{3M}.
$$

Interpretation: a centroid's **AND** channel fires when *both* rate-of-change views point to it (agreement); its **OR** channel fires when *either* does (coverage). Note this requires switching those channels from **softmax** (a competing distribution) to **sigmoid** (independent memberships) — a deliberate change, not a drop-in. The injection projection $W_{\text{inj}}$ keeps its shape because $r$ stays 384-d. No results are reported for this variant yet.

---

## Installation

```bash
git clone https://github.com/[TODO-your-handle]/hypernet-gpt2.git
cd hypernet-gpt2
pip install -r requirements.txt
```

## Reproducing the experiments

### Data

- **WikiText-103:** download and place the raw files (`wiki.train.txt`, `wiki.valid.txt`, `wiki.test.txt`) under a data directory; `dataset.py` handles Moses-token cleanup and chunking.
- **Dyck-k:** generated on the fly by `dyck_dataset.py` — no download needed.

### Configuration A — WikiText-103

```bash
# TODO: replace with your actual training command
python train.py \
  --task wikitext \
  --data_dir [TODO: path to WikiText-103] \
  --num_layers 20 \
  --hypernet_start_layer 10 \
  --num_centroids 128 

# Ablation (hypernet OFF): set --hypernet_start_layer 20
```

### Configuration B — Dyck-k

```bash
# TODO: add once the Dyck training script is uploaded
python train_dyck.py \
  --num_layers 12 \
  --hypernet_start_layer 6 \
  --use_memory --memory_slots 8 --memory_output_dim 32 \
  --train_max_depth [10 or 20] \
  [TODO: other flags]

# Evaluation at deeper depths (load the same checkpoint, test on depth 100–400):
python eval_dyck.py \
  --checkpoint [TODO] \
  --eval_max_depth 300 \
  [TODO: eval flags]

# Ablation (hypernet OFF): set --hypernet_start_layer 12 and drop --use_memory
```

## Limitations

We state these plainly, because the computational claim invites them:

- **This is evidence, not a proof.** Depth extrapolation and an ablation gap are consistent with recursive computation, but they do not constitute a formal proof of Turing-completeness. We claim the former and *motivate* with the latter.
- **Bounded memory and finite precision.** The memory bank has a fixed number of slots and the model runs at finite precision. A common objection is "memory is bounded" — the standard response is that any physical Turing machine is also bounded, and universality is about *scalability* (memory size and depth as parameters) rather than a fixed instantiation. We do not resolve this; we flag it.
- **"It's just a pushdown automaton."** Dyck is solvable by a PDA, so success on Dyck alone does not demonstrate universality. Dyck is used here as a *witness* for recursive state, not as a universality certificate.
- **Clusters are over hidden states, not tokens.** The abstraction story is motivated by token synonymy, but the centroids actually cluster contextual mid-network hidden states. The usage-entropy evidence should be read accordingly.
- **Non-monotonic depth curves.** As noted above, eval sets differ across depths; treat the depth sweep as extrapolation evidence, not a scaling law.

---

## Citation

```bibtex
@software{qian_speedlitegate_2026,
  author = {Qian, Songnian},
  title  = {Hypernet: A GPT-2 transformer augmented with a shared clustered hypernetwork controller},
  year   = {2026},
  url    = {https://github.com/songnianqian/Hypernet}
}
```

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).
