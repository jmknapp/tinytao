# Tiny-NN Lab

A computational-science experiment: train large populations of *independent* tiny
neural networks on one exactly defined discrete problem, then ask what
algorithms appear.

This is not a benchmark chase. Each network is an experimental organism.
The population is the experiment.

Phases 1–9 are the additive-ReLU control (Model A) on \(\mathbb{Z}/13\mathbb{Z}\).
A vs B changes one hidden primitive. Model C supplies complex
multiplication of a shared 2-D embedding.

---

## 1. First task

**Modular multiplication** \(y = (a \cdot b) \bmod p\) with prime \(p = 13\).

| Candidate | Enumerable | Tiny exact solvers plausible | Room for distinct algorithms | Reverse-engineering |
|---|---|---|---|---|
| Integer addition | only on a truncated box | yes, often linear | low | easy, usually uninteresting |
| Integer multiplication | same | mixed | medium | polynomial structure, unbounded |
| Modular addition | yes, \(p^2\) | yes | medium (clock / Fourier) | well-studied |
| **Modular multiplication** | **yes, \(p^2\)** | **yes at small \(p\)** | **higher** | **rich, not exhausted** |
| Parity / XOR | yes | yes | low | essentially one circuit |
| Binary arithmetic / FSM | yes if width is fixed | unclear | high | messy state |

Why this one:

- The complete input space is \(p^2 = 169\) pairs. Every claim about
  accuracy is exhaustive, not a sample.
- The rule is rigid. A network is an *exact solver* only if every pair
  is correct.
- Several *different* computations can implement the same table:
  bilinear / Fourier features on both residues, discrete-log then modular
  addition, a lookup table, and a hybrid that special-cases the absorbing
  element \(0\).
- Modular addition is kept as a second task (`src/tasks/modular.py`) for
  later comparison. It is easier and already reverse-engineered in the
  grokking literature; it is a better *validation* problem than a first
  discovery problem.
- \(p = 13\) is large enough that memorizing 70% of the table does not
  automatically give the rest, and small enough that activation heatmaps
  are \(13 \times 13\).

Tao’s modular-arithmetic challenge is about *scaling* learned multipliers
to huge primes. This lab is the complementary microscope: freeze a tiny
prime and vary the organisms.

## 2. Encoding

- Inputs: concatenated one-hots, \(x = [\mathrm{onehot}(a)\,\|\,\mathrm{onehot}(b)] \in \mathbb{R}^{2p}\).
- Outputs: \(p\)-way classification over residues. Loss is cross-entropy.
- No learned embedding table. The first linear layer *is* the embedding.

One-hot classification keeps “correct” binary. Regression onto the
integer \(0,\ldots,p-1\) would mix numerical tolerance with competence.
Binary encodings are a later ablation: they change the inductive bias
before we have a baseline.

## 3. Starting architecture

One hidden layer, ReLU, biases on:

```
26  --W1[P,26,8]-->  8  --ReLU-->  8  --W2[P,8,13]-->  13 logits
```

333 parameters per organism. Depth and width are configurable
(`hidden_dims: [8]` or `[8, 8]`, later sweeps `{2,3,4,5,6,8,12,16,24,32}`).

Phase 1 does **not** sweep. Diversity is independent Xavier initialization
only.

## 4. Vectorization

The population is an extra leading axis, not a Python loop.

```
W1: [P, 26, 8]
b1: [P, 8]
W2: [P, 8, 13]
b2: [P, 13]
```

Forward:

```python
h = relu(einsum("bi,pih->bph", x, W1) + b1)   # [B, P, 8]
logits = einsum("bph,pho->bpo", h, W2) + b2   # [B, P, 13]
```

Adam sees ordinary parameter tensors; slices do not share gradients
(checked in `src/correctness.py`). Init uses *last-two-dim* Xavier so
`P` is not mistaken for a feature axis.

## 5. VRAM / compute (order of magnitude)

For \(p=13\), hidden \(8\), \(P=10^4\), full domain \(B=169\), float32:

| Object | Bytes (approx.) |
|---|---|
| Parameters | 13 MB |
| Adam moments | 26 MB |
| Hidden acts | 54 MB |
| Logits | 88 MB |
| Backward leftovers | a few × activations |

Comfortably under **1 GiB**. A 16 GB RTX 5060 Ti can hold this population
with room for wider nets or \(P \sim 10^5\). The implementation still
halves `P` on OOM down to `min_size`.

Default compute is float32. Mixed precision is off: for these systems
the numerics may be part of the science, and exact-solver detection
must not depend on fp16 rounding.

## 6. First experiments

1. **Phase 1 (this repo).** \(P \approx 4\mathrm{k}\), hidden \(8\), random
   70/30 split. Prove batched training, isolate gradients, log
   complete-domain trajectories, bin outcomes.
2. **Phase 2.** Scale to \(P=10^4\), report step time / VRAM / utilization.
3. **Phase 3.** Width sweep, estimate \(\Pr(\text{exact solver})\) vs
   parameter count. Look for a capacity threshold.
4. **Phase 4–5.** Cluster exact solvers by *behavior* (activations over
   the full table, Fourier spectra, ablation signatures), not raw
   weights. Reverse-engineer cluster exemplars.
5. **Phase 6.** Structured holdouts (below) and shrinking of exact
   solvers (prune / quantize / narrower retrain).

Structured holdouts already implemented in `src/tasks/modular.py`:

| Split | What a success would mean |
|---|---|
| `random` | Interpolates unseen pairs; a smoothed table can pass |
| `held_out_operand` | Completes missing rows/columns of the multiplication table |
| `held_out_quadrant` | Extrapolates the rule beyond a rectangular training block |
| `held_out_zero` | Handles the absorbing element without having seen it |
| `held_out_product` | Produces reserved output residues it never trained on |

Phase 1 uses only `random`. The others are for later falsification.

## 7. What would count as *different algorithms*

Different weights are not different algorithms. Hidden units permute;
scales and signs flip; two networks can implement the same circuit.

Evidence that would actually support distinct strategies (none of this
is claimed yet):

1. **Permutation-aligned clustering of full-domain activations** yields
   several well-separated clusters that remain separate after Hungarian
   matching of hidden units.
2. **Different Fourier structure.** Some exact solvers concentrate on a
   few frequencies (or on \(\log a + \log b\) after a discrete-log
   relabeling); others are broadband / lookup-like.
3. **Different causal signatures.** Ablating unit \(i\) destroys a
   whole residue class in cluster A and a geometric band in cluster B.
4. **Different structured-holdout profiles.** One cluster generalizes to
   a held-out operand; another collapses to chance. Same training split,
   different internal rule.
5. **A falsification test for each story.** If we claim “these units are
   \(\sin(2\pi k a / p)\)”, then a phase shift of the one-hot embedding
   along that character should move the unit as predicted, and a
   frequency the model does not use should not.

Until those checks exist, a 100% network is reported as an exact solver,
not as an algorithm.

---

## Layout

```
configs/                 YAML experiment specs
src/tasks/               enumerable problems + structured splits
src/population/          batched tiny MLPs
src/training/            vectorized loop, outcome bins, transition flags
src/analysis/            outcomes, embedding-family fits, ablation, shrink
src/visualization/       matplotlib figures
src/storage.py           run directories
src/repro.py             seeds, torch/CUDA/GPU/git metadata
src/correctness.py       batched == sequential, no gradient leak
experiments/             Phase runners
results/runs/<id>/       config, metrics, checkpoints, figures
notebooks/               exploratory analysis only
```

Storage policy: every network’s *final* scalars and subsampled
trajectories; weights only for exact solvers (capped) and a few
representatives from each outcome bin. No per-step dumps of 10k nets.

## Commands

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python src/correctness.py
PYTHONPATH=. python experiments/phase1_vectorized_train.py --smoke
PYTHONPATH=. python experiments/phase1_vectorized_train.py --config configs/phase1_modmul.yaml
PYTHONPATH=. python experiments/benchmark.py --config configs/phase2_scale.yaml --epochs 50
PYTHONPATH=. python experiments/phase3_width_sweep.py --config configs/phase3_width_sweep.yaml
PYTHONPATH=. python experiments/phase3_width_sweep.py --config configs/phase3_split_sweep.yaml
PYTHONPATH=. python experiments/phase45_mechanistic.py --widths 4 5 6 8 16
PYTHONPATH=. python experiments/phase6_holdouts.py
PYTHONPATH=. python experiments/phase6_minimal.py
PYTHONPATH=. python experiments/phase7_falsify.py
PYTHONPATH=. python experiments/phase8_substitute.py
PYTHONPATH=. python experiments/phase9_modadd.py
```

A network is an exact solver only if complete-domain accuracy is \(1\):
every \((a,b)\) pair is classified correctly. Training accuracy of \(1\)
is not enough.

## Phase 1 observations (not interpretations)

Run `results/runs/20260909_020511_phase1_modmul` on an RTX 5060 Ti 16 GB,
PyTorch 2.11.0+cu128. Hidden width 8, \(P=4096\), 1500 Adam steps, random
70/30 split, float32.

| Quantity | Value |
|---|---|
| Exact solvers | 0 / 4096 |
| Mean train / test / domain acc | 0.693 / 0.159 / 0.532 |
| Max domain acc | 0.663 |
| Chance | \(1/13 \approx 0.077\) |
| Wall time | 4.1 s |
| Time per epoch | 2.7 ms |
| Network-evaluations / s | \(1.8 \times 10^8\) |
| Peak allocated VRAM | 0.18 GiB |

At \(P=10{,}000\), same architecture, full-domain batch: ~10.6 ms/step,
0.48 GiB allocated, ~843 MiB nvidia-smi, and kernel bursts at
99–100% GPU utilization. The card is not VRAM-limited.

Every network beat chance. Train accuracy is far above held-out
accuracy, so under this baseline the population is fitting the training
table more than the multiplication rule. That is a starting fact for
the Phase 3 capacity sweep, not a claim about algorithms.

Phase 1 used 1500 steps at \(lr=10^{-3}\) on a random split. A later
probe showed that the same architecture, trained on the *full* table at
\(lr=3\times 10^{-3}\) for a few thousand steps, does become an exact
solver. The Phase 1 zero is therefore an under-training / protocol
result, not a proof that width 8 cannot represent the function.

## Phase 3 observations (not interpretations)

Two matched sweeps, \(P=1024\) per width, 8000 Adam steps, \(lr=3\times 10^{-3}\),
ReLU MLP, one hidden layer. Wilson 95% intervals on \(\Pr(\text{exact})\).

### Full-domain protocol

Train on all 169 pairs. An exact solver is an implementation of the
complete function.

| hidden | params | exact / 1024 | \(\hat p\) [95% CI] | mean domain | median first exact epoch |
|---|---:|---:|---|---:|---:|
| 2 | 93 | 0 | 0.000 [0.000, 0.004] | 0.348 | — |
| 3 | 133 | 0 | 0.000 [0.000, 0.004] | 0.464 | — |
| 4 | 173 | 7 | 0.007 [0.003, 0.014] | 0.595 | 2950 |
| 5 | 213 | 11 | 0.011 [0.006, 0.019] | 0.736 | 3250 |
| 6 | 253 | 48 | 0.047 [0.036, 0.062] | 0.859 | 4350 |
| 8 | 333 | 443 | 0.433 [0.403, 0.463] | 0.982 | 3800 |
| 12 | 493 | 1011 | 0.987 [0.978, 0.993] | 1.000 | 1250 |
| 16 | 653 | 1024 | 1.000 [0.996, 1.000] | 1.000 | 650 |
| 24 | 973 | 1024 | 1.000 [0.996, 1.000] | 1.000 | 350 |
| 32 | 1293 | 1024 | 1.000 [0.996, 1.000] | 1.000 | 250 |

Smallest exact solver found: **width 4, 173 parameters** (7 of 1024
random inits). Width 3 reached domain accuracy 0.953 but not 1.0.
Runs: `results/runs/20260909_021014_phase3_width_sweep`.

### Random-split protocol

Same optimizer and widths; train on a random 70% (118 pairs). Exact
still means every one of the 169 pairs is correct.

| hidden | params | exact | memorize | mean train / test / domain | max domain |
|---|---:|---:|---:|---|---:|
| 2 | 93 | 0 | 0 | 0.385 / 0.181 / 0.323 | 0.609 |
| 4 | 173 | 0 | 0 | 0.672 / 0.160 / 0.517 | 0.769 |
| 8 | 333 | 0 | 916 | 0.997 / 0.113 / 0.730 | 0.775 |
| 16 | 653 | 0 | 1024 | 1.000 / 0.135 / 0.739 | 0.763 |
| 32 | 1293 | 0 | 1024 | 1.000 / 0.169 / 0.749 | 0.763 |

Zero exact solvers at every width. As width grows, the population
saturates the training table and held-out accuracy stays near chance.
Complete-domain accuracy plateaus near \(0.70 + 0.30/13 \approx 0.72\),
which is what perfect train memorization plus chance on the held-out
pairs predicts.

Observation only: under this encoding, optimizer, and random split,
extra capacity makes memorization reliable and does not produce a
domain-exact rule. The networks that *can* implement the function
(full-domain sweep) are the objects for later reverse-engineering.

Runs: `results/runs/20260909_021327_phase3_split_sweep`.

## Phase 4–5 observations (not interpretations)

Source: full-domain exact-solver checkpoints from Phase 3.
Method: hidden maps over the \(13\times 13\) table; Hungarian alignment of
units; \(R^2\) of each first-layer embedding \(u_i(a)\), \(v_i(b)\) against
three equal-complexity 1-D families (sinusoid on the residue, one-hot
spike, sinusoid on \(\log_g a\) for \(a\neq 0\), \(g=2\)). A family “wins”
a side of a unit only if \(R^2 \ge 0.85\) and beats the next family by
\(\ge 0.15\).

Every loaded checkpoint is still domain-exact.

| width | n | mean aligned distance | family wins (F / one-hot / dlog / weak) | acc after \(a \mapsto a+1\) |
|---:|---:|---:|---|---:|
| 4 | 7 | 0.76 | 0 / 0 / 36 / 20 of 56 | 0.077 (chance) |
| 5 | 11 | 0.74 | 0 / 0 / 71 / 38 of 110 | 0.077 |
| 6 | 48 | 0.73 | 0 / 0 / 358 / 218 of 576 | 0.077 |
| 8 | 256 | 0.72 | 0 / 0 / 2172 / 1923 of 4096 | 0.077 |
| 16 | 256 | 0.65 | 3 / 0 / 4993 / 3196 of 8192 | 0.077 |

On the seven width-4 solvers, mean \(R^2\) is 0.32 (residue sinusoid),
0.25 (one-hot), **0.87 (dlog sinusoid)**. Among dlog fits with
\(R^2\ge 0.85\), the only frequencies that appear are \(k=1\) and
\(k=5\) on \((\mathbb{Z}/13\mathbb{Z})^*\), and \(u\) and \(v\) of the
same unit share \(k\). Ablating any one of the four units drops
complete-domain accuracy from 1.0 into the 0.34–0.76 range.

Aligned hidden-map distance among those seven nets is 0.76; at
threshold 0.12 they are seven singletons. They share a *family* of
1-D embeddings, not one circuit up to hidden permutation.

Caveats that block an algorithm claim:

- A 3-parameter sinusoid on 12 nonzero points is flexible. The
  comparison is still informative because the residue-sinusoid model
  has the same flexibility and loses.
- High dlog \(R^2\) plus a linear readout is not a proof that the net
  computes \(\chi(a)\chi(b)\). That would need a causal test that
  survives other multiplicative relabelings.
- The 4 clusters imposed on width 8/16 are a partition of a diffuse
  cloud (mean distance \(> 0.6\)), not four recovered algorithms.

Hypothesis, not a result: exact solvers in this architecture implement
multiplication using characters of the multiplicative group. The
Phase 3 random-split failure is consistent with that requiring the
full multiplication table to pin down the character, but that is a
story, not a test.

Runs: `results/runs/20260909_022109_phase45_mechanistic` (100 figures).

## Phase 6 observations (not interpretations)

Two questions: does any structured hole in the table produce a
domain-exact rule when a random 70/30 split does not, and does an
exact solver smaller than width 4 / 173 parameters exist.

### Structured holdouts

Width 16 (653 params), \(P=512\), 8000 Adam steps, \(lr=3\times 10^{-3}\).
On the full table this architecture is 1024/1024 exact. Same optimizer
family as Phase 3. Wilson 95% upper bound at 0/512 is 0.007.

| split | train / test | exact / 512 | mean train / test / domain | max domain | max test |
|---|---:|---:|---|---:|---:|
| random_70 | 118 / 51 | 0 | 1.000 / 0.134 / 0.739 | 0.781 | 0.275 |
| held_operand_12 | 144 / 25 | 0 | 1.000 / 0.084 / 0.864 | 0.888 | 0.240 |
| held_zero | 144 / 25 | 0 | 1.000 / 0.000 / 0.852 | 0.852 | 0.000 |
| held_quadrant | 49 / 120 | 0 | 1.000 / 0.111 / 0.369 | 0.402 | 0.158 |
| held_product_0 | 144 / 25 | 0 | 1.000 / 0.000 / 0.852 | 0.852 | 0.000 |
| held_dlog_odd | 91 / 78 | 0 | 1.000 / 0.079 / 0.575 | 0.592 | 0.115 |

Zero exact solvers on every hole. Every network saturates the training
pairs (`memorize` = 512/512).

`held_zero` and `held_product_0` have mean test accuracy 0 and domain
accuracy \(144/169 \approx 0.852\): perfect on the seen pairs, nothing
on the held class. `held_dlog_odd` holds out the six residues of \(a\)
with odd \(\log_2 a\) (zeros stay in train). Mean held-out accuracy is
chance; max domain 0.592 versus the train-only floor \(91/169 \approx 0.538\).

Observation only: under this encoding and optimizer, no tested hole —
including the one aimed at discrete-log interpolation — produces a
domain-exact rule. Extra capacity still memorizes the visible table.

Runs: `results/runs/20260909_022642_phase6_holdouts`.

### Minimal exact networks

Source width-4 solvers: the 7 exact nets from the Phase 3 full-domain
sweep. Quantization is per-network symmetric min-max onto
\(\{0,\ldots,2^b-1\}\) reconstructed to float32.

| bits | exact / 7 | mean domain |
|---:|---:|---:|
| 16 | 7 | 1.000 |
| 8 | 7 | 1.000 |
| 6 | 0 | 0.924 |
| 5 | 0 | 0.728 |
| 4 | 0 | 0.516 |
| 3 | 0 | 0.274 |
| 2 | 0 | 0.150 |

Dropping the least-critical hidden unit (highest post-ablation
accuracy) yields width 3, 133 params, 0/7 exact, max domain 0.757.
Retraining those 7 pruned inits on the full table for 8000 steps at
\(lr=3\times 10^{-3}\) stays 0/7 exact.

From-scratch width-3 hunt: \(P=4096\), 12000 Adam steps, \(lr=5\times 10^{-3}\),
seed 1, full table.

| Quantity | Value |
|---|---|
| Exact solvers | **1 / 4096** |
| \(\hat p\) [95% CI] | 0.00024 [0.00004, 0.00138] |
| Mean / max domain | 0.476 / 1.000 |
| First exact (logged) | epoch 10300 |
| Params | 133 |

The Phase 3 width-3 zero (\(P=1024\), 8000 steps, max domain 0.953) is
therefore a budget / sample-size result, not a proof that width 3
cannot represent the table.

The one exact solver (hunt index 697), same embedding-family test as
Phase 5:

| unit | \(u\) dlog \(R^2\) (\(k\)) | \(v\) dlog \(R^2\) (\(k\)) | residue-Fourier \(R^2\) | one-hot \(R^2\) | acc if ablated |
|---:|---|---|---:|---:|---:|
| 0 | 0.977 (\(k=1\)) | 0.971 (\(k=1\)) | 0.29 / 0.31 | 0.22 / 0.24 | 0.645 |
| 1 | 0.493 (\(k=1\)) | 0.467 (\(k=1\)) | 0.43 / 0.44 | 0.56 / 0.53 | 0.308 |
| 2 | 0.970 (\(k=1\)) | 0.967 (\(k=1\)) | 0.30 / 0.34 | 0.21 / 0.20 | 0.485 |

Cycling \(a \mapsto a+1\) in the first-layer embeddings drops accuracy
to chance (0.077). Ablating any of the three units destroys exactness.

Two of three units match the high-dlog, shared-\(k\) pattern seen at
width 4. Unit 1 wins no family. That is a measurement on one
organism, not a recovered algorithm.

Smallest exact architecture found: **width 3, 133 parameters** (1 of
4096 random inits under the longer protocol). Smallest quantized
exact: the seven width-4 solvers at 8 bits (still 173 parameters).

Runs: `results/runs/20260909_022909_phase6_minimal`
(`metrics/width3_exact.json`, `width3_hunt/checkpoints/successful.pt`).

## Phase 7 observations (not interpretations)

Stated falsifier for the Phase 5 dlog measurement: if embeddings with
dlog \(R^2 \ge 0.85\) keep a high dlog \(R^2\) after additive
(\(a\mapsto a+k\)) or random permutations of \(\mathbb{F}_p^*\), the
3-parameter sinusoid is too flexible and the multiplicative-group
fit is not about group structure.

The test is applied to the residue axis of each first-layer
embedding, then the same two families are re-fit. Multiplicative
relabeling is \(a\mapsto c\cdot a\) with 0 fixed, \(c\in\{1,\ldots,12\}\).
Random: 12 independent permutations of the 12 nonzero residues.

### Orbit of high-dlog sides

| source | n sides ≥ 0.85 | base dlog | after \(\times c\) | after \(+k\) | after random | frac still ≥ 0.85 (\(\times c\) / \(+k\) / rand) |
|---|---:|---:|---:|---:|---:|---|
| width 3 (1 net) | 4 / 6 | 0.971 | 0.971 | 0.403 | 0.476 | 1.00 / 0.00 / 0.02 |
| width 4 (7 nets) | 36 / 56 | 0.929 | 0.929 | 0.399 | 0.455 | 1.00 / 0.00 / 0.00 |
| width 8 (256 nets) | 2173 / 4096 | 0.930 | 0.930 | 0.379 | 0.449 | 1.00 / 0.00 / 0.006 |

After \(\times c\), residue-Fourier \(R^2\) stays ~0.31 and never
clears 0.85. After \(+k\), no high-dlog side remains above 0.85
(width-8 max 0.79). Random permutations land above 0.85 on
well under 1% of draws.

The equality of base and \(\times c\) means is the phase-shift
invariance of the fitted family: if a vector is already a dlog
sinusoid, multiplying the domain is a phase change and the re-fit
recovers the same \(R^2\). The discriminating number is the
collapse under additive and random relabelings.

This particular flexibility falsifier does **not** fire.

### Hidden maps are not a function of the product

Each unit is exactly \(\mathrm{relu}(u[a]+v[b]+c)\). Factor \(R^2\)
is the variance explained by replacing every entry with the mean of
its class.

| source | product | dlog sum | \(a+b\) | \(a\) | \(b\) | zero pattern |
|---|---:|---:|---:|---:|---:|---:|
| width 3 | 0.218 | 0.218 | 0.014 | 0.393 | 0.422 | 0.162 |
| width 4 | 0.192 | 0.193 | 0.014 | 0.408 | 0.411 | 0.128 |
| width 8 | 0.192 | 0.195 | 0.017 | 0.397 | 0.398 | 0.110 |

Product-class means explain ~20% of hidden-map variance; additive
class means explain ~1.5%. The maps are much more a function of
\(a\) or of \(b\) alone, which the architecture already forces.

Width-3 unit 1 (the side that wins no 1-D family) is not a zero
detector: zero-pattern \(R^2 = 0.19\), one-hot \(R^2 \approx 0.56\).

Observation only: the dlog fit on \(u,v\) is stable under the
multiplicative group and unstable under additive / random
relabelings. The hidden units are not computing a function of
\(ab\) or of \(\chi(ab)\). That is compatible with
\(\mathrm{relu}(\chi(a)+\chi(b)+c)\) and incompatible with a hidden
layer that has already multiplied characters.

Runs: `results/runs/20260909_023439_phase7_falsify`.

## Phase 8 observations (not interpretations)

Stated falsifier: replace each first-layer side \(u_i,v_i\) with its
fitted dlog sinusoid on \(\mathbb{F}_p^*\) (keep the original value
at 0), leave hidden bias and readout untouched. If complete-domain
accuracy collapses, the residual on 12 points was load-bearing and
the dlog fit is correlational.

Control: the same protocol with residue-axis Fourier fits (same
3-parameter family that lost Phase 5). Salvage: least-squares refit
of \(W_2,b_2\) onto one-hot labels from the projected hidden units.

Counts below are among networks that had at least one side replaced
(`exact_touched`). One width-8 net has no side with dlog \(R^2\ge 0.85\),
so `dlog_high` is a no-op there and is excluded.

| source | identity | dlog high (frozen \(W_2\)) | dlog all | Fourier all | dlog high + LS \(W_2\) |
|---|---|---|---|---|---|
| width 3 | 1/1, mean 1.00 | **0/1**, 0.680 | 0/1, 0.503 | 0/1, 0.112 | 0/1, 0.290 |
| width 4 | 7/7, 1.00 | **0/7**, 0.628 | 0/7, 0.564 | 0/7, 0.126 | 0/7, 0.334 |
| width 8 | 256/256, 1.00 | **0/255**, 0.715 | 0/256, 0.541 | 0/256, 0.125 | 0/255, 0.553 |

The falsifier fires. No projected net remains exact. Fourier projection
falls to ~0.12, near chance. Least-squares readout does not restore a
domain-exact classifier, so the projected hidden layer is not linearly
sufficient for the multiplication table.

dlog projection still leaves mean domain accuracy far above chance
(0.63–0.72 with frozen \(W_2\)). The sinusoid is not irrelevant; it
is not enough for exactness.

Observation only: \(u,v\) *look like* characters under \(R^2\) and
the Phase 7 orbit test. Exactness uses the residual of that fit.
That blocks an algorithm claim of the form “the first layer *is* a
dlog sinusoid.”

Runs: `results/runs/20260909_101444_phase8_substitute`.

## Phase 9 observations (not interpretations)

Matched control: modular **addition** \(y=(a+b)\bmod 13\), same
one-hot encoding, ReLU MLP, Adam \(lr=3\times 10^{-3}\), 8000
full-domain steps, \(P=1024\), seed 0 as Phase 3 multiplication.

### Capacity

| hidden | params | add exact / 1024 | mul exact / 1024 (Phase 3) | add mean domain |
|---:|---:|---:|---:|---:|
| 2 | 93 | 0 | 0 | 0.223 |
| 3 | 133 | 0 | 0 | 0.320 |
| 4 | 173 | 4 | 7 | 0.445 |
| 8 | 333 | 181 | 443 | 0.934 |
| 16 | 653 | 1023 | 1024 | 1.000 |

Smallest exact adder found: **width 4, 173 params** (4 of 1024).
Width 3 reached max domain 0.787, not 1.0. Under this encoding,
addition is not easier to solve exactly than multiplication at
matched width.

### Embedding family swaps with the task

Same vote rule as Phase 5 (\(R^2\ge 0.85\) and gap \(\ge 0.15\)).

| width | add Fourier / dlog / weak | mul Fourier / dlog / weak | add mean \(R^2\) (F / dlog) | mul mean \(R^2\) (F / dlog) |
|---:|---|---|---|---|
| 4 | 20 / 0 / 12 of 32 | 0 / 36 / 20 of 56 | 0.88 / 0.37 | 0.32 / 0.87 |
| 8 | 1573 / 0 / 1323 of 2896 | 0 / 2172 / 1923 of 4096 | 0.83 / 0.38 | 0.36 / 0.82 |
| 16 | 3984 / 0 / 4208 of 8192 | 3 / 4993 / 3196 of 8192 | 0.80 / 0.39 | 0.36 / 0.83 |

Adders: **zero** dlog wins. Multipliers: **zero** residue-Fourier
wins at width 4 and 8. Mean \(R^2\) swaps. Cycling \(a\mapsto a+1\)
in \(W_1\) drops adder accuracy to **0** (systematically off by 1)
and multiplier accuracy to chance (0.077).

Hidden-map factor \(R^2\): adders sum 0.10 vs product 0.05;
multipliers product 0.19 vs sum 0.014. Neither hidden layer is a
function of the binary operation.

### Substitution of the winning family

Replace high-\(R^2\) sides with the fitted sinusoid; freeze readout.
Counts among nets that had at least one side replaced.

| source | Fourier-high (adders) | dlog-high (multipliers, Phase 8) |
|---|---|---|
| width 4 | **0/4** exact, mean 0.55 | **0/7**, 0.63 |
| width 8 | **0/181**, 0.65 | **0/255**, 0.72 |
| width 16 | **0/256**, 0.88 | (not in Phase 8; remeasured 20/256, mean 0.96) |

Cross-family projection (dlog onto adders, Fourier onto multipliers)
is a no-op on “high” sides: there are none. Forcing it on every side
drops adders to ~0.12 and multipliers to ~0.13, near chance.

Least-squares readout after Fourier projection does not restore an
exact adder.

Observation only: the 1-D family that wins tracks the group of the
task (additive cycle vs multiplicative characters). Substituting the
fitted sinusoid still destroys exactness on both tasks at small
width. “High \(R^2\) plus a load-bearing residual” is not special to
multiplication.

Runs: `results/runs/20260909_101659_phase9_modadd`.

## A vs B observations (not interpretations)

Question: does replacing the hidden primitive \(u[a]+v[b]\) with
\(u[a]\,v[b]\) move the population from table-fitting to
compositional rule learning?

Domain is \(\mathbb{F}_{13}^\times\) (144 pairs, 12-way labels).
Both models: one hidden layer, width 4, **160 parameters**, Adam
\(lr=3\times 10^{-3}\), 8000 full-batch steps, \(P=1024\), seed 0.
Model A: \(h_j=\mathrm{ReLU}(u_j[a]+v_j[b]+c_j)\).
Model B: \(h_j=\mathrm{ReLU}(u_j[a]\,v_j[b]+c_j)\).
No discrete-log initialization.

Success is **domain-exact on all 144 pairs**, including those held
out. Chance is \(1/12\approx 0.083\). Checkerboard
memorize-plus-chance domain is \(\approx 0.542\).

| model | protocol | exact / 1024 | mean train | mean test | mean domain | max domain | max test |
|---|---|---:|---:|---:|---:|---:|---:|
| A add | full table | 1 | 0.479 | 0.479 | 0.479 | 1.000 | 1.000 |
| B product | full table | **507** | 0.967 | 0.967 | 0.967 | 1.000 | 1.000 |
| A add | random 70/30 (101/43) | 0 | 0.608 | 0.025 | 0.434 | 0.819 | 0.395 |
| B product | random 70/30 (101/43) | **0** | 0.988 | 0.235 | 0.763 | 0.993 | 0.977 |
| A add | dlog checkerboard (72/72) | 0 | 0.902 | **0.000** | 0.451 | 0.500 | **0.000** |
| B product | dlog checkerboard (72/72) | **0** | 1.000 | **0.000** | 0.500 | 0.500 | **0.000** |

Wilson 95% on \(\Pr(\text{exact})\): A full-table \(1/1024\) in
\([0.0002, 0.0055]\); B full-table \(507/1024\) in \([0.465, 0.526]\).
Every holdout cell is \(0/1024\), upper bound \(0.0037\).

On the checkerboard, Model B memorizes the even-\(\mathrm{dlog}\)
half (1013/1024 train \(\ge 0.99\)) and is wrong on **every** held
pair in **every** network. Model A is the same on the test half,
with weaker train fit (90/1024 memorize). A 12-way guesser would
almost never zero 72 test pairs; this is not chance.

On random 70/30, Model B fits train (756/1024 memorize) and leaks a
little (mean test 0.235 vs chance 0.083). Best domain is
\(143/144\). Three nets reach test \(\ge 0.90\) without becoming
domain-exact. That is still Level 0 plus leakage, not Level 2.

The product primitive makes complete-table exactness easy. It does
not produce a solver of pairs it was never shown.

Runs: `results/runs/20260909_103432_ab_product`.

## Model C observations (not interpretations)

Architecture: one shared 2-D table \(e(a)=(x_a,y_a)\), interaction
the complex product, linear readout, **no** discrete-log init.
Heads \(H\) are independent copies concatenated. Planted
roots-of-unity plus nearest-root readout is domain-exact (the
architecture can express \(z(ab)=z(a)z(b)\)). Same optimizer,
\(P=1024\), 8000 steps, seed 0, \(\mathbb{F}_{13}^\times\).

| model | params | protocol | exact / 1024 | mean train | mean test | max domain | max test |
|---|---:|---|---:|---:|---:|---:|---:|
| A add | 160 | full table | 1 | 0.479 | 0.479 | 1.000 | 1.000 |
| B product | 160 | full table | 507 | 0.967 | 0.967 | 1.000 | 1.000 |
| C \(H=1\) | 60 | full table | 372 | 0.755 | 0.755 | 1.000 | 1.000 |
| C \(H=3\) | 156 | full table | **1024** | 1.000 | 1.000 | 1.000 | 1.000 |
| A add | 160 | random 70/30 | 0 | 0.608 | 0.025 | 0.819 | 0.395 |
| B product | 160 | random 70/30 | 0 | 0.988 | 0.235 | 0.993 | 0.977 |
| C \(H=1\) | 60 | random 70/30 | **375** | 0.818 | 0.646 | 1.000 | 1.000 |
| C \(H=3\) | 156 | random 70/30 | **931** | 1.000 | 0.990 | 1.000 | 1.000 |
| A add | 160 | dlog checkerboard | 0 | 0.902 | 0.000 | 0.500 | 0.000 |
| B product | 160 | dlog checkerboard | 0 | 1.000 | 0.000 | 0.500 | 0.000 |
| C \(H=1\) | 60 | dlog checkerboard | 0 | 0.873 | 0.000 | 0.500 | 0.000 |
| C \(H=3\) | 156 | dlog checkerboard | 0 | 1.000 | 0.001 | 0.556 | 0.111 |

Wilson 95% on random-70 \(\Pr(\text{exact})\): C \(H=1\) \(375/1024\)
in \([0.337, 0.396]\); C \(H=3\) \(931/1024\) in \([0.890, 0.925]\).
A and B remain \(0/1024\) (upper bound \(0.0037\)).

This is the first time a network trained on an incomplete table is
domain-exact on all 144 pairs. Extra heads make that the typical
outcome (\(91\%\) at \(H=3\)), not a rare organism.

The checkerboard column stays at 0. That split withholds every
odd-\(\mathrm{dlog}\) **label**: train products are 6 residues, test
products the other 6. Zero test accuracy does not by itself rule
out a group embedding; the readout never sees those 6 classes.

### Substitution on the smallest generalizers

256 saved domain-exact \(H=1\) nets from random 70/30 (all still
exact on reload). Each 2-D table is fit to a scaled, rotated
(and optionally conjugated) dlog sinusoid
\(\mathrm{scale}\cdot R(\phi)\,(\cos(2\pi k \log_g a/12),\sin(\cdot))\).

| measurement | value |
|---|---|
| mean / min circle \(R^2\) | 0.996 / 0.986 |
| fraction \(R^2\ge 0.95\) | 256/256 |
| winding \(k\) | 120 of \(k=1\), 133 of \(k=5\), 2 of \(k=7\), 1 of \(k=11\) |
| mean radius CV | 0.059 |
| replace \(e\) by the fit, freeze readout | **256/256 still exact** |

\(k=1,5,7,11\) are the units of \(\mathbb{Z}/12\mathbb{Z}\)
(\(7\equiv -5\), \(11\equiv -1\)). The residual off the fitted
circle is not load-bearing. That is the opposite of Phase 8 on
Model A, where dlog projection destroyed exactness.

Runs: `results/runs/20260909_104121_complex_composition`.

## Held row / column (class-preserving hole)

Hold residue 12 in **one** slot only: \(a=12\) (row) or \(b=12\)
(column). Train 132 / test 12. All 12 product classes appear in
both splits. The held residue still occurs in the other slot, so a
shared embedding of 12 is trained; a first-slot-only map of 12 is
not.

Checkerboard confirmed as a 6-vs-6 label split. This hole is not.

Same optimizer, \(P=1024\), 8000 steps, seed 0.

| model | params | hole | exact / 1024 | mean train | mean test | max domain | max test |
|---|---:|---|---:|---:|---:|---:|---:|
| A add | 160 | row | 0 | 0.498 | 0.027 | 0.854 | 0.333 |
| A add | 160 | col | 0 | 0.506 | 0.025 | 0.875 | 0.250 |
| B product | 160 | row | 0 | 0.972 | 0.085 | 0.958 | 0.500 |
| B product | 160 | col | 0 | 0.975 | 0.083 | 0.951 | 0.500 |
| C \(H=1\) | 60 | row | **375** | 0.791 | 0.689 | 1.000 | 1.000 |
| C \(H=1\) | 60 | col | **375** | 0.791 | 0.689 | 1.000 | 1.000 |
| C \(H=3\) | 156 | row | **1016** | 1.000 | 0.999 | 1.000 | 1.000 |
| C \(H=3\) | 156 | col | **1016** | 1.000 | 0.999 | 1.000 | 1.000 |

Wilson 95%: C \(H=1\) \(375/1024\) in \([0.337, 0.396]\); C \(H=3\)
\(1016/1024\) in \([0.985, 0.996]\). A and B are \(0/1024\).

Model B's mean test is chance (\(1/12\)). It memorizes the 132
seen pairs and does not compute \(\times 12\). Model C \(H=3\) is
domain-exact in \(99\%\) of random inits, including the missing
factor. Row and column match for C, as required by a shared
\(e(a)=e(b)\).

This is structured generalization with every output class in
train. It is not the checkerboard (missing-class) test.

Runs: `results/runs/20260909_104844_held_row`.

## Model C certificate (held-row H=1, p=13)

Source: 256 saved domain-exact H=1 nets from held-row residue 12
(`successful.pt`; 60 params). Primitive root \(g=2\). No retraining.
Raw \(\delta=\mathrm{wrap}(\theta(ab)-\theta(a)-\theta(b))\) is a
nonzero constant when the embedding has a global phase; the
**centered** residual is \(\delta\) minus its circular mean.

### Homomorphism (head-0 2-D table)

| quantity | exact H=1 (256) | held-row failures in representatives (9) |
|---|---:|---:|
| mean \(\lvert\delta_{\mathrm{c}}\rvert\) | **0.28°** | 13.1° |
| mean \(\lvert\delta_{\mathrm{c}}\rvert\) on train pairs | 0.29° | 12.7° |
| mean \(\lvert\delta_{\mathrm{c}}\rvert\) on held row \(a=12\) | **0.23°** | 17.2° |
| mean of per-net max \(\lvert\delta_{\mathrm{c}}\rvert\) | 0.80° | 37.0° |
| fraction with max \(\lvert\delta_{\mathrm{c}}\rvert<2^\circ\) | 254/256 | 0/9 |
| circle \(R^2\) mean / min | 0.995 / 0.968 | 0.643 / 0.592 |
| winding \(k\) | 124 of 1, 129 of 5, 1 of 7, 2 of 11 | 7 of 2, 1 of 6, 1 of 10 |
| \(\gcd(k,12)=1\) | **256/256** | **0/9** |
| radius CV | 0.063 | 0.71 |
| corr(radius, dlog) | \(-0.005\) | \(-0.05\) |

The multiplication law on **held-out** pairs is as tight as on
train. Failed nets sit on non-units of \(\mathbb{Z}/12\mathbb{Z}\)
and do not close the homomorphism.

### Substitution (frozen readout unless noted)

| variant | domain exact / 256 | held-row exact / 256 |
|---|---:|---:|
| learned \(e\), frozen \(W\) | 256 | 256 |
| LS sinusoid (scale+rot+conj), frozen \(W\) | 256 | 256 |
| **exact** roots of unity with that scale/rot, frozen \(W\) | **256** | **256** |
| unit exact roots, no scale/rot, frozen \(W\) | 0 (mean acc 0.084) | 0 |
| unit exact roots, LS readout only | 256 | 256 |
| unit-normalize learned \(e\), frozen \(W,b\) | 0 (mean 0.294) | 0 |
| unit-normalize \(e\), rescale \(W\) by \(\bar r^2\) | 256 | 256 |
| exact aligned roots + geometric nearest-root \(W\), \(b=0\) | **256** | **256** |

No residue-specific residual is required. Scale and global phase
are gauges of the representation (they set the hidden radius
against \(b\) and the doubled angle in the product). They are not
a lookup table on 12 points.

### Causal interventions (predictions first)

| intervention | prediction | observed domain exact / 256 |
|---|---|---:|
| rotate every \(e(a)\) by \(\phi\in\{15,30,45,90\}^\circ\), freeze \(W\) | hidden rotates by \(2\phi\); freeze fails | 0 |
| same + \(W\leftarrow R_{2\phi}W\) | exactness restored | **256** |
| \(\phi=180^\circ\), freeze \(W\) | \(2\phi=360^\circ\); product unchanged | **256** |
| replace \(k=1\) nets by exact \(k=5\) (aligned), freeze \(W\) | those 124 fail; other 132 stay | 132 |
| same + geometric \(W\) for \(k=5\) | all 256 exact | **256** |
| \(k=5\to 1\), freeze \(W\) | 129 fail | 127 |
| \(k=5\to 1\) + predicted \(W\) | 256 | **256** |
| \(\theta(a_0)\leftarrow\theta(a_0)+\varepsilon\) | errors only on pairs using \(a_0\) | affected ~0.45–1.0; **unaffected 0.000** |
| embedding automorphism \(a\mapsto a^q\) (\(q=5,7,11\)), freeze \(W\) | winding \(kq\); freeze fails | 0 |
| same + predicted new winding \(E,W\) | exact | **256** |
| random permutation of the 12 slots, freeze \(W\) | fail | 0 |

H=3 saved exact nets still admit exact-circle substitution of every
head (256/256), but a single head is not a faithful character
(only ~38% of heads have unit \(k\); mean \(\lvert\delta_{\mathrm{c}}\rvert\)
11.8°). The clean certificate is H=1.

Symbolic form of an exact H=1 solver, 60 neural parameters:

```
k in {1,5,7,11} = (Z/12Z)*
theta(a) = 2 pi k log_2(a) / 12     # or conjugate
z(a)     = s * exp(i (phi + theta(a)))
h        = z(a) z(b)
class    = argmax_c  h · (s^2 R_{2 phi} z(c))
```

Level: **4 (symbolic extraction)** for Model C, H=1, \(p=13\),
held-row generalizers. Not yet Level 5 (one prime, one split).

Runs: `results/runs/20260909_111156_phase_c_certificate`.

## Restore zero (full \(\mathbb{Z}/13\mathbb{Z}\) table)

Same Model C, now on all 169 pairs including the absorbing element.
13-way labels. Chance \(1/13\approx 0.077\). Held row 12 is a missing
nonzero factor (13 test pairs, all 13 classes in train). Held row 0
is a missing absorbing factor; test labels are all 0, while \(e(0)\)
is still trained as the second operand.

A is the ReLU additive MLP, width 4, 173 params (full-table 7/1024
matches Phase 3). C \(H=1\) has 65 params; \(H=3\) has 169.

| model | params | protocol | exact / 1024 | mean train | mean test | max domain | \(r_0/\bar r_{*}\) |
|---|---:|---|---:|---:|---:|---:|---:|
| A add | 173 | full table | 7 | 0.595 | 0.595 | 1.000 | — |
| C \(H=1\) | 65 | full table | **380** | 0.799 | 0.799 | 1.000 | \(2\times 10^{-5}\) |
| C \(H=3\) | 169 | full table | **1024** | 1.000 | 1.000 | 1.000 | 0.200 (median 0.004) |
| A add | 173 | random 70/30 | 0 | 0.672 | 0.160 | 0.769 | — |
| C \(H=1\) | 65 | random 70/30 | **375** | 0.885 | 0.784 | 1.000 | \(4\times 10^{-4}\) |
| C \(H=3\) | 169 | random 70/30 | **407** | 1.000 | 0.964 | 1.000 | 0.062 |
| A add | 173 | held row 12 | 0 | 0.618 | 0.103 | 0.953 | — |
| C \(H=1\) | 65 | held row 12 | **371** | 0.829 | 0.720 | 1.000 | \(8\times 10^{-5}\) |
| C \(H=3\) | 169 | held row 12 | **1004** | 1.000 | 0.998 | 1.000 | 0.218 (median 0.009) |
| A add | 173 | held row 0 | 0 | 0.556 | 0.086 | 0.929 | — |
| C \(H=1\) | 65 | held row 0 | **373** | 0.783 | **1.000** | 1.000 | \(3\times 10^{-5}\) |
| C \(H=3\) | 169 | held row 0 | **998** | 1.000 | 0.998 | 1.000 | 0.009 |

Among exact C \(H=1\) nets, \(\|e(0)\|\) is four orders of magnitude
below the mean nonzero radius: zero is the origin, so
\(e(0)e(b)=0\). On held row 0, **every** one-head net scores 1.0 on
the 13 test pairs (373 domain-exact, 651 test-perfect but not
domain-exact). The absorbing row is not a second algorithm; it is
the zero vector.

C \(H=3\) still fills both missing rows (\(\ge 97\%\) exact) but
random-70 exactness drops from 931/1024 on \(\mathbb{F}_{13}^\times\)
to 407/1024 on the 169-cell table (mean test remains 0.964). Extra
heads plus the absorbing element make complete-domain exactness
from a random hole less typical, not more.

A remains 0 exact on every incomplete table.

Runs: `results/runs/20260909_105910_restore_zero` (C);
A ReLU control `results/runs/20260909_110437_restore_zero`.

## Bridge (exploratory; not Collatz)

A neural modulus-shortlist plus symbolic extractor was run on three
approved Collatz-like maps. Level 5 on
`medium_five_on_1_mod_16` and `hard_five_on_1_mod_16`; failed covering
on the matched negative `control_five_on_1_mod_4` (`agreed_negative`).
Do not train \(3n+1\). Write-up:
`tiny_tao_results/bridge_neural_certificates.md`.
