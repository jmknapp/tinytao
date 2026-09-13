# Tiny Tao Project Context

**Purpose:** Durable research context for Cursor/Claude/other coding
agents working on this repository.\
**Last consolidated:** 2026-09-10

> **Agent instruction:** Read this file and `README.md` before modifying
> experiments. Treat preregistered predictions, stop rules, exactness
> gates, control definitions, and completed-vs-uncompleted status as
> scientific constraints. Do not silently change them after observing
> results.

## 1. Research objective

This project tests Terence Tao's idea of studying populations of very
small neural networks as experimental mathematical systems.

Pipeline:

1.  Train large populations of tiny networks on precisely defined
    mathematical tasks.
2.  Identify recurrent or rare exact solvers.
3.  Distinguish memorization from genuine compositional rules.
4.  Mechanistically reverse-engineer successful networks.
5.  Replace the neural implementation with a compact exact symbolic
    algorithm.
6.  Verify the symbolic algorithm independently.
7.  Use extracted mechanisms to make prospective predictions.
8.  Only after calibration should the method be applied to substantially
    harder/open problems.

Key principle: **the neural network should ideally disappear from the
final mathematical result.**

## 2. Original additive-ReLU modular experiment

Initial task: `(a*b) mod 13`, with concatenated one-hot inputs and tiny
one-hidden-layer ReLU networks:

`h_j(a,b) = ReLU(u_j[a] + v_j[b] + c_j)`.

The population dimension was vectorized on GPU.

Full-table results:

-   width 4: 7/1024 exact, 173 parameters
-   width 8: 443/1024 exact
-   width 16: 1024/1024 exact
-   width 3 can be exact: 1/4096 at 12,000 steps, 133 parameters
-   8-bit quantization of the seven width-4 multipliers remained exact;
    6-bit did not

Structured holdouts (random 70/30, held row/column, zeros, product-0,
odd-dlog holes) produced no exact generalizers. Wider networks could
memorize visible pairs while remaining near chance on holes.

First-layer embeddings often fit sinusoidal functions of discrete log
very well (mean R² about 0.87 at width 4), but replacing learned
embeddings by fitted sinusoids destroyed exactness. The residual
finite-table information was load-bearing.

**Conclusion:** `symmetry-organized memorization`.

Addition controls showed the analogous effect with residue-Fourier
structure. Therefore a pretty/group-like representation or high R² alone
is not evidence of an algorithm.

## 3. Architecture experiment

Central question:

> What minimal architectural change causes a transition from
> symmetry-organized memorization to genuine compositional rule
> learning?

### Model A

Original additive ReLU control.

### Model B

Minimal multiplicative/product interaction with roughly comparable
parameter count. It still failed to give the desired structured
generalization.

### Model C

Shared learned 2-D embedding `e(a)=(x_a,y_a)`, interpreted as
`z(a)=x_a+i y_a`, followed by fixed complex multiplication:

`z(a)z(b) = (x_a x_b - y_a y_b) + i(x_a y_b + y_a x_b)`.

A tiny learned readout maps the 2-D product to the output residue.

There is no discrete-log initialization. The architecture supplies a
compositional primitive capable of representing the group law; SGD must
discover the embedding.

## 4. Model C breakthrough at p=13

On `F_13^*` (12 elements), random 70/30:

  model     params   exact/1024
  ------- -------- ------------
  A            160            0
  B            160            0
  C H=1         60          375
  C H=3        156          931

For 256 saved H=1 exact generalizers:

-   mean dlog-circle R² ≈ 0.996, minimum ≈ 0.986
-   dominant windings k=1 and k=5
-   replacing embeddings by fitted sinusoids while freezing readout left
    256/256 exact

Thus the structured representation became functionally sufficient.

A checkerboard split still gave zero exact because six of twelve output
labels were absent from training; this is a missing-class test, not a
clean counterexample.

Held-row/held-column, with residue 12 omitted from only one operand slot
while its shared embedding remained trained in the other:

  model     row exact/1024   col exact/1024
  ------- ---------------- ----------------
  A                      0                0
  B                      0                0
  C H=1                375              375
  C H=3               1016             1016

All output classes remained present in training.

## 5. Level-4 faithful certificate at p=13

With generator g=2 and `a=g^j mod 13`, successful H=1 networks
implement, up to gauge:

`z(a) = s exp(i(phi + 2*pi*k*j/12))`

with `k in {1,5,7,11}`, exactly the units modulo 12.

Gauge-corrected homomorphism residual:
`delta = wrap(theta(ab)-theta(a)-theta(b))`, after subtracting its
circular mean.

For 256 exact H=1 networks:

-   mean centered \|delta\| all pairs ≈ 0.28°
-   held row ≈ 0.23°
-   254/256 had maximum error \<2°
-   radius CV ≈ 0.063
-   radius essentially uncorrelated with dlog

Exact substitutions/interventions:

-   exact roots of unity + fitted scale/rotation + frozen readout:
    256/256
-   unit roots + LS readout: 256/256
-   unit-normalized embeddings + appropriate readout rescaling: 256/256
-   exact aligned roots + geometric nearest-root readout, bias 0:
    256/256
-   rotation and automorphism interventions behaved exactly as complex
    multiplication predicts

**Classification:** Level 4 mechanistic solver for H=1, p=13.

## 6. Mechanism classes

Use these names consistently.

**CLASS 0 --- FAILURE:** not domain exact.

**CLASS 1 --- DOMAIN-EXACT NON-SYMBOLIC:** domain exact but no compact
exact symbolic mechanism extracted.

**CLASS 2 --- ALTERNATIVE SYMBOLIC SOLVER:** domain exact and
replaceable by a compact exact symbolic mechanism, but not a faithful
one-circle character.

**CLASS 3 --- FAITHFUL LEVEL-4 SOLVER:** domain exact, faithful unit
winding, homomorphism verified, exact symbolic/root substitution
succeeds.

Class numbers classify mechanism, not quality. Do not describe Class 2
as objectively better than Class 3.

## 7. Cross-prime Stage A

H=1 Model C, held-row, population 1024:

  p      exact   faithful/unit-k
  ---- ------- -----------------
  7        554               395
  11       553               458
  13       375   all saved exact
  17       516               516

Faithful mechanism replicated across all four primes.

p=7 and p=11 also produced exact non-unit windings with `gcd(k,p-1)=2`,
leading to a second symbolic mechanism.

## 8. Class-2 quotient + radial lift

For p=11, `F_11^* ≅ C10`. Among 553 exact networks:

-   458 faithful
-   95 non-unit with gcd(k,10)=2

The non-unit angular character collapses `{a,-a}`, yet the networks
remain exact.

Stage A.5 showed:

-   phase represents the C5 quotient
-   each angular fiber is `{a,-a}`
-   radius distinguishes the two members
-   radius partition exactly tracks dlog parity
-   product radius creates three shells `rho0²`, `rho0*rho1`, `rho1²`
-   same-vs-mixed shell status encodes the missing parity bit

Algebraically: `C10 ≅ C5 × C2`.

Writing `a=g^j`, angle stores `j mod 5`, radius class stores `j mod 2`.
The binary coordinate is also the quadratic character / Legendre-symbol
bit.

## 9. Stage A.5 Step 2: p=11 Class 2 certified

All 95 p=11 non-unit exact networks admit a compact neural-free
quotient+radial program reproducing the full C10 table exactly.

Important findings:

-   within-parity radius noise is not load-bearing
-   two radius levels by dlog parity preserve the mechanism
-   exact quotient phases + two radii + symbolic decoder solve 95/95
-   unit normalization removes the binary information and reduces
    performance to the phase-only bound
-   parity-preserving radius permutations preserve exactness
-   parity-destroying permutations destroy it

Where XOR lives:

1.  embedding radius stores `s(a)=log_g(a) mod 2`
2.  multiplication creates three product shells
3.  same-vs-mixed shell gives `s(ab)=s(a) XOR s(b)`

The positive radii are not themselves a finite group representation; the
mathematical representation is the CRT decomposition.

For g=2, Class-2 symbolic computation:

`u = (j_a + j_b) mod 5` `s = (j_a mod 2) XOR (j_b mod 2)`
`j = u + 5*((s-u) mod 2)` `answer = 2^j mod 11`

The group theory is classical. The interesting result is independent
emergence as an SGD attractor and mechanistic extraction.

## 10. Mod-4 hypothesis

For primes `p ≡ 3 (mod 4)`, write `p-1=2m` with m odd. Then:

`C_(2m) ≅ C_m × C2`.

So the p=11-style quotient+binary-lift representation is algebraically
available.

For `p ≡ 1 (mod 4)`, m is even and the analogous direct-product
decomposition does not hold.

Existence of the representation does **not** guarantee SGD will discover
it.

## 11. Prospective H_CLASS2_MOD4 test

Preregistered before training unseen primes.

Eligible p≡3 mod4: 19, 23, 31.\
Matched p≡1 mod4: 29, 37, 41.

Locked protocol: H=1 Model C, held row p-1, P=1024, 8000 Adam steps,
lr=3e-3, Xavier, no dlog init, unchanged float32 exactness gate.

Results:

  p      mod4   exact   faithful   Class2
  ---- ------ ------- ---------- --------
  19        3     409        348       61
  23        3       0          0        0
  31        3     319        271       48
  29        1     439        439        0
  37        1     355        355        0
  41        1     416        416        0

At p=19 and p=31 every gcd-2 exact network certified as Class 2 with
exact quotient phase + two dlog-parity radii + neural-free decoder.

At p=29/37/41 all exact solvers were faithful and no gcd-2 exact family
appeared.

At p=23, algebra permits Class 2 but 0/1024 passed the locked float32
exactness gate; 558 were at approximately 0.99999994. The gate was **not
relaxed**.

Correct wording:

> Class 2 appeared only where the decomposition was algebraically
> available, but not at every eligible prime under the locked protocol.

Do not say it appeared "exactly whenever p≡3 mod4."

### 11.1 Three-torus probe at p=31 (preregistered 2026-09-10, before training)

`C_30 ≅ C_2 × C_3 × C_5`. H=1 cannot store three phases; p=31 H=1
already found F and Q. H=3 is the same Model C with three heads.
Certificate: gcd triple `{15,10,6}`, neural-free CRT decoder 100%,
drop-one-factor fails, cyclotomic snap preserves. Zero certified nets
is a miss; do not raise H. Predictions:
`tiny_tao_results/preregistration_p31_h3_three_torus.json`.

**Scored (run `20260910_125735_p31_h3_three_torus`).** H=3 exact
1022/1024. Faithful head on 687 nets (F-like, not 3-torus). Certified
3-torus: **3/1022**, all three gcd triples `{15,10,6}` (in some
order) with CRT 900/900. Rare attractor, real. Do not raise H.
Score: `tiny_tao_results/preregistration_p31_h3_three_torus.scored.json`.

### 11.2 Valuation-min GCD probe (preregistered 2026-09-10, before training)

Not another torus. Model T is shared real embeddings and **fixed
coordinatewise min**, H=5 locked primes `{2,3,5,7,11}` on `{1..12}`.
Held row `a0=6` (not 12: that row is the only source of class 12).
Certificate: affine assignment of heads to `v_p`, neural-free
`product p^{min(v_p(a),v_p(b))}` 100%, drop-one-prime fails, **sum**
of the same coordinates fails. Destroyed-law control: symmetric
shuffle of off-diagonal gcd entries (seed 20260910). Zero certified
on the meet table is a miss; do not add primes or Euclid/Stein
decoders. Predictions:
`tiny_tao_results/preregistration_gcd_valuation_min.json`.

**Scored (run `20260910_134642_gcd_valuation_min`).** GCD meet: exact
659/1024, certified valuation-min **165/659**. Sum decoder 0/165.
Destroyed-law table: exact 0/1024, certified 0. Lattice algorithm is
an SGD attractor under fixed min; not a character of a torus. Do not
add primes. Score:
`tiny_tao_results/preregistration_gcd_valuation_min.scored.json`.

### 11.3 Spanning-tree Kirchhoff probe (preregistered 2026-09-10, before training)

Not a torus and not valuation-min. Domain: all labeled simple graphs
on n=5 vertices (`2^{10}=1024`). Labels `τ(G)` by Kirchhoff cofactor
(independent DC oracle). Held edge `e=(0,1)`; `K_5` is the unique
graph with `τ=125` and contains every edge, so class 125 is
test-only. Model K stacks the cofactor of a learned weighted
Laplacian (`w_e = scale_e x_e + bias_e`); SGD must learn 0-1 edge
weights. Certificate: binarize learned `w` at 0.5, cofactor equals
`τ` on every graph, drop-one-edge fails, `det(A[1:,1:])` is not `τ`.
Destroyed-law control: seed-20260910 permutation of the `τ` vector.
Model A is a width-8 ReLU MLP with no Laplacian. Do not raise n to 6
or add a recursive DC net after a miss. Predictions:
`tiny_tao_results/preregistration_spanning_trees_n5.json`.

## 12. OSC replication

OSC OnDemand/Pitzer is now used for GPU-backed Jupyter experiments.

OSC home: `/users/PAS2038/jmknapp`\
Remote project target: `~/tao`

`tiny_tao_osc_gpu.ipynb` implements Model C H=1 population training,
exact counts, winding analysis, homomorphism diagnostics, and radial
diagnostics.

OSC p=11 run:

-   P=1024
-   8000 steps
-   lr=3e-3
-   held row
-   g=2
-   557/1024 exact
-   exact count stabilized around step 2500
-   training time ≈0.61 min
-   PyTorch peak allocation ≈0.04 GB

Winding census among exact:

-   446 gcd(k,10)=1 faithful candidates
-   111 gcd(k,10)=2 Class-2 candidates

Earlier local p=11: 553 exact = 458 faithful + 95 Class 2.

The OSC gcd-2 population should not be called formally Class-2 certified
solely from the gcd census; apply radial/symbolic replacement if a
formal replication claim is needed.

## 13. Stage B controls --- PREREGISTRATION FROZEN

Locked preregistration:
`results/runs/20260909_125737_stage_b_controls/control_predictions.json`

Locked primes: 11, 19, 29. p=31 deliberately excluded before table
generation. No p=23, no Collatz, Model C unchanged.

Locked protocol:

-   8000 Adam steps
-   lr 3e-3
-   Xavier
-   H=1
-   held-row last symbol
-   float32 domain accuracy gate \>=1.0
-   instance 0: 1024 nets
-   instances 1--4: 256 nets
-   master seed 20260909

Baseline priors:

-   p=11: 553 exact / 458 faithful / 95 Class2
-   p=19: 409 / 348 / 61
-   p=29: 439 / 439 / 0

Stop before controls if:

-   exact rate is zero
-   exact rate differs by \>0.20 absolute
-   Class2 vanishes at p=11 or p=19
-   Class2 appears at p=29

Control families:

**genuine:** actual cyclic multiplication.

**A isomorphic relabeling:** `a★b = pi^-1(pi(a)*pi(b))`. Preserves the
same cyclic group/all axioms; destroys numerical residue alignment. This
is scrambled labels, **not scrambled law**.

**B output permutation:** preserves latent genuine products, Latin
structure, commutativity; breaks output-label compatibility.

**C independent slot relabeling:** independently relabeled operand
slots; preserves Latin structure but breaks compatibility with a single
shared embedding.

**D random commutative balanced:** preserves size, commutativity, exact
class frequencies; destroys associativity, identity, inverses, Latin,
cyclicity.

**E nonassociative Latin:** generic and commutative variants; preserves
Latin/class balance, optionally commutativity; destroys
associativity/group axioms.

**F associativity damage:** genuine table corrupted at locked 1/5/10/25%
doses while preserving commutativity/global frequencies and most of the
table. At n=10, 1% cannot alter a frequency-preserving table as one
unordered pair, so locked fallback is a 2-cycle (2/55 pairs); p=11 1%
and 5% overlap. Do not retrospectively change this.

**G Ca×Cb:** preserves abelian group axioms/Latin but destroys cyclicity
and a faithful one-circle character. Skipped at order 10 because the
only abelian group of order 10 is C10.

155/155 intended tables were generated and verified. Every held-row
training set contains every output class.

**Scientific stop rule:** if a new exact family appears on a
destroyed-law control, stop aggregate analysis and extract it before
proceeding.

Strong expected pattern:
`genuine ≈ isomorphic relabeling >> destroyed-law controls`.

## 14. Possible third p=11 mechanism

Observed exact H=1 p=11 populations are exhausted empirically by:

1.  faithful one-circle family
2.  quotient+radial Class-2 family

This does not prove a third exact representation cannot exist.

Potential mathematical problem:

> Classify all exact H=1 Model-C representations of the C10
> multiplication table, up to gauge/equivalence.

Ask whether the exact solution space contains only faithful-circle and
quotient+radial components, or additional non-character geometries that
SGD rarely/never reaches.

## 15. Collatz status

Collatz is a long-term target, **not the next experiment**.

Do not attack it by trajectory prediction. Do not train `3n+1`.

Useful proof target: `for every n>1, exists k>0 such that T^k(n)<n`.

Candidate discoverable structures:

-   residue-class descent rules
-   finite transition graphs
-   Lyapunov/potential functions
-   recursive modular certificates
-   exact affine residue formulas `n=Mq+r -> T^k(n)=Aq+B`

A Discoverer--Falsifier loop was designed: DISCOVERER → compact rule →
FALSIFIER → counterexample/exceptional class → DISCOVERER.

Before genuine Collatz, use a bridge experiment on Collatz-like maps
with known hidden termination/descent proofs. The system should recover
an exact infinite-domain certificate, not merely fit finite
trajectories.

### 15.1 Neural bridge (exploratory, 2026-09-10)

Frozen protocol, black box only, feature bank
`{2,3,4,5,6,7,8,9,10,12,15,16,24,32}`, population 1024, 1500 steps,
seed `20260909`, oracle JSON read last. Notebook
`tiny_tao_bridge_neural.ipynb`. Write-up
`tiny_tao_results/bridge_neural_certificates.md`. Index
`tiny_tao_results/bridge_neural_index.json`.

The neural stage is a modulus shortlist (top 8 by residue-feature
energy). Success is the extracted covering after Falsifier + all-q
verifier, not a low potential loss. Energy ranking put `M=4` first on
every map; 16 was never the neural top.

  map                          role       extracted M   Level 5   oracle relation
  ---------------------------- ---------- ------------- --------- ------------------------------------
  `medium_five_on_1_mod_16`    positive   16            yes       smaller mixed covering than affine 32
  `hard_five_on_1_mod_16`      positive   16            yes       discoverer covered, affine oracle did not
  `control_five_on_1_mod_4`    negative   16 incomplete no        `agreed_negative`

Hard is not a copy of medium: residue 9 is `F=12q+7` (the `3n+1`
class) instead of `8q+5`. Control Lyapunov did not saturate (frac
`V(F)<V(n)` = 0.887 vs 1.0 on both positives); uncovered classes
`{5,9,13}`.

Do not enlarge the feature bank after seeing these maps. Do not retune
the Discoverer from hidden oracle JSON. `trivial_plus_one` and
`easy_half_collatz` were Level 5 in the symbolic suite
(`bridge_suite_20260909_224156`) and were not part of this neural trio.

### 15.2 Prospective map (preregistered 2026-09-10, before training)

Map `prospective_seven_on_1_mod_32`: `7n+1` only on `n≡1 (mod 32)`,
`n+1` on other odd classes. Feature bank unchanged. Predictions locked
in `tiny_tao_results/bridge_preregistration_seven_on_1_mod_32.json`.

Locked predictions: `M=16` incomplete; mixed covering at `M=32` with
`F=28q+1` on residue 1; Level 5 if and only if 32 is in the neural
top-8 shortlist. If 32 is missing, record a shortlist miss — do not
raise `TOP_K` or add moduli. Neural `top_M` need not be 32. Affine-only
oracle may fail on residue 31. Not Collatz.

**Scored (run `bridge_neural_prospective_seven_on_1_mod_32_20260910_012136`).**
Preregistered gate: **False**. P3 missed: 32 ranked 10th, shortlist
`(2,4,6,8,12,10,3,16)`. P1 held: chosen `M=16` uncovered `{1}`. P5
held: `top_M=2`. P6 applied: do not raise `TOP_K`. Lyapunov still
saturated on the finite sample (frac `V(F)<V(n)` = 1.0). Hidden affine
oracle verified at `M=32`. Score file:
`tiny_tao_results/bridge_preregistration_seven_on_1_mod_32.scored.json`.

### 15.3 Thicker prospective map (preregistered 2026-09-10, before training)

Map `prospective_seven_and_three_mod_32`: `7n+1` on `n≡1 (mod 32)`,
`3n+1` on `n≡17 (mod 32)`, `n+1` elsewhere. Expanding density 1/16 of
odds (about 124 of 999 samples), matching medium, while `M=16` still
mixes two expanding laws on residue 1. Not a rerun of the 1/32 miss
with higher `TOP_K`. Predictions locked in
`tiny_tao_results/bridge_preregistration_seven_and_three_mod_32.json`.

Same frozen bank and top-8. Gate: shortlist contains 32 and Level 5 at
`M=32`. If 32 is missing, stop again.

**Scored (run `bridge_neural_prospective_seven_and_three_mod_32_20260910_012529`).**
Preregistered gate: **False**. P3 missed again: 32 ranked 10th (energy
0.00503, same as the 1/32 miss). Shortlist `(4,2,8,6,12,10,3,24)`.
Chosen `M=8`, uncovered `{1}`. Lyapunov saturated. Hidden affine oracle
verified at `M=32`. Score:
`tiny_tao_results/bridge_preregistration_seven_and_three_mod_32.scored.json`.
Do not raise `TOP_K`.

### 15.4 Neural lemma heads (preregistered 2026-09-10, before training)

Energy ranking is no longer the shortlist. Tiny affine heads propose
integer `(M,r,k,A,B)` on the **locked** feature bank; odd-part fill is
the existing leftover. Falsifier and Level 5 unchanged. `TOP_K`
unused. Bank unchanged. Not Collatz.

Predictions:
`tiny_tao_results/bridge_preregistration_neural_lemmas.json`.

P1: Level 5 on `prospective_seven_and_three_mod_32` (predicted `M=32`).
P2: medium still Level 5. P3: control still fails.

**Scored (run `bridge_neural_lemmas_20260910_013006`).**
Preregistered gate: **True** (P1). P1 held: `M=32`,
`LEVEL_5_universal_descent`; `r=1` `odd_part(7n+1)` (worst-case
`28q+1`), `r=17` `odd_part(3n+1)` (worst-case `24q+13`). P3 held:
control uncovered `{1}`, `agreed_negative`. P2 missed: medium chosen
`M=16` uncovered `{1}`. Heads did not recover the known
`F^2=10q+1`; `odd_part(5n+1)` does not descend. Do not restore
algebraic `_try_affine` after seeing this miss. Do not raise `TOP_K`.
Score: `tiny_tao_results/bridge_preregistration_neural_lemmas.scored.json`.

### 15.5 Exact affine leftover (preregistered 2026-09-10, before training)

Same heads and locked bank as 15.4. Residues the rounded heads miss
get the existing exact `_try_affine` leftover, then odd-part. New
protocol, not a rescore of 15.4 P2. `TOP_K` unused. Bank unchanged.
Not Collatz.

Predictions:
`tiny_tao_results/bridge_preregistration_neural_lemmas_exact_fill.json`.

P1: thicker still Level 5 at `M=32`. P2: medium Level 5 at `M=16` with
`r=1` `F^2=10q+1` from `exact_affine`. P3: control still fails. P4:
hard Level 5 at `M=16`. P5: 15.4 P2 stays a miss.

**Scored (run `bridge_neural_lemmas_exact_fill_20260910_014058`).**
Preregistered gate: **True**. P1–P4 held. Thicker `r=1` is now
`F=28q+1` and `r=17` is `F=24q+13`, both `exact_affine` (heads still
miss those classes). Medium `r=1` is `F^2=10q+1` from `exact_affine`.
Hard `r=9` is `F=12q+7` from `exact_affine`. Control incomplete at
`M=32`, uncovered `{5,13,21,25,29}`, `agreed_negative`. 15.4 P2
remains a miss. Do not raise `TOP_K`. Score:
`tiny_tao_results/bridge_preregistration_neural_lemmas_exact_fill.scored.json`.

### 15.6 Locked-bank symbolic control (preregistered 2026-09-10, before the run)

No neural stage. Exact `propose_for_modulus` on the **locked** feature
bank, `k_max=4`. Same Falsifier and Level 5. Tests whether 15.5
certificates require the lemma heads. Includes the thin map
`prospective_seven_on_1_mod_32` that energy ranking missed. Bank
unchanged. Not Collatz.

Predictions:
`tiny_tao_results/bridge_preregistration_locked_bank_symbolic.json`.

P1–P4: same Level 5 pattern as 15.5. P5: thin map Level 5 at `M=32`,
`M=16` incomplete. P6: if those hold, heads are not load-bearing here.

**Scored (run `bridge_locked_bank_symbolic_20260910_102936`).**
Preregistered gate: **True**. P1–P6 held. Thin map `M=16` uncovered
`{1}`; Level 5 at `M=32` with `F=28q+1`. Same certificates as 15.5
with no neural stage. Lemma heads are not load-bearing on this bank.
Do not raise `TOP_K`. Do not start Collatz. Score:
`tiny_tao_results/bridge_preregistration_locked_bank_symbolic.scored.json`.

### 15.7 Covering outside the locked bank (preregistered 2026-09-10, before the run)

Map `prospective_seven_on_1_mod_64`: `7n+1` only on `n≡1 (mod 64)`.
Hand analysis: covering at `M=64` with `F=56q+1`; `M=32` residue 1
mixes two laws. Same 15.6 extractor, **locked bank unchanged** (max
32). No neural stage. Oracle JSON after discovery. Not Collatz.

Predictions:
`tiny_tao_results/bridge_preregistration_seven_on_1_mod_64.json`.

P1: Level 5 false; residue 1 uncovered on the bank. P2: medium still
Level 5. P3: control still fails. P4: hidden oracle verifies at
`M=64` (`discoverer_failed_oracle_verified`). P5: do not add 64 to
the bank after the miss.

**Scored (run `bridge_seven_on_1_mod_64_20260910_123215`).**
Preregistered gate: **True**. Locked-bank sweep chose `M=32`,
uncovered `{1}`, Level 5 false. Oracle (written after discovery)
verified at `M=64`. Medium still Level 5. Control still
`agreed_negative`. Do not add 64 to `FEATURE_MODULI`. Score:
`tiny_tao_results/bridge_preregistration_seven_on_1_mod_64.scored.json`.

## 16. Methodological rules

### Exactness

Do not loosen an exactness threshold after observing results. p=23 is
the canonical example.

### Preregistration

Do not alter locked predictions, primes, controls, stop rules, or
primary analysis after training starts. Exploratory follow-ups must be
labeled exploratory.

### Mechanistic evidence

Do not infer an algorithm merely from high R², PCA, Fourier/dlog-looking
embeddings, clustering, or high test accuracy. Require interventions and
ideally exact symbolic replacement.

### Gauge freedom

Account for global rotation, scale, automorphisms/winding choice, and
permutation symmetries.

### Neural-free certificate

Strongest evidence: 1. identify mathematical representation 2. replace
learned details by exact structure 3. remove/refit neural readout as
appropriate 4. reproduce full domain exactly 5. independently verify
symbolic rule

### Representation vs implementation

A learned W,b need not itself equal the clean symbolic decoder. p=11
Class2 showed that mathematically equivalent interventions can break
frozen W,b while a neural-free decoder remains exact.

### Negative results matter

Preserve Model A/B failures, checkerboard failure, p=23 null, the
bridge control `control_five_on_1_mod_4` (`agreed_negative`), and any
destroyed-law surprises.

### Stop on unexpected exact families

Do not average away a new mechanism.

## 17. Success levels

-   **Level 0:** table fitting
-   **Level 1:** structured representation
-   **Level 2:** structured generalization
-   **Level 3:** mechanistic generalization
-   **Level 4:** exact symbolic extraction and independent verification
-   **Level 5:** discovery/extraction platform generalizes prospectively
    and survives serious controls

The modular work has strong Level-4 evidence. Stage B is part of
assessing a broader Level-5 claim.

## 18. Mathematical terminology

For prime p, `F_p^* ≅ C_(p-1)`.

If g is primitive and `a=g^j`, a faithful complex character is:

`chi_k(a)=exp(2*pi*i*k*j/(p-1))`

with `gcd(k,p-1)=1`.

In words: assign nonzero residues unique equally spaced points on a
circle in generator-power order. Complex multiplication adds angles, so
the product point identifies modular multiplication.

For p=11, g=2 gives dlog order: `1, 2, 4, 8, 5, 10, 9, 7, 3, 6`.

Class2 instead uses a 5-position quotient clock plus one binary
coordinate.

## 19. Computing environment

Local Linux GPU: NVIDIA RTX 5060 Ti, 16 GB.

OSC/Pitzer: Cursor can connect by Remote SSH and edit the OSC copy
directly. Heavy GPU work must run through an OSC GPU allocation /
OnDemand / Slurm, **not on a login node**.

Local `/home/jmknapp/tao` and OSC `~/tao` are independent unless
explicitly synchronized.

## 20. Agent continuation checklist

Before coding:

1.  Read this file and README.md.
2.  Inspect relevant preregistration JSON.
3.  Identify whether work is replication, preregistered primary
    analysis, mechanistic extraction, or exploratory follow-up.
4.  Preserve that status in outputs.

For each new exact population report:

-   total population
-   exact count/rate
-   faithful/Class3 count/rate
-   Class2 count/rate
-   unclassified exact count/rate

Do not equate `gcd(k,n)=2` with fully certified Class2 until
quotient+radial symbolic certification is checked.

For any proposed mechanism ask:

-   what information does each coordinate carry?
-   is it injective/faithful?
-   what gauge freedoms exist?
-   does exact replacement work?
-   do targeted interventions behave predictably?
-   can the neural readout be removed?
-   can the resulting rule be verified/proved exactly?

## 21. Current scientific conclusion

With an appropriate minimal compositional primitive, SGD repeatedly
discovers exact classical representations of finite cyclic
multiplication:

1.  faithful complex-character representations
2.  when algebraically available, quotient + binary-lift representations

These can be mechanistically extracted, converted to exact symbolic
algorithms, and used to make prospective predictions about unseen
primes.

The algorithms themselves are classical. The potentially novel
contribution is the machine-discovery methodology and optimization
phenomenon:

> Populations of tiny neural networks can independently converge to
> multiple exact mathematical representations; those representations can
> be reverse-engineered, certified, and used to make prospective
> predictions about where the same solver families will or will not
> emerge.

Stage B tests whether this behavior specifically depends on genuine
algebraic law rather than superficial finite-table structure.

## 22. Preserve scientific history

Keep raw run artifacts and preregistration files immutable when
possible.

For follow-ups:

-   create a new run directory
-   reference the parent run
-   state whether predictions were written before training
-   preserve seeds/configuration
-   write a concise machine-readable summary
-   keep symbolic verification separate from neural training where
    practical

Unexpected results are not bugs unless independently shown to be
implementation errors.
