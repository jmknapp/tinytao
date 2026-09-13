# Neural Discoverer certificates (exploratory)

Status: **exploratory**. Not Collatz. Not a claim that SGD wrote the
lemmas. The neural stage is a modulus shortlist. The certificate is
the symbolic covering after the Falsifier and the all-\(q\) verifier.

Notebook: `tiny_tao_bridge_neural.ipynb`.
Machine-readable index: `tiny_tao_results/bridge_neural_index.json`.

## Frozen protocol

Same settings on every map, chosen before the control was run:

| knob | value |
|---|---|
| sample | odd \(n\le 2000\) (999 points) |
| Falsifier | odd \(n\le 20{,}000\) |
| population | 1024 independent linear nets |
| steps / lr / seed | 1500 / \(3\cdot 10^{-3}\) / `20260909` |
| feature bank | \(M\in\{2,3,4,5,6,7,8,9,10,12,15,16,24,32\}\) |
| shortlist | top 8 by mean \(W^2\) on the \(n\bmod M\) one-hots |
| \(V(n)\) | \(\alpha\log n + \sum_M c^{(M)}_{n\bmod M}\) |
| loss | \(\mathrm{ReLU}(V(F(n))-V(n)+0.1)\) |

The Discoverer sees the black box \(T\) / \(F_{\mathrm{odd}}\) only.
Hidden oracle JSON is read after the Level 5 gate. The feature bank
includes 16 as one of several \(2^k\) options; the loss is not told
that 16 is correct. Do not enlarge the bank after seeing these maps.

Success is **not** a low potential loss. Level 5 means: some shortlist
\(M\) extracts to a covering with no uncovered odd class, the Falsifier
finds no counterexample, and each lemma is proved for every integer
\(q\).

## Outcome

| map | role | frac \(V(F)<V(n)\) | energy top \(M\) | shortlist | extracted \(M\) | Level 5 | oracle |
|---|---|---:|---:|---|---:|---|---|
| `medium_five_on_1_mod_16` | positive | 1.000 | 4 | `(4,8,32,16,2,12,10,24)` | **16** | yes | smaller mixed covering than hidden affine \(M=32\) |
| `hard_five_on_1_mod_16` | positive | 1.000 | 4 | `(4,32,8,16,2,12,24,6)` | **16** | yes | discoverer covered; hidden affine oracle did not (\(M=65536\), 4096 failures) |
| `control_five_on_1_mod_4` | negative | 0.887 | 4 | `(4,10,2,8,5,6,12,16)` | 16, incomplete | **no** | `agreed_negative` |

Energy ranking never put 16 first. On both positives it was fourth.
The certificate is the extracted covering, not the bar chart.

## Medium: `medium_five_on_1_mod_16`

Law: \(5n+1\) on \(n\equiv 1\pmod{16}\); \(n+1\) on the other odd
classes.

Run: `tiny_tao_results/bridge_neural_medium_five_on_1_mod_16_20260910_005550`.
Replication after the plot-cell ylabel fix:
`..._20260910_010546` (same gate).

By step 400 every net had \(V(F(n))<V(n)\) on the sample. First
complete shortlist covering that survived the Falsifier: \(M=16\),
cover 1.000, `LEVEL_5_universal_descent`.

| \(n\bmod 16\) | lemma | all-\(q\) note |
|---|---|---|
| 1 | \(F^2=10q+1\) | locked at lift 32; \(A=10<16\); worst \(n=17\to 11\) |
| 3 | \(F=4q+1\) | \(A=4<16\); worst \(n=3\to 1\) |
| 5 | \(F=8q+3\) | \(A=8<16\); worst \(n=5\to 3\) |
| 7 | \(F=2q+1\) | \(A=2<16\); worst \(n=7\to 1\) |
| 9 | \(F=8q+5\) | \(A=8<16\); worst \(n=9\to 5\) |
| 11 | \(F=4q+3\) | \(A=4<16\); worst \(n=11\to 3\) |
| 13 | \(F=8q+7\) | \(A=8<16\); worst \(n=13\to 7\) |
| 15 | \(\mathrm{odd\_part}(n+1)\) | worst \(n=15\to 1\) |

Hidden oracle: verified all-affine covering at \(M=32\). The extracted
covering is smaller and mixed (odd-part on \(r=15\)). Allowed: the
discoverer need not match the hidden proof.

## Hard: `hard_five_on_1_mod_16`

Law: \(5n+1\) on \(n\equiv 1\pmod{16}\); \(3n+1\) on \(n\equiv 9\pmod{16}\);
\(n+1\) elsewhere.

Run: `tiny_tao_results/bridge_neural_hard_five_on_1_mod_16_20260910_011000`.

Same gate. The certificate is **not** a copy of the medium lemmas.
Residue 9 is the second expanding class: \(F=12q+7\) instead of
\(8q+5\).

| \(n\bmod 16\) | lemma | all-\(q\) note |
|---|---|---|
| 1 | \(F^2=10q+1\) | same as medium |
| 3 | \(F=4q+1\) | |
| 5 | \(F=8q+3\) | |
| 7 | \(F=2q+1\) | |
| 9 | \(F=12q+7\) | \(A=12<16\); worst \(n=9\to 7\) |
| 11 | \(F=4q+3\) | |
| 13 | \(F=8q+7\) | |
| 15 | \(\mathrm{odd\_part}(n+1)\) | |

Hidden affine oracle: **not verified** at \(M=65536\) (28672 lemmas,
4096 failures). Relation: `discoverer_covered_oracle_did_not`.

## Control: `control_five_on_1_mod_4`

Law: \(5n+1\) on every \(n\equiv 1\pmod{4}\); \(n+1\) on
\(n\equiv 3\pmod{4}\). Matched-looking expanding branch on a class
that does not admit a small 2-adic covering. Not claimed terminating.

Run: `tiny_tao_results/bridge_neural_control_five_on_1_mod_4_20260910_011150`.

Expected Level 5 gate: **False**. Observed: False.

The Lyapunov did not saturate. Loss stuck at \(0.023\); only \(0.887\)
of the sample had \(V(F(n))<V(n)\). On both positives that fraction
reached 1.0 by step 400.

No shortlist modulus produced a complete covering. Fallback \(M=16\),
cover \(0.625\), uncovered \(\{5,9,13\}\),
`LEVEL_1_survives_finite_testing`. Those three odd classes are the
expanding \(5n+1\) residues at this modulus.

Lemmas that did fit:

| \(n\bmod 16\) | lemma |
|---|---|
| 1 | \(F^2=10q+1\) |
| 3 | \(F=4q+1\) |
| 7 | \(F=2q+1\) |
| 11 | \(F=4q+3\) |
| 15 | \(\mathrm{odd\_part}(n+1)\) |

Hidden oracle: also not verified, \(M=32\), 81 failures.
Relation: `agreed_negative`.

## What this is not

- Not Collatz. Do not train \(3n+1\).
- Not a neural proof. The nets propose a shortlist of \(M\); the
  extractor and verifier produce the certificate.
- Not evidence that energy ranking identifies the working modulus.
  Top \(M\) was 4 on all three maps, including the control.
- Not a reason to enlarge the feature bank. The control's failure is
  the result. Adding \(M=256\) after seeing the symbolic suite would
  leak the catalog.
- `trivial_plus_one` and `easy_half_collatz` were not part of this
  neural trio. The symbolic suite already extracted Level 5 coverings
  there (`bridge_suite_20260909_224156`).

## Lemma heads (separate protocol)

Energy ranking is not the shortlist here. Affine heads propose integer
\((M,r,k,A,B)\) on the locked bank; odd-part fill is leftover only.
Preregistration:
`tiny_tao_results/bridge_preregistration_neural_lemmas.json`.
Run: `tiny_tao_results/bridge_neural_lemmas_20260910_013006`.
Score: `tiny_tao_results/bridge_preregistration_neural_lemmas.scored.json`.

| map | chosen \(M\) | Level 5 | uncovered | oracle |
|---|---:|---|---|---|
| `prospective_seven_and_three_mod_32` | **32** | yes | none | `covering_found` |
| `medium_five_on_1_mod_16` | 16 | **no** | \(\{1\}\) | `discoverer_failed_oracle_verified` |
| `control_five_on_1_mod_4` | 4 | no | \(\{1\}\) | `agreed_negative` |

P1 held without putting 32 in an energy top-8. On residue 1 the
certificate is \(\mathrm{odd\_part}(7n+1)\) (verifier worst-case
\(28q+1\)); residue 17 is \(\mathrm{odd\_part}(3n+1)\) (worst-case
\(24q+13\)).

P2 missed. The energy+symbolic extractor had \(F^2=10q+1\) on medium
residue 1. Adam+round did not land on that lemma. Leftover odd-part
cannot cover an expanding \(5n+1\) class. Do not splice algebraic
`_try_affine` back in after seeing the miss.

## Exact affine leftover (15.5, separate protocol)

Same heads and bank. Misses get exact `_try_affine`, then odd-part.
Preregistration:
`tiny_tao_results/bridge_preregistration_neural_lemmas_exact_fill.json`.
Run: `tiny_tao_results/bridge_neural_lemmas_exact_fill_20260910_014058`.
Score: `tiny_tao_results/bridge_preregistration_neural_lemmas_exact_fill.scored.json`.
15.4 P2 remains a miss.

| map | chosen \(M\) | Level 5 | leftover that heads missed | oracle |
|---|---:|---|---|---|
| `prospective_seven_and_three_mod_32` | **32** | yes | \(r=1\) \(F=28q+1\), \(r=17\) \(F=24q+13\) | `covering_found` |
| `medium_five_on_1_mod_16` | **16** | yes | \(r=1\) \(F^2=10q+1\) | smaller than affine 32 |
| `hard_five_on_1_mod_16` | **16** | yes | \(r=1\) \(F^2=10q+1\), \(r=9\) \(F=12q+7\) | discoverer covered; oracle did not |
| `control_five_on_1_mod_4` | 32 incomplete | **no** | still uncovered \(\{5,13,21,25,29\}\) | `agreed_negative` |

Rounded heads fit the \(n+1\) majority. Expanding classes still need
exact two-point leftover.

## Locked-bank symbolic control (15.6)

No neural stage. Exact extractor on the same locked bank, \(k_{\max}=4\).
Preregistration:
`tiny_tao_results/bridge_preregistration_locked_bank_symbolic.json`.
Run: `tiny_tao_results/bridge_locked_bank_symbolic_20260910_102936`.
Score: `tiny_tao_results/bridge_preregistration_locked_bank_symbolic.scored.json`.

| map | chosen \(M\) | Level 5 | note |
|---|---:|---|---|
| thicker prospective | **32** | yes | \(r=1\) \(F=28q+1\) |
| medium | **16** | yes | \(r=1\) \(F^2=10q+1\) |
| hard | **16** | yes | \(r=9\) \(F=12q+7\) |
| control | 32 incomplete | **no** | uncovered \(\{5,13,21,25,29\}\) |
| thin prospective | **32** | yes | \(M=16\) uncovered \(\{1\}\); closes the energy miss |

Same certificates as 15.5. Lemma heads are not load-bearing on this
bank.

## Covering outside the bank (15.7)

Map `prospective_seven_on_1_mod_64`: `7n+1` on `n\equiv 1\pmod{64}`.
Hand analysis: covering at \(M=64\) with \(F=56q+1\). Locked bank
stops at 32. Same 15.6 extractor. Oracle after discovery.
Preregistration:
`tiny_tao_results/bridge_preregistration_seven_on_1_mod_64.json`.
Run: `tiny_tao_results/bridge_seven_on_1_mod_64_20260910_123215`.
Score: `tiny_tao_results/bridge_preregistration_seven_on_1_mod_64.scored.json`.

| map | chosen \(M\) | Level 5 | oracle |
|---|---:|---|---|
| `prospective_seven_on_1_mod_64` | 32, uncovered \(\{1\}\) | **no** | verified at 64; `discoverer_failed_oracle_verified` |
| medium | **16** | yes | smaller covering |
| control | 32 incomplete | **no** | `agreed_negative` |

Two different failures: control has no small covering; this map has a
covering the bank cannot name. Do not add 64.

## Stop

On the locked bank, exact affine + odd-part + Falsifier already writes
the certificate. Energy ranking missed \(M=32\). Adam+round missed
expanding classes. A covering at \(M=64\) is refused, not a reason to
grow the bank. Do not add 64. Do not stack more neural lemma machinery
here. Do not raise `TOP_K`. Do not start Collatz. Do not retune from
oracle JSON. Do not loosen exactness.
