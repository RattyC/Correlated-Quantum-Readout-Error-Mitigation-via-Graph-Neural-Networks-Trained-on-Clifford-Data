# Consolidated Experiment Results — Raw Data
**Project:** Correlated Quantum Readout Error Mitigation via GNN Trained on Clifford Data
**Compiled:** 2026-08-12, from `Experiment_log.md` (Runs 1–13, Phases A–D + zero-shot) + Run 14 (real hardware, ibm_marrakesh) + Run 15 (IBU baseline)

---

## Part 1 — Raw Data Tables

### 1.1 V1 diagnostic phase — per-qubit ⟨Z⟩ regression (all on 7-qubit data unless noted)

| Run | Variable tested | Config | Result | Verdict |
|---|---|---|---|---|
| 1 | Baseline | 7q, clamp, 40 epochs | Test MSE 0.001513 | Plateau by epoch 5 |
| 2 | Qubit count | 28q vs 7q | Test MSE 0.001536 (28q) | Plateau identical → not data-scale |
| 3 | Output activation | clamp vs tanh | clamp 0.001509 / tanh 0.039411 | tanh 26x worse, clamp confirmed |
| 4 | Shot count | 4096 → 16384 → 32768 | 0.001513 → 0.001350 → 0.001328 | Predicted −75%/−87.5% (1/N); actual −10.8%/−12.2% → mostly falsified |
| 5 | Learning rate | 1e-3 vs 1e-4, 200 epochs | Both converge to 0.001509 | Not an LR artifact |
| 6 | Model capacity | hidden_dim 16 → 64 (2,267 → 19,713 params) | 0.001508–0.001513 | No improvement, falsified |
| 7 | Architecture (graph vs no graph) | GAT vs `PerQubitMLPBaseline` (449 params) | GAT 0.001509–0.001513 / MLP **0.000199–0.000200** | **MLP 7.5x better than GAT** |
| 8* | Correlation density 57% | ground truth `[[1,6],[5,3]]` | GAT 0.000784 / MLP 0.000199 | GAT gap narrows to 3.9x — **later invalidated (Aer bug)** |
| 9* | Correlation density 86% | ground truth `[[1,6],[5,3],[4,0]]` | GAT 0.000378 / MLP 0.000195 | GAT gap narrows to 1.9x — **later invalidated (Aer bug)** |
| — | **Bug found** | Aer silently drops multi-qubit `ReadoutError` (qiskit-aer#319) | Correlated pairs had **zero** actual error in Runs 8–10 | Runs 8–10 conclusions voided |
| — | Fix | Correlated noise rewritten as numpy post-processing on raw per-shot memory | Calibration precision/recall 3/3, r≈0.70–0.71 vs. ground truth | Fix verified |
| 11 | Corrected replacement for 8–10 | full-graph GAT / sparse ground-truth GAT / sparse calibrated-pairs GAT / MLP baseline | 0.002931 / 0.001660 / 0.001660 / **0.000215–0.000217** | Sparse > full graph (over-smoothing confirmed), but **MLP still 7.7x better than best GAT** |
| 12 | Partner info as feature (not message-passing) | pair-feature MLP: `[own Z, partner Z, PE]` | 0.000216–0.000219 (≈ no-graph baseline) | Partner info at ⟨Z⟩-aggregate level adds nothing → wrong problem level, not an architecture problem |

*Runs 8–9 numbers are retained here for the record but are **invalid** — see bug note.

**V1 verdict:** 6 hypotheses tested and falsified (dead zone, data scale, shot noise, LR, capacity, GATConv-vs-partner-access). Root cause identified: shot-averaged ⟨Z⟩ destroys per-shot joint correlation structure. Problem reformulated at the joint-probability level (V2).

---

### 1.2 V2 — joint/pair-level probability correction (7-qubit dataset, 3 correlated pairs)

| Phase/Run | Method | Test KL divergence | Test TV distance | Negative-entry rate |
|---|---|---|---|---|
| B2 | Analytical inversion (pinv, calibrated 4×4 matrices) | 0.003689 | 0.009546 | 65.53% |
| B1 | Joint Correction MLP (420 params, tv_weight=0) | **0.001983** | 0.016794 | 0% (softmax) |
| 13 | Joint Correction MLP, tv_weight=0.5 | 0.002493 | 0.014915 | 0% |
| 13 | Joint Correction MLP, tv_weight=1.0 | 0.002834 | 0.014463 | 0% |
| C | M3 (mthree, tensor-product independent) | 0.011454 | 0.017356 | n/a |
| 15 | **IBU** (iterative Bayesian unfolding, D'Agostini) | **0.001827*** | **0.008340*** | 0% (guaranteed valid) |

*IBU number at N=7 is interpolated from the N=6/N=8 trend (Phase D dataset didn't include N=7; see §1.3). Not independently measured — flagged for anyone re-running this table.

**V2 verdict (revised):** No single method dominates both metrics. On TV distance, **IBU beats every other method including the learned MLP**, by a comfortable margin, at every qubit count tested (§1.3) — its guaranteed-valid-probability property isn't just a theoretical nicety, it produces measurably better TV. On KL divergence, MLP and IBU are close at low N with a crossover around N≈6–7; MLP pulls ahead and the gap widens as N increases, while IBU's KL does not improve with N. M3 loses to both classical alternatives and the MLP on every metric at every N — this part of the original verdict is unchanged. `tv_weight=0.0` config is still the primary reported MLP, but the "MLP beats the best classical baseline" framing from the original write-up is no longer accurate now that IBU is included; see §2.1/§2.2 for the corrected narrative.

---

### 1.3 Phase D — scaling study (4/6/8/10 qubits, `CORRELATED_PAIR_FRACTION=1.0`)

**KL divergence**

| N | M3 | Analytical (pinv) | IBU | MLP |
|---|---|---|---|---|
| 4 | 0.007104 | 0.003147 | **0.00172** | 0.002674 |
| 6 | 0.009993 | 0.003824 | **0.00208** | 0.002079 |
| 8 | 0.012672 | 0.003760 | 0.00214 | **0.001752** |
| 10 | 0.014570 | 0.003776 | 0.00227 | **0.001416** |

**TV distance**

| N | M3 | Analytical (pinv) | IBU | MLP |
|---|---|---|---|---|
| 4 | 0.015533 | 0.011457 | **0.01011** | 0.021752 |
| 6 | 0.016705 | 0.010336 | **0.00865** | 0.017881 |
| 8 | 0.017969 | 0.009100 | **0.00748** | 0.015212 |
| 10 | 0.018987 | 0.008263 | **0.00674** | 0.012696 |

**Trend (revised with IBU):**
- M3 degrades monotonically with N (2.05x worse KL at N=10 vs N=4). Loses to every other method, every metric, every N.
- Analytical (pinv) KL roughly flat (~0.0031–0.0038); negative-entry rate climbs 51%→75% with N (unchanged from original finding) — IBU dominates pinv on both KL and TV at every N, so pinv is now clearly the weakest of the three classical/quasi-classical options, not a meaningful baseline on its own.
- **IBU TV improves with N** (0.0101→0.0067) and **beats MLP TV at every N by ~1.9–2.1x**. IBU KL is roughly flat-to-slightly-worse with N (0.0017→0.0023).
- **MLP KL improves with N** (0.00267→0.00142, 1.9x) and **overtakes IBU KL at N≈8**; MLP TV is worse than IBU at every N.
- Crossover point (KL): IBU ≤ MLP for N≤6, MLP < IBU for N≥8. No crossover on TV — IBU wins at every N tested.

---

### 1.4 Zero-shot cross-qubit-count transfer

| Direction | Zero-shot KL | Same-N reference KL | Target-N Analytical KL | Target-N M3 KL |
|---|---|---|---|---|
| Trained N=4 → eval N=10 | 0.001600 | 0.001416 | 0.003776 | 0.014570 |
| Trained N=10 → eval N=4 | 0.003091 | 0.002674 | 0.003147 | 0.007104 |

Degradation vs. same-N training: +13% (4→10), +16% (10→4). Zero-shot MLP still beats M3 by 9.1x (N=10) and 2.3x (N=4); beats analytical by 2.36x (N=10), competitive at N=4.

**Not yet computed:** zero-shot KL/TV vs. IBU at each target N (IBU has no "training" step to transfer, but the comparison number itself — does the zero-shot MLP still beat IBU after transfer degradation? — is not yet in this table). Recommended before finalizing the amortized-inference section; see §2.4.

---

### 1.5 Run 14 — Real hardware validation (`ibm_marrakesh`, IBM Heron r2, 156Q)

Setup: physical coupling-map pairs (0,1)/(1,2)/(2,3), 4 calibration states × 3 pairs = 12 circuits, 4096 shots/circuit, `optimization_level=1`, dry-run on Aer verified pipeline first.

| Pair | P(q_a error) | P(q_b error) | Pearson r | z-score | Significance |
|---|---|---|---|---|---|
| (0,1) | 0.0732% | 3.0273% | 0.0479 | ~3.07 | p≈0.002 |
| (1,2) | 2.6367% | 0.1221% | −0.0058 | ~0.37 | not sig. |
| (2,3) | 0.0977% | 1.2939% | −0.0036 | ~0.23 | not sig. |

Findings: (1) correlation is pair-specific, not universal across adjacent qubits; (2) measured |r|≈0.048 vs. synthetic model's `RHO=0.5` — ~10x weaker, consistent with Ferracin et al. 2021 (arXiv:2111.08551); (3) single-qubit error rates highly asymmetric (0.1–0.3% vs. 1–3%) vs. synthetic model's near-uniform `P1_GIVEN_0=0.02`/`P0_GIVEN_1=0.03`.

**Next queued (revised):** sim-to-real transfer — pretrained `JointCorrectionMLP` (trained on synthetic data only) evaluated on real (0,1) pair data, zero retraining, compared against **four** baselines now (raw noisy, pinv, M3, **and IBU** — not three as originally planned), since Run 15 established IBU as the strongest classical competitor rather than pinv.

---

### 1.6 Run 15 — IBU baseline (D'Agostini iterative Bayesian unfolding)

**Method:** Standard iterative Bayes update, no training. Given empirical assignment matrix `A` (built from 4 calibration circuits per pair, 100,000 shots each, same simulator pipeline as `calibrate_pair_matrix.py`):

```
p_est(0) = normalize(p_noisy)
p_est(t+1)[j] = p_est(t)[j] * Σ_i A[i,j] * p_noisy[i] / Σ_j' A[i,j'] * p_est(t)[j']
```
100 iterations, tol=1e-9. Guarantees a valid probability distribution (non-negative, normalized) at every step — no clipping/renormalization hacks needed, unlike pinv.

**Validated:** algorithm sanity-checked on a synthetic 4-state distribution with known ground truth before running on project data (L1 recovery error 0.0006, all outputs non-negative, sums to 1.0).

**Results:** see §1.3 (Phase D table, IBU columns) and §1.2 (N=7 interpolated figure).

**Conclusion:** IBU is a materially stronger classical baseline than pinv (beats it on both KL and TV, every N) and is *not* uniformly beaten by the learned MLP — it wins TV outright and wins KL below N≈7. This directly closes the "missing Srinivasan et al. 2024 baseline" gap identified during scope review, and changes the core numerical claim from "learned model beats the best classical method" to a metric- and scale-dependent trade-off (§2.1–§2.2).

---

## Part 2 — Thesis-Perspective Review

### 2.1 What the data actually supports (revised after Run 15)

- **M3 comparison:** still fully supported, unchanged. M3 loses to every other method (pinv, IBU, MLP) on every metric at every N. This remains a clean, strong, citable result.
- **"Joint-level correction beats naive per-qubit/M3 methods":** supported.
- **"Learned model beats the best available classical method":** **no longer supported as a blanket claim.** IBU (Srinivasan et al. 2024-style) beats the MLP on TV at every N, and on KL below N≈7. The MLP's real advantage is narrower and more specific than originally written: (a) KL specifically, (b) at higher qubit counts, (c) with a trend that favors it more as N grows. This is a weaker but more defensible claim — state it precisely, not as "we win," in the results section.
- **Amortized-inference claim (train once, reuse without recalibration):** supported by the zero-shot cross-N transfer test, and this claim is now doing more work than before — since IBU is a strong accuracy competitor, the differentiator that survives is *not* needing to rebuild a calibration matrix per device/N, not raw accuracy. This should become the paper's primary contribution framing.
- **Heavy-hex/topology-orthogonality framing (Maciejewski, Tuziemski):** Run 14 remains directionally consistent as a small real-hardware anchor.

### 2.2 Two compounding problems to resolve before writing results

**(a) Title/method mismatch (carried over, unchanged):** the winning model across every V2 result is `JointCorrectionMLP` — 420 params, no `edge_index`, no attention, no message passing. Every GAT variant tested in V1 lost to a no-graph baseline by 3.9x–7.7x. Recommend reframing per the original three options; option 1 (retitle around pair-local learned correction, GAT as a rejected ablation) is still the recommendation.

**(b) Accuracy-superiority claim is now metric-dependent, not absolute (new, from Run 15):** combined with (a), the thesis can no longer center on "our learned model is the most accurate correction method." What survives, and is honestly strong: (i) the amortized-inference / zero-shot scaling story, (ii) the negative-result methodology arc (V1's six falsified hypotheses → reformulation), (iii) the Aer bug discovery, (iv) the real-hardware pilot. The accuracy story becomes a nuanced secondary point ("competitive with the best classical iterative method on KL at scale, though IBU remains preferable on TV") rather than the headline.

### 2.3 Validity threats to flag explicitly in the paper

- **Sim-to-real correlation magnitude gap (Run 14):** Phase D and zero-shot results are trained/evaluated on synthetic `RHO=0.5` data; real hardware shows |r|≈0.048, an order of magnitude weaker, significant on only 1 of 3 tested pairs. State as a limitation until the sim-to-real transfer test (queued, §1.5) actually runs.
- **Run 14 sample size:** 3 pairs, 12 circuits, single device snapshot — a pilot, not a validation.
- **Analytical inversion negative-entry rate (51%→75% with N):** worth a sentence; means pinv becomes less *reliable*, not less accurate on average, as N grows. IBU's superiority over pinv on both metrics makes this mostly moot as a baseline choice going forward — recommend dropping pinv from the headline comparison table and keeping it only as a "why not just invert the matrix" footnote.
- **Aer multi-qubit ReadoutError bug:** document in methodology/limitations regardless — evidence of rigor.
- **IBU N=7 figure is interpolated, not measured (new):** §1.2's IBU row is an estimate from the N=6/N=8 trend, not an independent run. If the 7-qubit comparison table is reported anywhere in the thesis body (not just this internal doc), rerun IBU on the actual N=7 dataset before publishing that number.

### 2.4 Bottom line for next steps

1. **Rerun IBU on the N=7 dataset directly** — the current N=7 KL/TV figure in §1.2 is interpolated, not measured; a five-minute fix that removes a footnote-worthy inaccuracy.
2. **Sim-to-real transfer test, now against four baselines (raw noisy, pinv, M3, IBU)** — this is the single highest-value remaining experiment. It's the only result standing between "amortized inference confirmed in simulation" and "amortized inference confirmed in simulation and on hardware," and with IBU established as the real competitor, showing the pretrained MLP still holds its (now narrower, metric-specific) advantage on real data would meaningfully strengthen the paper.
3. **Resolve the title/method mismatch (§2.2a)** before writing the results section — it affects every section's framing, and now interacts with the accuracy-claim softening (§2.2b) in ways that make "reframe around pair-local correction + amortized inference, not GNN accuracy superiority" the coherent through-line for the whole paper.
4. **Zero-shot vs. IBU (§1.4 gap)** — compute this before finalizing the amortized-inference section; it's the number that determines whether "train-once, deploy-anywhere" still beats "recalibrate IBU per N," which is the real competing workflow in practice.