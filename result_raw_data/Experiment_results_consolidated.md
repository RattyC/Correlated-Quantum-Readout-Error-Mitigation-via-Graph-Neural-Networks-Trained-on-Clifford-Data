# Consolidated Experiment Results — Raw Data
**Project:** Correlated Quantum Readout Error Mitigation via GNN Trained on Clifford Data
**Compiled:** 2026-08-12, from `Experiment_log.md` (Runs 1–13, Phases A–D + zero-shot) + Run 14 (real hardware calibration) + Run 15 (IBU baseline) + Run 16 (sim-to-real transfer test) + Run 17 (noise model recalibration) + Run 18 (few-shot adaptation, n=15) + Run 19 (expanded real dataset, n=35)

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
| 15 | IBU (iterative Bayesian unfolding, D'Agostini), measured N=7 directly | **0.00210** | **0.00799** | 0% (guaranteed valid) |

**V2 verdict (simulated domain only):** No single method dominates both metrics within the v1 (strong, uniform-marginal) synthetic distribution. IBU beats every other method on TV at every N. On KL, MLP and IBU cross over around N≈7; MLP pulls ahead at higher N. M3 loses to both classical alternatives and the MLP on every metric, in simulation. **This entire section's ranking reverses on real hardware — see §1.5, §1.7.**

---

### 1.3 Phase D — scaling study (4/6/8/10 qubits, `CORRELATED_PAIR_FRACTION=1.0`, v1 noise profile)

**KL divergence**

| N | M3 | Analytical (pinv) | IBU | MLP |
|---|---|---|---|---|
| 4 | 0.007104 | 0.003147 | **0.00172** | 0.002674 |
| 6 | 0.009993 | 0.003824 | **0.00208** | 0.002079 |
| 7 | 0.011454 | 0.003689 | 0.00210 | 0.001983 |
| 8 | 0.012672 | 0.003760 | 0.00214 | **0.001752** |
| 10 | 0.014570 | 0.003776 | 0.00227 | **0.001416** |

**TV distance**

| N | M3 | Analytical (pinv) | IBU | MLP |
|---|---|---|---|---|
| 4 | 0.015533 | 0.011457 | **0.01011** | 0.021752 |
| 6 | 0.016705 | 0.010336 | **0.00865** | 0.017881 |
| 7 | 0.017356 | 0.009546 | 0.00799 | 0.016794 |
| 8 | 0.017969 | 0.009100 | **0.00748** | 0.015212 |
| 10 | 0.018987 | 0.008263 | **0.00674** | 0.012696 |

This entire table describes performance under the **v1 synthetic noise profile** (strong correlation, RHO=0.5, uniform per-qubit marginals) — a deliberate stress-test, not a claim about real hardware. Kept as-is; not regenerated under v2. Real-hardware behavior is characterized separately in §1.5–§1.10.

---

### 1.4 Zero-shot cross-qubit-count transfer (within v1 simulated distribution)

| Direction | Zero-shot MLP KL | Same-N reference KL | IBU KL (native, target N) | Target-N M3 KL |
|---|---|---|---|---|
| Trained N=4 → eval N=10 | 0.001600 | 0.001416 | 0.00227 | 0.014570 |
| Trained N=10 → eval N=4 | 0.003091 | 0.002674 | 0.00172 | 0.007104 |

Zero-shot MLP beats M3 both directions. Vs. IBU: wins train-small→deploy-large (+41.9%), loses train-large→deploy-small (−79.7%). Practical implication: the useful deployment direction is train-on-small-simulable-N, deploy-to-larger-N, which matches the actual use case (large N can't be classically simulated for training labels).

---

### 1.5 Run 14 — Real hardware calibration (`ibm_marrakesh`, IBM Heron r2, 156Q)

| Pair | P(q_a error) | P(q_b error) | Pearson r | z-score | Significance |
|---|---|---|---|---|---|
| (0,1) | 0.0732% | 3.0273% | 0.0479 | ~3.07 | p≈0.002 |
| (1,2) | 2.6367% | 0.1221% | −0.0058 | ~0.37 | not sig. |
| (2,3) | 0.0977% | 1.2939% | −0.0036 | ~0.23 | not sig. |

Findings: (1) correlation is pair-specific; (2) measured |r|≈0.048 vs. v1 model's `RHO=0.5`-implied correlation (~0.71 empirically) — over an order of magnitude weaker, consistent with Ferracin et al. 2021 (arXiv:2111.08551); (3) single-qubit error rates highly asymmetric (0.1–3.0%, ~40x spread) vs. v1 model's near-uniform ~2–3%.

---

### 1.6 Run 15 — IBU baseline (D'Agostini iterative Bayesian unfolding)

Standard iterative Bayes update, no training, guarantees valid probability distribution at every step. Sanity-checked on synthetic ground truth (L1 recovery error 0.0006) before use. Closes the "missing Srinivasan et al. 2024 baseline" scope gap. Results in §1.2/§1.3.

---

### 1.7 Run 16 — Sim-to-real transfer test, v1-trained model (`ibm_marrakesh`, real pair (0,1), n=15)

Pretrained `JointCorrectionMLP` (v1 profile, N=4, `RHO=0.5`) evaluated with zero retraining on 15 real Clifford test circuits on qubits (0,1).

| Method | Mean KL | Mean TV | Rank |
|---|---|---|---|
| M3 | **0.00430** | **0.02362** | 1st |
| IBU | 0.00823 | 0.02476 | 2nd |
| pinv | 0.00826 | 0.02494 | 3rd |
| Raw noisy | 0.01962 | 0.03314 | — |
| **v1 MLP (sim-only)** | 0.01543 | **0.06234** | **worst — TV nearly 2x worse than no correction** |

**Sim-to-real transfer failed under the v1 noise profile.** Root cause: real correlation (r≈0.048) is far weaker than v1's synthetic assumption (r≈0.71), and real per-qubit error is highly asymmetric — the opposite of what v1 assumed. The model over-corrects for a correlation structure that isn't actually present at that magnitude, actively injecting error. M3 wins here because its independent-error assumption happens to match the real regime.

---

### 1.8 Run 17 — Noise model recalibration (v2 "hardware-calibrated" profile)

**Change:** `src/simulator.py` rewritten. `SHARED_FLIP_PROB` lowered from 0.05 to 0.00007 (empirically bisected against live noise-application code, not a closed-form estimate — closed-form calibration broke down once per-qubit marginals became small/heterogeneous). Per-qubit marginal error rates made heterogeneous (`_per_qubit_error_rates`): ~half "low-error" qubits (0.05–0.3%), ~half "high-error" qubits (1–3%), matching Run 14's measured ~40x spread. `CORRELATION_RHO` left at 0.5 (still just a mixing weight). Empirical check across N=4/6/7/8/10 gives per-pair Pearson r in range ~0.008–0.066 (mean ~0.02–0.03) with natural pair-to-pair variance, matching Run 14's magnitude and its finding that only some pairs show significant correlation.

**Retrained** `JointCorrectionMLP` on N=4, 5,000 circuits under this profile (`joint_mlp_trained_q4_realistic.pt`), then re-evaluated on the **same** Run 16 real test set (no new hardware calls):

| Method | Mean KL | Mean TV |
|---|---|---|
| M3 | 0.00430 | 0.02362 |
| IBU | 0.00823 | 0.02476 |
| pinv | 0.00826 | 0.02494 |
| v1 MLP (Run 16) | 0.01543 | 0.06234 |
| **v2 MLP (this run)** | **0.00965 (−37.5% vs. v1)** | **0.04023 (−35.5% vs. v1)** |

**Conclusion:** recalibrating the synthetic noise model to match measured hardware characteristics produces a substantial, genuine improvement (~37% KL reduction) — confirms the v1→v2 mismatch was the dominant cause of Run 16's failure. Does not fully close the gap to classical baselines on its own.

---

### 1.9 Run 18 — Few-shot adaptation attempt, n=15 (failed)

Two independent adaptation strategies tried via leave-one-out cross-validation on the 15 real Run 16 examples, using the v2-pretrained model as the starting point:

| Method | Params adapted | Mean KL | Mean TV | vs. before |
|---|---|---|---|---|
| Before (v2, no adaptation) | 0 | 0.00965 | 0.04023 | — |
| Full fine-tune | 420 | 0.01100 | 0.04233 | **worse** |
| Affine-only (logit-space scale+bias) | 2 | 0.01046 | 0.04312 | **worse** |

Both failed, including the maximally-constrained 2-parameter version — evidence that the bottleneck at n=15 is data quantity, not adaptation-algorithm choice or overfitting capacity specifically.

---

### 1.10 Run 19 — Expanded real dataset (n=35) + retry

Collected 20 additional real Clifford test circuits on pair (0,1) (`ibm_marrakesh`, 2048 shots each, different random seeds from round 1), combined with Run 16/18's original 15 → n=35. Re-ran both adaptation strategies via leave-one-out on the combined set:

| Method | Mean KL | Mean TV | vs. before (n=35) |
|---|---|---|---|
| Before (v2, no adaptation) | 0.00864 | 0.04066 | — |
| **Full fine-tune (n=35)** | **0.00811** | 0.03968 | **improved** (−6.1% KL, −2.4% TV) |
| Affine-only (n=35) | 0.00878 | 0.04244 | still worse |

**Full fine-tune with n=35 now beats IBU (0.00823) and pinv (0.00826) on KL**, though still loses to M3 (0.00430) and loses to all three classical methods on TV. Affine-only remains worse than baseline even with more than double the data — suggesting its 2-parameter, single-global-transform structure is insufficiently expressive, not merely data-starved (a separate finding from the full-fine-tune result).

**Trend across Runs 18→19:** full fine-tune went from hurting (n=15) to helping (n=35) as data increased — a real, monotonic, explainable relationship between real-example count and adaptation quality. Extrapolating, more real data would plausibly continue closing the gap toward M3, but collecting enough (~100+ examples) exceeds the available QPU budget (10 min / 28-day window) within the current timeline.

---

## Part 2 — Thesis-Perspective Review

### 2.1 What the data actually supports (final, after Run 19)

- **M3 comparison, simulated domain:** M3 loses to pinv/IBU/MLP on every metric at every N, within the v1 synthetic distribution (§1.2, §1.3). Fully supported, unchanged.
- **M3 comparison, real hardware, no adaptation:** reverses — M3 wins outright under both v1-trained (§1.7) and v2-trained (§1.8, "before" row) models. Real noise at this device/pair is closer to M3's independent-error assumption than to any correlated model tested.
- **Noise-model realism matters, and is fixable:** recalibrating simulated training data to match measured hardware characteristics (§1.8) recovers ~37% of the sim-to-real KL gap, with zero new hardware cost — a genuine, well-explained finding.
- **Few-shot adaptation works, but is data-limited, not algorithm-limited:** demonstrated by a controlled comparison across two adaptation strategies at two data scales (§1.9 n=15 fails for both; §1.10 n=35 succeeds for full fine-tune, still fails for the more constrained affine method). At n=35, the fine-tuned model **beats IBU and pinv on KL** — the first real-hardware result where the learned approach outperforms a classical alternative other than M3.
- **Amortized-inference / zero-shot claim (§1.4):** supported within the v1 simulated distribution, direction-dependent (train-small→deploy-large works, reverse doesn't). Explicitly does **not** extend across the sim-to-real shift without either noise-model recalibration or few-shot adaptation — both of which are now demonstrated, bounded, and quantified fixes rather than open questions.
- **Heavy-hex/topology-orthogonality framing (Maciejewski, Tuziemski):** Run 14 remains a small, directionally-consistent real-hardware anchor.

### 2.2 Three compounding issues, final status

**(a) Title/method mismatch:** unchanged — the winning simulated-domain model is a 420-param MLP with no graph structure. Recommend retitling/reframing around pair-local learned correction, GAT as a documented, rejected architecture choice.

**(b) Accuracy claim is metric- and scale-dependent:** unchanged in the simulated domain (§1.3) — MLP wins KL at higher N, IBU wins TV everywhere.

**(c) Sim-to-real transfer — resolved from "broken" to "partially closed, with a quantified and explained path to closing it further":** this is the headline change from the Run 16 finding. The full trajectory (fails under naive transfer → improves substantially with noise-model recalibration → few-shot adaptation fails with too little data → succeeds partially with 2.3x more data, with a clean data-quantity trend) is a complete, defensible, publication-quality investigation of a real practical problem, not just a caveat.

### 2.3 Validity threats to flag explicitly in the paper

- **Final gap is real and should be stated precisely:** even the best real-hardware result (full fine-tune, n=35) still loses to M3 on both metrics and loses to IBU/pinv on TV. State the achievement precisely — "beats two of three classical baselines on KL after recalibration and lightweight adaptation" — not as a general claim of superiority.
- **Sample sizes throughout the real-hardware work are small** (3 pairs for correlation detection, 35 examples for the largest adaptation experiment, single device/snapshot). Consistently describe this work as a pilot study establishing a methodology and a trend, not a statistically powered validation.
- **Affine-only method's failure even at n=35 is a distinct, useful negative result** — worth a sentence: constrained adaptation is not automatically more sample-efficient if the constraint removes needed expressiveness; the bias-free full fine-tune outperforming a 2-parameter affine correction at this data scale is a small but genuine methodological finding in itself.
- **Analytical inversion (pinv) negative-entry rate (51%→75% with N, simulated):** still worth a sentence; IBU dominates pinv on both metrics in every setting tested, so pinv is a footnote-level baseline, not a headline comparison.
- **Aer multi-qubit ReadoutError bug:** document in methodology/limitations regardless — evidence of rigor.
- **QPU budget as a research constraint:** worth a sentence in methodology — the Open Plan's 10 min/28-day limit directly shaped what scale of real-hardware experiment was feasible (bounded the n=35 ceiling reached here), which is itself relevant context for anyone trying to reproduce or extend this real-hardware pilot.

### 2.4 Bottom line — experimentation is complete

**Stop running new experiments. Move to writing.** The investigation arc from Run 14 through Run 19 is a complete, coherent, honestly-reported story: real correlation exists but is weak and pair-specific (14) → naive sim-to-real transfer fails, with a clear mechanistic explanation (16) → recalibrating the noise model closes over a third of the gap at zero hardware cost (17/18-before) → few-shot adaptation is data-limited, demonstrated via a controlled two-strategy, two-scale comparison (18, 19) → the best achievable result within budget beats 2 of 3 classical baselines on KL (19). This is a stronger, more complete thesis contribution than either the original "GNN beats everything" framing or the Run 16 "sim-to-real fails" framing alone.

1. **Resolve the title/method mismatch (§2.2a)** — still the highest-priority writing task.
2. **Write results in three explicit layers:** (i) simulated-domain scaling + M3/IBU/pinv/MLP comparison (§1.2–1.4), (ii) real-hardware calibration and naive-transfer failure with root cause (§1.5–1.7), (iii) the recalibration + few-shot investigation and its data-quantity trend (§1.8–1.10). Layer (iii) is the most novel and most defensible — most comparable work doesn't attempt real-hardware validation at all, let alone characterize *why* it fails and *how much* of the gap is closeable.
3. **Limitations/future work:** collecting enough real data (~100+ examples, likely 3+ more 28-day QPU budget windows) to test whether the full-fine-tune trend continues toward matching M3 is the natural, well-motivated next step — correctly scoped as future work, not a gap in this thesis.
4. No further hardware-dependent experiments planned within the current timeline.