# Consolidated Experiment Results — Raw Data
**Project:** Correlated Quantum Readout Error Mitigation via GNN Trained on Clifford Data
**Compiled:** 2026-08-12, from `Experiment_log.md` (Runs 1–13, Phases A–D + zero-shot) + Run 14 (real hardware, ibm_marrakesh)

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
| C | **M3** (mthree, tensor-product independent) | 0.011454 | 0.017356 | n/a |

**V2 verdict:** Joint Correction MLP beats M3 on KL by 5.8x and beats analytical inversion by 1.86x. TV distance shows a genuine Pareto trade-off (KL-optimal ≠ TV-optimal solution); `tv_weight=0.0` kept as primary reported config per plan.

---

### 1.3 Phase D — scaling study (4/6/8/10 qubits, `CORRELATED_PAIR_FRACTION=1.0`)

| N | M3 KL | Analytical KL | MLP KL | M3 TV | Analytical TV | MLP TV | Analytical neg-entry rate |
|---|---|---|---|---|---|---|---|
| 4 | 0.007104 | 0.003147 | 0.002674 | 0.015533 | 0.011457 | 0.021752 | 51.0% |
| 6 | 0.009993 | 0.003824 | 0.002079 | 0.016705 | 0.010336 | 0.017881 | 61.1% |
| 7 | 0.011454 | 0.003689 | 0.001983 | 0.017356 | 0.009546 | 0.016794 | 65.5% |
| 8 | 0.012672 | 0.003760 | 0.001752 | 0.017969 | 0.009100 | 0.015212 | 69.6% |
| 10 | 0.014570 | 0.003776 | 0.001416 | 0.018987 | 0.008263 | 0.012696 | 75.2% |

**Trend:** M3 degrades monotonically with N (2.05x worse at N=10 vs N=4). Analytical KL flat but negative-entry rate climbs 51%→75%. MLP KL *improves* with N (1.9x better). M3-vs-MLP gap widens 2.7x (N=4) → 10.3x (N=10).

---

### 1.4 Zero-shot cross-qubit-count transfer

| Direction | Zero-shot KL | Same-N reference KL | Target-N Analytical KL | Target-N M3 KL |
|---|---|---|---|---|
| Trained N=4 → eval N=10 | 0.001600 | 0.001416 | 0.003776 | 0.014570 |
| Trained N=10 → eval N=4 | 0.003091 | 0.002674 | 0.003147 | 0.007104 |

Degradation vs. same-N training: +13% (4→10), +16% (10→4). Zero-shot model still beats M3 by 9.1x (at N=10) and 2.3x (at N=4); beats analytical by 2.36x (N=10), competitive at N=4.

---

### 1.5 Run 14 — Real hardware validation (`ibm_marrakesh`, IBM Heron r2, 156Q)

Setup: physical coupling-map pairs (0,1)/(1,2)/(2,3), 4 calibration states × 3 pairs = 12 circuits, 4096 shots/circuit, `optimization_level=1`, dry-run on Aer verified pipeline first.

| Pair | P(q_a error) | P(q_b error) | Pearson r | z-score | Significance |
|---|---|---|---|---|---|
| (0,1) | 0.0732% | 3.0273% | 0.0479 | ~3.07 | p≈0.002 |
| (1,2) | 2.6367% | 0.1221% | −0.0058 | ~0.37 | not sig. |
| (2,3) | 0.0977% | 1.2939% | −0.0036 | ~0.23 | not sig. |

Findings: (1) correlation is pair-specific, not universal across adjacent qubits; (2) measured |r|≈0.048 vs. synthetic model's `RHO=0.5` — ~10x weaker, consistent with Ferracin et al. 2021 (arXiv:2111.08551); (3) single-qubit error rates highly asymmetric (0.1–0.3% vs. 1–3%) vs. synthetic model's near-uniform `P1_GIVEN_0=0.02`/`P0_GIVEN_1=0.03`.

Next queued (not yet run): sim-to-real transfer — pretrained `JointCorrectionMLP` (trained on synthetic data only) evaluated on real (0,1) pair data, zero retraining.

---

## Part 2 — Thesis-Perspective Review

### 2.1 What the data actually supports

- **Core claim (correlated readout ≠ independent per-qubit error, and joint-level correction beats M3):** well-supported. Phase A–D + M3 comparison is a clean, reproducible three-way ranking (M3 < analytical < learned) with a monotonic N-scaling trend in the MLP's favor. This is your strongest, most citable result.
- **Amortized-inference claim (train once, reuse without recalibration):** supported by the zero-shot cross-N transfer test — real evidence, not just an architectural argument. This is the second-strongest result and directly differentiates you from M3's design.
- **Heavy-hex/topology-orthogonality framing (Maciejewski, Tuziemski):** Run 14, though tiny, is directionally consistent — correlation on real hardware did *not* appear uniformly across physically adjacent pairs. Useful as a real-hardware anchor for a claim you'd otherwise only cite from literature.

### 2.2 Problem you need to resolve before writing this up as a "GNN" thesis

**The winning model across every V2 result (B1, Run 13, Phase D, zero-shot) is `JointCorrectionMLP` — 420 params, no `edge_index`, no attention, no message passing.** Every GAT variant tested in V1 (full-graph, sparse ground-truth, sparse calibrated) lost to a no-graph baseline, by 3.9x–7.7x depending on the run. Your thesis title is "...via Graph Neural Networks Trained on Clifford Data," but the artifact that actually produces your headline numbers is not a graph neural network. This is not a wording nitpick — it's a mismatch between the claimed method and the method that works, and a committee will catch it. You have three real options, not a formatting fix:

1. Retitle/reframe around what actually worked: pair-local learned correction, GAT as a rejected/ablated architecture choice, with the V1 negative-result arc as a methodology section explaining *why* graph message-passing was abandoned. This is honest and the ablation story (6 falsified hypotheses, then the architecture-vs-problem-level pivot) is genuinely strong writing material.
2. Reintroduce a real GNN into the V2 (joint-probability) formulation and show it beats the 420-param MLP — untested. Given V1's consistent over-smoothing result on this same qubit topology, I would not expect this to succeed without deliberate architecture changes (e.g., restricting message passing to calibrated pairs only, which is close to what "sparse calibrated" already was and still lost).
3. Keep "GNN" in the title only if you scope it explicitly as "graph-structured pair-correction" and are precise that the deployed/best-performing instantiation is the degenerate no-edge case — this is a weaker framing and reviewers may ask why you didn't just call it an MLP from the start.

I'd recommend (1). Don't force a GNN result you don't have; the negative-result arc is publishable and honest, and matches what your own log already argues in the "Note for whoever reads this log" entry after Run 12.

### 2.3 Validity threats to flag explicitly in the paper

- **Sim-to-real correlation magnitude gap (Run 14):** your entire scaling study (Phase D) and zero-shot result are trained/evaluated on synthetic data with `RHO=0.5`. Real hardware shows |r|≈0.048 — an order of magnitude weaker, and only significant on 1 of 3 tested pairs. Until the sim-to-real transfer test (queued) actually runs and the pretrained MLP is shown to work at this much lower real-world correlation strength, the amortized-inference claim is validated only in simulation. State this as a limitation, not a footnote.
- **Run 14 sample size:** 3 pairs, 12 circuits, single device snapshot. This is a pilot, not a validation — don't let the paper's language imply more than that.
- **Analytical inversion negative-entry rate (51%→75% with N):** worth a sentence — it means analytical inversion becomes *less reliable*, not less accurate on average, as N grows. Keep KL/TV/negative-rate as three separate reported numbers rather than collapsing to one metric, as your own Run 13 conclusion already argues.
- **Aer multi-qubit ReadoutError bug:** document this in the methodology/limitations section regardless — it's evidence of rigor (you caught and fixed a real simulator bug that silently invalidated 3 runs), and reviewers who know Qiskit-Aer's issue tracker will trust the paper more for disclosing it than if you omit the false starts.

### 2.4 Bottom line for next steps

Before writing the results section, resolve the title/method mismatch (2.2) — it affects every other section's framing. After that, the highest-value remaining experiment is the sim-to-real transfer test already queued in Run 14, since it's the only thing standing between "amortized inference confirmed in simulation" and "amortized inference confirmed in simulation and on hardware."