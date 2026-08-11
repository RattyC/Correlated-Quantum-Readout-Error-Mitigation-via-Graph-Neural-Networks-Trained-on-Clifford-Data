# Experiment Log

Running log of infra changes and experiment results. Not the thesis/README — that
gets written after correlated noise model + M3 baseline are in place. This file
is the raw record so nothing gets lost between now and then.

---

## Infra

### Docker
- **Added:** `Dockerfile` (repo root) — base image `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`,
  installs `requirements.txt` + `torch_geometric`, `git`. Built and verified on workstation
  (Ryzen 9 9900X, RTX 5060, CUDA 12.8) — `docker build -t qrem-gnn:latest .` succeeds,
  `torch.cuda.is_available() == True` inside container.
- Run pattern: `docker run --rm -it --gpus all -v "$(pwd)":/app -w /app qrem-gnn:latest bash`

### Code changes

| File | Change | Why |
|---|---|---|
| `main.py` | Added `argparse` (`--num-qubits`, `--num-circuits`, `--depth`, `--shots`, `--output`). Output filename now auto-suffixed with qubit count (`quantum_dataset_q{N}.json`). | Was hardcoded to `NUM_QUBITS = 28`, single fixed output file — blocked running multiple qubit-count datasets without overwriting. |
| `prepare_gnn_dataset.py` | (1) Replaced O(n^2) pure-Python nested loop for edge index with vectorized `torch.meshgrid` + `@lru_cache` per qubit count. (2) Replaced one-hot qubit ID (`torch.eye(num_qubits)`, dim = `num_qubits`) with fixed-dim `sinusoidal_positional_encoding()` (`D_MODEL = 8`, module constant). (3) Added `argparse` (`--input`). | One-hot tied `in_channels` to a fixed qubit count — blocked zero-shot transfer across qubit counts, which is required for the planned scaling study. Nested loop was fine at 10k x 7 but not going forward. |
| `train_gnn.py` | (1) `QuantumReadoutMitigationGAT.__init__` now takes `in_channels` directly (derived from `train_dataset[0].x.shape[1]`) instead of `num_qubits=10` default that computed `in_channels = 1 + num_qubits`. (2) Added `argparse` (`--input`, `--epochs`, `--activation clamp|tanh`). (3) Fixed indentation bug from original where `model.eval()` block sat outside the epoch loop and `average_test_loss` was computed but never used — `_evaluate()` now called properly every epoch and printed. | Old `in_channels` calc broke the moment `prepare_gnn_dataset.py` stopped using one-hot. `--activation` flag added specifically for the clamp-vs-tanh ablation below. |

Confirmed already correct, no change needed:
- `src/simulator.py` — `AerSimulator(method="stabilizer")` was already set at module level (not `statevector`/`automatic`). Verified before running 28-qubit generation.

---

## Runs

### Run 1 — 7-qubit baseline (clamp), first Docker smoke test
- Command: `python main.py` (pre-argparse version) -> `prepare_gnn_dataset.py` -> `train_gnn.py`
- Config: 7 qubits, depth 30, 10,000 circuits, 4096 shots, 40 epochs, `clamp` activation
- Result: Train MSE 0.001489 / Test MSE 0.001513 (epoch 40). Plateaued by epoch 5 (0.001490/0.001514) and stayed flat.

### Run 2 — 28-qubit generation + train, full pipeline w/ argparse
- Command:
  ```
  python main.py --num-qubits 28 --num-circuits 10000 --depth 30 --shots 4096
  python prepare_gnn_dataset.py --input output_data/quantum_dataset_q28.json
  python train_gnn.py --input output_data/quantum_dataset_q28.json --epochs 40
  ```
- Result: Train MSE 0.001526 / Test MSE 0.001536 (epoch 40). Plateaued by epoch 5 — nearly
  identical to the 7-qubit run despite 4x the qubit count and completely different data.
  Conclusion: plateau is not a data-scale artifact.

### Run 3 — Ablation: clamp vs tanh output activation (7-qubit dataset, controls held fixed)
- Command:
  ```
  python train_gnn.py --input output_data/quantum_large_dataset.json --activation clamp
  python train_gnn.py --input output_data/quantum_large_dataset.json --activation tanh
  ```
- Result:

  | | clamp | tanh |
  |---|---|---|
  | Test MSE (epoch 40) | 0.001509 | 0.039411 |
  | Saturated-qubit example (noisy +0.9624, ideal +1.0000) | mitigated +0.9373 (moves toward ideal) | mitigated +0.7129 (moves away from ideal) |
  | Plateau onset | ~epoch 5 | ~epoch 5 |

- Conclusion: clamp-gradient-dead-zone hypothesis is falsified. tanh has no dead zone
  and still plateaus at the same epoch, at ~26x worse MSE — it compresses already-near-saturated
  inputs toward 0, which is the wrong direction. clamp is confirmed the better choice; keep it.
- New working hypothesis (untested): both `ideal_outputs` and `noisy_outputs` in `main.py`
  are generated from finite `shots` (4096) via `AerSimulator` — the *ideal* target itself carries
  shot-sampling noise whenever the final state isn't an exact Z-eigenstate. This could be an
  irreducible noise floor in the training labels that no model capacity fixes.

---

## Open items (priority order)

1. Shot-count sweep (next, not yet run) — rerun 7-qubit gen at `--shots 16384` / `32768`,
   retrain, check if MSE plateau drops proportionally. If yes, confirms label shot-noise floor,
   not a model/architecture problem. If no, go back to model capacity / learning rate.
2. Correlated noise model — `src/simulator.py`'s `_build_readout_noise_model()` is currently
   per-qubit uncorrelated (`add_readout_error` applied independently per qubit). This contradicts
   the project's "correlated readout error" framing and blocks any real result claim. Needs a
   joint/correlated `ReadoutError` across qubit pairs before further scaling work is worth doing.
3. M3 baseline (Nation et al. 2021, `mthree` package) — not yet implemented. Required before
   any amortized-inference claim.
4. Scaling study proper (4/6/8/10+ qubits, zero-shot cross-qubit-count eval) — infra is ready
   (`main.py --num-qubits`, sinusoidal PE) but not yet run as an actual study; only ran 7 and 28
   as isolated smoke tests so far.
5. README + thesis writeup — deliberately deferred until items 2-3 above are done.

### Run 4 — Shot-count sweep (7-qubit dataset, clamp, controls held fixed)
- Command:
  ```
  python main.py --num-qubits 7 --num-circuits 10000 --depth 30 --shots 16384 --output quantum_dataset_q7_s16384.json
  python prepare_gnn_dataset.py --input output_data/quantum_dataset_q7_s16384.json
  python train_gnn.py --input output_data/quantum_dataset_q7_s16384.json --epochs 40 --activation clamp

  python main.py --num-qubits 7 --num-circuits 10000 --depth 30 --shots 32768 --output quantum_dataset_q7_s32768.json
  python prepare_gnn_dataset.py --input output_data/quantum_dataset_q7_s32768.json
  python train_gnn.py --input output_data/quantum_dataset_q7_s32768.json --epochs 40 --activation clamp
  ```
- Result:

  | shots | Test MSE (epoch 40) | drop from baseline (4096) |
  |---|---|---|
  | 4,096 (baseline, Run 1) | 0.001513 | - |
  | 16,384 (4x) | 0.001350 | -10.8% |
  | 32,768 (8x) | 0.001328 | -12.2% |

  Pure shot-noise-variance scaling (~1/N) predicts -75% at 4x shots and -87.5% at 8x shots.
  Actual drop is an order of magnitude smaller than predicted, and shows diminishing returns
  (16384->32768 only drops a further 1.6% despite doubling shots again).

- Conclusion: label shot-noise floor hypothesis is **mostly falsified** — it explains a small
  slice of the plateau (~0.00016 of the ~0.0015 MSE) but not the dominant ~0.0013 component,
  which stays flat regardless of shots, qubit count (Run 2), or activation (Run 3).
- Observation carried into next hypothesis: plateau onset is epoch 5 in every run regardless of
  what was varied (dataset, qubit count, shots, activation). This is a consistent optimization
  signature, not a data-noise signature — points at learning rate / model capacity next.
- Qualitative note: mitigated correction magnitude looks roughly constant (~0.02-0.03) across
  most samples regardless of which circuit produced them — consistent with the model having
  converged to something close to a constant/near-linear correction rather than a per-sample
  learned function. Worth checking directly once the LR/capacity experiments below are run.

### Ruled out so far
1. ~~Clamp gradient dead zone~~ — falsified by Run 3 (tanh is worse, same plateau epoch).
2. ~~Data scale (qubit count)~~ — falsified by Run 2 (28q plateau == 7q plateau).
3. ~~Label shot noise~~ — mostly falsified by Run 4 (predicted vs actual drop off by ~10x).

## Open items (priority order) — updated

1. **Learning rate / model capacity sweep** (next) — add `--lr` CLI flag to `train_gnn.py`
   (default 1e-3, matching current). Rerun 7-qubit baseline at `--lr 1e-4 --epochs 200` to check
   whether plateau is Adam converging too fast at the current LR vs. a genuine capacity ceiling.
   Follow up with a hidden-dim increase (16 -> 64) if lowering LR doesn't move the floor.
2. Correlated noise model — unchanged, still open (see above).
3. M3 baseline — unchanged, still open.
4. Scaling study proper — unchanged, still open.
5. README + thesis writeup — deliberately deferred until items 2-3 above are done.

### Run 5 — Learning-rate sweep (7-qubit baseline dataset, clamp, controls held fixed)
- Command: `python train_gnn.py --input output_data/quantum_large_dataset.json --lr 0.0001 --epochs 200 --log-every 10`
- Result: Test MSE converges to 0.001509 by ~epoch 30-40, then stays flat through epoch 200.
  Essentially identical to the lr=1e-3/epochs=40 baseline floor (0.001509-0.001513) — only
  difference is convergence speed (slower with lower LR, as expected), not the floor reached.
- Conclusion: learning-rate-too-aggressive hypothesis is **falsified**. 10x lower LR and 5x more
  epochs land at the same floor. The plateau is not an optimizer-speed artifact.

### Ruled out so far — updated
1. ~~Clamp gradient dead zone~~ — falsified by Run 3.
2. ~~Data scale (qubit count)~~ — falsified by Run 2.
3. ~~Label shot noise~~ — mostly falsified by Run 4.
4. ~~Learning rate too high~~ — falsified by Run 5.

Remaining leading hypothesis: **model capacity ceiling** — 2,267 params / hidden_dim=16 / 2 GAT
layers (kept shallow specifically to limit over-smoothing on the complete graph, per README) may
simply not have enough capacity to fit anything beyond the current floor. Next test: widen
hidden_dim (16 -> 64) with lr/epochs held at original baseline, same 7-qubit dataset.

## Open items (priority order) — updated

1. **Model capacity sweep** (next) — add `--hidden-dim` CLI flag to `train_gnn.py` (default 16).
   Rerun 7-qubit baseline at `--hidden-dim 64` (lr=1e-3, epochs=40, clamp). If MSE floor drops
   meaningfully, capacity was the bottleneck — consider trade-off against over-smoothing risk
   before permanently widening. If floor stays ~0.0015, the architecture may be at/near a genuine
   Bayes-optimal floor for this synthetic uncorrelated-noise task, which would itself be a
   noteworthy (if less exciting) finding to write up.
2. Correlated noise model — unchanged, still open.
3. M3 baseline — unchanged, still open.
4. Scaling study proper — unchanged, still open.
5. README + thesis writeup — deliberately deferred until items 2-3 above are done.

### Run 6 — Model-capacity sweep (7-qubit baseline dataset, clamp, lr=1e-3, controls held fixed)
- Command: `python train_gnn.py --input output_data/quantum_large_dataset.json --hidden-dim 64 --epochs 40`
- Result: 19,713 params (8.7x baseline's 2,267). Test MSE converges to 0.001508-0.001513 by
  epoch 5, with minor noise mid-training (spike to 0.001577 at epoch 20) that settles back to the
  same floor by epoch 40. No meaningful improvement over the 16-dim baseline.
- Conclusion: model-capacity-ceiling hypothesis is **falsified**. 8.7x more parameters lands at
  the identical floor.

### Ruled out so far — updated
1. ~~Clamp gradient dead zone~~ — falsified by Run 3.
2. ~~Data scale (qubit count)~~ — falsified by Run 2.
3. ~~Label shot noise~~ — mostly falsified by Run 4.
4. ~~Learning rate too high~~ — falsified by Run 5.
5. ~~Model capacity~~ — falsified by Run 6.

Five independent variables changed, all landing on the same ~0.0015 floor. Working theory now:
the floor is not a training/architecture defect at all — it may simply equal what a **trivial
per-qubit-only regressor** (no graph, no cross-qubit information) could achieve, because the
current noise model (`src/simulator.py`) is per-qubit uncorrelated. If there is no cross-qubit
correlation signal in the data, the GAT's attention mechanism has nothing extra to exploit versus
a plain per-node function of its own noisy ⟨Z⟩ — the architecture's core premise would simply be
untested by this dataset, not disproven by it.

## Open items (priority order) — updated

1. **Trivial baseline test** (next) — add `train_baseline_mlp.py`: a plain per-qubit MLP with no
   `edge_index` input (ignores all other qubits), same node features (noisy Z + PE), same
   residual+clamp output head, same training config. Train on the same 7-qubit dataset. If its
   floor MSE lands at ~0.0015 (same as the GAT), it proves the GAT is gaining nothing from the
   graph/attention on this data — conclusively pointing at the uncorrelated noise model as the
   real blocker, not model design. If the baseline is meaningfully worse, the GAT is extracting
   some signal and the floor has a different explanation.
2. **Correlated noise model** — likely to be promoted to priority #1 depending on the baseline
   result above. `_build_readout_noise_model()` needs a joint/correlated `ReadoutError` across
   qubit pairs before any claim tied to the project's "correlated readout error" framing is valid.
3. M3 baseline — unchanged, still open.
4. Scaling study proper — unchanged, still open.
5. README + thesis writeup — deliberately deferred until items 2-3 above are done.

### Run 7 — Trivial baseline: per-qubit MLP, no graph (7-qubit baseline dataset, clamp, controls held fixed)
- Command: `python train_baseline_mlp.py --input output_data/quantum_large_dataset.json --epochs 40`
- Model: `PerQubitMLPBaseline` — 449 params, same 2-hidden-layer depth as the GAT, `edge_index`
  never touched, each qubit sees only its own `[noisy Z, positional encoding]` features.
- Result: Test MSE 0.000199-0.000200 (epoch 40) vs. GAT baseline's 0.001509-0.001513 — the
  no-graph baseline is **~7.5x better**, not merely equal as hypothesized.
- Sample predictions: baseline hits saturated qubits essentially exactly (noisy -0.9321 ->
  mitigated -1.0000, ideal -1.0000; noisy +0.9624 -> mitigated +1.0000, ideal +1.0000) — GAT on
  the same sample only reaches -0.9573/+0.9373.

- **Conclusion — revised, stronger than the pre-registered hypothesis.** The original theory was
  "no correlation signal exists in this uncorrelated-noise dataset, so the GAT and a trivial
  baseline should land at the same floor." The actual result is that the GAT performs *worse*
  than the trivial baseline, not merely equal to it. This points at the all-to-all graph +
  2-layer attention actively degrading each qubit's own (already informative) signal via
  over-smoothing — exactly the risk the README's own design-choices table flagged
  ("all-to-all edges... accelerates over-smoothing") but had not empirically confirmed until now.
- Practical implication: none of Runs 2-6 (data scale, activation, shots, LR, capacity) could
  have found this, because they all kept the complete-graph GAT architecture fixed and varied
  everything else. The bottleneck was the architecture's interaction with this dataset the whole
  time.

### Ruled out so far — final for this diagnostic phase
1. ~~Clamp gradient dead zone~~ — falsified by Run 3.
2. ~~Data scale (qubit count)~~ — falsified by Run 2.
3. ~~Label shot noise~~ — mostly falsified by Run 4.
4. ~~Learning rate too high~~ — falsified by Run 5.
5. ~~Model capacity~~ — falsified by Run 6.
6. **Confirmed (not falsified): GAT/complete-graph architecture underperforms a trivial per-qubit
   baseline on the current (uncorrelated-noise) dataset — Run 7.**

## Open items (priority order) — updated

1. **Correlated noise model** — now the clear top priority, for two combined reasons: (a) required
   for the project's "correlated readout error" framing to be valid at all, and (b) it is the only
   way to test whether the GAT's attention mechanism has *any* real signal to exploit — Run 7
   shows it currently has none, and the complete graph is actively harmful in that regime.
2. **Re-run GAT vs. MLP-baseline comparison once correlated noise exists.** If GAT still loses to
   the trivial baseline even with real cross-qubit correlation in the data, the architecture
   itself (all-to-all edges, 2-layer depth) needs revisiting — e.g. sparser/learned edge topology,
   gating to let the model ignore irrelevant neighbors, or fewer effective hops.
3. M3 baseline — unchanged, still open; the per-qubit MLP result above is a useful interim
   sanity-check baseline but is not a substitute for M3.
4. Scaling study proper — unchanged, still open; should not be run until item 1-2 are resolved,
   since scaling a broken architecture just reproduces the same failure at more qubit counts.
5. README + thesis writeup — deliberately deferred until items 1-2 above are done. This finding
   (GAT underperforming a trivial baseline under uncorrelated noise) is itself a legitimate,
   citable result for the eventual paper's ablation section — do not discard it once the
   correlated-noise experiments are run; keep both results for contrast.

---

## Correlated noise model (design + first results)

### Design
- **Added:** `src/simulator.py` rewritten to build a correlated-pair readout noise model
  instead of independent per-qubit noise. Mechanism: a fraction of qubits
  (`CORRELATED_PAIR_FRACTION=0.5`) are randomly paired (NOT tied to coupling map, per
  Maciejewski et al. 2021 finding that real hardware correlations don't follow physical
  layout). Each pair shares a single "crosstalk event" (`SHARED_FLIP_PROB=0.05`) that flips
  both qubits together — genuinely correlated, not decomposable into independent per-qubit
  noise. Mixed with the old independent model at weight `CORRELATION_RHO=0.5`. Unpaired
  qubits keep the original independent single-qubit error. Pair topology is cached/seeded
  per qubit count (`NOISE_MODEL_SEED`) so it's fixed across all circuits in one dataset run,
  like a real device's (unknown) fixed correlation structure.
- **Added:** `get_correlated_pairs(num_qubits)` — ground-truth metadata, now saved into every
  record via `main.py` (`correlated_pairs` field) for later analysis of whether GAT attention
  actually learns the true correlation structure.
- Smoke-tested at 4 qubits / 200 circuits first (no errors, `[[2, 1]]` pair as expected) before
  the full run.

### Run 8 — GAT vs. MLP baseline on correlated-noise dataset (7 qubits, 10,000 circuits)
- Ground truth: `correlated_pairs = [[1, 6], [5, 3]]` (4 of 7 qubits actually correlated; 0, 2, 4
  unpaired/independent).
- Command:
  ```
  python main.py --num-qubits 7 --num-circuits 10000 --depth 30 --shots 4096 --output quantum_dataset_q7_correlated.json
  python prepare_gnn_dataset.py --input output_data/quantum_dataset_q7_correlated.json
  python train_gnn.py --input output_data/quantum_dataset_q7_correlated.json --epochs 40 --activation clamp
  python train_baseline_mlp.py --input output_data/quantum_dataset_q7_correlated.json --epochs 40
  ```
- Result:

  | | uncorrelated (Run 1 / Run 7) | correlated (Run 8) |
  |---|---|---|
  | GAT Test MSE | 0.001509-0.001513 | 0.000784 |
  | MLP baseline Test MSE | 0.000199-0.000200 | 0.000199 |
  | GAT vs baseline gap | GAT 7.5x worse | GAT 3.9x worse |

- Conclusion: **GAT improved ~48% once real cross-qubit correlation exists in the data, while
  the MLP baseline stayed essentially flat** (as expected — its marginal per-qubit error
  statistics didn't change, only the joint pattern did, which it cannot see). This is the first
  positive evidence that the GAT's attention mechanism is extracting real correlation signal,
  not just noise. However, GAT still underperforms the trivial baseline by ~3.9x. Working
  explanation: only 2 of the 21 possible qubit pairs in the 7-qubit complete graph carry real
  signal (4 of 7 qubits are in a correlated pair at all); the other ~19 edges still force
  attention to mix in uninformative neighbors every layer, and over-smoothing on the dense graph
  is still winning against the (now partially real) signal.

## Open items (priority order) — updated

1. **Correlation-density sweep** (next) — rerun the same 7-qubit generation at
   `CORRELATED_PAIR_FRACTION` closer to 1.0 (most/all qubits paired) as an upper-bound test: if
   GAT closes the gap to or beats the MLP baseline when correlation is dense, it confirms the
   architecture works but is signal-starved at 50% pairing. If it still loses even at high
   density, the complete-graph/over-smoothing problem is the dominant issue regardless of how
   much real signal exists, and the graph topology itself (not just the data) needs to change —
   e.g. sparse/learned edges restricted to actually-correlated pairs instead of all-to-all.
2. M3 baseline — unchanged, still open.
3. Scaling study proper — unchanged, still open; still blocked on resolving item 1.
4. README + thesis writeup — deliberately deferred. Runs 1-8 (5 falsified hypotheses + 2
   confirmed findings: GAT loses to trivial baseline under no correlation, GAT improves but still
   loses under partial correlation) are themselves a coherent ablation narrative worth keeping
   intact for the eventual paper, regardless of how the density sweep turns out.

### Run 9 — Correlation-density upper bound (CORRELATED_PAIR_FRACTION 0.5 -> 1.0, 7-qubit dataset)
- Ground truth: `correlated_pairs = [[1, 6], [5, 3], [4, 0]]` (6 of 7 qubits correlated; only
  qubit 2 unpaired — near-maximum density possible at odd qubit count).
- Command:
  ```
  python main.py --num-qubits 7 --num-circuits 10000 --depth 30 --shots 4096 --output quantum_dataset_q7_correlated_full.json
  python prepare_gnn_dataset.py --input output_data/quantum_dataset_q7_correlated_full.json
  python train_gnn.py --input output_data/quantum_dataset_q7_correlated_full.json --epochs 40 --activation clamp
  python train_baseline_mlp.py --input output_data/quantum_dataset_q7_correlated_full.json --epochs 40
  ```
- Result:

  | correlation density | GAT Test MSE | MLP baseline Test MSE | GAT vs baseline |
  |---|---|---|---|
  | 0% (Run 7) | 0.001509 | 0.000199 | 7.5x worse |
  | ~57%, 4/7 q (Run 8) | 0.000784 | 0.000199 | 3.9x worse |
  | ~86%, 6/7 q (Run 9) | 0.000378 | 0.000195 | 1.9x worse |

- Conclusion: **clean monotonic trend — GAT gap to baseline roughly halves each time correlation
  density increases, but GAT still loses even at near-maximum density (6 of 7 qubits paired).**
  This rules out "not enough correlation signal" as the remaining explanation. The bottleneck is
  now conclusively the **complete-graph topology itself**: even with almost every qubit
  genuinely correlated with another, attention over 21 possible pairs (only 3 of which are real)
  still dilutes/over-smooths the signal enough to lose to a model that ignores neighbors
  entirely.

## Open items (priority order) — updated

1. **Sparse/ground-truth-topology graph test** (next, decisive) — modify
   `prepare_gnn_dataset.py`'s edge construction to build edges ONLY between the qubit pairs
   listed in each record's `correlated_pairs` (instead of all-to-all), and retrain the same GAT
   on the same correlated dataset. If GAT now beats the MLP baseline decisively, it proves the
   architecture is sound and all-to-all edges were the actual defect the whole time (matches the
   over-smoothing risk the README already flagged in the design-choices table). This result also
   directly informs the real scaling study later: production noise-model correlation structure
   is unknown in advance on real hardware, so an all-to-all fallback vs. a learned/sparse edge
   predictor becomes a real design decision, not just a training nicety.
2. M3 baseline — unchanged, still open.
3. Scaling study proper — unchanged, still open; blocked on resolving item 1.
4. README + thesis writeup — deliberately deferred. The full arc (Runs 1-9: 5 falsified
   hyperparameter/data hypotheses, then a clean correlation-density sweep isolating the
   complete-graph topology as the actual bottleneck) is a strong, coherent ablation story for the
   eventual paper — keep all of it.