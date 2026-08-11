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