# Experiment Log

Running log of infra changes and experiment results. Not the thesis/README — that
gets written after correlated noise model + M3 baseline are in place. This file
is the raw record so nothing gets lost between now and then.

---

## Infra

### Docker
- **Added:** `Dockerfile` (repo root) — base image `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`,
  installs `requirements.txt` + `torch_geometric`, `git`. Built and verified on workstation
  (Ryzen 9 9900X, RTX 3080, CUDA 12.8) — `docker build -t qrem-gnn:latest .` succeeds,
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