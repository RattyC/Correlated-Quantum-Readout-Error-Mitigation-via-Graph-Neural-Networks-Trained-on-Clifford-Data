# Correlated Quantum Readout Error Mitigation via Graph Neural Networks Trained on Clifford Data

> Status: **v2 — joint-distribution correction, validated in simulation and on real IBM hardware.** The originally-planned GAT was tested, ablated, and **lost to a plain 420-parameter MLP in every controlled comparison** — this is documented and defended below, not hidden. The scaling study (4/6/8/10 qubits), zero-shot cross-N generalization, and M3 / iterative-Bayesian-unfolding benchmarks referenced as "not yet done" in v1.0 are now complete. A real-hardware sim-to-real validation pipeline (IBM Quantum) is included and its results — including a naive-transfer failure, its root cause, and a partial, quantified fix — are reported honestly. See [Roadmap](#roadmap--known-limitations) for what's still open.

Learned, pair-local correction for **correlated** (not just independent, per-qubit) readout
error on NISQ quantum hardware, trained entirely on classically-simulable Clifford circuit
data. No quantum hardware access is required to reproduce the core (simulated-domain) results;
a separate, optional pipeline lets you validate against real IBM Quantum hardware yourself.

**Full experimental record, every run (including the ones that failed or were later
invalidated), and an honest thesis-perspective review of what the results do and don't
support:** [`result_raw_data/Experiment_log.md`](result_raw_data/Experiment_log.md) and
[`result_raw_data/Experiment_results_consolidated.md`](result_raw_data/Experiment_results_consolidated.md).

## Table of contents

- [Correlated Quantum Readout Error Mitigation via Graph Neural Networks Trained on Clifford Data](#correlated-quantum-readout-error-mitigation-via-graph-neural-networks-trained-on-clifford-data)
  - [Table of contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Methodology](#methodology)
    - [v1: per-qubit ⟨Z⟩ regression (historical — superseded)](#v1-per-qubit-z-regression-historical--superseded)
    - [v2: joint pair-distribution correction (current method)](#v2-joint-pair-distribution-correction-current-method)
    - [Classical baselines](#classical-baselines)
    - [Real hardware validation](#real-hardware-validation)
  - [Installation](#installation)
    - [Docker (recommended)](#docker-recommended)
    - [Without Docker](#without-docker)
  - [Usage](#usage)
  - [Project structure](#project-structure)
  - [Results summary](#results-summary)
  - [Roadmap / known limitations](#roadmap--known-limitations)
  - [References](#references)
  - [License](#license)

## Motivation

Readout error on NISQ hardware is rarely independent per qubit — crosstalk correlates the
error across qubits, which classical bit-flip correction cannot capture. Matrix-based
mitigation (naive inversion of the 2ⁿ×2ⁿ assignment matrix) is not the right straw-man
baseline here: matrix-free methods such as M3 (Nation et al., 2021) already solve the *scaling*
problem for **independent** per-qubit error. What M3 does **not** solve — by construction, its
`single_qubit_cals` architecture is a tensor product with no cross-qubit term — is *correlated*
error. Iterative Bayesian unfolding (Srinivasan et al., 2024) goes further and also solves
naive inversion's negative-probability problem without any learning at all, which turned out
to be a materially stronger baseline than initially assumed (see [Results
summary](#results-summary)). The case for a learned model rests on two narrower, tested
properties: it can exploit *pairwise correlation structure* that M3 cannot represent, and once
trained it can be deployed across qubit counts without per-device recalibration (amortized
inference) — both are measured, not assumed, in this repo.

## Methodology

Two formulations were built and tested, in order. The second one supersedes the first for all
current results; the first is kept because the ablation that motivated abandoning it is itself
a real finding.

### v1: per-qubit ⟨Z⟩ regression (historical — superseded)

**Files:** `prepare_gnn_dataset.py`, `train_gnn.py`, `train_baseline_mlp.py`, `train_pair_feature_mlp.py`

Each circuit becomes one graph: nodes = qubits, node features = `[noisy ⟨Z⟩, sinusoidal
positional encoding]`, edges = fully-connected, target = ideal per-qubit ⟨Z⟩. A 2-layer GAT
with a residual connection and `clamp(-1, 1)` output was trained to predict the correction.

```
Input (x): [noisy ⟨Z⟩ | positional encoding]
        │
        ▼
GATConv(in, 16, heads=4)  →  ELU
        │
        ▼
GATConv(64, 16, heads=1)  →  ELU
        │
        ▼
Linear(16 → 1)                          # Δ correction
        │
        ▼
mitigated = clamp(noisy_z + Δ, −1, 1)
```

**This failed, systematically, and the failure is the actual finding.** Six hypotheses were
tested and falsified in turn (clamp dead zone, dataset scale, shot noise, learning rate, model
capacity, GATConv-vs-direct-partner-access) — the GAT never beat a 449-parameter no-graph MLP
baseline (`train_baseline_mlp.py`), by 3.9–7.7x depending on the run. The root cause: **shot-
averaged ⟨Z⟩ destroys the per-shot joint correlation structure the model needed to exploit.**
Averaging over shots before training throws away exactly the signal that makes readout error
correlated in the first place. Full ablation log: `result_raw_data/Experiment_log.md`, Runs
1–13.

### v2: joint pair-distribution correction (current method)

**Files:** `prepare_joint_dataset.py`, `train_joint_mlp.py`

Reformulated at the joint-probability level instead of the expectation-value level. For each
correlated qubit pair, extract the **joint 4-outcome distribution** `[P(00), P(01), P(10),
P(11)]` per circuit — this preserves the per-shot correlation structure that v1 destroyed —
and train a model to correct the noisy joint distribution toward the ideal one.

```
Input (x): noisy joint 4-vector [P(00), P(01), P(10), P(11)]
        │
        ▼
Linear(4 → 16) → ReLU → Linear(16 → 16) → ReLU → Linear(16 → 4)
        │
        ▼
logits = fc_out(hidden) + log(x.clamp_min(1e-6))     # residual in logit space
        │
        ▼
corrected = softmax(logits)                           # always a valid distribution
```

**420 parameters. No graph, no edges, no attention.** This is deliberate, not an oversight —
see [Roadmap](#roadmap--known-limitations) for the direct statement on why this repo is titled
"Graph Neural Networks" when the model that actually works isn't one. Loss is KL divergence
between corrected and ideal distributions, with an optional `--tv-weight` to trade off against
total-variation distance (the two are not simultaneously optimal — see `train_joint_mlp.py
--help` and the results doc).

Zero-shot cross-qubit-count transfer (`eval_zero_shot_transfer.py`) — train at one N, evaluate
at a different N with no retraining — is the amortized-inference claim this whole approach is
actually staked on; it holds in the train-small→deploy-large direction and does not in reverse
(see results doc §1.4).

### Classical baselines

Three baselines, of increasing strength, are implemented for honest comparison — **not** just
naive matrix inversion:

- **`eval_analytical_baseline.py`** — naive `pinv(A)` on the empirical assignment matrix.
  Frequently returns negative "probabilities" (51–75% of test cases, worsening with qubit
  count). Weakest baseline; kept for completeness, not as the headline comparison.
- **`eval_ibu_baseline.py`** — Iterative Bayesian Unfolding (D'Agostini algorithm), no
  training, guarantees a valid distribution at every step. This is the Srinivasan et al.
  (2024) style baseline and turned out to be materially stronger than `pinv` on every metric —
  it beats the learned MLP on total-variation distance at every qubit count tested.
- **`calibrate_single_qubit_m3.py` / `eval_m3_baseline.py`** — M3 (`mthree`), tensor-product
  independent-qubit calibration (Nation et al., 2021). Cannot represent correlation by
  construction; included as the standard reference method.

### Real hardware validation

**Files:** `setup_ibm_account.py`, `run_real_calibration.py`, `analyze_real_correlation.py`,
`build_sim_to_real_test.py`(`_round2`), `analyze_sim_to_real_test.py`,
`fewshot_finetune_loo.py`, `fewshot_affine_loo.py`, `fewshot_combined_loo.py`

Optional, requires a free [IBM Quantum Platform](https://quantum.cloud.ibm.com/) account. This
pipeline: (1) measures real pairwise correlation on physically-adjacent qubits (selected from
the device's actual coupling map, not a synthetic assignment), (2) tests whether a
purely-simulation-trained model transfers to real noisy data with zero retraining, (3)
diagnoses *why* it does or doesn't, and (4) tests two few-shot adaptation strategies. All of
this is reported honestly, including the parts that didn't work — see [Results
summary](#results-summary).

## Installation

### Docker (recommended)

This is how the project was actually built and how the real-hardware scripts were run.

```bash
git clone <repo-url>
cd Correlated-Quantum-Readout-Error-Mitigation-via-Graph-Neural-Networks-Trained-on-Clifford-Data
docker build -t qrem-gnn:latest .
docker run --gpus all -it --rm -v $(pwd):/app -w /app qrem-gnn:latest bash
```

Drop `--gpus all` if you don't have `nvidia-container-toolkit` set up — the pipeline is
CPU-bound; GPU only meaningfully helps `train_gnn.py` (the ablated v1 GAT, not the method that
actually works). `-v $(pwd):/app` bind-mounts the project directory so every generated dataset,
checkpoint, and result persists on your host, not just inside the (disposable) container.

Verify the environment:

```bash
python3 -c "import torch, qiskit, qiskit_aer; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### Without Docker

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch-geometric` is only needed for `train_gnn.py` (v1 ablation). `mthree` is only needed for
the M3 baseline. `qiskit-ibm-runtime` is only needed for the real-hardware scripts — skip all
three if you only want the core v2 simulated-domain pipeline.

## Usage

```bash
# 1. Generate a Clifford circuit dataset (repeat per qubit count for a scaling study)
python main.py --num-qubits 4 --num-circuits 10000 --output quantum_dataset_q4.json

# 2. Train the joint-correction model (the method that actually works)
python train_joint_mlp.py --input output_data/quantum_dataset_q4.json --epochs 40 --save-model output_data/joint_mlp_q4.pt

# 3. Build the classical baselines for comparison
python calibrate_pair_matrix.py --input output_data/quantum_dataset_q4.json
python eval_analytical_baseline.py --input output_data/quantum_dataset_q4.json
python eval_ibu_baseline.py --input output_data/quantum_dataset_q4.json       # no training needed
python calibrate_single_qubit_m3.py --input output_data/quantum_dataset_q4.json
python eval_m3_baseline.py --input output_data/quantum_dataset_q4.json

# 4. (Optional) cross-N zero-shot generalization
python eval_zero_shot_transfer.py --model output_data/joint_mlp_q4.pt --eval-input output_data/quantum_dataset_q10.json

# 5. (Optional, historical) the ablated v1 GAT, for reproducing the negative result
python prepare_gnn_dataset.py
python train_gnn.py

# 6. (Optional, requires IBM Quantum account) real hardware validation
python setup_ibm_account.py          # one-time auth — edit the token/instance fields first
python run_real_calibration.py       # confirm real correlated pairs + build calibration matrices
python build_sim_to_real_test.py     # submit real test circuits
python analyze_sim_to_real_test.py   # compare raw / pinv / IBU / M3 / MLP against real ideal targets
```

**Real hardware budget note:** IBM's Open (free) plan gives 10 QPU-minutes per rolling 28-day
window. Every real-hardware script here is written to be budget-conscious (small circuit
counts, a dry run on Aer before every real submission, calibration data reused across
experiments rather than re-measured). Check your remaining budget on the IBM Quantum Platform
dashboard before running any of them.

## Project structure

```
.
├── README.md
├── requirements.txt
├── Dockerfile
├── LICENSE
├── main.py                          # parallel Clifford circuit generation + noise simulation
├── src/
│   ├── generator.py                  # generate_random_clifford_circuit
│   ├── simulator.py                  # noise model + execute_circuit_pipeline (v2, hardware-calibrated)
│   ├── simulator_v1_stress_test.py.bak  # original strong/uniform noise profile (scaling study)
│   └── data_exporter.py              # save_dataset_to_json
│
├── prepare_gnn_dataset.py           # v1 (historical): bitstrings -> per-qubit PyG graph Data
├── train_gnn.py                      # v1: QuantumReadoutMitigationGAT — lost to the MLP below
├── train_baseline_mlp.py             # v1: PerQubitMLPBaseline — beat GAT by 3.9-7.7x
├── train_pair_feature_mlp.py         # v1: isolates architecture vs. problem-formulation failure
│
├── prepare_joint_dataset.py         # v2 (current method): bitstrings -> joint 4-vector examples
├── train_joint_mlp.py                # v2: JointCorrectionMLP (420 params) — the method that works
├── eval_zero_shot_transfer.py        # cross-qubit-count generalization, zero retraining
│
├── calibrate_pair_matrix.py         # empirical assignment matrix A per correlated pair
├── eval_analytical_baseline.py      # naive pinv(A) baseline
├── eval_ibu_baseline.py              # iterative Bayesian unfolding baseline (Srinivasan et al. 2024 style)
├── calibrate_single_qubit_m3.py     # per-qubit calibration for M3
├── eval_m3_baseline.py               # M3 (mthree) baseline
├── calibrate_correlations.py        # blind correlation detection from measurement statistics
│
├── setup_ibm_account.py             # IBM Quantum Platform auth (real hardware, optional)
├── verify_auth.py
├── run_real_calibration.py          # real pairwise correlation measurement
├── analyze_real_correlation.py
├── build_sim_to_real_test.py         # real test circuits, round 1 (n=15)
├── build_sim_to_real_test_round2.py  # round 2 (n=20 more, for few-shot experiments)
├── analyze_sim_to_real_test.py       # raw/pinv/IBU/M3/MLP comparison on real data
├── fewshot_finetune_loo.py           # full-model fine-tune, leave-one-out
├── fewshot_affine_loo.py             # constrained 2-param affine adaptation, leave-one-out
├── fewshot_combined_loo.py           # both strategies on the merged round 1+2 real dataset
│
├── output_data/                      # generated datasets, model checkpoints, calibration matrices (gitignored)
└── result_raw_data/
    ├── Experiment_log.md              # full chronological run-by-run log
    └── Experiment_results_consolidated.md   # corrected summary + honest thesis-perspective review
```

## Results summary

Full tables and derivations: `result_raw_data/Experiment_results_consolidated.md`. Headlines:

- **Simulated domain, scaling study (N=4/6/8/10):** M3 loses to every other method at every N.
  IBU beats the learned MLP on total-variation distance at every N; MLP overtakes IBU on KL
  divergence above N≈7 and the gap widens with N. No method dominates both metrics.
- **Zero-shot cross-N transfer:** works training-small→deploying-large (beats IBU by 41.9%),
  fails in reverse (79.7% worse than IBU) — state the validated direction only.
- **Real hardware, naive transfer:** fails outright. The purely-simulation-trained model is the
  *worst* of five methods tested, actively worse than doing nothing (M3 wins by a wide margin).
  Root cause: real correlation (r≈0.048) is an order of magnitude weaker than the synthetic
  training assumption, and real per-qubit error is highly asymmetric where the synthetic model
  assumed uniformity.
- **Real hardware, after fixing the noise model + few-shot adaptation:** recalibrating the
  synthetic training distribution to match measured hardware characteristics recovers ~37% of
  the gap for free (no new hardware data). Few-shot fine-tuning on real examples fails at n=15
  (not enough data — even a 2-parameter-only adaptation fails at this scale) and starts working
  at n=35, where the fine-tuned model **beats both IBU and pinv on KL** (still loses to M3, and
  to all three classical methods on TV). The trend is real, monotonic, and explainable; closing
  it further requires more real data than fits in this project's QPU budget.

## Roadmap / known limitations

- **Title/method mismatch, stated directly:** the model that wins every controlled comparison
  in this repo is a 420-parameter MLP, not a GNN. Every GAT variant tested lost to it. If
  you're citing or building on this work, cite the joint-distribution-correction method, not
  "GNN for readout mitigation."
- **Sim-to-real gap is real and only partially closed.** See Results summary — the best
  real-hardware result still loses to M3 overall. Collecting substantially more real
  calibration data (~100+ examples, likely several 28-day QPU budget windows) is the natural
  next step to test whether the observed few-shot trend continues toward closing the gap.
- **k>2 correlation groups are untested.** Everything here is pairwise (k=2); larger correlated
  clusters are out of scope for this repo.
- **The v1 GAT ablation is kept for reproducibility of the negative result**, not because it's
  recommended — see [Methodology](#v1-per-qubit-z-regression-historical--superseded).
- **`clamp` vs. `tanh` output activation** was resolved for the v1 model (`clamp` confirmed
  better by ablation, Run 3 in the log) but v1 itself was superseded before this mattered for
  final results.
- Real hardware validation to date covers a single device (`ibm_marrakesh`) and a single
  confirmed-correlated pair — a pilot, not a statistically powered validation.

## References

- Nation, Kang, Sundaresan, Gambetta (2021). *Scalable Mitigation of Measurement Errors on
  Quantum Computers.* PRX Quantum 2, 040326. [arXiv:2108.12518](https://arxiv.org/abs/2108.12518)
- Pokharel, Srinivasan, Quiroz, Boots (2024). *Scalable measurement error mitigation via
  iterative Bayesian unfolding.* Phys. Rev. Research 6, 013187. [arXiv:2210.12284](https://arxiv.org/abs/2210.12284)
- Czarnik, Arrasmith, Coles, Cincio (2021). *Error mitigation with Clifford quantum-circuit
  data.* Quantum 5, 592. [arXiv:2005.10189](https://arxiv.org/abs/2005.10189) — cited as
  inspiration only; this project's method is not CDR (see the note this repo previously
  carried in its Methodology section: the regression target here is readout error on stabilizer
  states directly, not an arbitrary observable extrapolated from near-Clifford circuits).
- Maciejewski, Baccari, Zimborás, Oszmaniec (2021). *Modeling and mitigation of cross-talk
  effects in readout noise with applications to the Quantum Approximate Optimization
  Algorithm.* Quantum 5, 464. Measured real-hardware readout correlations do not follow the
  physical coupling/heavy-hex map — the basis for this repo's synthetic pair topology being
  randomly assigned rather than coupling-map-derived.
- van den Berg, Minev, Temme (2021). *Qubit readout error mitigation with bit-flip averaging.*
  Sci. Adv. 7, eabi8009. [arXiv:2106.05800](https://arxiv.org/abs/2106.05800)
- Ferracin, Yang, McNulty, et al. (2021). *Using classical bit-flip correction for error
  mitigation including 2-qubit correlations.* Reports that 2-qubit correlation was small enough
  to neglect on their IBMQ device — consistent with this repo's own real-hardware measurement
  (§1.5 of the results doc) that correlation was pair-specific and an order of magnitude weaker
  than the initial synthetic assumption.
- Veličković et al. (2018). *Graph Attention Networks.* ICLR.
- Alon & Yahav (2021). *On the Bottleneck of Graph Neural Networks and its Practical
  Implications.* ICLR. — the over-squashing/over-smoothing trade-off motivating (and ultimately
  not saving) the v1 all-to-all graph design.
- Gottesman (1998). *The Heisenberg Representation of Quantum Computers.*
  [quant-ph/9807006](https://arxiv.org/abs/quant-ph/9807006) — Gottesman–Knill theorem,
  the basis for polynomial-time Clifford circuit simulation used throughout this project.

## License

MIT — see [LICENSE](LICENSE).