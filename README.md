# Correlated Quantum Readout Error Mitigation via Graph Neural Networks Trained on Clifford Data

> Status: **v1.0 — reproducible baseline.** Data generation and training pipeline run end-to-end. Scaling study (4/6/8/10+ qubits), zero-shot generalization, and M3/iterative-Bayesian-unfolding benchmarks are **not** included in this release — see [Roadmap](#roadmap--known-limitations).

A Graph Attention Network (GAT) with a residual connection (~2,267 parameters) that learns to correct correlated readout errors on multi-qubit quantum circuits, trained entirely on classically-simulable Clifford circuit data.

## Table of contents

- [Motivation](#motivation)
- [Methodology](#methodology)
  - [1. Data generation](#1-data-generation)
  - [2. Graph representation](#2-graph-representation)
  - [3. Model architecture](#3-model-architecture)
  - [4. Training](#4-training)
- [Installation](#installation)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Roadmap / known limitations](#roadmap--known-limitations)
- [References](#references)
- [License](#license)

## Motivation

Readout error on NISQ hardware is rarely independent per qubit — crosstalk correlates the error across qubits, which classical bit-flip correction cannot capture. Matrix-based mitigation (naive inversion of the 2ⁿ×2ⁿ assignment matrix) is not the right straw-man baseline here: matrix-free methods such as M3 (Nation et al., 2021) already solve the scaling problem. The case for a learned model instead rests on two properties matrix inversion does not have: it amplifies statistical variance (negative entries in the inverse assignment matrix inflate the variance of the mitigated expectation value), and it must be recalibrated per device/measurement setting. This project tests whether a GNN trained once on simulated Clifford data can amortize that cost.

## Methodology

### 1. Data generation

- **Circuits:** randomly generated Clifford-dominated circuits (`generate_random_clifford_circuit`), 7 qubits × depth 30 in the current dataset, 10,000 circuits total.
- **Why Clifford circuits:** the Gottesman–Knill theorem guarantees Clifford circuits can be simulated classically in polynomial time by tracking stabilizer generators (not full 2ⁿ state amplitudes). This makes it possible to generate massive `{ideal, noisy}` label pairs without exponential blowup.
- **Simulation:** each circuit is run twice via Qiskit-Aer — once ideal (no noise), once with a readout/bit-flip noise model — producing bitstring count dictionaries for both.
- **Terminology note:** this is *not* CDR (Clifford Data Regression, Czarnik et al. 2021). CDR uses near-Clifford circuits because pure-Clifford expectation values converge to zero for arbitrary observables, which makes them useless for that regression target. Here the target is the *readout* error on stabilizer states directly (⟨Z⟩ ∈ {0, ±1}), so pure Clifford circuits remain informative. This is referred to as **Clifford-generated training data**, with CDR cited only as inspiration.

### 2. Graph representation

Each circuit becomes one graph:

- **Nodes** = qubits. Node feature vector = `[noisy ⟨Z⟩, one-hot qubit ID]`, i.e. `in_channels = 1 + num_qubits`.
- **Edges** = fully-connected (all-to-all). This is a deliberate trade-off, not a free win — see the layer-depth note below.
- **Target (y)** = ideal ⟨Z⟩ per qubit, computed from the ideal bitstring counts.

**Bitstring → Pauli-Z expectation.** For a qubit's marginal counts over `shots`:

```
⟨Z⟩ = P(measured 0) − P(measured 1)
```

implemented by iterating each bitstring, weighting by `count / total_shots`, and adding/subtracting per bit (Qiskit indexes qubit 0 as the rightmost character).

### 3. Model architecture

```
Input (x): [noisy ⟨Z⟩ | one-hot qubit ID]  →  dim = 1 + N
        │
        ▼
GATConv(in, 16, heads=4)  →  ELU        # attention over all-to-all graph
        │
        ▼
GATConv(64, 16, heads=1)  →  ELU        # aggregate heads
        │
        ▼
Linear(16 → 1)                          # Δ correction
        │
        ▼
mitigated = clamp(noisy_z + Δ, −1, 1)   # residual + bounded output
```

**Design choices and why:**

| Choice | Reasoning |
|---|---|
| GAT over GCN | Crosstalk between qubit pairs is not symmetric or uniform, so fixed degree-normalized averaging (GCN) is the wrong inductive bias — attention lets the model weight each pair. |
| Residual connection | The noisy ⟨Z⟩ is usually already close to the ideal value, so the model only needs to learn the *correction* (Δ), not the value from scratch. |
| All-to-all edges | Removes the message-passing bottleneck (over-squashing) for long-range crosstalk, at a real cost: it also accelerates over-smoothing, since every node now averages against every other node each layer. This is why the network is kept to 2 GAT layers — that depth is a forced compensation for the dense graph, not an independent hyperparameter choice. |
| Positional encoding (one-hot qubit ID) | Needed because on a complete graph, without a way to identify *which* qubit a node is, nodes become indistinguishable to the attention mechanism. |
| `clamp(-1, 1)` instead of `tanh` | `tanh` cannot reach exactly ±1 (it only approaches it asymptotically) and its gradient vanishes near saturation, penalizing the model more as it approaches the target while teaching it less. `clamp` reaches ±1.0000 exactly, at the cost of zero gradient outside the clamp range — flagged explicitly under [Known limitations](#roadmap--known-limitations), not treated as solved. |
| GAT on a complete graph + positional encoding | Architecturally, this is a Transformer over qubits. Not disguised as a graph-topology-aware model — stated directly here so it doesn't read as an omission. |

Note this is **mitigation, not correction**: it is statistical post-processing on measured expectation values, not a fault-tolerant quantum error correction scheme, and the underlying quantum state is untouched.

### 4. Training

- Loss: MSE between predicted and ideal ⟨Z⟩ per qubit.
- Optimizer: Adam, lr = 1e-3.
- Split: 80/20 train/test, seeded (`torch.manual_seed(42)`).
- Batching: PyG `DataLoader`, batch size 128.

## Installation

```bash
git clone <repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate the Clifford circuit dataset (ideal vs. noisy bitstring counts)
python main.py

# 2. (Optional sanity check) inspect the graph conversion on a single sample
python prepare_gnn_dataset.py

# 3. Train the GAT
python train_gnn.py
```

## Project structure

```
.
├── README.md
├── requirements.txt
├── main.py                    # parallel Clifford circuit generation + noise simulation
├── prepare_gnn_dataset.py     # bitstring counts -> PyG graph Data objects
├── train_gnn.py                # QuantumReadoutMitigationGAT model + training loop
├── src/
│   ├── generator.py            # generate_random_clifford_circuit
│   ├── simulator.py            # execute_circuit_pipeline (ideal / noisy)
│   └── data_exporter.py        # save_dataset_to_json
└── output_data/                 # generated dataset (gitignored)
```

## Roadmap / known limitations

This v1.0 is a reproducible baseline, not a publication-ready result. Before claiming any research result, the following are open:

- **One-hot qubit encoding is not zero-shot.** `in_channels = 1 + num_qubits` ties the model to a fixed qubit count — a model trained at N qubits cannot run on a graph of a different size. Needs to be replaced with a fixed-dimension sinusoidal positional encoding before any scaling study (4/6/8/10+ qubits, zero-shot transfer).
- **O(n²) edge construction.** `prepare_gnn_dataset.py` builds the fully-connected edge index with a pure-Python nested loop per circuit — fine at 7 qubits × 10k circuits, but should be vectorized (`torch.combinations` / cached per qubit-count) before scaling up.
- **`clamp` gradient dead zone unverified.** ±1.0000 output looks clean but `clamp` has zero gradient outside [−1, 1] — this needs an ablation against `tanh` and "linear during training, clamp at inference" before it's reported as a solved problem.
- **No baseline benchmark yet.** Not yet compared against M3 (Nation et al., 2021) or iterative Bayesian unfolding (Srinivasan et al., 2024) — required before any claim of improvement.
- **Superposition preservation unmeasured.** Current evaluation only spot-checks pure-state recovery; needs explicit reporting for mid-range ⟨Z⟩ (e.g. ≈ 0.5) to rule out the model just pushing outputs to the nearest extreme.
- **Simulator-only.** No real hardware (IBM Quantum) validation yet; noise model is simulated, not measured.

## References

- Nation, Kang, Sundaresan, Gambetta (2021). *Scalable Mitigation of Measurement Errors on Quantum Computers.* PRX Quantum 2, 040326. [arXiv:2108.12518](https://arxiv.org/abs/2108.12518)
- Srinivasan, Pokharel, Quiroz, Boots (2024). *Scalable measurement error mitigation via iterative Bayesian unfolding.* Phys. Rev. Research 6, 013187.
- Czarnik, Arrasmith, Coles, Cincio (2021). *Error mitigation with Clifford quantum-circuit data.* Quantum 5, 592. [arXiv:2005.10189](https://arxiv.org/abs/2005.10189)
- Veličković et al. (2018). *Graph Attention Networks.* ICLR.
- Alon & Yahav (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR.
- Giraldo et al. (2023). *On the Trade-off between Over-smoothing and Over-squashing in Deep GNNs.* [arXiv:2212.02374](https://arxiv.org/abs/2212.02374)
- Gottesman (1998). *The Heisenberg Representation of Quantum Computers.* [quant-ph/9807006](https://arxiv.org/abs/quant-ph/9807006)

## License

MIT — see [LICENSE](LICENSE).
