# src/data_exporter.py
"""Persist a generated circuit dataset to disk as JSON."""
import json


def save_dataset_to_json(dataset: list[dict], output_path: str) -> None:
    """Write the dataset (list of per-circuit dicts) to output_path as JSON.

    Note: bitstring count dicts from Qiskit are JSON-serializable as-is
    (str keys, int values), so no conversion step is needed here.
    """
    with open(output_path, "w") as f:
        json.dump(dataset, f)
    print(f"Saved {len(dataset)} samples to {output_path}")
