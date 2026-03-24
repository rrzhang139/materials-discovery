"""
Preprocess MP-20 CSV data into pickle format expected by CliqueFlowmer.
Uses M3GNet oracle to compute formation energy targets (matching the paper's setup).

Usage:
    python preprocess_mp20.py [--use_csv_targets]

    --use_csv_targets: Use DFT formation energy from the CSV instead of M3GNet predictions.
                       Faster but differs from the paper's setup.
"""
import pandas as pd
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from io import StringIO
from tqdm import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure

USE_CSV_TARGETS = "--use_csv_targets" in sys.argv

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(base_dir, "data", "preprocessed", "mp20")
out_dir = csv_dir  # output pickles go alongside CSVs


def cif_to_structure(cif_str):
    parser = CifParser(StringIO(cif_str))
    return parser.parse_structures(primitive=True)[0]


def process_split_csv_targets(csv_path, out_path):
    """Fast path: use formation_energy_per_atom from CSV directly."""
    df = pd.read_csv(csv_path)
    structures = []
    targets = []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_path)}"):
        try:
            structure = cif_to_structure(row["cif"])
            target = float(row["formation_energy_per_atom"])
            structures.append(structure)
            targets.append(target)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error: {e}")

    print(f"  Processed {len(structures)} structures ({errors} errors)")
    data = {"inputs": structures, "targets": targets}
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved to {out_path}")


def process_split_m3gnet(csv_path, out_path):
    """Slow path: use M3GNet oracle to predict formation energy (matches paper)."""
    import torch
    from matgl import load_model

    device = torch.device("cpu")
    oracle = load_model("M3GNet-MP-2018.6.1-Eform")
    oracle_model = oracle.model.to(device)
    oracle_model.eval()

    df = pd.read_csv(csv_path)
    structures = []
    targets = []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_path)}"):
        try:
            structure = cif_to_structure(row["cif"])
            target = oracle.predict_structure(structure)
            structures.append(structure)
            targets.append(float(target))
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error: {e}")

    print(f"  Processed {len(structures)} structures ({errors} errors)")
    data = {"inputs": structures, "targets": targets}
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    process_fn = process_split_csv_targets if USE_CSV_TARGETS else process_split_m3gnet
    mode = "CSV targets" if USE_CSV_TARGETS else "M3GNet oracle"
    print(f"Preprocessing MP-20 with {mode}")

    for split in ["train", "val", "test"]:
        csv_path = os.path.join(csv_dir, f"{split}.csv")
        out_path = os.path.join(out_dir, f"{split}.pickle")

        if not os.path.exists(csv_path):
            print(f"Skipping {split}: {csv_path} not found")
            continue

        if os.path.exists(out_path):
            print(f"Skipping {split}: {out_path} already exists")
            continue

        process_fn(csv_path, out_path)

    print("Done!")
