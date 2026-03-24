import pandas as pd
import pickle
import os
from tqdm import tqdm

from pymatgen.io.cif import CifParser
from matgl import load_model
from matgl.ext.pymatgen import Structure2Graph
from matgl.config import DEFAULT_ELEMENTS

import torch

#
# Setup device
#
device = torch.device("cpu")

#
# Load M3GNet oracle
#
oracle = load_model("M3GNet-MP-2018.6.1-Eform")
oracle = oracle.model.to(device)
oracle.eval()

#
# Converter for pymatgen Structure -> graph
#
converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=5.0)

def cif_to_structure(cif_str):
    parser = CifParser.from_string(cif_str)
    return parser.get_structures()[0]


def predict_m3gnet(structure):
    #
    # Convert to graph
    #
    g, _, state_feats = converter.get_graph(structure)
    g, state_feats = g.to(device), state_feats.to(device)
    
    #
    # Predict
    #
    with torch.no_grad():
        result = oracle(g=g, state_attr=state_feats)
        # For formation energy, this is usually result["e_form"]
        return result["e_form"].cpu().item()

def process_split(csv_path, out_path):
    df = pd.read_csv(csv_path)
    structures = []
    targets = []
    for cif_str in tqdm(df["cif"], desc=f"Processing {os.path.basename(csv_path)}"):
        try:
            structure = cif_to_structure(cif_str)
            structure = structure.get_primitive_structure()
            target = oracle.predict_structure(structure)
            structures.append(structure)
            targets.append(target)
        except Exception as e:
            print(f"Error processing structure: {e}")
    
    #
    # Save as pickle
    #
    data = {"inputs": structures, "targets": targets}
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

base_dir = "mp20"
for split in ["train", "val", "test"]:
    csv_path = os.path.join(base_dir, f"{split}.csv")
    out_path = os.path.join(base_dir, f"{split}.pkl")
    process_split(csv_path, out_path)
