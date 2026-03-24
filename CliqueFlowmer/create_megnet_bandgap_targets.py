# make_mp20_2d_targets.py

import os
import math
import pickle
import warnings

import pandas as pd
from tqdm import tqdm

import torch
from pymatgen.io.cif import CifParser
from matgl import load_model

from data.tools import predict_structure_megnet


warnings.filterwarnings(
    "ignore",
    message=r"Issues encountered while parsing CIF:.*rounded to ideal values.*",
    category=UserWarning,
    module=r"pymatgen\.io\.cif",
)

DEVICE = torch.device("cpu")


def is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def to_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().view(-1)[0].item())
    return float(x)


def cif_to_structure(cif_str):
    parser = CifParser.from_string(cif_str)
    # Do NOT primitive-reduce — this is a major NaN source
    s = parser.parse_structures(primitive=False)[0]
    return s


def assert_finite(name, x):
    try:
        x = float(x)
    except Exception:
        raise RuntimeError(f"{name} is not convertible to float: {x}")
    if not math.isfinite(x):
        raise RuntimeError(f"{name} is NaN/Inf: {x}")
    return x

#
# Load formation-energy oracle once
#
m3gnet = load_model("M3GNet-MP-2018.6.1-Eform")
m3gnet.model.to(DEVICE)
m3gnet.model.eval()


@torch.no_grad()
def predict_eform(structure) -> float:
    y = m3gnet.predict_structure(structure)
    return to_float(y)


def process_split(csv_path, out_path, *, method="PBE"):
    df = pd.read_csv(csv_path)

    inputs = []
    targets = []

    for cif_str in tqdm(df["cif"], desc=f"Processing {os.path.basename(csv_path)}"):
        try:
            s = cif_to_structure(cif_str)

            bg = predict_structure_megnet(s, method=method, device=str(DEVICE))
            ef = predict_eform(s)

            bg = assert_finite("bandgap", bg)
            ef = assert_finite("eform", ef)

            bg = to_float(bg)
            ef = to_float(ef)

            # HARD STOP: never write NaNs
            if not (is_finite(bg) and is_finite(ef)):
                continue

            inputs.append(s)
            targets.append([bg, ef])

        except Exception:
            # skip malformed CIFs / oracle crashes
            continue

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "inputs": inputs,
                "targets": targets,
            },
            f,
        )

    print(
        f"{os.path.basename(csv_path)}: saved {len(targets)} valid samples "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    base_dir = "mp20-bandgap"

    for split in ["train", "val", "test"]:
        process_split(
            os.path.join(base_dir, f"{split}.csv"),
            os.path.join(base_dir, f"{split}.pkl"),
            method="PBE",
        )
