#!/usr/bin/env python3

import argparse
import wandb
import gzip
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import random
import numpy as np

from pymatgen.core import Structure
from pymatgen.core.structure import Structure as PMGStructure
from pymatgen.io.cif import CifWriter

# Your project import
from data import tools
from optimization.sun import build_reference_metadata, classify_sun_for_optimized
from matgl import load_model
from m3gnet.models import M3GNet


def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument(
        '--model', 
        type=str, 
        help="Which model is evaluated."
    )
    ap.add_argument(
        "--structs",
        type=str,
        default=None,
        help="The path to structures to evaluate..",
    )
    ap.add_argument(
        "--bucket",
        type=str,
        default='<Your Google Bucket>',
        help="Bucket with reference data."
    )
    ap.add_argument(
        "--task_cls",
        type=str,
        default="mp20"
    )
    ap.add_argument(
        "--eform_reg",
        action="store_true",
        help="Whether the task uses eform as a regularizer."
    )
    ap.add_argument(
        "--subsample",
        action="store_true",
        help="Whether to subsample structures."
    )
    ap.add_argument(
        "--save_top",
        type=int,
        default=100
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu"
    )
    args = ap.parse_args()

    device = torch.device(args.device)

    #
    # Load the structures
    #
    with open(args.structs, 'rb') as f:
        structs = pickle.load(f)

    structs = structs.get("refined_structures") if isinstance(structs, dict) else structs
    print(f"There is {len(structs)} refined structs!")

    #
    # Create a run for logging
    #
    wandb.init(project=f'{args.task_cls}-optimize', name=args.model)
    
    seed = 1
    torch.manual_seed(seed) 
    np.random.seed(seed)
    random.seed(seed)

    #
    # Load the data
    #
    data = []
    data_dir = os.path.join('materials', 'data', 'preprocessed', args.task_cls)

    for split in ['train', 'val', 'test']:
        data_path = os.path.join(data_dir, split)
        subset = tools.load_pickled_object_from_gcs(args.bucket, data_path)
        data.append(subset)

    #
    # Extract the train val split
    #
    train_data, val_data, test_data = data
    train_inputs, train_targets = list(train_data.values())[:2]
    val_inputs, val_targets = list(val_data.values())[:2]
    test_inputs, test_targets = list(test_data.values())[:2]

    train_targets = torch.tensor(train_targets).to(device)
    val_targets = torch.tensor(val_targets).to(device)
    test_targets = torch.tensor(test_targets).to(device)

    #
    # Extract Eform regularizer if necessary
    #   
    if args.eform_reg:
        train_eform = train_targets[..., -1]
        val_eform = val_targets[..., -1]
        test_eform = test_targets[..., -1]

        train_targets = train_targets[..., 0]
        val_targets = val_targets[..., 0]
        test_targets = test_targets[..., 0]
    
    else:
        train_eform = train_targets.clone()
        val_eform = val_targets.clone()
        test_eform = test_targets.clone()

    #
    # Create the reference set
    #
    inputs = train_inputs + val_inputs + test_inputs
    eform = torch.cat([train_eform, val_eform, test_eform])
    targets = torch.cat([train_targets, val_targets, test_targets])

    #
    # Find the min and max of train targets for later normalization
    #
    min_val = targets.min()
    max_val = targets.max()
    print(f'Min: {min_val.item()} Max: {max_val.item()} Mean: {targets.mean().item()}')

    #
    # Initialize the oracle
    #
    oracle = load_model("M3GNet-MP-2018.6.1-Eform")
    oracle = oracle.model
    oracle.eval()
    print("Loaded the oracle!")

    #
    # Set the ground-truth evaluation
    #
    bandgap_task = ("bandgap" in args.task_cls)
    eval_gt = lambda struct: tools.bandgap_from_primitive(struct) if bandgap_task else oracle.predict_structure(struct)

    #
    # Build the reference set
    #
    ref_device = torch.device("cpu")
    hull_splits = ("train", "val", "test")

    print("Building reference metadata for SUN...")
    ref_entries, by_elemset = build_reference_metadata(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        test_inputs=test_inputs,
        train_targets=train_eform,
        val_targets=val_eform,
        test_targets=test_eform,
        batch_size=128,
        device=ref_device,
        hull_splits=hull_splits,
    )
    print(f"Built {len(ref_entries)} reference entries.")

    #
    # Calculate the target value
    #
    with torch.no_grad():
        true_val = [eval_gt(s) for s in structs]
        band_gap = [tools.bandgap_from_primitive(s) for s in structs]


    # 
    # Compute SUN metrics on optimized structures
    # 
    print("Evaluating SUN metrics on optimized structures...")

    if args.subsample:
        n_subsample = min(1000, len(structs))
        idx = np.random.choice(len(structs), size=n_subsample, replace=False)
        structs = [structs[i] for i in idx]
        true_val = [true_val[i] for i in idx]
        band_gap = [band_gap[i] for i in idx]

    indexes, sun_results = classify_sun_for_optimized(
        optimized_structs=structs,
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        test_inputs=test_inputs,
        ref_entries=ref_entries,
        by_elemset=by_elemset,
        oracle=oracle,
        stable_threshold=0.0,
        metastable_threshold=0.08,
        min_dist=0.5,
    )


    #
    # Calculate in-SUN metrics
    #
    sun_indexes = indexes[:sun_results['N_SUN_stable']]
    
    sun_true_val = [true_val[i] for i in sun_indexes]
    sun_band_gap = [band_gap[i] for i in sun_indexes]

    sun_results['Target-per-SUN'] = float(np.mean(sun_true_val))
    sun_results['Target-per-SUN-min'] = float(np.min(sun_true_val))
    sun_results['BandGap-per-SUN'] = float(np.mean(sun_band_gap))
    sun_results['BandGap-per-SUN-min'] = float(np.min(sun_band_gap))

    print("SUN evaluation (strictly stable, Ehull < 0.0 eV/atom):")
    print(f"Target property                    = {np.mean(true_val)}")
    print(f"Band Gap                 = {np.mean(band_gap)}")
    print(f"N_gen                    = {sun_results['N_gen']}")
    print(f"N_stable                 = {sun_results['N_stable']}")
    print(f"N_SUN_stable             = {sun_results['N_SUN_stable']}")
    print(f"StabilityRate_stable     = {sun_results['StabilityRate_stable']:.4f}")
    print(f"SUNRate_stable           = {sun_results['SUNRate_stable']:.4f}")

    print("\nSUN evaluation (metastable-or-better, Ehull < 0.08 eV/atom):")
    print(f"N_metastable             = {sun_results['N_metastable']}")
    print(f"N_SUN_metastable         = {sun_results['N_SUN_metastable']}")
    print(f"StabilityRate_metastable = {sun_results['StabilityRate_metastable']:.4f}")
    print(f"SUNRate_metastable       = {sun_results['SUNRate_metastable']:.4f}")

    #
    # Clean the results up for logging
    #
    info = {}

    for k, v in sun_results.items():
        if isinstance(v, float) or isinstance(v, int):
            info[k] = v 

    info["TargetProperty"] = np.mean(true_val)
    info["BandGap"] = np.mean(band_gap)

    #
    # Log the final structures
    #
    wandb.log({f'Stability/{k}': v for k, v in info.items()})

    #
    # Save the most stable structures as a pickle file
    #
    top_stable_idx = sun_indexes[:args.save_top] 
    top_structs = [structs[i] for i in top_stable_idx]

    structs_dir, _ = os.path.split(args.structs)
    structs_save = os.path.join(structs_dir, f"stable-{args.save_top}.pkl")

    with open(structs_save, "wb") as f:
        pickle.dump(structs, f)

    #
    # Save the most stable structures as CIF files
    #
    cifs_dir = os.path.join(structs_dir, f"{args.model}-stable-cifs-{args.save_top}")
    os.makedirs(cifs_dir, exist_ok=True)

    for i, struct in enumerate(top_structs):
        cif_path = os.path.join(cifs_dir, f"Struct_{i}-{struct.composition.reduced_formula}")
        writer = CifWriter(struct)
        writer.write_file(cif_path)

    #
    # End the run
    #
    wandb.finish()


if __name__ == "__main__":
    main()