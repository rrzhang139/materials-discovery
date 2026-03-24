"""
End-to-end evaluation pipeline for CliqueFlowmer.
Loads a checkpoint, runs ES optimization, decodes materials, evaluates with M3GNet oracle.
Skips W&B. Outputs results to JSON + pickle.
"""
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import json
import time
import sys

from absl import app, flags
from ml_collections import config_flags

from models import CliqueFlowmer, Transformer
import models.graphops as graphops
import data.tools as tools
import saving

from matgl import load_model
from optimization.design import Design
from optimization.learner import ES

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('design_batch_size', 100, 'Number of materials to optimize.')
flags.DEFINE_integer('top_k', 10, 'Top-k best designs.')
flags.DEFINE_integer('design_steps', 200, 'ES optimization steps.')
flags.DEFINE_bool('refine', False, 'Whether to refine structures with M3GNet.')
flags.DEFINE_bool('sun', False, 'Whether to compute SUN metrics (slow).')
flags.DEFINE_bool('offset_atoms', True, 'Whether to offset lattice lengths by atoms.')
flags.DEFINE_string('output_dir', 'eval_results', 'Output directory.')

config_flags.DEFINE_config_file(
    'config',
    'configs/mp20/cliqueflowmer.py',
    'Hyperparameter config.',
    lock_config=False
)


def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Parse config
    kwargs = dict(**FLAGS.config)
    data_kwargs = dict(kwargs['data'])
    model_kwargs = dict(kwargs['model'])
    learner_kwargs = dict(kwargs['learner'])

    task_cls = data_kwargs.pop('task')
    model_cls = model_kwargs.pop('cls')
    learner_cls = learner_kwargs.pop('cls')
    learner_kwargs.pop('design_steps')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'preprocessed', task_cls)
    model_dir = os.path.join(base_dir, 'models', 'states', model_cls, task_cls)
    out_dir = os.path.join(base_dir, FLAGS.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(data_dir, f'{split}.pickle')
        with open(path, 'rb') as f:
            data.append(pickle.load(f))

    train_data, val_data, test_data = data
    train_inputs, train_targets = list(train_data.values())[:2]
    val_inputs, val_targets = list(val_data.values())[:2]
    test_inputs, test_targets = list(test_data.values())[:2]

    all_inputs = train_inputs + val_inputs + test_inputs
    all_targets = train_targets + val_targets + test_targets
    targets_t = torch.tensor(all_targets).to(device)
    min_val, max_val = targets_t.min(), targets_t.max()
    print(f"Target range: [{min_val.item():.3f}, {max_val.item():.3f}], mean: {targets_t.mean().item():.3f}")

    # Build model
    length_mle_vals = tools.length_mle(train_inputs, FLAGS.offset_atoms)
    if model_kwargs.pop('mle_prior'):
        model_kwargs["initial_length_dist"] = tools.normal_lengths_from_mle(length_mle_vals, device)
    lower_p, upper_p = tools.find_percentiles(train_targets)
    model_kwargs["lower_percentile"] = lower_p
    model_kwargs["upper_percentile"] = upper_p

    model = globals()[model_cls](**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Load checkpoint
    model_path = os.path.join(model_dir, 'checkpoint')
    response = saving.load_model_state_dict(model_path, model)
    if response is not None:
        model = response
        print(f"Loaded checkpoint from {model_path}")
    else:
        print(f"WARNING: No checkpoint at {model_path}, using random weights!")

    model = nn.DataParallel(model).to(device)
    model.eval()

    # Load oracle
    print("Loading M3GNet oracle...")
    oracle = load_model("M3GNet-MP-2018.6.1-Eform")
    oracle_model = oracle.model
    oracle_model.eval()

    eval_gt = lambda struct: oracle_model.predict_structure(struct)

    # Sample initial designs
    print(f"Sampling {FLAGS.design_batch_size} initial materials...")
    indices = np.random.choice(len(all_inputs), FLAGS.design_batch_size, replace=False)
    init_structs_raw = [all_inputs[i] for i in indices]
    init_data = tools.unpack_structures(init_structs_raw)
    abc, angles, atomic, pos, mask = tools.move_to_device(init_data, device)
    structures_start = tools.tensors_to_structure(abc, angles, atomic, pos, mask)

    # Evaluate starting materials
    print("Evaluating starting materials with oracle...")
    start_vals = []
    for i, s in enumerate(structures_start):
        try:
            v = float(eval_gt(s))
        except Exception:
            v = float('inf')
        start_vals.append(v)
        if i < 5:
            print(f"  {s.formula}: eform = {v:.4f}")
    start_vals_t = torch.tensor(start_vals).to(device)
    print(f"Starting: mean={start_vals_t[start_vals_t.isfinite()].mean():.4f}")

    # Encode to latent space
    print("Encoding to latent space...")
    with torch.no_grad():
        atomic_emb = model.module.atomic_emb(atomic)
        design_z = model.module.encode(abc, angles, atomic_emb, pos, mask, separate=False)
    print(f"Latent shape: {design_z.shape}")

    # Initialize ES optimizer
    design = Design(design_z)
    index_matrix = model.module.index_matrix.to(device)
    learner_kwargs['structure_fn'] = lambda x: graphops.separate_latents(x, index_matrix)
    learner_model = nn.DataParallel(model.module.target_regressor.to(device))
    learner = ES(design, learner_model, **learner_kwargs)

    # Run ES optimization
    print(f"Running {FLAGS.design_steps} ES optimization steps...")
    t0 = time.time()
    for step in range(FLAGS.design_steps):
        if step > 0:
            learner.train_step()
        if step % max(1, FLAGS.design_steps // 10) == 0:
            with torch.no_grad():
                val = learner.value()
            print(f"  Step {step}/{FLAGS.design_steps}: predicted = {val.item():.4f}")
    es_time = time.time() - t0
    print(f"ES optimization: {es_time:.1f}s")

    # Decode optimized latents
    print("Decoding optimized materials...")
    t0 = time.time()
    with torch.no_grad():
        design_opt = learner.design_fn()
        abc_d, angles_d, atomic_d, pos_d, mask_d = model.module.decode(design_opt)
        structures_opt = tools.tensors_to_structure(abc_d, angles_d, atomic_d, pos_d, mask_d)
        best_values, best_indices = learner.best(FLAGS.top_k)
    decode_time = time.time() - t0
    print(f"Decoded {len(structures_opt)} structures in {decode_time:.1f}s")

    # Optionally refine
    if FLAGS.refine:
        print("Refining structures...")
        for i in range(len(structures_opt)):
            try:
                structures_opt[i] = tools.refine_to_primitive_fast_strong(structures_opt[i])
            except Exception as e:
                if i < 3:
                    print(f"  Refine failed for {i}: {e}")

    # Evaluate optimized materials with oracle
    print("Evaluating optimized materials with oracle...")
    opt_vals = []
    for i, s in enumerate(structures_opt):
        try:
            v = float(eval_gt(s))
        except Exception:
            v = float('inf')
        opt_vals.append(v)
    opt_vals_t = torch.tensor(opt_vals).to(device)

    finite_mask = opt_vals_t.isfinite()
    opt_mean = opt_vals_t[finite_mask].mean().item() if finite_mask.any() else float('inf')
    opt_min = opt_vals_t[finite_mask].min().item() if finite_mask.any() else float('inf')

    # Top-k
    opt_vals_topk, topk_idx = torch.topk(opt_vals_t, FLAGS.top_k, largest=False)
    topk_mean = opt_vals_topk.mean().item()

    # Improvement rate
    improvement_rate = (opt_vals_t < start_vals_t).float().mean().item()

    print(f"\n{'='*60}")
    print(f"RESULTS (Formation Energy)")
    print(f"{'='*60}")
    print(f"Starting mean:       {start_vals_t[start_vals_t.isfinite()].mean():.4f}")
    print(f"Optimized mean:      {opt_mean:.4f}")
    print(f"Optimized min:       {opt_min:.4f}")
    print(f"Top-{FLAGS.top_k} mean:        {topk_mean:.4f}")
    print(f"Improvement rate:    {improvement_rate:.2%}")
    print(f"ES time:             {es_time:.1f}s")
    print(f"Decode time:         {decode_time:.1f}s")

    # Print top-k structures
    print(f"\nTop-{FLAGS.top_k} optimized materials:")
    best_idx_list = best_indices.cpu().tolist()
    for rank, idx in enumerate(best_idx_list[:FLAGS.top_k]):
        s = structures_opt[idx]
        v = opt_vals[idx]
        print(f"  {rank+1}. {s.formula} ({len(s)} atoms) — eform = {v:.4f}")

    # Save artifacts
    results = {
        "config": {
            "design_batch_size": FLAGS.design_batch_size,
            "design_steps": FLAGS.design_steps,
            "top_k": FLAGS.top_k,
            "seed": FLAGS.seed,
            "device": str(device),
        },
        "start_mean": float(start_vals_t[start_vals_t.isfinite()].mean()),
        "opt_mean": opt_mean,
        "opt_min": opt_min,
        "topk_mean": topk_mean,
        "improvement_rate": improvement_rate,
        "es_time": es_time,
        "decode_time": decode_time,
        "opt_vals": [float(v) for v in opt_vals],
        "start_vals": [float(v) for v in start_vals],
        "top_k_formulas": [structures_opt[i].formula for i in best_idx_list[:FLAGS.top_k]],
        "top_k_eform": [opt_vals[i] for i in best_idx_list[:FLAGS.top_k]],
    }

    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/eval_results.json")

    with open(os.path.join(out_dir, 'optimized_structures.pkl'), 'wb') as f:
        pickle.dump(structures_opt, f)
    print(f"Structures saved to {out_dir}/optimized_structures.pkl")

    # SUN evaluation (optional, slow)
    if FLAGS.sun:
        from optimization.sun import build_reference_metadata, classify_sun_for_optimized

        print("\nBuilding reference metadata for SUN...")
        eform_all = torch.tensor(all_targets).to(torch.device("cpu"))
        train_eform = eform_all[:len(train_targets)]
        val_eform = eform_all[len(train_targets):len(train_targets)+len(val_targets)]
        test_eform = eform_all[len(train_targets)+len(val_targets):]

        ref_entries, by_elemset = build_reference_metadata(
            train_inputs=train_inputs, val_inputs=val_inputs, test_inputs=test_inputs,
            train_targets=train_eform, val_targets=val_eform, test_targets=test_eform,
            batch_size=128, device=torch.device("cpu"),
        )
        print(f"Built {len(ref_entries)} reference entries")

        print("Running SUN evaluation...")
        indexes, sun_results = classify_sun_for_optimized(
            optimized_structs=structures_opt,
            train_inputs=train_inputs, val_inputs=val_inputs, test_inputs=test_inputs,
            ref_entries=ref_entries, by_elemset=by_elemset,
            oracle=oracle_model, stable_threshold=0.0, metastable_threshold=0.08, min_dist=0.5,
        )

        print(f"\nSUN Results:")
        print(f"  N_gen (valid):     {sun_results['N_gen']}")
        print(f"  N_stable:          {sun_results['N_stable']}")
        print(f"  N_SUN_stable:      {sun_results['N_SUN_stable']}")
        print(f"  SUNRate_stable:    {sun_results['SUNRate_stable']:.4f}")
        print(f"  N_metastable:      {sun_results['N_metastable']}")
        print(f"  N_SUN_metastable:  {sun_results['N_SUN_metastable']}")
        print(f"  SUNRate_meta:      {sun_results['SUNRate_metastable']:.4f}")

        sun_json = {k: v for k, v in sun_results.items()
                    if isinstance(v, (int, float, str))}
        with open(os.path.join(out_dir, 'sun_results.json'), 'w') as f:
            json.dump(sun_json, f, indent=2)
        print(f"SUN results saved to {out_dir}/sun_results.json")

    print("\nDone!")


if __name__ == '__main__':
    app.run(main)
