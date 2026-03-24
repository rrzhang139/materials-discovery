"""
Single-GPU / CPU training script for CliqueFlowmer.
Strips out DDP logic from train.py for local development and single-GPU runs.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
import pickle
import os

from absl import app, flags
from ml_collections import config_flags

from models import CliqueFlowmer, Transformer
import models.graphops as graphops
import data.tools as tools
import saving

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', int(37), 'Random seed.')
flags.DEFINE_integer('batch_size', int(128), 'Batch size.')
flags.DEFINE_integer('N_eval', int(100), 'Evaluation frequency.')
flags.DEFINE_integer('N_save', int(1000), 'Saving frequency.')
flags.DEFINE_integer('N_epochs', int(25000), 'Number of epochs.')
flags.DEFINE_bool('From_scratch', bool(True), 'Whether to start training from scratch.')
flags.DEFINE_bool('offset_atoms', bool(True), 'Whether to offset the lattice lengths by atoms.')
flags.DEFINE_bool('standardize', bool(False), 'Whether to standardize the targets.')
flags.DEFINE_bool('augment', bool(False), 'Whether to augment the structures.')
flags.DEFINE_bool('eform_reg', bool(False), 'Whether the primary target is blended with Eform.')
flags.DEFINE_float('eform_tau', -0.2, 'Threshold to start penalizing Eform.')
flags.DEFINE_integer('eform_lambda', 0, 'Strength of the Eform penalty.')
flags.DEFINE_bool('no_wandb', bool(False), 'Disable W&B logging.')

config_flags.DEFINE_config_file(
    'config',
    'configs/mp20/cliqueflowmer.py',
    'File with hyperparameter configurations.',
    lock_config=False
)


def main(_):
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #
    # Parse configs of the run
    #
    kwargs = dict(**FLAGS.config)

    data_kwargs = dict(kwargs['data'])
    model_kwargs = dict(kwargs['model'])
    storage_kwargs = dict(kwargs['storage'])

    #
    # Build model spec string for saving (simplified for local)
    #
    model_spec = 'checkpoint'

    #
    # Extract specific kwargs
    #
    task_cls = data_kwargs.pop('task')
    model_cls = model_kwargs.pop('cls')

    #
    # Create data path for loading (local)
    #
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'preprocessed', task_cls)
    model_dir = os.path.join(base_dir, 'models', 'states', model_cls, task_cls)

    #
    # Initialize W&B
    #
    use_wandb = not FLAGS.no_wandb
    if use_wandb:
        wandb.init(project=f'{task_cls}', name=f'{model_cls}-local')
        wandb.config.update(FLAGS)

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    #
    # Load the data
    #
    data = []
    for split in ['train', 'val']:
        data_path = os.path.join(data_dir, f'{split}.pickle')
        with open(data_path, 'rb') as f:
            subset = pickle.load(f)
        data.append(subset)

    train_data, val_data = data
    train_inputs, train_targets = list(train_data.values())[:2]
    val_inputs, val_targets = list(val_data.values())[:2]

    #
    # Maybe regularize with formation energy
    #
    def _form_target(x):
        value, eform = x
        penalty = np.maximum(0, eform - FLAGS.eform_tau)
        target = value + FLAGS.eform_lambda * penalty
        return target

    form_target = lambda x: _form_target(x) if FLAGS.eform_reg else x
    train_targets = [form_target(x) for x in train_targets]
    val_targets = [form_target(x) for x in val_targets]

    #
    # Maybe standardize
    #
    mean_targets = np.mean(train_targets)
    std_targets = np.std(train_targets)
    preprocess_target = lambda x: (x - mean_targets) / std_targets if FLAGS.standardize else x
    train_targets = [preprocess_target(x) for x in train_targets]
    val_targets = [preprocess_target(x) for x in val_targets]

    #
    # Create data loaders (standard, not distributed)
    #
    train_dataset = tools.MatbenchDataset(train_inputs, train_targets, augment=FLAGS.augment)
    val_dataset = tools.MatbenchDataset(val_inputs, val_targets, augment=FLAGS.augment)

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=tools.collate_structure,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        collate_fn=tools.collate_structure,
        num_workers=0,
        drop_last=True
    )

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"Batch size: {FLAGS.batch_size}, Train batches/epoch: {len(train_loader)}")

    #
    # Estimate the initial length distribution
    #
    length_mle_vals = tools.length_mle(train_inputs, FLAGS.offset_atoms)

    print("MLE For Log-Lengths")
    for k, v in length_mle_vals.items():
        if k != "cov":
            print(f"{k}: mu {v[0]} sigma {v[1]}")

    if model_kwargs.pop('mle_prior'):
        model_kwargs["initial_length_dist"] = tools.normal_lengths_from_mle(length_mle_vals, device)

    #
    # Estimate the target percentiles
    #
    lower_p, upper_p = tools.find_percentiles(train_targets)
    model_kwargs["lower_percentile"] = lower_p
    model_kwargs["upper_percentile"] = upper_p
    print(f"Percentile Estimation: Lower {lower_p:.3f} Upper {upper_p:.3f}")

    #
    # Initialize the model
    #
    model = globals()[model_cls](**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if not FLAGS.From_scratch:
        model_path = os.path.join(model_dir, model_spec)
        response = saving.load_model_state_dict(model_path, model)
        if response is not None:
            model = response
            model.beta_vae = 1.
            model.beta_mse = 1.

    #
    # Train the model
    #
    total_steps = 0

    for epoch in range(FLAGS.N_epochs):

        model.train()
        val_iter = iter(val_loader)

        for batch in train_loader:
            abc, angles, spec, pos, mask, targets = tools.move_to_device(batch, device)

            train_info = model.training_step(abc, angles, spec, pos, mask, targets)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            #
            # Evaluate
            #
            if total_steps == 0 or (total_steps + 1) % FLAGS.N_eval == 0:
                model.eval()

                try:
                    batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    batch = next(val_iter)

                abc, angles, spec, pos, mask, targets = tools.move_to_device(batch, device)
                eval_info = model.eval_step(abc, angles, spec, pos, mask, targets)

                # Log
                train_log = {f'train/{k}': v.cpu().item() if hasattr(v, 'cpu') else v for k, v in train_info.items()}
                eval_log = {f'eval/{k}': v.cpu().item() if hasattr(v, 'cpu') else v for k, v in eval_info.items()}

                if use_wandb:
                    wandb.log({**train_log, **eval_log}, step=total_steps)

                if total_steps % (FLAGS.N_eval * 10) == 0:
                    loss_str = ', '.join([f'{k}: {v:.4f}' for k, v in train_log.items()])
                    print(f"Step {total_steps}: {loss_str}")

                model.train()

            #
            # Save checkpoint
            #
            if (total_steps + 1) % FLAGS.N_save == 0:
                model_path = os.path.join(model_dir, model_spec)
                saving.save_model_state_dict(model_path, model)

            total_steps += 1

    #
    # Final save
    #
    model_path = os.path.join(model_dir, model_spec)
    saving.save_model_state_dict(model_path, model)

    if use_wandb:
        wandb.finish()

    print(f"Training complete. Total steps: {total_steps}")

if __name__ == '__main__':
    app.run(main)
