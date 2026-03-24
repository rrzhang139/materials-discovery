"""
Diagnostic script: Measure the factorization gap in CliqueFlowmer's latent space.

Loads a pretrained checkpoint, trains an unconstrained predictor h(z) on the full
latent vector, and measures how much h(z) disagrees with the additive predictor
Σ g_c(Z_c). A large gap means the property is highly non-additive in latent space.

Usage:
    python measure_factorization_gap.py --batch_size=256
    python measure_factorization_gap.py --batch_size=256 --no_wandb
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import wandb
import pickle
import os

from absl import app, flags
from ml_collections import config_flags

from models import CliqueFlowmer
import models.graphops as graphops
import data.tools as tools
import saving

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 37, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for h(z) training.')
flags.DEFINE_integer('n_epochs', 100, 'Epochs to train h(z).')
flags.DEFINE_float('lr', 1e-3, 'Learning rate for h(z).')
flags.DEFINE_integer('n_optim_steps', 200, 'Gradient steps for latent optimization.')
flags.DEFINE_integer('n_optim_samples', 500, 'Number of z samples to optimize.')
flags.DEFINE_bool('no_wandb', False, 'Disable W&B logging.')

config_flags.DEFINE_config_file(
    'config',
    'configs/mp20/cliqueflowmer.py',
    'File with hyperparameter configurations.',
    lock_config=False
)


def encode_dataset(model, inputs, batch_size, device):
    """Encode all structures into flat latent vectors."""
    dataset = tools.MatbenchDataset(inputs, [0.0] * len(inputs))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=tools.collate_structure, num_workers=0)

    all_z = []
    with torch.no_grad():
        for batch in loader:
            abc, angles, spec, pos, mask, _ = tools.move_to_device(batch, device)
            atomic_emb = model.atomic_emb(spec)
            z = model.encode(abc, angles, atomic_emb, pos, mask, separate=False)
            all_z.append(z.cpu())

    return torch.cat(all_z, dim=0)


def measure_gap(model, z_all, t_all, split_name, batch_size, device):
    """Measure factorization gap and prediction errors on a dataset."""
    loader = DataLoader(TensorDataset(z_all, t_all), batch_size=batch_size)
    gaps, add_errors, unc_errors = [], [], []

    with torch.no_grad():
        for z_batch, t_batch in loader:
            z_batch, t_batch = z_batch.to(device), t_batch.to(device)

            pred_add = model.predict(z_batch)
            pred_unc = model.predict_unconstrained(z_batch)

            gaps.append(((pred_unc - pred_add) ** 2).cpu())
            add_errors.append(torch.abs(pred_add - t_batch).cpu())
            unc_errors.append(torch.abs(pred_unc - t_batch).cpu())

    gaps = torch.cat(gaps)
    add_errors = torch.cat(add_errors)
    unc_errors = torch.cat(unc_errors)

    results = {
        f'{split_name}/factorization_gap_mean': gaps.mean().item(),
        f'{split_name}/factorization_gap_std': gaps.std().item(),
        f'{split_name}/factorization_gap_max': gaps.max().item(),
        f'{split_name}/additive_mae': add_errors.mean().item(),
        f'{split_name}/unconstrained_mae': unc_errors.mean().item(),
    }

    print(f"\n{split_name}:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    return results


def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Parse config
    kwargs = dict(**FLAGS.config)
    data_kwargs = dict(kwargs['data'])
    model_kwargs = dict(kwargs['model'])

    task_cls = data_kwargs.pop('task')
    model_cls = model_kwargs.pop('cls')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'preprocessed', task_cls)
    model_dir = os.path.join(base_dir, 'models', 'states', model_cls, task_cls)

    # W&B
    use_wandb = not FLAGS.no_wandb
    if use_wandb:
        wandb.init(project=f'{task_cls}', name='factorization-gap-diagnostic')

    # Load data
    data = []
    for split in ['train', 'val']:
        data_path = os.path.join(data_dir, f'{split}.pickle')
        with open(data_path, 'rb') as f:
            subset = pickle.load(f)
        data.append(subset)

    train_data, val_data = data
    train_inputs, train_targets = list(train_data.values())[:2]
    val_inputs, val_targets = list(val_data.values())[:2]

    # Build model
    length_mle_vals = tools.length_mle(train_inputs, False)
    if model_kwargs.pop('mle_prior'):
        model_kwargs["initial_length_dist"] = tools.normal_lengths_from_mle(length_mle_vals, device)

    lower_p, upper_p = tools.find_percentiles(train_targets)
    model_kwargs["lower_percentile"] = lower_p
    model_kwargs["upper_percentile"] = upper_p

    model = CliqueFlowmer(**model_kwargs).to(device)

    # Load checkpoint (strict=False handles missing unconstrained_regressor keys)
    model_path = os.path.join(model_dir, 'checkpoint')
    response = saving.load_model_state_dict(model_path, model)
    if response is not None:
        model = response
    model.eval()
    print("Loaded pretrained checkpoint.")

    # Freeze everything, unfreeze only unconstrained_regressor
    for param in model.parameters():
        param.requires_grad = False
    for param in model.unconstrained_regressor.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_trainable:,} / {n_total:,} parameters")

    # Encode all data into flat latent vectors
    print("Encoding training data...")
    z_train = encode_dataset(model, train_inputs, FLAGS.batch_size, device)
    t_train = torch.tensor(train_targets, dtype=torch.float32)
    print(f"Encoded {z_train.shape[0]} train samples -> z shape {z_train.shape}")

    print("Encoding validation data...")
    z_val = encode_dataset(model, val_inputs, FLAGS.batch_size, device)
    t_val = torch.tensor(val_targets, dtype=torch.float32)
    print(f"Encoded {z_val.shape[0]} val samples -> z shape {z_val.shape}")

    # Measure pre-training gap (random h(z))
    with torch.no_grad():
        z_sample = z_train[:1000].to(device)
        pred_add = model.predict(z_sample)
        pred_unc = model.predict_unconstrained(z_sample)
        pre_gap = ((pred_unc - pred_add) ** 2).mean().item()
        print(f"Pre-training factorization gap (random h): {pre_gap:.4f}")

    # Train h(z) on (z, target) pairs
    train_dataset = TensorDataset(z_train, t_train)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    h_optimizer = optim.Adam(model.unconstrained_regressor.parameters(), lr=FLAGS.lr)
    criterion = nn.MSELoss()

    print(f"\nTraining unconstrained predictor h(z) for {FLAGS.n_epochs} epochs...")
    for epoch in range(FLAGS.n_epochs):
        model.unconstrained_regressor.train()
        epoch_loss = 0
        n_batches = 0

        for z_batch, t_batch in train_loader:
            z_batch, t_batch = z_batch.to(device), t_batch.to(device)

            pred = model.unconstrained_regressor(z_batch)
            loss = criterion(pred, t_batch)

            h_optimizer.zero_grad()
            loss.backward()
            h_optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{FLAGS.n_epochs}: MSE = {avg_loss:.6f}")

        if use_wandb:
            wandb.log({'h_train/mse': avg_loss}, step=epoch)

    # Measure factorization gap on train + val
    model.unconstrained_regressor.eval()
    results = {}
    results.update(measure_gap(model, z_train, t_train, 'train', FLAGS.batch_size, device))
    results.update(measure_gap(model, z_val, t_val, 'val', FLAGS.batch_size, device))

    # Measure gap on gradient-optimized latent codes (proxy for ES)
    print(f"\nOptimizing {FLAGS.n_optim_samples} latent codes for {FLAGS.n_optim_steps} steps...")
    n_samples = min(FLAGS.n_optim_samples, len(z_train))
    indices = np.random.choice(len(z_train), n_samples, replace=False)
    z_opt = z_train[indices].clone().to(device).requires_grad_(True)

    z_optimizer = optim.Adam([z_opt], lr=3e-4)
    index_matrix = model.index_matrix.to(device)

    for step in range(FLAGS.n_optim_steps):
        z_cliques = graphops.separate_latents(z_opt, index_matrix)
        pred = model.target_regressor(z_cliques)
        loss = pred.mean()
        loss.backward()
        z_optimizer.step()
        z_optimizer.zero_grad()

    with torch.no_grad():
        pred_add = model.predict(z_opt)
        pred_unc = model.predict_unconstrained(z_opt)
        opt_gap = (pred_unc - pred_add) ** 2

    opt_results = {
        'optimized/factorization_gap_mean': opt_gap.mean().item(),
        'optimized/factorization_gap_std': opt_gap.std().item(),
        'optimized/factorization_gap_max': opt_gap.max().item(),
        'optimized/additive_pred_mean': pred_add.mean().item(),
        'optimized/unconstrained_pred_mean': pred_unc.mean().item(),
    }

    print("\nOptimized latent codes:")
    for k, v in opt_results.items():
        print(f"  {k}: {v:.6f}")

    results.update(opt_results)

    if use_wandb:
        wandb.log(results)
        wandb.finish()

    print("\nDone.")


if __name__ == '__main__':
    app.run(main)
