import ml_collections
from ml_collections.config_dict import config_dict
import torch
import torch.nn as nn
import architectures.ops as ops

def get_config():
    """Factorization experiment v2: max batch, phase-2 gated h(z) training.

    Scaling from paper baseline (batch=1024, lr=1.4e-4, warmup=1e5):
      batch 1024 → 16384 (16x)
      lr    1.4e-4 → 2.24e-3 (linear scaling)
      warmup 1e5 → 6250 (same data volume per phase)
    """
    config = ml_collections.ConfigDict()

    config.model = {
        'cls': 'CliqueFlowmer',
        'n_cliques': 8,
        'clique_dim': 16,
        'knot_dim': 1,
        'transformer_dim': 256,
        'n_registers': 2,
        'mlp_dim': 128,
        'n_mlp': 2,
        'n_blocks': 4,
        'n_heads': 4,
        'dropout_rate': 0.1,
        'alpha_vae': 1e-4,
        'alpha_mse': 1,
        'beta_mse': 1e-4,
        'warmup': 6250,           # 1e5 / 16 — same data volume per warmup phase
        'temp_atom': 1,
        'temp_flow': 16,
        'mle_prior': True,
        'temp_distance': 0,
        'drop_type': 0,
        'drop_latent': 0.1,
        'submodule': ops.SwiGLU,
        'act': nn.GELU(),
        'lr': 2.24e-3,            # 1.4e-4 * 16 — linear scaling for 16x batch
        'alpha_fact': 0.1
    }

    config.data = {
        'task': 'mp20'
    }

    config.learner = {
        'cls': 'ES',
        'design_steps': 2000,
        'decay': 0.4,
        'lr': 3e-4,
        'n_pert': 20,
        'scale_pert': 0.05,
        'antithetic': True,
        'rank': True
    }

    config.storage = {
        'bucket': '<your_Google_bucket>'
    }

    return config
