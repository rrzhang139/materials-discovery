import ml_collections
from ml_collections.config_dict import config_dict
import torch 
import torch.nn as nn 

def get_config():

    config = ml_collections.ConfigDict()

    config.model = {
        'cls': 'Transformer',
        'transformer_dim': 128,
        'n_blocks': 1, 
        'n_heads': 4,
        'dropout_rate': 0.2,
        'act': nn.GELU(),
        'lr': 1e-4
    }

    config.data = {
        'task': 'mp20'
    }

    config.learner = {
        'cls': 'GradientAscent',
        'design_steps': 1000,
        'decay': 0.5,
        'lr': 3e-4
    }

    config.storage = {
        'bucket': 'kubair'
    }

    return config 