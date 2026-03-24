import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import math 

import architectures.ops as ops 
import architectures.blocks as blocks 
import architectures.backbones as backbones

import saving
import data.tools as data_tools
import models.tools as tools 


class Transformer(nn.Module):

    def __init__(self, transformer_dim, n_blocks, n_heads, dropout_rate=0.1, lr=1e-4, act=nn.GELU()):

        super().__init__()
        self.act = act 
        self.atomic_emb = ops.AtomicEmbedding(transformer_dim)

        #
        # The projection steps from the lattice to the transformer spaces
        #
        self.abc_to_model = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.angle_to_model = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.pos_to_model = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)

        #
        # Transformer and structure that produce the latent space
        #
        self.transformer = backbones.Transformer(transformer_dim, n_blocks, n_heads, dropout_rate, act)

        #
        # The projection from the transformer features
        #
        self.abc_from_model = ops.MLP(transformer_dim, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.angles_from_model = ops.MLP(transformer_dim, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.pos_from_model = ops.MLP(transformer_dim, transformer_dim, (2 * transformer_dim,), act_final=True)
        
        #
        # Projection onto scalars
        #
        self.pred = ops.MLP(transformer_dim, 1, (transformer_dim // 2,))

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)


    def forward(self, abc, angles, atomic, pos, mask):

        #
        # Compute distances for the attention shift
        #
        distances = data_tools.compute_pairwise_distances(abc, angles, pos)

        #
        # Embed molecular information in HD spaces
        #
        atomic = self.atomic_emb(atomic)
        abc = self.abc_to_model(abc)
        angles = self.angle_to_model(angles)
        pos = self.pos_to_model(pos) 

        #
        # Merge the embeddings of geometry information
        #
        x = tools.into_structure_tensor(abc, angles, pos) 

        #
        # Add necessary padding
        #
        atomic = F.pad(atomic, (0, 0, 2, 0), value=0)
        mask = F.pad(mask, (2, 0), value=1)
        distances = F.pad(distances, (2, 0, 2, 0), value=0)

        #
        # Process with a transformer
        #
        x = self.transformer(x, atomic, shift=distances, mask=mask)

        #
        # Separate the geometry information and process independently
        #
        abc, angles, pos = tools.from_structure_tensor(x)

        abc = self.abc_from_model(abc)
        angles = self.angles_from_model(angles)
        pos = self.pos_from_model(pos)

        pos = tools.masked_mean(pos, mask[:, 2:], -2, sqrt=True)

        #
        # Make a prediction
        #
        x = self.pred(abc + angles + pos)
        
        return x.squeeze(-1)

    def training_step(self, abc, angles, atomic, pos, mask, target):
        #
        # Compute importance weights of the examples 
        #
        outlier = torch.relu(target) + torch.relu(-3 - target)
        weight = 1 + 10 * outlier.view(-1)**2

        #
        # Compute weights normalized globally
        #
        global_weight_sum = weight.clone()
        torch.distributed.all_reduce(global_weight_sum, op=torch.distributed.ReduceOp.SUM)
        weight = weight / global_weight_sum.sum()

        #
        # Compute the error of the model
        #
        pred = self(abc, angles, atomic, pos, mask).view(-1)
        error = pred - target.view(-1)
        
        #
        # Take the gradient step
        #
        loss = (weight * (error**2)).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)

        self.optimizer.step()
        self.optimizer.zero_grad()

        #
        # Log the info
        #
        error = error.detach()

        return {
            'loss': loss.item(),
            'mse': (error**2).mean(),
            'mae': torch.abs(error).mean(),
            'max_ae': torch.abs(error).max(),
            'r2': tools.r2(pred.detach(), target)
        }

    def eval_step(self, abc, angles, atomic, pos, mask, target):
        
        self.eval()

        with torch.no_grad():
            pred = self(abc, angles, atomic, pos, mask).view(-1)

        error = pred - target.view(-1)

        mse = (error**2).mean()
        mae = torch.abs(error).mean()
        max_ae = torch.abs(error).max()

        self.train()

        return {
            'mse': mse,
            'mae': mae,
            'max_ae': max_ae,
            'r2': tools.r2(pred, target)
        }
