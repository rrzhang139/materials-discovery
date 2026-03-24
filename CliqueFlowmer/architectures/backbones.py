import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math

import architectures.ops as ops
import architectures.blocks as blocks


class DMLP(nn.Module):

    def __init__(self, n_input, input_dim, model_dim, n_layers, dropout_rate= 0.1, act=nn.GELU()):

        super().__init__()
        self.n_input = n_input
        self.input_dim = input_dim
        self.layers = nn.Sequential()

        self.act = act 
        dims = (input_dim,) + n_layers * (model_dim,) 

        for i in range(n_layers):

            layer = nn.Sequential(
                nn.Linear(dims[i], dims[i+1]), act
            )

            if i + 1 < n_layers:
                layer.append(nn.BatchNorm1d(n_input))  
                layer.append(nn.Dropout(dropout_rate))

            self.layers.append(layer)
        
        self.proj = nn.Linear(dims[-1], 1)
        self.clique_emb = ops.IndexEmbedding(model_dim)(n_input)
        self.clique_mlp = ops.MLP(model_dim, model_dim, n_layers * (model_dim,), act)

    
    def forward(self, x):

        clique_emb = self.clique_mlp(self.clique_emb.to(x.device))
        scale = 1 / math.sqrt(len(self.layers))

        for i, layer in enumerate(self.layers):

            v = layer(x)
            v = v + clique_emb if i == 0 else v 
            x = x + scale * v if i > 0 else v
        
        x = self.proj(x).squeeze(-1) / math.sqrt(self.n_input)

        return x.sum(-1)


class PMLP(nn.Module):

    def __init__(self, n_input, input_dim, output_dim, model_dim, n_layers, dropout_rate= 0.1, act=nn.GELU()):

        super().__init__()
        self.n_input = n_input
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.layers = nn.Sequential()

        self.act = act 
        dims = (input_dim,) + n_layers * (model_dim,) 

        for i in range(n_layers):

            layer = nn.Sequential(
                nn.Linear(dims[i], dims[i+1]), act
            )

            if i + 1 < n_layers:
                layer.append(nn.LayerNorm(dims[i+1], elementwise_affine=False, bias=False))  
                layer.append(nn.Dropout(dropout_rate))

            self.layers.append(layer)
        
        self.proj = nn.Linear(dims[-1], self.output_dim)
        self.clique_emb = ops.IndexEmbedding(model_dim)(n_input)
        self.clique_mlp = ops.MLP(model_dim, model_dim, n_layers * (model_dim,), act)

    
    def forward(self, x):

        clique_emb = self.clique_mlp(self.clique_emb.to(x.device))
        scale = 1 / math.sqrt(len(self.layers))

        for i, layer in enumerate(self.layers):

            v = layer(x)
            v = v + clique_emb if i == 0 else v 
            x = x + scale * v if i > 0 else v
        
        return self.proj(x)  


class DDSwiGLU(nn.Module):

    def __init__(self, n_input, input_dim, model_dim, n_layers, expansion=1.33, dropout_rate= 0.1, norm_last=False):  

        super().__init__()
        self.n_input = n_input
        self.input_dim = input_dim
        self.n_layers = n_layers

        self.submodule = ops.DeepSwiGLU(input_dim, 1, model_dim, expansion=1, n_layers=n_layers, dropout=dropout_rate)

        self.clique_emb = ops.IndexEmbedding(model_dim)(n_input)
        self.clique_submodule = ops.SwiGLU(model_dim, model_dim, expansion=expansion) 
        self.norm_last = norm_last

    def forward(self, x):

        clique_emb = self.clique_submodule(self.clique_emb.to(x.device))
        scale = 1 / math.sqrt(len(self.submodule.layers))

        for i, layer in enumerate(self.submodule.layers):
            a, b = layer(x).chunk(2, dim=-1)
            v = F.silu(a) * b 
            if self.submodule.dropout:
                v = F.dropout(v, p=self.submodule.dropout, training=self.training)
        
            v = v + clique_emb if i == 0 else v 
            x = x + scale * v if i > 0 else v 

            if i < self.n_layers - 1 or self.norm_last:
                x = ops.rmsnorm(x)

        x = self.submodule.w_out(x).squeeze(-1) / math.sqrt(self.n_input)
        return x.sum(-1)


class PDSwiGLU(nn.Module):

    def __init__(self, n_input, input_dim, output_dim, model_dim, n_layers, expansion=2.67, dropout_rate= 0.1, norm=True, norm_last=True):

        super().__init__()
        self.n_input = n_input
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.n_layers = n_layers
        
        self.submodule = ops.DeepSwiGLU(input_dim, output_dim, model_dim, expansion=1, n_layers=n_layers, dropout=dropout_rate)

        self.clique_emb = ops.IndexEmbedding(model_dim)(n_input)
        self.clique_submodule = ops.SwiGLU(model_dim, model_dim, expansion=expansion) 
        self.norm = norm 
        self.norm_last = norm_last

    def forward(self, x):

        clique_emb = self.clique_submodule(self.clique_emb.to(x.device))
        scale = 1 / math.sqrt(len(self.submodule.layers))

        for i, layer in enumerate(self.submodule.layers):
            a, b = layer(x).chunk(2, dim=-1)
            v = F.silu(a) * b 
            if self.submodule.dropout:
                v = F.dropout(v, p=self.submodule.dropout, training=self.training)

            v = v + clique_emb if i == 0 else v 
            x = x + scale * v if i > 0 else v 
            
            if self.norm and (i < self.n_layers - 1 or self.norm_last):
                x = ops.rmsnorm(x)

        return self.submodule.w_out(x)


class Transformer(nn.Module):

    def __init__(self, dim, n_blocks, n_heads, dropout_rate, submodule=ops.SwiGLU, act=nn.GELU(), n_registers=0):

        super().__init__()

        self.blocks = nn.Sequential()
        self.norm = nn.LayerNorm(dim)

        for _ in range(n_blocks):
            self.blocks.append(
                blocks.TransformerBlock(dim, n_heads, dropout_rate, submodule, act)
            )
        
        self.n_registers = n_registers

        if n_registers > 0:
            self.registers = ops.Registers(n_registers, dim)

    def forward(self, x, z=None, c=None, shift=None, mask=None):

        if len(mask.shape) < len(x.shape):
            mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        if self.n_registers > 0:
            x, z, c, _, shift, mask = ops.attach_registers(self.registers(), x, z, c, None, shift, mask)

        for block in self.blocks:
            x = block(x, z, c, shift=shift, mask=mask)
        
        if self.n_registers > 0:
            x = ops.remove_registers(self.n_registers, x)

        x = self.norm(x)

        return x


class TransformerDecoder(Transformer):

    def __init__(self, dim, n_blocks, n_heads, dropout_rate, submodule=ops.SwiGLU, act=nn.GELU(), n_registers=0):

        super().__init__(dim, n_blocks, n_heads, dropout_rate, submodule, act, n_registers)

        self.blocks = nn.Sequential()

        for _ in range(n_blocks):
            self.blocks.append(
                blocks.TransformerDecoderBlock(dim, n_heads, dropout_rate, submodule, act)
            )
    
    def forward(self, x, z=None, c1=None, c2=None, shift=None, mask=None):

        if self.n_registers > 0:
            x, _, c1, c2, shift, mask = ops.attach_registers(self.registers(), x, None, c1, c2, shift, mask)

        for block in self.blocks:
            x = block(x, z, c1, c2, shift=shift, mask=mask)
        
        if self.n_registers > 0:
            x = ops.remove_registers(self.n_registers, x)

        x = self.norm(x)

        return x 