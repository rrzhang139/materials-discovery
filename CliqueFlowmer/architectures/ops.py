import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 


from torch.optim.lr_scheduler import LambdaLR
from typing import Callable
from data.constants import atomic_numbers

def rmsnorm(x, dim=-1, eps=1e-6):
    normed = x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + eps)
    return normed


class IndexEmbedding:

    def __init__(self, dim):

        self.dim = dim 
    
    def __call__(self, horizon):

        steps = torch.arange(horizon)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp( (-emb) * torch.arange(half_dim))

        emb = steps.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb.unsqueeze(0)


class TimeEmbedding:
    
    def __init__(self, dim):

        self.dim = dim 
    
    def __call__(self, t):

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp( (-emb) * torch.arange(half_dim)).to(t.device)

        time_emb = t.reshape(-1, 1) * emb.reshape(1, -1)
        time_emb = torch.cat([torch.sin(time_emb), torch.cos(time_emb)], dim=-1)

        return time_emb.unsqueeze(-2)


class AtomicEmbedding(nn.Module):
    
    def __init__(self, dim):

        super().__init__()

        self.dim = dim 
        self.embedding = nn.Embedding(len(atomic_numbers), dim)
    
    def forward(self, tokens):
        self.embedding = self.embedding.to(tokens.device)
        return self.embedding(tokens)


class Registers(nn.Module):

    def __init__(self, n, dim):

        super().__init__()

        self.n = n 
        self.dim = dim 
        self.val = nn.Parameter(torch.randn(n, dim))
    
    def forward(self):
        return self.val


def attach_registers(registers, x, z, c1, c2, shift, mask):

    B, L, D = x.shape
    N = registers.shape[0]

    x = torch.cat([x, registers.unsqueeze(0).repeat(B, 1, 1)], dim=1)

    if z is not None: 
        if z.shape[0] == 1:
            z = z.repeat(B, 1, 1)
        z = torch.cat([z, torch.zeros(B, N, D, device=z.device)], dim=1)

    if c1 is not None: 
        if c1.shape[0] == 1:
            c1 = c1.repeat(B, 1, 1)
        c1 = torch.cat([c1, torch.zeros(B, N, D, device=c1.device)], dim=1)

    if c2 is not None: 
        if c2.shape[0] == 1:
            c2 = c2.repeat(B, 1, 1)
        c2 = torch.cat([c2, torch.zeros(B, N, D, device=c2.device)], dim=1)

    if shift is not None:
        shift = F.pad(shift, (0, N, 0, N), value=0)

    if mask is not None:
        if len(mask.shape) == len(x.shape):
            mask = F.pad(mask, (0, 0, 0, N), value=0)
            mask = F.pad(mask, (0, N, 0, 0), value=1)
        else:
            mask = F.pad(mask, (0, N), value=1)
        
    return x, z, c1, c2, shift, mask 


def remove_registers(n_registers, x):

    x = x[..., :-n_registers, :]    

    return x
    


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, act = nn.GELU(), act_final=False):
        super().__init__()

        self.model = nn.Sequential()
        dims = (input_dim,) + hidden_dims

        for i in range(len(hidden_dims)):

            self.model.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]), act
                )
            )
        
        self.model.append(nn.Linear(dims[-1], output_dim))
        
        if act_final:
            self.model.append(act)

    def forward(self, x):
        return self.model(x).squeeze(-1)    


class SwiGLU(nn.Module):

    def __init__(self, input_dim, output_dim=None, base_dim=None, expansion=2.67, dropout=0.0, bias_in=False, bias_out=True):
        
        super().__init__()
        
        base_dim = input_dim if base_dim is None else base_dim 
        hidden_dim = round(expansion * base_dim)

        self.w = nn.Linear(input_dim, 2 * hidden_dim, bias=bias_in)
        
        output_dim = input_dim if output_dim is None else output_dim

        self.w_out = nn.Linear(hidden_dim, output_dim, bias=bias_out)
        self.dropout = dropout

    def forward(self, x):

        a, b = self.w(x).chunk(2, dim=-1)
        
        y = F.silu(a) * b
        if self.dropout:
            y = F.dropout(y, p=self.dropout, training=self.training)
        return self.w_out(y)


class DeepSwiGLU(nn.Module):

    def __init__(self, input_dim, output_dim, base_dim=None, expansion=2.67, n_layers=2, dropout=0.0, bias_in=False, bias_out=True, norm=True, norm_last=True):

        super().__init__()

        base_dim = input_dim if base_dim is None else base_dim 
        hidden_dim = round(expansion * base_dim)
        self.n_layers = n_layers
        self.layers = nn.Sequential()

        w_in = nn.Linear(input_dim, 2 * hidden_dim, bias=bias_in)

        self.layers.append(w_in)

        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, 2 * hidden_dim, bias=False))

        self.w_out = nn.Linear(hidden_dim, output_dim, bias=bias_out)
        self.dropout = dropout
        self.norm = norm
        self.norm_last = norm_last

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            a, b = layer(x).chunk(2, dim=-1)
            x = F.silu(a) * b 
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            if self.norm and (i < self.n_layers - 1 or self.norm_last):
                x = rmsnorm(x)
        
        return self.w_out(x)


#
# Attention operation for a ready q, k, v tuple
#
def attention(q, k, v, shift=None, mask=None):

    d = q.shape[-1]

    #
    # Compute the attention logits: Q of shape (*, N, D) and K of shape (*, M, D) are mapped to Att of shape (*, N, M)
    #
    log_att = q @ k.transpose(-1, -2) / math.sqrt(d)

    #
    # Apply the shift (e.g., ALiBi) to the logits
    #
    log_att = log_att if shift is None else log_att - shift
    
    #
    # Apply the mask by assigning ~= -inf to masked logits
    # 
    log_att = log_att if mask is None else log_att * mask - 1e8 * (1-mask)

    #
    # Obtain attention weights from the logits
    #
    att = F.softmax(log_att, dim=-1)

    #
    # Compute the attention output by mixing v with attention weights
    #
    return att @ v  


class Attention(nn.Module):

    def __init__(self, model_dim: int, n_heads: int = 2, act: Callable = nn.GELU()):

        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.act = act 

        self.lin = nn.Linear(model_dim, 3 * model_dim)
        self.lin_out = nn.Identity() if n_heads == 1 else nn.Linear(model_dim, model_dim)


    def forward(self, x, shift=None, mask=None):
        #
        # Compute q, k, v from x at once; shape (*, N, 3 x D)
        #
        qkv = self.lin(x)

        #
        # Chunk qkv into different heads; shape (H, *, N, 3 x D / H)
        #
        qkv = torch.stack(
            qkv.chunk(self.n_heads, dim=-1)
        )

        #
        # Duplicate mask and shift for each head
        #
        if mask is not None: 
            mask = mask.unsqueeze(0).repeat((self.n_heads,) + len(mask.shape) * (1,))

        if shift is not None:
            shift = shift.unsqueeze(0).repeat((self.n_heads,) + len(shift.shape) * (1,))
            scale = torch.exp(-math.log(2) * torch.arange(self.n_heads)).reshape((self.n_heads,) + len(shift.shape[:-1]) * (1,))
            shift = scale.to(shift.device) * shift

        #
        # Extract q, k, v
        #
        q, k, v = qkv.chunk(3, dim=-1)

        #
        # Compute self-attention
        #
        o = attention(q, k, v, shift=shift, mask=mask)

        #
        # Combine outputs from all heads
        #
        o = o.chunk(self.n_heads, dim=0)
        o = torch.cat(o, dim=-1).squeeze(0)
        o = self.lin_out(o)

        return o


class CrossAttention(Attention):

    def __init__(self, model_dim: int, n_heads: int = 2, act: Callable = nn.GELU()):

        super().__init__(model_dim, n_heads, act)

        del self.lin 

        self.lin_q = nn.Linear(model_dim, model_dim)
        self.lin_kv = nn.Linear(model_dim, 2 * model_dim)

    def forward(self, x, z, mask=None):
        #
        # Compute q from the processed sequence and k, v from the latent
        #
        q = self.lin_q(x)
        kv = self.lin_kv(z)

        #
        # Chunk q, k, v into different heads
        #
        q = torch.stack(q.chunk(self.n_heads, dim=-1))
        kv = torch.stack(kv.chunk(self.n_heads, dim=-1))
    
        #
        # Separate k and v
        #
        k, v = kv.chunk(2, dim=-1)

        #
        # Duplicate mask for each head
        #
        if mask is not None: 
            mask = mask.unsqueeze(0).repeat((self.n_heads,) + len(mask.shape) * (1,))

        #
        # Compute the cross-attention
        #
        o = attention(q, k, v, mask=mask)

        #
        # Combine outputs from all heads
        #
        o = o.chunk(self.n_heads, dim=0)
        o = torch.cat(o, dim=-1).squeeze(0)
        o = self.lin_out(o)

        return o


#
# Given a sequence of atoms create a list of indexes it attains
#
def sequence_timer(x):

    sequence_length = x.shape[-2]

    return torch.arange(sequence_length).to(x.device)


#
# Given a structure (abc, angles, pos) create an embedding of indexes in the sequences
#
def structure_timer(x):
    
    structure_length = x.shape[-2]

    return torch.arange(structure_length).to(x.device) - 2


class AttentionPool(nn.Module):
    def __init__(self, dim, n_heads=4, expand_ratio=1):

        super().__init__()

        self.hidden_dim = expand_ratio * dim 

        self.q = nn.Parameter(torch.randn(1, 1, dim))
        self.proj_q = nn.Linear(dim, self.hidden_dim)
        self.proj_k = nn.Linear(dim, self.hidden_dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.scale = (self.hidden_dim // n_heads) ** -0.5
        self.out = nn.Linear(dim, dim)


    def split(self, h):  # (B,*,D) -> (B,heads,*,Dh)
        B, L, D = h.shape 
        H = self.n_heads 
        Dh = D // H
        return h.view(B, -1, H, Dh).transpose(1, 2)

    def forward(self, x, mask):
        #
        # Project the input (q, k, v)
        #
        B, L, D = x.shape
        q = self.proj_q(self.q.expand(B, -1, -1))          # (B,1,HD)
        k = self.proj_k(x)                                  # (B,L,HD)
        v = self.proj_v(x)                                  # (B,L,HD)

        qh, kh, vh = self.split(q), self.split(k), self.split(v)           # (B,H,1,Dh), (B,H,L,Dh), ...
        attn = (qh @ kh.transpose(-2, -1)) * self.scale     # (B,H,1,L)

        #
        # Remove attention from the masked tokens
        #
        attn = attn.masked_fill(mask[:, None, None, :] == 0, -1e9)
        w = torch.softmax(attn, dim=-1)                     # (B,H,1,L)
        out = (w @ vh).transpose(1, 2).reshape(B, 1, D)     # (B,1,D)
        return self.out(out).squeeze(1)                     # (B,D)


class AdaLN(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.weight = nn.Linear(dim, 2 * dim)

    def forward(self, x, c):

        x = self.norm(x)
        scale, shift = self.weight(c).chunk(2, dim=-1)

        return x * (1 + scale) + shift


def linear_warmup_decay(optimizer, max_lr, warmup_steps, decay_steps):
    #
    # Set base lr in optimizer param groups
    #
    for pg in optimizer.param_groups:
        pg["lr"] = max_lr

    total_steps = warmup_steps + decay_steps

    def lr_lambda(current_step: int):
        
        if current_step < warmup_steps:
            #
            # Warmup 0 -> 1
            #
            return float(current_step) / float(max(1, warmup_steps))
        
        if current_step >= total_steps:
            return 0.0
        
        #
        # Decay 1 -> 0
        #
        progress = float(current_step - warmup_steps) / float(max(1, decay_steps))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def mask_in_sequence(x, val=0, p_mask=0.1):
    #
    # Get the shape information
    #
    shape = x.shape 
    n = len(shape)
    B, L = shape[:2]

    #
    # Create the random 
    #
    drop_mask = (torch.rand((B, L) + (n-2) * (1,), device=x.device) < p_mask).float().repeat((1, 1) + shape[2:])

    #
    # Mask the input with the given value
    #
    x = (1 - drop_mask) * x + drop_mask * val 

    return x


def mask_in_batch(x, p_mask=0.1):
    #
    # Get the shape information 
    #
    shape = x.shape 
    n = len(shape)
    B = shape[0]

    #
    # Create the random noising mask
    #
    drop_mask = (torch.rand((B,) + (n-1) * (1,), device=x.device) < p_mask).float().repeat((1,) + shape[1:])

    #
    # Mask the input with noise
    #
    noise = torch.randn_like(x, device=x.device)
    x = (1 - drop_mask) * x + drop_mask * noise 

    return x