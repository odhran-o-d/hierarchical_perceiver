'''this code describes modules for the hierarchical perceiver'''
import numpy as np
from math import pi, log
from functools import wraps #this just stops decorators fucking up the naming of the underlying

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce


large_HPC = {
    'encode':                   [None,'a','b',None,None,None,None],
    'decode':                   [None,   None,    None,    None,    None,    'b',    'a'],
    'groups':                   [16,  4,    1,    1,    1,    4,    16],  
    'self-attention_layers':    [2,   2,    18,   2,    1,    1,    1],
    'heads':                    [4,   8,    16,   32,   16,   8,    4],
    'latent_channels':          [128, 256,  512,  1024, 512,  256,  128],
    'latent_vectors_per_group': [128, 256,  256,  64,   256,  256,  128],
    'num_ff':                   2,
    'attn_dropout':             0.,
    'ff_dropout':               0.,
    'num_input_features':       32,
}





# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f): # this is a generic cache function, that just caches all reslts to a dictionary. It is beautiful! 
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class Old_PreNorm(nn.Module): #applies pre-layer norm as an operation. For why pre-layer beats post see https://arxiv.org/pdf/2002.04745.pdf
    #NOTE:change PreNorm behaviour if there are two inputs, not if there is an argument for context. 
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class PreNorm(nn.Module): #applies pre-layer norm as an operation. For why pre-layer beats post see https://arxiv.org/pdf/2002.04745.pdf
    #NOTE:change PreNorm behaviour if there are two inputs, not if there is an argument for context. 
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        if type(x) == list:
          x, context = x 

        x = self.norm(x)


        if exists(self.norm_context):
            normed_context = self.norm_context(context)
            x = [x, normed_context]

        return self.fn(x)

class GEGLU(nn.Module):   # Gated version of the gaussian error linear units function 
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module): #the FFN component of any layer of a transformer 
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, output_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        output_dim = default(output_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, output_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        if type(x) == list:
          x, context = x 

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        # The map applies the rearranging into heads to each of Q, K and V 
        # here b = batch size, n = sequence length, and (h d) is the embedding dimension accross the heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # rules of Einsum:
            # 1: Repeating letters in different inputs means those values will be multiplied
            # 2: Given rule 1 repeating letters must be the same length
            # 3: Omitting a letter means that axis will be summed
            # 4: We can return unsummed axes in any order
        # b i j is a stack of h head matrices along b where I and J are attention between positions in the sequence n
        # I & J are different lengths in cross-attention
        # but I is the latent length, which should be preserved 
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # accross batches this multiplies each of value embeddings by the  n * n query-key scores, and then sums them by dropping the j.  
        # latent length I is preserved. 
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class HPC_layer(nn.Module):
  def __init__(self, group_modules, latent_vector_size, dim_emb):
    super().__init__() 
    self.group_modules = group_modules
    self.num_groups = len(group_modules)
    self.ind = nn.Parameter(torch.Tensor(self.num_groups, latent_vector_size, dim_emb)) 
    nn.init.xavier_uniform_(self.ind)
  def forward(self, x):
    batches = torch.chunk(x, self.num_groups, dim=1) #NOTE: chunk is super janky if your input is small - may return only 15 batches
    return torch.cat([self.group_modules[i]([self.ind[i].repeat(batches[i].size(0), 1, 1), batches[i]]) for i in range(self.num_groups)], dim=1)  



class Hierarchical_Perceiver(nn.Module):
  def __init__(self, dim_in, c):
    super().__init__()
    if c.model_type == 'h-npt-large':
        self.HPC = large_HPC
    else:
        raise NotImplementedError

    self.HPC['input_channels'] = [dim_in] + HPC['latent_channels'][:-1]
    self.layers = nn.ModuleList([])
    self.final_attention = Attention(query_dim=dim_in, context_dim=HPC['latent_channels'][-1], output_dim=HPC['latent_channels'][-1])


    for index, layer in enumerate(self.HPC['encode']):
      layer_group = nn.ModuleList([])

      for g_total in range(self.HPC['groups'][index]):

        context_dim = self.HPC['input_channels'][index]
        latent_dim = self.HPC['latent_channels'][index]
        heads = HPC['heads'][index]
        dim_head = int(self.HPC['latent_channels'][index] / self.HPC['heads'][index])
        attn_dropout = self.HPC['attn_dropout']
        ff_dropout = self.HPC['ff_dropout']
        num_ff = self.HPC['num_ff']

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, context_dim,
                                                              heads = heads, dim_head = dim_head, dropout = attn_dropout), 
                                                              context_dim = context_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_self_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))
        get_self_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))


        layer_sub_section = nn.ModuleList([])
        layer_sub_section.append(get_cross_attn())    
        layer_sub_section += [get_cross_ff() for _ in range(num_ff)]

        for _ in range(HPC['self-attention_layers'][index]):
          layer_sub_section.append(get_self_attn()) 
          layer_sub_section += [get_self_ff() for _ in range(num_ff)]

        layer_group.append(nn.Sequential(*layer_sub_section))
      
      self.layers.append(HPC_layer(layer_group, HPC['latent_vectors_per_group'][index], HPC['latent_channels'][index]))    

  def forward(self, x):
    new_x = x
    cache = {}
    for i, layer in enumerate(self.layers):

      if HPC['decode'][i] != None:
        new_x = layer(new_x + cache[HPC['decode'][i]])
      else:
        new_x = layer(new_x)
      if HPC['encode'][i] != None: 
        cache[HPC['encode'][i]] = new_x

    return self.final_attention(x, new_x)




if __name__ == '__main__':

    HPC = {
        'encode':                   [None,'a','b',None,None,None,None],
        'decode':                   [None,   None,    None,    None,    None,    'b',    'a'],
        'groups':                   [16,  4,    1,    1,    1,    4,    16],  
        'self-attention_layers':    [2,   2,    18,   2,    1,    1,    1],
        'heads':                    [4,   8,    16,   32,   16,   8,    4],
        'latent_channels':          [128, 256,  512,  1024, 512,  256,  128],
        'latent_vectors_per_group': [128, 256,  256,  64,   256,  256,  128],
        'num_ff':                   2,
        'attn_dropout':             0.,
        'ff_dropout':               0.,
        'num_input_features':       32,
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.ones(1,160, 32).to(device)
    print(x)
    model = Hierarchical_Perceiver(HPC)
    model = model.to(device)
    y = model(x)
