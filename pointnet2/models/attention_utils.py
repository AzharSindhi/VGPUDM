import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import wraps
from einops import einsum


# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
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

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        temp = self.to_kv(context)
        k,v = temp.chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# Convolutional Position Encoding
class ConvPosEnc(nn.Module):
    def __init__(self, dim_q, dim_content, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj_q = nn.Conv1d(
            in_channels=dim_q,
            out_channels=dim_q,
            kernel_size=k,
            stride=1,
            padding=k//2,
            groups=dim_q
        )

        self.proj_content = nn.Conv1d(
            in_channels=dim_content,
            out_channels=dim_content,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim_content
        )

    def forward(self,q,content):
        q = q.permute(0,2,1)
        q = self.proj_q(q) + q
        q = q.permute(0,2,1)

        # B,C,H,W = content.shape
        content = content.permute(0, 2, 1)
        content = self.proj_content(content) + content
        content = content.permute(0,2,1)

        return q,content

# main class
class AttentionFusion(nn.Module):
    def __init__(
        self,
        depth,                                  # Self-Attention deep
        dim,                                    # Q dim
        latent_dim = 512,                       # Content dim
        cross_heads = 1,                        # Cross-Attention Head
        latent_heads = 8,                       # Self-Attention Head
        cross_dim_head = 64,                    # Cross-Attention Head dim
        latent_dim_head = 64,                   # Self-Attention Head dim
        weight_tie_layers = False,
        pe=False
    ):
        super().__init__()

        self.pe = pe
        if(pe):
            # position encoding
            self.cpe = ConvPosEnc(
                dim_q=latent_dim,
                dim_content=dim
            )

        # Cross-Attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        #
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        # Self-Attention
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(
        self,
        data,                           # Content data
        mask = None,                    # mask
        queries_encoder = None,         # Q data
    ):
        b, *_, device = *data.shape, data.device
        x = queries_encoder

        # ---- position encoding ----
        if(self.pe):
            x,data = self.cpe(
                q=x,
                content=data,
            )
        # ---- position encoding ----

        # ---- Cross-Attention----
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x
        # ---- Cross-Attention----


        #  ---- Self-Attention ----
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        #  ---- Self-Attention ----

        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

class PointNorm(nn.Module):
    def __init__(self,dim,t=True):
        super().__init__()
        self.t = t
        self.norm = nn.BatchNorm1d(dim)
    def forward(self,x):
        if(self.t):
            x = x.permute(0,2,1)
            return self.norm(x).permute(0,2,1)
        return self.norm(x)
