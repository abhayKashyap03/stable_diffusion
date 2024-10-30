import torch
from torch import nn
from torch.nn import functional as f
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, channels: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(channels, 3 * channels, bias=in_proj_bias)  # Weights to be applied to the input pixels
        self.out_proj = nn.Linear(channels, channels, bias=out_proj_bias)  # Weight to be applied to the final output
        self.n_heads = n_heads
        self.d_heads = channels // n_heads
    
    def forward(self, x: torch.Tensor, causal_mask=True) -> torch.Tensor:
        """
        Feedforward function for self-attention block

        Args:
        x (torch.Tensor): Input tensor (batch_size, sequence_length, channels)
        """

        inp_shape = x.shape
        batch_size, seq_len, channels = inp_shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

        # (batch_size, seq_len, channels) -> (batch_size, seq_len, channels * 3) -> 3 * (batch_size, seq_len, channels)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, channels) -> (batch_size, seq_len, n_heads, d_heads) -> (batch_size, n_heads, seq_len, d_heads)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (batch_size, n_heads, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where upper triangle is 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_heads)

        weight = f.softmax(weight, dim=-1)

        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_heads) -> (batch_size, n_heads, seq_len, d_heads)
        out = weight @ v

        # (batch_size, n_heads, seq_len, d_heads) -> (batch_size, seq_len, n_heads, d_heads) -> (batch_size, seq_len, channels)
        out = out.transpose(1, 2).reshape(inp_shape)

        return out


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, embedding_size: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(embedding_size, embedding_size, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, embedding_size, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, embedding_size, bias=in_proj_bias)
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = embedding_size // n_heads
    
    def forward(self, x, y):
        """
        Feedforward function for cross attention layer
        
        Args:
        x (torch.Tensor): Latent tensor (batch_size, seq_len_q, dim_q)
        y (torch.Tensor): Context tensor (batch_size, seq_len_kv, dim_kv) = (batch_size, 77, 768)
        """

        inp_shape = x.shape
        batch_size, seq_len, embedding_size = inp_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = f.softmax(weight, dim=-1)

        out = weight @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(inp_shape)
        out = self.out_proj(out)

        return out
