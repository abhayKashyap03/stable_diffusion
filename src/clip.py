import torch
from torch import nn
from torch.nn import functional as f
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_tokens: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, embedding_size))
    
    def forward(self, tokens: torch.Tensor):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x


class CLIPLayer(nn.Module):
    def __init__(self, embedding_size, n_heads: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(embedding_size)
        self.attention = SelfAttention(n_heads, embedding_size)
        self.layernorm_2 = nn.LayerNorm(embedding_size)
        self.linear_1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.linear_2 = nn.Linear(4 * embedding_size, embedding_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feedforward function for CLIP layer

        Args:
        x (torch.Tensor): Input tensor (batch_size, seq_len, dims)
        """
        
        residue = x

        # Self-attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        # Feedforward
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function
        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(768, 12) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        out = self.layernorm(state)

        return out
