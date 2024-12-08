import torch
import torch.nn as nn
from torch.nn import functional as f
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feedforward function for the attention block

        Args:
        x (torch.Tensor): Input tensor (batch_size, channels, height, width)
        """
        
        residue = x

        x = self.groupnorm(x)
        
        n, c, h, w = x.shape

        # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        x = x.view((n, c, h * w))

        # (batch_size, channels, height * width) -> (batch_size, height * width, channels)
        x = x.transpose(-1, -2)

        # (batch_size, height * width, channels) -> (batch_size, height * width, channels)
        x = self.attention(x)

        # (batch_size, height * width, channels) -> (batch_size, channels, height * width)
        x = x.transpose(-1, -2)

        # (batch_size, channels, height * width) -> (batch_size, channels, height, width)
        x = x.view((n, c, h, w))

        return x + residue


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feedforward function for the residual block

        Args:
        x (torch.Tensor): Input tensor (batch_size, in_channels, height, width)
        """

        residue = x

        x = self.groupnorm_1(x)
        x = f.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = f.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (batch_size, 4, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/4, width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/2, width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (batch_size, 512, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height, width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # (batch_size, 256, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (batch_size, 128, height, width) -> (batch_size, 3, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feedforward function for decoder layer

        Args:
        x (torch.Tensor): Input tensor (batch_size, 4, height/8, width/8)
        """

        x /= 0.18215

        for module in self:
            x = module(x)
        
        return x  # (batch_size, 3, height, width)
