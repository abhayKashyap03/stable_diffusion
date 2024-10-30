import torch
from torch import nn
from torch.nn import functional as f
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = nn.Linear(4 * embedding_dim, 4 * embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feedforward function for TimeEmbedding layer

        Args:
        x (torch.Tensor): Input tensor (1, 320)
        """

        x = self.linear1(x)
        x = f.silu(x)
        x = self.linear2(x)
        return x  # (1, 1280)


class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        
        return x


class UpSample(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Feedforward function for upsampling layer
        
        Args:
        x (torch.Tensor): Input tensor (batch_size, channels, height, width)
        """

        x = f.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1290):
        super().__init__()

        self.group_norm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.group_norm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        """
        Feedforward function for UNet residual block

        Args:
        feature (torch.Tensor): Input feature tensor (batch_size, channels, height, width)
        time (torch.Tensor): Input time tensor (batch_size, 1280)
        time (torch.Tensor): Input time tensor (1, 1280)
        """

        residue = feature

        feature = self.group_norm_feature(feature)
        feature = f.silu(feature)
        feature = self.conv_feature(feature)

        time = f.silu(time)
        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.group_norm_merged(merged)
        merged = f.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, embedding_size:int, d_context=768):
        super().__init__()
        channels = n_head * embedding_size

        self.group_norm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layer_norm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layer_norm3 = nn.LayerNorm(channels)
        self.linear_geglu1 = nn.Linear(channels, 4 * channels)
        self.linear_geglu2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        """
        Feedforward function for UNet attention block

        Args:
        x (torch.Tensor): Input feature tensor (batch_size, channels, height, width)
        context (torch.Tensor): Input context tensor (batch_size, seq_len, dim)
        """

        residue_long = x

        x = self.group_norm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (batch_size, channels, height, width) -> (batch_size, height * width, channels)
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        # Normalization + self-attention with skip connection
        residue_short = x

        x = self.layer_norm1(x)
        self.attention1(x)
        x += residue_short

        # Normalization + cross attention with skip connection
        residue_short = x

        x = self.layer_norm2(x)
        self.attention2(x, context)  # cross attention
        x += residue_short

        # Normalization + feedforward with GeGLU and skip connection
        residue_short = x
        x = self.layer_norm3(x)
        x, gate = self.linear_geglu1(x).chunk(2, dim=-1)
        x = x * f.gelu(gate)
        x = self.linear_geglu2(x)
        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module([
            # (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            # (batch_size, 320, height/8, width/8) -> (batch_size, 640, height/16, width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),

            # (batch_size, 640, height/16, width/16) -> (batch_size, 1280, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),

            # (batch_size, 1280, height/32, width/32) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),
            UNet_AttentionBlock(8, 160),
            UNet_ResidualBlock(1280, 1280)
        )

        self.decoder = nn.ModuleList([
            # (batch_size, 2560, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), UpSample(1280)),
            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), UpSample(640)),
            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
        ])


class UNet_OutLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Feedforward function for UNet output layer

        Args:
        x (Tensor): Input tensor (batch_size, 320, height/8, width/8)
        """

        x = self.group_norm(x)
        x = f.silu(x)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_OutLayer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Feedforward function for diffusion model

        Args:
        latent (torch.Tensor): Latent vector (batch_size, 4, height/8, width/8)
        context (torch.Tensor): Context vector (batch_size, seq_len, dim)
        time (torch.Tensor): Time step (1, 320)
        """

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
        out = self.unet(latent, context, time)

        # (batch_size, 320, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        out = self.final(out)

        return out
