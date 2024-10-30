import torch
import numpy as np


class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, 
                 beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps / self.num_inference_steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def add_noise(self, original: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alphas_cumprod.to(device=original.device, dtype=original.dtype)
        timesteps = timesteps.to(original.device)

        sqrt_alpha_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()

        while len(sqrt_alpha_cumprod.shape) < len(original.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()

        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        noise = torch.randn(original.shape, generator=self.generator, device=original.device, dtype=original.dtype)
        noisy_sample = (sqrt_alpha_cumprod * original) + (sqrt_one_minus_alpha_cumprod * noise)

        return noisy_sample

    def set_strength(self, strength: int):
        pass

    def step(self):
        pass
