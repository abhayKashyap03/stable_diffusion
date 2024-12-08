import torch
import numpy as np


class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
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
        # Eqn 4 of DDPM paper & distribution transformation
        noisy_sample = (sqrt_alpha_cumprod * original) + (sqrt_one_minus_alpha_cumprod * noise)

        return noisy_sample

    def _get_prev_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.num_training_steps // self.num_inference_steps
        return prev_t

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """
        Something here
        Eqn 11 in paper describes backward process essentially - how to retrieve image with less noise than current image
        Also see `algorithm 2 - sampling`
        """
        t = timestep
        prev_t = self._get_prev_timestep(t)
        
        alpha_prod_t = self.alphas_cumprod[t]  # Alpha cumulative till timestep t
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t  # Beta cumulative till timestep t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t  # Value of alpha at timestep t
        current_beta_t = 1 - current_alpha_t  # Value of beta at timestep t

        # Calculate predicted original sample - eqn 15
        pred_orig_sample = (latents - (beta_prod_t ** 0.5)*model_output) / (alpha_prod_t ** 0.5)

        # Eqn 6/7 for less-noisy ("previous") sample
        pred_orig_sample_coeff = (alpha_prod_prev_t ** 0.5) * current_beta_t / beta_prod_t
        curr_sample_coeff = (current_alpha_t ** 0.5) * beta_prod_prev_t / beta_prod_t

        pred_prev_sample = (pred_orig_sample_coeff * pred_orig_sample) + (curr_sample_coeff * latents)

        variance = 0
        if t > 0:
            variance = beta_prod_prev_t * current_beta_t / beta_prod_t
            variance = torch.clamp(variance, min=1e-20)
            noise = torch.randn(model_output.shape, generator=self.generator, device=model_output.device, dtype=model_output.dtype)
            variance = (noise * (variance ** 0.5))
        
        # Distribution transformation
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def set_strength(self, strength: int = 1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step