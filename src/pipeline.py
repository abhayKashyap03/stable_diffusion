import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8


def generate(prompt: str, uncond_prompt: str, inp_img=None, strength=0.8, do_cfg=True, 
             cfg_weight=7.5, sampler_name="ddpm", n_inference_steps=50, models={}, 
             seed=None, device=None, idle_device=None, tokenizer=None):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
    
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    
    clip = models["clip"]
    clip.to(device)

    if do_cfg:
        # Tokenizes the prompt
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        # Converts tokens into tensor (batch_size, seq_len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        cond_context = clip(cond_tokens)

        uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)

        # (2, seq_len, dim) = (2, 77, 768)
        context = torch.cat([cond_context, uncond_context])
    else:
        tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        context = clip(tokens)
    to_idle(clip)

    if sampler_name == "ddpm":
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(n_inference_steps)
    else:
        raise ValueError(f"Unknown sampler {sampler_name}")
    
    latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

    if inp_img:
        encoder = models["encoder"]
        encoder.to(device)

        inp_img_tensor = inp_img.resize((WIDTH, HEIGHT))
        inp_img_tensor = np.array(inp_img_tensor)
        inp_img_tensor = torch.tensor(inp_img_tensor, dtype=torch.float32, device=device)
        inp_img_tensor = rescale(inp_img_tensor, (0, 255), (-1, 1))
        inp_img_tensor = inp_img_tensor.unsqueeze(0)
        inp_img_tensor = inp_img_tensor.permute(0, 3, 1, 2)
        
        encoder_noise = torch.randn(latent_shape, generator=generator, device=device)

        latents = encoder(inp_img_tensor, encoder_noise)

        sampler.set_strength(strength=strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])

        to_idle(encoder)
    else:
        # If text-to-image, then start with random noise N(0, I)
        latents = torch.randn(latent_shape, generator=generator, device=device)

    diffusion = models["diffusion"]
    diffusion.to(device)

    timesteps = tqdm(sampler.timesteps)

    for i, timestep in enumerate(timesteps):
        time_embedding = get_time_embedding(timestep).to(device)

        model_inputs = latents
        if do_cfg:
            model_inputs = model_inputs.repeat(2, 1, 1, 1)
        
        # Predicted noise by UNet
        model_output = diffusion(model_inputs, context, time_embedding)

        if do_cfg:
            output_cond, output_uncond = model_output.chunk(2)
            model_output = cfg_weight * (output_cond - output_uncond) + output_uncond
        
        # Remove noise predicted by UNet
        latents = sampler.step(timestep, latents, model_output)
    
    to_idle(diffusion)

    decoder = models["decoder"]
    decoder.to(device)

    imgs = decoder(latents)
    to_idle(decoder)

    imgs = rescale(imgs, (-1, 1), (0, 255), clamp=True)
    imgs = imgs.permute(0, 2, 3, 1)
    imgs = imgs.to("cpu", torch.uint8).numpy()

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
