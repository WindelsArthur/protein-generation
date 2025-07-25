import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def denoise_step(model, x, noise_schedule, t_current, t_next):
    """A denoising step with selective correction."""
    B, L = x.shape
    noise_curr = noise_schedule.get_noise_level(t_current)
    noise_next = noise_schedule.get_noise_level(t_next)
    
    if noise_curr > 1e-6:
        correct_prob = (noise_curr - noise_next) / noise_curr
    else:
        correct_prob = 1.0
    
    # Model predictions
    t_tensor = torch.full((B, 1), t_current, device=x.device)
    logits = model(x, t_tensor)
    probs = F.softmax(logits, dim=-1)
    
    # Decide which positions to correct
    correct_mask = (torch.rand(B, L, device=x.device) < correct_prob)
    
    # Correct selected positions
    x_new = x.clone()
    if correct_mask.any():
        # Sample from model predictions
        samples = torch.multinomial(probs[correct_mask], 1).squeeze(-1)
        x_new[correct_mask] = samples
    
    return x_new


@torch.no_grad()
def generate_sequences(model, vocab, n_samples, seq_length, noise_schedule, dt):
    """Generates sequences through iterative denoising."""
    model.eval()
    device = next(model.parameters()).device
    
    # Start with completely random sequences (except PAD)
    x = torch.randint(0, len(vocab.ALPHABET), (n_samples, seq_length), device=device)
    
    # Iterative denoising
    t = 1.0
    while t > 0:
        t_next = max(t - dt, 0.0)
        x= denoise_step(model, x, noise_schedule, t, t_next)
        t = t_next
    
    return x