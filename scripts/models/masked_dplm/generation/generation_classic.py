import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# TODO: make combination chosen instead of independent logits

@torch.no_grad()
def denoise_step(model, x, noise_schedule, vocab, t_current, t_next):
    """No attention mask needed as no PAD tokens in generation"""
    B, L = x.shape
    noise_curr = noise_schedule.get_noise_level(t_current)
    noise_next = noise_schedule.get_noise_level(t_next)
    
    if noise_curr > 1e-6:
        reveal_prob = (noise_curr - noise_next) / noise_curr
    else:
        reveal_prob = 1.0
    
    # Model predictions
    t_tensor = torch.full((B, 1), t_current, device=x.device)
    logits = model(x, t_tensor)
    probs = F.softmax(logits, dim=-1)
    
    # Currently masked positions
    mask_pos = (x == vocab.MASK_TOKEN)
    
    # Decide which positions to reveal
    reveal_mask = (torch.rand(B, L, device=x.device) < reveal_prob) & mask_pos
    
    # Sample new tokens
    x_new = x.clone()
    if reveal_mask.any():
        samples = torch.multinomial(probs[reveal_mask], 1).squeeze(-1)
        x_new[reveal_mask] = samples
    
    return x_new

@torch.no_grad()
def generate_sequences(model, vocab, n_samples, seq_length, noise_schedule, dt):
    """Generate sequences through iterative denoising"""
    model.eval()
    device = next(model.parameters()).device
    
    x = torch.full((n_samples, seq_length), vocab.MASK_TOKEN, dtype=torch.long, device=device)
    
    # Iterative denoising
    t = 1.0
    while t > 0:
        t_next = max(t - dt, 0.0)
        x = denoise_step(model, x, noise_schedule, vocab, t, t_next)
        t = t_next
    
    # Clean remaining masks
    if (x == vocab.MASK_TOKEN).any():
        t_tensor = torch.zeros((n_samples, 1), device=x.device)
        logits = model(x, t_tensor)
        probs = F.softmax(logits, dim=-1)
        
        mask_pos = (x == vocab.MASK_TOKEN)
        if mask_pos.any():
            samples = torch.multinomial(probs[mask_pos], 1).squeeze(-1)
            x[mask_pos] = samples
    
    return x