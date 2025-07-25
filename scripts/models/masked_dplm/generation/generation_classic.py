import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# TODO faire que combinaison choisie et pas logits independant

@torch.no_grad()
def denoise_step(model, x, noise_schedule, vocab, t_current, t_next):
    """ pas besoin de mask d'attention car aps de PAD dans la génération"""
    B, L = x.shape
    
    noise_curr = noise_schedule.get_noise_level(t_current)
    noise_next = noise_schedule.get_noise_level(t_next)
    
    if noise_curr > 1e-6:
        reveal_prob = (noise_curr - noise_next) / noise_curr
        # j'ai bien vérifié avec ma feuille de calculs prise en photo
    else:
        reveal_prob = 1.0
    
    # Prédictions du modèle
    t_tensor = torch.full((B, 1), t_current, device=x.device)
    logits = model(x, t_tensor)
    probs = F.softmax(logits, dim=-1)
    
    # Positions actuellement masquées
    mask_pos = (x == vocab.MASK_TOKEN)
    
    # Décider quelles positions révéler
    reveal_mask = (torch.rand(B, L, device=x.device) < reveal_prob) & mask_pos
    
    # Échantillonner de nouveaux tokens
    x_new = x.clone()
    if reveal_mask.any():
        samples = torch.multinomial(probs[reveal_mask], 1).squeeze(-1)
        x_new[reveal_mask] = samples
    
    return x_new


@torch.no_grad()
def generate_sequences(model, vocab, n_samples, seq_length, noise_schedule, dt):
    """Génère des séquences par débruitage itératif.
    pas besoin de mask d'attention car pas de PAD dans la génaration"""
    model.eval()

    device = next(model.parameters()).device
    x = torch.full((n_samples, seq_length), vocab.MASK_TOKEN, dtype=torch.long, device=device)
    
    # Débruitage itératif
    t = 1.0
    while t > 0:
        t_next = max(t - dt, 0.0)
        x = denoise_step(model, x, noise_schedule, vocab, t, t_next)
        t = t_next
    
    # Nettoyer les masques restants
    if (x == vocab.MASK_TOKEN).any():
        t_tensor = torch.zeros((n_samples, 1), device=x.device)
        logits = model(x, t_tensor)
        probs = F.softmax(logits, dim=-1)
        mask_pos = (x == vocab.MASK_TOKEN)
        if mask_pos.any():
            samples = torch.multinomial(probs[mask_pos], 1).squeeze(-1)
            x[mask_pos] = samples
    
    return x
