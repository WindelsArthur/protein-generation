import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def forward_diffusion(x, t, noise_schedule, vocab):
    B, L = x.shape
    
    # Masque pour identifier les tokens qui peuvent être masqués
    maskable = (x != vocab.PAD_TOKEN)
    
    # Probabilité de masquage pour chaque séquence
    mask_probs = torch.stack([
        torch.tensor(noise_schedule.get_noise_level(float(ti)), device=x.device) for ti in t
    ]).view(B, 1).expand(B, L)
    
    # Masquage aléatoire seulement sur les positions maskables
    mask = (torch.rand(B, L, device=x.device) < mask_probs) & maskable
    
    # Appliquer les masques
    xt = x.clone()
    xt[mask] = vocab.MASK_TOKEN
    
    return xt, mask

def compute_loss(model, x0, noise_schedule, vocab):
    """Calcule la loss pour un batch."""
    B, L = x0.shape
    
    # Timesteps aléatoires
    t = torch.rand(B, device=x0.device)
    
    # Forward diffusion
    xt, mask = forward_diffusion(x0, t, noise_schedule, vocab)
    
    # Prédictions du modèle en ne prenant en contexte que les positions non PAD (faut dans le forward du modele)
    logits = model(xt, t.unsqueeze(1))
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x0.device, requires_grad=True), 0.0
    
    # Loss seulement sur les positions masquées
    loss = F.cross_entropy(logits[mask], x0[mask], reduction='mean')
    mask_ratio = mask.sum().item() / mask.numel()
    
    return loss, mask_ratio


def train_step(model, batch, optimizer, noise_schedule, vocab):
    """Un pas d'entraînement."""
    model.train()
    
    loss, mask_ratio = compute_loss(model, batch, noise_schedule, vocab)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), mask_ratio


def train_model(model, dataloader, optimizer, noise_schedule, n_epochs, vocab):
    """Boucle d'entraînement complète."""
    losses = []
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        epoch_losses = []
        
        for batch_data in dataloader:
            batch = batch_data[0].to(next(model.parameters()).device)
            loss, _ = train_step(model, batch, optimizer, noise_schedule, vocab)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return losses

