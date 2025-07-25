import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

    

def forward_diffusion(x, t, noise_schedule, vocab, device):
    """Forward diffusion by random mutation of amino acids."""
    B, L = x.shape
    
    # Calculate mutation probabilities for each timestep in the batch
    mutate_probs = torch.tensor([
        noise_schedule.get_noise_level(float(t_i)) for t_i in t
    ], device=device).unsqueeze(1)  # [B, 1]
    
    # Generate mutation masks for the entire batch
    random_vals = torch.rand(B, L, device=device)
    mutate = random_vals < mutate_probs  # [B, L]
    
    # Don't mutate PAD positions
    pad_mask = (x == vocab.PAD_TOKEN)
    mutate = mutate & (~pad_mask)
    
    # Clone input sequences
    xt = x.clone()
    
    # Find all positions to modify in the batch
    batch_indices, pos_indices = torch.where(mutate)
    
    if len(batch_indices) == 0:
        return xt, mutate
    
    # Generate random amino acids for all positions to modify
    num_mutations = len(batch_indices)
    random_aa_indices = torch.randint(
        0, len(vocab.ALPHABET), (num_mutations,), device=device
    )
    
    # Apply mutations in a single vectorized operation
    xt[batch_indices, pos_indices] = random_aa_indices
    
    return xt, mutate




def compute_loss(model, x0, noise_schedule, vocab):
    """Compute loss for a batch using diffusion by noising."""
    B, L = x0.shape
    device = x0.device
    
    # Random timesteps
    t = torch.rand(B, device=device)
    
    # Forward diffusion (noising)
    xt, mutate_mask = forward_diffusion(x0, t, noise_schedule, vocab, device)
    
    # Model predictions
    logits = model(xt, t.unsqueeze(1))
    
    # Calculate loss only on positions that were mutated
    if mutate_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # Cross-entropy loss on mutated positions
    loss = F.cross_entropy(logits[mutate_mask], x0[mutate_mask], reduction='mean')
    mutation_ratio = mutate_mask.sum().item() / mutate_mask.numel()
    
    return loss, mutation_ratio


def train_step(model, batch, optimizer, noise_schedule, vocab):
    """One training step."""
    model.train()
    
    loss, mutation_ratio = compute_loss(model, batch, noise_schedule, vocab)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), mutation_ratio


def train_model(model, dataloader, optimizer, noise_schedule, n_epochs, vocab):
    """Complete training loop."""
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