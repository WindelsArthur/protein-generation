import torch
import torch.nn.functional as F
from functools import lru_cache
import hashlib
import numpy as np
from tqdm import tqdm


class ESMCache:
    """Simple cache pour les prédictions ESM"""
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def _get_key(self, sequence_str):
        """Crée une clé de cache basée sur la séquence"""
        return hashlib.md5(sequence_str.encode()).hexdigest()
    
    def get(self, sequence_str):
        """Récupère les prédictions du cache"""
        key = self._get_key(sequence_str)
        return self.cache.get(key, None)
    
    def set(self, sequence_str, probs):
        """Stocke les prédictions dans le cache"""
        if len(self.cache) >= self.max_size:
            # Simple FIFO: supprime le premier élément
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        key = self._get_key(sequence_str)
        self.cache[key] = probs.clone()

# Cache global (simple mais efficace)
esm_cache = ESMCache(max_size=2000)



def get_esm_predictions_batch(sequences, ppl_model, ppl_tokenizer, device, cache=None):
    """
    Obtient les prédictions ESM pour un batch de séquences avec cache
    """
    if cache is None:
        cache = esm_cache
    
    # Séparer les séquences cachées des non-cachées
    cached_results = {}
    sequences_to_process = []
    indices_to_process = []
    
    for i, seq_str in enumerate(sequences):
        cached_probs = cache.get(seq_str)
        if cached_probs is not None:
            cached_results[i] = cached_probs
        else:
            sequences_to_process.append(seq_str)
            indices_to_process.append(i)
    
    # Traiter les séquences non-cachées en batch
    batch_results = {}
    if sequences_to_process:
        # Tokenize toutes les séquences en une fois
        inputs = ppl_tokenizer(sequences_to_process, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ppl_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Stocker les résultats et les mettre en cache
        for i, (seq_idx, seq_str) in enumerate(zip(indices_to_process, sequences_to_process)):
            seq_probs = probs[i]  # [seq_len, vocab_size]
            batch_results[seq_idx] = seq_probs
            cache.set(seq_str, seq_probs.cpu())
    
    # Combiner résultats cachés et nouveaux
    all_results = {**cached_results, **batch_results}
    return all_results


def forward_diffusion_perp_naive(x, t, noise_schedule, vocab, device, ppl_model, ppl_tokenizer):
    B, L = x.shape
    
    # Calculer tous les masques de mutation
    mutate_probs = torch.tensor([
        noise_schedule.get_noise_level(float(t_i)) for t_i in t
    ], device=device).unsqueeze(1)
    
    random_vals = torch.rand(B, L, device=device)
    mutate = random_vals < mutate_probs
    pad_mask = (x == vocab.PAD_TOKEN)
    mutate = mutate & (~pad_mask)
    
    # Identifier les séquences qui ont des mutations
    sequences_with_mutations = []
    sequence_strings = []
    
    for batch_idx in range(B):
        if mutate[batch_idx].any():
            seq_str = vocab.decode_sequence(x[batch_idx].cpu().numpy())
            sequences_with_mutations.append(batch_idx)
            sequence_strings.append(seq_str)
    
    # Traiter toutes les séquences avec mutations en une fois
    xt = x.clone()
    
    if sequence_strings:
        esm_predictions = get_esm_predictions_batch(
            sequence_strings, ppl_model, ppl_tokenizer, device
        )
        
        for i, batch_idx in enumerate(sequences_with_mutations):
            probs = esm_predictions[i].to(device)
            positions_to_modify = torch.where(mutate[batch_idx])[0]
            
            for pos in positions_to_modify:
                esm_pos = pos + 1
                
                # Trouver l'AA avec la plus faible probabilité
                aa_probs = {
                    aa: probs[esm_pos, ppl_tokenizer.convert_tokens_to_ids(aa)].item()
                    for aa in vocab.ALPHABET
                }
                
                lowest_prob_aa = min(aa_probs, key=aa_probs.get)
                xt[batch_idx, pos] = vocab.vocab_to_id[lowest_prob_aa]
    
    return xt, mutate


# Fonction de training modifiée pour utiliser la version optimisée
def compute_loss(model, x0, noise_schedule, vocab, ppl_model, ppl_tokenizer):
    """Version modifiée de compute_loss utilisant ESM diffusion"""
    B, L = x0.shape
    device = x0.device
    
    # Random timesteps
    t = torch.rand(B, device=device)
    
    # Forward diffusion avec ESM (version optimisée)
    xt, mutate_mask = forward_diffusion_perp_naive(
        x0, t, noise_schedule, vocab, device, ppl_model, ppl_tokenizer
    )
    
    # Model predictions
    logits = model(xt, t.unsqueeze(1))
    
    # Calculate loss only on positions that were mutated
    if mutate_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # Cross-entropy loss on mutated positions
    loss = F.cross_entropy(logits[mutate_mask], x0[mutate_mask], reduction='mean')
    mutation_ratio = mutate_mask.sum().item() / mutate_mask.numel()
    
    return loss, mutation_ratio

def train_step(model, batch, optimizer, noise_schedule, vocab, ppl_model, ppl_tokenizer):
    """One training step."""
    model.train()
    
    loss, mutation_ratio = compute_loss(model, batch, noise_schedule, vocab, ppl_model, ppl_tokenizer)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), mutation_ratio


def train_model(model, dataloader, optimizer, noise_schedule, n_epochs, vocab, ppl_model, ppl_tokenizer):
    """Complete training loop."""
    losses = []
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        epoch_losses = []
        
        for batch_data in dataloader:
            batch = batch_data[0].to(next(model.parameters()).device)
            loss, _ = train_step(model, batch, optimizer, noise_schedule, vocab, ppl_model, ppl_tokenizer)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return losses