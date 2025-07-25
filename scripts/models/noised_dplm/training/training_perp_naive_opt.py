import torch
import torch.nn.functional as F
from functools import lru_cache
import hashlib
import numpy as np
from tqdm import tqdm

class ESMCache:
    """Simple cache for ESM predictions"""
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def _get_key(self, sequence_str):
        """Create cache key based on sequence"""
        return hashlib.md5(sequence_str.encode()).hexdigest()
    
    def get(self, sequence_str):
        """Retrieve predictions from cache"""
        key = self._get_key(sequence_str)
        return self.cache.get(key, None)
    
    def set(self, sequence_str, probs):
        """Store predictions in cache"""
        if len(self.cache) >= self.max_size:
            # Simple FIFO: remove first element
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        key = self._get_key(sequence_str)
        self.cache[key] = probs.clone()

# Global cache
esm_cache = ESMCache(max_size=2000)

def get_esm_predictions_batch(sequences, ppl_model, ppl_tokenizer, device, cache=None):
    """
    Get ESM predictions for a batch of sequences with caching
    """
    if cache is None:
        cache = esm_cache
    
    # Separate cached and non-cached sequences
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
    
    # Process non-cached sequences in batch
    batch_results = {}
    if sequences_to_process:
        inputs = ppl_tokenizer(sequences_to_process, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ppl_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Store results and cache them
        for i, (seq_idx, seq_str) in enumerate(zip(indices_to_process, sequences_to_process)):
            seq_probs = probs[i]
            batch_results[seq_idx] = seq_probs
            cache.set(seq_str, seq_probs.cpu())
    
    # Combine cached and new results
    all_results = {**cached_results, **batch_results}
    return all_results

def forward_diffusion_perp_naive(x, t, noise_schedule, vocab, device, ppl_model, ppl_tokenizer):
    B, L = x.shape
    
    # Calculate all mutation masks
    mutate_probs = torch.tensor([
        noise_schedule.get_noise_level(float(t_i)) for t_i in t
    ], device=device).unsqueeze(1)
    
    random_vals = torch.rand(B, L, device=device)
    mutate = random_vals < mutate_probs
    
    pad_mask = (x == vocab.PAD_TOKEN)
    mutate = mutate & (~pad_mask)
    
    # Identify sequences with mutations
    sequences_with_mutations = []
    sequence_strings = []
    
    for batch_idx in range(B):
        if mutate[batch_idx].any():
            seq_str = vocab.decode_sequence(x[batch_idx].cpu().numpy())
            sequences_with_mutations.append(batch_idx)
            sequence_strings.append(seq_str)
    
    # Process all sequences with mutations at once
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
                
                # Find AA with lowest probability
                aa_probs = {
                    aa: probs[esm_pos, ppl_tokenizer.convert_tokens_to_ids(aa)].item()
                    for aa in vocab.ALPHABET
                }
                
                lowest_prob_aa = min(aa_probs, key=aa_probs.get)
                xt[batch_idx, pos] = vocab.vocab_to_id[lowest_prob_aa]
    
    return xt, mutate

def compute_loss(model, x0, noise_schedule, vocab, ppl_model, ppl_tokenizer):
    """Modified compute_loss using ESM diffusion"""
    B, L = x0.shape
    device = x0.device
    
    t = torch.rand(B, device=device)
    
    # Forward diffusion with ESM (optimized version)
    xt, mutate_mask = forward_diffusion_perp_naive(
        x0, t, noise_schedule, vocab, device, ppl_model, ppl_tokenizer
    )
    
    logits = model(xt, t.unsqueeze(1))
    
    if mutate_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    # Cross-entropy loss on mutated positions
    loss = F.cross_entropy(logits[mutate_mask], x0[mutate_mask], reduction='mean')
    mutation_ratio = mutate_mask.sum().item() / mutate_mask.numel()
    
    return loss, mutation_ratio

def train_step(model, batch, optimizer, noise_schedule, vocab, ppl_model, ppl_tokenizer):
    """Single training step"""
    model.train()
    
    loss, mutation_ratio = compute_loss(model, batch, noise_schedule, vocab, ppl_model, ppl_tokenizer)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), mutation_ratio

def train_model(model, dataloader, optimizer, noise_schedule, n_epochs, vocab, ppl_model, ppl_tokenizer):
    """Complete training loop"""
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