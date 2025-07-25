import torch
import torch.nn.functional as F
from functools import lru_cache
import hashlib
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor
import threading
import os

# Disable Triton warnings
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'


class ESMCache:
    """Thread-safe optimized cache for ESM predictions"""
    def __init__(self, max_size=5000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.access_count = {}
    
    def _get_key(self, sequence_str):
        """Create cache key based on sequence"""
        return hashlib.md5(sequence_str.encode()).hexdigest()
    
    def get(self, sequence_str):
        """Retrieve predictions from cache with LRU"""
        key = self._get_key(sequence_str)
        with self.lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
        return None
    
    def set(self, sequence_str, probs):
        """Store predictions in cache with LRU policy"""
        key = self._get_key(sequence_str)
        with self.lock:
            if len(self.cache) >= self.max_size:
                lru_key = min(self.access_count, key=self.access_count.get)
                del self.cache[lru_key]
                del self.access_count[lru_key]
            
            self.cache[key] = probs.cpu().half()
            self.access_count[key] = 1

esm_cache = ESMCache(max_size=20000)


class VocabCache:
    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.aa_to_token_id = {}
        self._precompute_conversions()
    
    def _precompute_conversions(self):
        """Pre-compute AA -> token_id conversions"""
        for aa in self.vocab.ALPHABET:
            self.aa_to_token_id[aa] = self.tokenizer.convert_tokens_to_ids(aa)
    
    def get_token_id(self, aa):
        return self.aa_to_token_id.get(aa)


@torch.jit.script
def compute_mutation_masks(B: int, L: int, t: torch.Tensor, x: torch.Tensor, 
                          pad_token: int, device: torch.device) -> torch.Tensor:
    """JIT-compiled version of mutation mask computation"""
    mutate_probs = t.unsqueeze(1)
    random_vals = torch.rand(B, L, device=device)
    mutate = random_vals < mutate_probs
    pad_mask = (x == pad_token)
    mutate = mutate & (~pad_mask)
    return mutate


def get_esm_predictions_batch(sequences, ppl_model, ppl_tokenizer, device, 
                             cache=None, vocab_cache=None):
    """Optimized version with parallel processing and mixed precision"""
    if cache is None:
        cache = esm_cache
    
    cached_results = {}
    sequences_to_process = []
    indices_to_process = []
    
    for i, seq_str in enumerate(sequences):
        cached_probs = cache.get(seq_str)
        if cached_probs is not None:
            cached_results[i] = cached_probs.to(device).float()
        else:
            sequences_to_process.append(seq_str)
            indices_to_process.append(i)
    
    batch_results = {}
    if sequences_to_process:
        inputs = ppl_tokenizer(sequences_to_process, return_tensors="pt", 
                              padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad(), autocast():
            outputs = ppl_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        for i, (seq_idx, seq_str) in enumerate(zip(indices_to_process, sequences_to_process)):
            seq_probs = probs[i].float()
            batch_results[seq_idx] = seq_probs
            cache.set(seq_str, seq_probs)
    
    all_results = {**cached_results, **batch_results}
    return all_results


def forward_diffusion_perp_optimized(x, t, noise_schedule, vocab, device, 
                                    ppl_model, ppl_tokenizer, vocab_cache):
    """Optimized version of forward diffusion"""
    B, L = x.shape
    
    noise_levels = torch.tensor([
        noise_schedule.get_noise_level(float(t_i)) for t_i in t
    ], device=device, dtype=torch.float32)
    
    mutate = compute_mutation_masks(B, L, noise_levels, x, vocab.PAD_TOKEN, device)
    
    has_mutations = mutate.any(dim=1)
    sequences_with_mutations = torch.where(has_mutations)[0].tolist()
    
    if not sequences_with_mutations:
        return x.clone(), mutate
    
    sequence_strings = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for batch_idx in sequences_with_mutations:
            future = executor.submit(
                vocab.decode_sequence, 
                x[batch_idx].cpu().numpy()
            )
            futures.append((batch_idx, future))
        
        for batch_idx, future in futures:
            sequence_strings.append(future.result())
    
    xt = x.clone()
    esm_predictions = get_esm_predictions_batch(
        sequence_strings, ppl_model, ppl_tokenizer, device, vocab_cache=vocab_cache
    )
    
    for i, batch_idx in enumerate(sequences_with_mutations):
        probs = esm_predictions[i]
        positions = torch.where(mutate[batch_idx])[0]
        
        if positions.numel() > 0:
            esm_positions = positions + 1
            
            aa_indices = torch.tensor([
                vocab_cache.get_token_id(aa) for aa in vocab.ALPHABET
            ], device=device)
            
            aa_probs = probs[esm_positions][:, aa_indices]
            min_indices = aa_probs.argmin(dim=1)
            
            for j, pos in enumerate(positions):
                lowest_aa_idx = min_indices[j].item()
                lowest_aa = vocab.ALPHABET[lowest_aa_idx]
                xt[batch_idx, pos] = vocab.vocab_to_id[lowest_aa]
    
    return xt, mutate


def compute_loss_optimized(model, x0, noise_schedule, vocab, ppl_model, 
                          ppl_tokenizer, vocab_cache, use_amp=True):
    """Optimized version with mixed precision"""
    B, L = x0.shape
    device = x0.device
    
    t = torch.rand(B, device=device)
    
    xt, mutate_mask = forward_diffusion_perp_optimized(
        x0, t, noise_schedule, vocab, device, ppl_model, ppl_tokenizer, vocab_cache
    )
    
    n_mutations = mutate_mask.sum()
    if n_mutations == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    if use_amp:
        with autocast():
            logits = model(xt, t.unsqueeze(1))
            loss = F.cross_entropy(logits[mutate_mask], x0[mutate_mask], reduction='mean')
    else:
        logits = model(xt, t.unsqueeze(1))
        loss = F.cross_entropy(logits[mutate_mask], x0[mutate_mask], reduction='mean')
    
    mutation_ratio = n_mutations.item() / mutate_mask.numel()
    
    return loss, mutation_ratio


def train_step_optimized(model, batch, optimizer, noise_schedule, vocab, 
                        ppl_model, ppl_tokenizer, vocab_cache, scaler, use_amp=True):
    """Optimized training step with gradient accumulation and mixed precision"""
    model.train()
    
    if use_amp:
        with autocast():
            loss, mutation_ratio = compute_loss_optimized(
                model, batch, noise_schedule, vocab, ppl_model, 
                ppl_tokenizer, vocab_cache, use_amp=True
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    else:
        loss, mutation_ratio = compute_loss_optimized(
            model, batch, noise_schedule, vocab, ppl_model, 
            ppl_tokenizer, vocab_cache, use_amp=False
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item(), mutation_ratio


def train_model_optimized(model, dataloader, optimizer, noise_schedule, n_epochs, 
                         vocab, ppl_model, ppl_tokenizer, use_amp=True, 
                         gradient_accumulation_steps=1):
    """Optimized training loop with all improvements"""
    losses = []
    
    scaler = GradScaler() if use_amp else None
    vocab_cache = VocabCache(vocab, ppl_tokenizer)
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='default', backend='inductor')
    
    if hasattr(dataloader, 'pin_memory'):
        dataloader.pin_memory = True
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        epoch_losses = []
        accumulated_loss = 0
        
        for step, batch_data in enumerate(dataloader):
            batch = batch_data[0].to(next(model.parameters()).device, non_blocking=True)
            
            is_accumulation_step = (step + 1) % gradient_accumulation_steps != 0
            
            if is_accumulation_step and step < len(dataloader) - 1:
                with model.no_sync():
                    loss, _ = train_step_optimized(
                        model, batch, optimizer, noise_schedule, vocab, 
                        ppl_model, ppl_tokenizer, vocab_cache, scaler, use_amp
                    )
            else:
                loss, _ = train_step_optimized(
                    model, batch, optimizer, noise_schedule, vocab, 
                    ppl_model, ppl_tokenizer, vocab_cache, scaler, use_amp
                )
            
            accumulated_loss += loss
            
            if not is_accumulation_step or step == len(dataloader) - 1:
                avg_accumulated_loss = accumulated_loss / min(gradient_accumulation_steps, step % gradient_accumulation_steps + 1)
                epoch_losses.append(avg_accumulated_loss)
                accumulated_loss = 0
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            if epoch % 50 == 0:
                torch.cuda.empty_cache()
    
    return losses


def get_optimal_settings(batch_size=32):
    """Return optimal parameters for RTX A6000"""
    return {
        'use_amp': True,
        'gradient_accumulation_steps': 1,
        'cache_size': 20000,
        'max_workers': 8,
        'compile_mode': 'default',
    }

def train_model(model, dataloader, optimizer, noise_schedule, n_epochs, 
               vocab, ppl_model, ppl_tokenizer):
    """Wrapper to use optimized version with RTX A6000 parameters"""
    settings = get_optimal_settings()
    
    global esm_cache
    esm_cache = ESMCache(max_size=settings['cache_size'])
    
    return train_model_optimized(
        model, dataloader, optimizer, noise_schedule, n_epochs,
        vocab, ppl_model, ppl_tokenizer,
        use_amp=settings['use_amp'],
        gradient_accumulation_steps=settings['gradient_accumulation_steps']
    )