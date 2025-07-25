import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def evaluate_combination(seq_tokens, positions, combination, vocab, ppl_model, ppl_tokenizer, device):
    """
    Evaluate perplexity of a specific combination at given positions
    """
    test_seq_tokens = seq_tokens.clone()
    for pos, aa in zip(positions, combination):
        test_seq_tokens[pos] = vocab.vocab_to_id[aa]
    
    log_prob_total = 0.0
    
    for i, pos in enumerate(positions):
        sequence_str = vocab.decode_sequence(test_seq_tokens.cpu().numpy())
        
        # Mask position to evaluate
        masked_sequence = sequence_str[:pos] + ppl_tokenizer.mask_token + sequence_str[pos+1:]
        
        inputs = ppl_tokenizer(masked_sequence, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ppl_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        esm_pos = pos + 1  # Account for [CLS] token
        
        aa_token_id = ppl_tokenizer.convert_tokens_to_ids(combination[i])
        prob = probs[0, esm_pos, aa_token_id].item()
        log_prob_total += torch.log(torch.tensor(prob))
    
    perplexity = torch.exp(-log_prob_total / len(positions))
    return perplexity.item()

def find_worst_combination(seq_tokens, positions_to_modify, vocab, ppl_model, ppl_tokenizer, device):
    """
    Test all possible combinations and return the one with maximum perplexity
    """
    from itertools import product
    
    all_combinations = list(product(vocab.ALPHABET, repeat=len(positions_to_modify)))
    
    best_combo = None
    max_perplexity = -1
    
    for combo in all_combinations:
        perp = evaluate_combination(seq_tokens, positions_to_modify, combo,
                                   vocab, ppl_model, ppl_tokenizer, device)
        if perp > max_perplexity:
            max_perplexity = perp
            best_combo = combo
    
    return best_combo, max_perplexity

def forward_diffusion_perp_comb(x, t, noise_schedule, vocab, device, ppl_model, ppl_tokenizer):
    """
    Forward diffusion that chooses combination maximizing perplexity
    """
    B, L = x.shape
    xt = x.clone()
    mutate = torch.zeros_like(x, dtype=torch.bool)
    
    for batch_idx in range(B):
        seq_tokens = x[batch_idx]
        timestep = t[batch_idx].item()
        
        mutate_prob = noise_schedule.get_noise_level(timestep)
        seq_mutate = torch.rand(L, device=device) < mutate_prob
        
        pad_mask = (seq_tokens == vocab.PAD_TOKEN)
        seq_mutate = seq_mutate & (~pad_mask)
        
        positions_to_modify = torch.where(seq_mutate)[0].tolist()
        
        if len(positions_to_modify) == 0:
            mutate[batch_idx] = seq_mutate
            continue
        
        # Find worst combination (highest perplexity)
        worst_combo, max_perp = find_worst_combination(
            seq_tokens, positions_to_modify, vocab, ppl_model, ppl_tokenizer, device
        )
        
        # Apply worst combination
        for pos, aa in zip(positions_to_modify, worst_combo):
            xt[batch_idx, pos] = vocab.vocab_to_id[aa]
        
        mutate[batch_idx] = seq_mutate
    
    return xt, mutate