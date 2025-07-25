import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def forward_diffusion_perp_naive(x, t, noise_schedule, vocab, device, ppl_model, ppl_tokenizer):
    """
    Version avec contexte: utilise la séquence complète sans masquage
    pour obtenir les probabilités de chaque position.
    """
    B, L = x.shape
    # Initialize output tensors
    xt = x.clone()
    mutate = torch.zeros_like(x, dtype=torch.bool)
    
    # Process each sequence in the batch
    for batch_idx in range(B):
        # Get current sequence and timestep
        seq_tokens = x[batch_idx] # [L]
        timestep = t[batch_idx].item()
        
        # Calculate mutation probability for this timestep
        mutate_prob = noise_schedule.get_noise_level(timestep)
        
        # Create mutation mask for this sequence
        seq_mutate = torch.rand(L, device=device) < mutate_prob
        
        # Don't mutate PAD positions
        pad_mask = (seq_tokens == vocab.PAD_TOKEN)
        seq_mutate = seq_mutate & (~pad_mask)
        
        # Find positions to modify
        positions_to_modify = torch.where(seq_mutate)[0]
        
        if len(positions_to_modify) == 0:
            mutate[batch_idx] = seq_mutate
            continue
        
        # Convert sequence to protein string
        sequence_str = vocab.decode_sequence(seq_tokens.cpu().numpy())
        
        # Get ESM predictions for the COMPLETE unmasked sequence
        inputs = ppl_tokenizer(sequence_str, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ppl_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Start with original sequence
        current_seq_tokens = seq_tokens.clone()
        
        # Process each position to modify and assign lowest probability amino acid
        for pos in positions_to_modify:
            # Account for [CLS] token in ESM (position shift by 1)
            esm_pos = pos + 1
            
            # Get probabilities for valid amino acids only
            aa_probs = {}
            for aa in vocab.ALPHABET:
                aa_token_id = ppl_tokenizer.convert_tokens_to_ids(aa)
                aa_probs[aa] = probs[0, esm_pos, aa_token_id].item()
            
            # Find amino acid with lowest probability
            lowest_prob_aa = min(aa_probs, key=aa_probs.get)
            
            # Apply mutation
            current_seq_tokens[pos] = vocab.vocab_to_id[lowest_prob_aa]
        
        # Copy final mutated sequence to output
        xt[batch_idx] = current_seq_tokens
        
        # Store mutation mask
        mutate[batch_idx] = seq_mutate
    
    return xt, mutate


def forward_diffusion_perp_naive_mutation_masked(x, t, noise_schedule, vocab, device, ppl_model, ppl_tokenizer):
    """
    Naive forward diffusion: replaces all positions to mutate simultaneously
    with the least probable tokens using parallel masking.
    """
    B, L = x.shape
    
    # Initialize output tensors
    xt = x.clone()
    mutate = torch.zeros_like(x, dtype=torch.bool)
    
    # Process each sequence in the batch
    for batch_idx in range(B):
        # Get current sequence and timestep
        seq_tokens = x[batch_idx]  # [L]
        timestep = t[batch_idx].item()
        
        # Calculate mutation probability for this timestep
        mutate_prob = noise_schedule.get_noise_level(timestep)
        
        # Create mutation mask for this sequence
        seq_mutate = torch.rand(L, device=device) < mutate_prob
        
        # Don't mutate PAD positions
        pad_mask = (seq_tokens == vocab.PAD_TOKEN)
        seq_mutate = seq_mutate & (~pad_mask)
        
        # Find positions to modify
        positions_to_modify = torch.where(seq_mutate)[0]
        
        if len(positions_to_modify) == 0:
            mutate[batch_idx] = seq_mutate
            continue
        
        # Convert sequence to protein string
        sequence_str = vocab.decode_sequence(seq_tokens.cpu().numpy())
        
        # Create masked sequence with ALL positions to mutate masked simultaneously
        masked_sequence = list(sequence_str)
        for pos in positions_to_modify:
            masked_sequence[pos] = ppl_tokenizer.mask_token
        masked_sequence_str = ''.join(masked_sequence)
        
        # Get ESM predictions for the multiply-masked sequence
        inputs = ppl_tokenizer(masked_sequence_str, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ppl_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Start with original sequence
        current_seq_tokens = seq_tokens.clone()
        
        # Process each masked position and assign lowest probability amino acid
        for pos in positions_to_modify:
            # Account for [CLS] token in ESM (position shift by 1)
            esm_pos = pos + 1
            
            # Get probabilities for valid amino acids only
            aa_probs = {}
            for aa in vocab.ALPHABET:
                aa_token_id = ppl_tokenizer.convert_tokens_to_ids(aa)
                aa_probs[aa] = probs[0, esm_pos, aa_token_id].item()
            
            # Find amino acid with lowest probability
            lowest_prob_aa = min(aa_probs, key=aa_probs.get)
            
            # Apply mutation
            current_seq_tokens[pos] = vocab.vocab_to_id[lowest_prob_aa]
        
        # Copy final mutated sequence to output
        xt[batch_idx] = current_seq_tokens
        
        # Store mutation mask
        mutate[batch_idx] = seq_mutate
    
    return xt, mutate

