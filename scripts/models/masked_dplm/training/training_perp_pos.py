import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def forward_diffusion(x, t, noise_schedule, mask_token, ppl_model, ppl_tokenizer, vocab, device):
    """
    Forward diffusion that masks positions with highest perplexity increase.
    Ignores PAD token positions which cannot be masked.
    """
    B, L = x.shape
    
    # Identify maskable positions (no PAD tokens)
    maskable = (x != vocab.PAD_TOKEN)
    
    # Calculate number of positions to mask for each sequence
    num_masks = []
    for b in range(B):
        ti = t[b]
        mask_prob = noise_schedule.get_noise_level(float(ti))
        num_maskable = maskable[b].sum().item()
        
        if num_maskable > 0:
            num_to_mask = torch.binomial(torch.tensor(num_maskable, dtype=torch.float), torch.tensor(mask_prob)).int().item()
        else:
            num_to_mask = 0
        num_masks.append(num_to_mask)
    
    mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    
    # Find positions with highest perplexity for each sequence
    for b in range(B):
        if num_masks[b] == 0:
            continue
            
        maskable_indices = torch.where(maskable[b])[0]
        
        if len(maskable_indices) == 0:
            continue
            
        # Convert sequence to string (non-PAD positions only)
        sequence_tokens = []
        for pos in range(L):
            if maskable[b, pos]:
                sequence_tokens.append(vocab.id_to_vocab[x[b, pos].item()])
        sequence_str = ''.join(sequence_tokens)
        
        # Calculate perplexity for each maskable position
        perplexities = []
        for i, pos in enumerate(maskable_indices):
            # Mask this position in the string sequence
            masked_sequence = sequence_str[:i] + ppl_tokenizer.mask_token + sequence_str[i+1:]
            
            # Evaluate with ESM model
            inputs = ppl_tokenizer(masked_sequence, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = ppl_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                esm_pos = i + 1  # Position after [CLS] in maskable sequence
                
                # Probability of original token
                original_aa = vocab.id_to_vocab[x[b, pos].item()]
                original_token_id = ppl_tokenizer.convert_tokens_to_ids(original_aa)
                original_prob = probs[0, esm_pos, original_token_id].item()
                
                # Perplexity = -log(prob)
                perplexity = -torch.log(torch.tensor(original_prob + 1e-8)).item()
                perplexities.append(perplexity)
        
        # Select positions with highest perplexity among maskable ones
        if len(perplexities) > 0:
            num_to_select = min(num_masks[b], len(perplexities))
            _, top_indices_in_maskable = torch.topk(torch.tensor(perplexities), num_to_select)
            # Convert relative indices to absolute positions
            selected_positions = maskable_indices[top_indices_in_maskable]
            mask[b, selected_positions] = True
    
    xt = x.clone()
    xt[mask] = mask_token
    
    return xt, mask


def compute_loss(model, x0, noise_schedule, vocab, ppl_model, ppl_tokenizer):
    """Compute loss for a batch"""
    B, L = x0.shape
    
    t = torch.rand(B, device=x0.device)
    xt, mask = forward_diffusion(x0, t, noise_schedule, vocab, ppl_model, ppl_tokenizer)
    
    # Model predictions (forward should handle non-PAD positions)
    logits = model(xt, t.unsqueeze(1))
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x0.device, requires_grad=True), 0.0
    
    # Loss only on masked positions
    loss = F.cross_entropy(logits[mask], x0[mask], reduction='mean')
    mask_ratio = mask.sum().item() / mask.numel()
    
    return loss, mask_ratio


def train_step(model, batch, optimizer, noise_schedule, vocab, ppl_model, ppl_tokenizer):
    """Single training step"""
    model.train()
    
    loss, mask_ratio = compute_loss(model, batch, noise_schedule, vocab, ppl_model, ppl_tokenizer)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), mask_ratio


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