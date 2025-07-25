import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def forward_diffusion(x, t, noise_schedule, mask_token, ppl_model, ppl_tokenizer, vocab, device):
    """
    Forward diffusion qui masque les positions qui augmentent le plus la perplexité.
    Ignore les positions avec des PAD tokens qui ne peuvent pas être masquées.
    """
    B, L = x.shape
    
    # Identifier les positions masquables (pas de PAD tokens)
    maskable = (x != vocab.PAD_TOKEN)
    
    # Calculer le nombre de positions à masquer pour chaque séquence
    num_masks = []
    for b in range(B):
        ti = t[b]
        mask_prob = noise_schedule.get_noise_level(float(ti))
        # Nombre de positions masquables pour cette séquence
        num_maskable = maskable[b].sum().item()
        # Binomiale sur les positions masquables uniquement
        if num_maskable > 0:
            num_to_mask = torch.binomial(torch.tensor(num_maskable, dtype=torch.float), torch.tensor(mask_prob)).int().item()
        else:
            num_to_mask = 0
        num_masks.append(num_to_mask)
    
    # Initialiser les masques
    mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
    
    # Pour chaque séquence, trouver les positions avec la plus haute perplexité
    for b in range(B):
        if num_masks[b] == 0:
            continue
            
        # Obtenir les indices des positions masquables
        maskable_indices = torch.where(maskable[b])[0]
        
        if len(maskable_indices) == 0:
            continue
            
        # Convertir la séquence en string (seulement les positions non-PAD)
        sequence_tokens = []
        for pos in range(L):
            if maskable[b, pos]:
                sequence_tokens.append(vocab.id_to_vocab[x[b, pos].item()])
        sequence_str = ''.join(sequence_tokens)
        
        # Calculer la perplexité pour chaque position masquable
        perplexities = []
        for i, pos in enumerate(maskable_indices):
            # Masquer cette position dans la séquence string
            masked_sequence = sequence_str[:i] + ppl_tokenizer.mask_token + sequence_str[i+1:]
            
            # Évaluer avec le modèle ESM
            inputs = ppl_tokenizer(masked_sequence, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = ppl_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                esm_pos = i + 1  # Position après [CLS] dans la séquence masquable
                
                # Probabilité du token original
                original_aa = vocab.id_to_vocab[x[b, pos].item()]
                original_token_id = ppl_tokenizer.convert_tokens_to_ids(original_aa)
                original_prob = probs[0, esm_pos, original_token_id].item()
                
                # Perplexité = -log(prob)
                perplexity = -torch.log(torch.tensor(original_prob + 1e-8)).item()
                perplexities.append(perplexity)
        
        # Sélectionner les positions avec la plus haute perplexité parmi les masquables
        if len(perplexities) > 0:
            num_to_select = min(num_masks[b], len(perplexities))
            _, top_indices_in_maskable = torch.topk(torch.tensor(perplexities), num_to_select)
            # Convertir les indices relatifs aux positions masquables en indices absolus
            selected_positions = maskable_indices[top_indices_in_maskable]
            mask[b, selected_positions] = True
    
    # Appliquer les masques
    xt = x.clone()
    xt[mask] = mask_token
    
    return xt, mask



def compute_loss(model, x0, noise_schedule, vocab, ppl_model, ppl_tokenizer):
    """Calcule la loss pour un batch."""
    B, L = x0.shape
    
    # Timesteps aléatoires
    t = torch.rand(B, device=x0.device)
    
    # Forward diffusion
    xt, mask = forward_diffusion(x0, t, noise_schedule, vocab, ppl_model, ppl_tokenizer)
    
    # Prédictions du modèle en ne prenant en contexte que les positions non PAD (faut dans le forward du modele)
    logits = model(xt, t.unsqueeze(1))
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x0.device, requires_grad=True), 0.0
    
    # Loss seulement sur les positions masquées
    loss = F.cross_entropy(logits[mask], x0[mask], reduction='mean')
    mask_ratio = mask.sum().item() / mask.numel()
    
    return loss, mask_ratio


def train_step(model, batch, optimizer, noise_schedule, vocab, ppl_model, ppl_tokenizer):
    """Un pas d'entraînement."""
    model.train()
    
    loss, mask_ratio = compute_loss(model, batch, noise_schedule, vocab, ppl_model, ppl_tokenizer)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), mask_ratio


def train_model(model, dataloader, optimizer, noise_schedule, n_epochs, vocab, ppl_model, ppl_tokenizer):
    """Boucle d'entraînement complète."""
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

