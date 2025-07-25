# Core imports
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import random
import time
import math
import warnings
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
import statistics as stats
import faiss #type: ignore
import numpy as np
import torch
from tqdm import tqdm
from typing import List
from typing import Union, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
from typing import Tuple, Optional

# Scientific computing
from scipy.spatial import ConvexHull
from scipy.linalg import eigvals, eig
from scipy.stats import entropy
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from sklearn.neighbors import KernelDensity

from top_pr import compute_top_pr as TopPR #type: ignore

# Hugging Face transformers
from transformers import EsmTokenizer, EsmForMaskedLM, EsmForProteinFolding #type: ignore

# External soft alignment
import sys, pathlib, os
project_root = pathlib.Path.home() / "projets" / "protein-generation"
sys.path.append(str(project_root))

from external.protein_embed_softalign.soft_align import soft_align

def load_perplexity_model(
    ppl_model_name: str, 
    device: str = "cuda"
) -> Tuple[EsmForMaskedLM, EsmTokenizer]:
    print("Loading perplexity model...")
    
    # Load tokenizer and model from Hugging Face
    ppl_tokenizer = EsmTokenizer.from_pretrained(ppl_model_name)
    ppl_model = EsmForMaskedLM.from_pretrained(ppl_model_name)
    
    # Set model to evaluation mode and move to specified device
    ppl_model.eval()
    ppl_model.to(device)
    
    print("✓ Perplexity model loaded")
    return ppl_model, ppl_tokenizer


def load_folding_model(
    fold_model_name: str, 
    device: str
) -> Tuple[EsmForProteinFolding, EsmTokenizer]:
    print("Loading folding model...")
    
    # Load tokenizer and folding model from Hugging Face
    fold_tokenizer = EsmTokenizer.from_pretrained(fold_model_name)
    fold_model = EsmForProteinFolding.from_pretrained(fold_model_name)
    
    # Set model to evaluation mode and move to specified device
    fold_model.eval()
    fold_model.to(device)
    
    print("✓ Folding model loaded")
    return fold_model, fold_tokenizer


def get_sequence_embeddings(
    sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: Union[str, torch.device],
    show_progress: bool = True
) -> np.ndarray:
    embeddings = []
    
    # Setup iterator with or without progress bar
    iterator = tqdm(sequences, desc="Extracting embeddings") if show_progress else sequences
    
    for seq in iterator:
        # Tokenize the protein sequence
        inputs = ppl_tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get model outputs with hidden states from all layers
            outputs = ppl_model(**inputs, output_hidden_states=True)
            
            # Extract layer 6 hidden states (0-indexed, so layer 6 is index 6)
            # Shape: [batch_size, seq_len, hidden_dim]
            hidden_states = outputs.hidden_states[6]
            
            # Get actual sequence length (excluding special tokens)
            seq_len = len(seq)
            
            # Extract embeddings for the actual sequence (excluding [CLS] and [SEP] tokens)
            # [CLS] is at position 0, sequence starts at position 1
            seq_embedding = hidden_states[0, 1:seq_len+1].mean(dim=0).cpu().numpy()
            
            embeddings.append(seq_embedding)
    
    return np.array(embeddings)


def calculate_plddt(seq: str, fold_model, fold_tokenizer, device="cuda") -> float:
    inputs = fold_tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        plddt = fold_model(**inputs).plddt[0].mul_(100).mean()
    
    return float(plddt)

def calculate_plddt_with_residue(
    seq: str,
    fold_model: EsmForProteinFolding,
    fold_tokenizer: EsmTokenizer,
    device: Union[torch.device, str] = "cuda"
) -> Tuple[float, torch.Tensor]:
    # Tokenize sequence without special tokens for folding
    inputs = fold_tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Get folding model outputs including pLDDT predictions
        outputs = fold_model(**inputs)

    # Extract pLDDT scores and scale to 0-100 range
    plddt_per_residue = outputs.plddt[0].mul_(100).cpu()
    plddt_mean = float(plddt_per_residue.mean())

    return plddt_mean, plddt_per_residue


def calculate_perplexity(
    seq: str,
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: Union[torch.device, str] = "cuda"
) -> float:
    # Tokenize sequence
    inputs = ppl_tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get sequence length (excluding special tokens)
    seq_len = len(seq)
    
    # Store original tokens for positions we'll mask
    original_input_ids = inputs['input_ids'].clone()
    
    # Create batch with all positions masked (one sample per position)
    batch_size = seq_len
    batch_input_ids = original_input_ids.repeat(batch_size, 1)
    batch_attention_mask = inputs['attention_mask'].repeat(batch_size, 1)
    
    # Mask each position in its corresponding batch sample
    for i in range(batch_size):
        batch_input_ids[i, i + 1] = ppl_tokenizer.mask_token_id  # +1 to skip [CLS]
    
    batch_inputs = {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask
    }
    
    with torch.no_grad():
        # Single forward pass for all masked positions
        outputs = ppl_model(**batch_inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    
    total_loss = 0.0
    
    # Calculate loss for each position
    for i in range(batch_size):
        # Get the original token at position i+1 (skip [CLS])
        original_token = original_input_ids[0, i + 1].item()
        
        # Get logits for the masked position in sample i
        position_logits = logits[i, i + 1]  # +1 to skip [CLS]
        
        # Calculate log probability
        log_probs = torch.nn.functional.log_softmax(position_logits, dim=-1)
        token_loss = -log_probs[original_token].item()
        total_loss += token_loss
    
    # Calculate perplexity
    avg_loss = total_loss / seq_len
    perplexity = math.exp(avg_loss)
    
    return perplexity



def calculate_convex_hull_ratio(
    generated_sequences: List[str],
    training_sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: str
) -> float:
    print("Extracting embeddings for convex hull calculation...")
    
    # Get embeddings for both sets of sequences
    generated_embeddings = get_sequence_embeddings(
        sequences=generated_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    
    training_embeddings = get_sequence_embeddings(
        sequences=training_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    
    def compute_volume(embeddings: np.ndarray) -> float:
        # Reduce dimensionality for convex hull calculation
        # ConvexHull requires fewer dimensions than samples
        target_dims = min(10, len(embeddings) - 1)
        
        if embeddings.shape[1] > target_dims:
            print(f"  Reducing dimensions from {embeddings.shape[1]} to {target_dims}")
            pca = PCA(n_components=target_dims)
            embeddings = pca.fit_transform(embeddings)
        
        # Calculate convex hull and return volume
        hull = ConvexHull(embeddings)
        return hull.volume
    
    # Calculate volumes for both sets
    print("Computing convex hull volumes...")
    generated_volume = compute_volume(generated_embeddings)
    training_volume = compute_volume(training_embeddings)
    
    print(f"  Generated volume: {generated_volume:.6f}")
    print(f"  Training volume: {training_volume:.6f}")
    
    return generated_volume / training_volume


def calculate_frechet_esm_distance(
    generated_sequences: List[str],
    test_sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: str
) -> float:
    print("Calculating Fréchet ESM distance...")
    
    # Get embeddings for both sets
    gen_embeddings = get_sequence_embeddings(
        sequences=generated_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    
    test_embeddings = get_sequence_embeddings(
        sequences=test_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    
    # Calculate means and covariance matrices for both distributions
    mu_gen = np.mean(gen_embeddings, axis=0)
    sigma_gen = np.cov(gen_embeddings, rowvar=False)
    mu_test = np.mean(test_embeddings, axis=0)
    sigma_test = np.cov(test_embeddings, rowvar=False)
    
    # Calculate difference in means
    diff = mu_gen - mu_test
    
    # Calculate covariance mean using matrix square root
    covmean = sqrtm(sigma_gen @ sigma_test)
    
    # Handle complex numbers that can arise from numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate Fréchet distance formula
    frechet_dist = np.sum(diff**2) + np.trace(sigma_gen + sigma_test - 2*covmean)
    
    return float(frechet_dist)


def calculate_min_hamming_distance(
    generated_sequences: List[str],
    training_sequences: List[str]
) -> float:
    print("Calculating minimum Hamming distances...")
    
    results = []
    
    for i, gen_seq in enumerate(tqdm(generated_sequences, desc="Hamming distances")):
        min_dist = 1.0  # Maximum possible normalized distance
        
        for train_seq in training_sequences:
            # Only compare sequences of the same length
            if len(gen_seq) == len(train_seq):
                # Calculate normalized Hamming distance (fraction of differing positions)
                dist = sum(c1 != c2 for c1, c2 in zip(gen_seq, train_seq)) / len(gen_seq)
                min_dist = min(min_dist, dist)
        
        results.append(min_dist)
    
    # Return average minimum distance across all generated sequences
    return (sum(results) / len(results), results)


def get_single_sequence_embedding(
    seq: str,
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: Union[str, torch.device]
) -> Dict[str, Dict[int, torch.Tensor]]:
    # Tokenize the sequence
    inputs = ppl_tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Get model outputs with hidden states
        outputs = ppl_model(**inputs, output_hidden_states=True)
        
        # Extract layer 6 hidden states
        hidden_states = outputs.hidden_states[6]
        seq_len = len(seq)
        
        # Return per-residue representations (not averaged)
        # Exclude [CLS] and [SEP] tokens
        rep = hidden_states[0, 1:seq_len+1].detach().cpu()
        
    return {"representations": {6: rep}}


def calculate_min_soft_alignment_distance(
    generated_sequences: List[str],
    training_sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: str = "cuda",
    k: int = 10
) -> Tuple[float, List[float]]:
    
    longest_paths = []
    
    # Step 1: Extract and normalize mean-pooled embeddings for all training sequences
    print("Extracting mean-pooled embeddings for training sequences...")
    train_vecs = []
    
    for seq in tqdm(training_sequences, desc="Embed train seqs"):
        # Get per-residue representations
        reprs = get_single_sequence_embedding(
            seq=seq,
            ppl_model=ppl_model,
            ppl_tokenizer=ppl_tokenizer,
            device=device
        )["representations"][6]
        
        # Mean-pool across residues and normalize
        vec = reprs.mean(dim=0).cpu().numpy().astype("float32")
        train_vecs.append(vec)
    
    # Stack into matrix and normalize for cosine similarity
    train_mat = np.vstack(train_vecs)
    train_mat /= (np.linalg.norm(train_mat, axis=1, keepdims=True) + 1e-8)

    # Step 2: Create GPU FAISS index for fast similarity search
    print("Building FAISS index...")
    d = train_mat.shape[1]
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexFlatIP(gpu_res, d)  # Inner product for cosine similarity
    gpu_index.add(train_mat)

    # Step 3: Process each generated sequence
    distances = []
    
    for gen_seq in tqdm(generated_sequences, desc="Query gen seqs"):
        # Get mean-pooled embedding for generated sequence
        reprs1 = get_single_sequence_embedding(
            seq=gen_seq,
            ppl_model=ppl_model,
            ppl_tokenizer=ppl_tokenizer,
            device=device
        )["representations"][6]
        
        # Normalize for cosine similarity
        vec1 = reprs1.mean(dim=0).cpu().numpy().astype("float32")
        vec1 /= (np.linalg.norm(vec1) + 1e-8)

        # Find top-k most similar training sequences
        _, I = gpu_index.search(vec1.reshape(1, -1), k)

        # Perform soft alignment with top-k sequences
        max_score = 0
        for idx in I[0]:
            # Get per-residue representations for training sequence
            reprs2 = get_single_sequence_embedding(
                seq=training_sequences[idx],
                ppl_model=ppl_model,
                ppl_tokenizer=ppl_tokenizer,
                device=device
            )["representations"][6]

            # Perform soft alignment between the two sequences
            path = soft_align(
                gen_seq,
                training_sequences[idx],
                {"representations": {6: reprs1}},
                {"representations": {6: reprs2}}
            )
            
            # Track maximum alignment score
            max_score = max(max_score, len(path))
        
        longest_paths.append(max_score)

        # Convert similarity to distance (1 - normalized_similarity)
        similarity = max_score / len(gen_seq)
        distances.append(1.0 - similarity)

    # Return average distance across all generated sequences
    return (float(np.mean(distances)), longest_paths)


def calculate_sliced_wasserstein(
    generated_sequences: List[str],
    training_sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: str = "cuda",
    n_projections: int = 100,
    random_seed: int = 0
) -> tuple[float, List[float]]:
    # 1) Extract embeddings for both sets
    gen_emb = get_sequence_embeddings(
        sequences=generated_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    train_emb = get_sequence_embeddings(
        sequences=training_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    
    # 2) Initialize random number generator
    rng = np.random.default_rng(random_seed)
    sw_distance = 0.0
    dim = gen_emb.shape[1]
    
    scale = np.sqrt(dim)
    gen_emb_scaled   = gen_emb   * scale
    train_emb_scaled = train_emb * scale
    
    distances = []
    # 3) Perform projections and compute 1-D Wasseen utilisantrstein
    for _ in range(n_projections):
        # Draw a random unit vector
        direction = rng.standard_normal(dim)
        direction /= np.linalg.norm(direction)
        
        # Project embeddings onto this direction
        proj_gen = gen_emb_scaled.dot(direction)   # shape: (n_gen,)
        proj_train = train_emb_scaled.dot(direction)  # shape: (n_train,)
        
        # Compute 1-D Wasserstein distance
        sw_distance += wasserstein_distance(proj_gen, proj_train)
        distances.append(sw_distance)
    
    # 4) Return average distance
    return ((sw_distance / n_projections), distances)

def calculate_knn_embedding_distance(
    generated_sequences: List[str],
    training_sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: str,
    k: int
) -> Tuple[float, List[float]]:

    
    # Extract and normalize training embeddings
    print("Extracting training embeddings...")
    train_embeddings = get_sequence_embeddings(
        sequences=training_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    
    # Normalize embeddings to unit length for cosine similarity
    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8
    train_norm = train_embeddings / norms.astype("float32")

    # Build FAISS index for fast KNN search
    d = train_norm.shape[1]
    gpu_res = faiss.StandardGpuResources()
    gpu_index = faiss.GpuIndexFlatIP(gpu_res, d)
    gpu_index.add(train_norm.astype("float32"))

    
    # Process generated sequences
    distances = []
    
    for gen_seq in tqdm(generated_sequences, desc="Computing KNN distances"):
        # Get generated sequence embedding
        gen_embedding = get_sequence_embeddings(
            sequences=[gen_seq],
            ppl_model=ppl_model,
            ppl_tokenizer=ppl_tokenizer,
            device=device,
            show_progress=False
        )[0]
        
        # Find k nearest neighbors
        D, I = gpu_index.search(gen_embedding.reshape(1, -1).astype("float32"), k)
        
        # Store distance to nearest neighbor (or mean of k distances)
        distances.append(float(D[0].mean()))
    
    return (float(np.mean(distances)), distances)


def get_embeddings_eigenvalues(
    sequences: List[str],
    ppl_model,
    ppl_tokenizer,
    device: str = "cuda"
) -> np.ndarray:
    
    embeddings = get_sequence_embeddings(
        sequences=sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=False
    )
    
    cov_matrix = np.cov(embeddings, rowvar=False)
    
    eigenvals = eigvals(cov_matrix)

    eigenvals = np.real(eigenvals[np.argsort(np.real(eigenvals))[::-1]])
    
    return eigenvals


def calculate_toppr_score(
    generated_sequences: List[str],
    training_sequences: List[str],
    ppl_model,
    ppl_tokenizer,
    device: str = "cuda",
    alpha: float = 0.1,
    kernel: str = "cosine",
    random_proj: bool = True
) -> Tuple[float, float, float]:
    print("Extracting embeddings for TopP&R calculation...")
    
    real_embeddings = get_sequence_embeddings(
        sequences=training_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )

    gen_embeddings = get_sequence_embeddings(
        sequences=generated_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )

    print("Computing TopP&R...")
    toppr_result = TopPR(
        real_features=real_embeddings,
        fake_features=gen_embeddings,
        alpha=alpha,
        kernel=kernel,
        random_proj=random_proj,
        f1_score=True
    )

    return (
        toppr_result["fidelity"],   # TopP
        toppr_result["diversity"], # TopR
        toppr_result["Top_F1"]     # F1-score
    )
    
    
def calculate_kde_kl_divergence(
    generated_sequences: List[str],
    training_sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    device: str,
    bandwidth: float = 0.1,
    n_samples: int = 1000,
    random_seed: int = 42
) -> Tuple[float, float]:
    
    print("Extracting embeddings for KDE KL divergence calculation...")
    
    # Extraire les embeddings
    gen_embeddings = get_sequence_embeddings(
        sequences=generated_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    
    train_embeddings = get_sequence_embeddings(
        sequences=training_sequences,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        device=device,
        show_progress=True
    )
    
    # Réduction de dimensionnalité avec PCA pour rendre le KDE plus stable
    print("Reducing dimensionality with PCA...")
    n_components = min(50, gen_embeddings.shape[1], len(training_sequences) - 1)
    pca = PCA(n_components=n_components)
    
    # Combiner les embeddings pour un PCA cohérent
    combined_embeddings = np.vstack([gen_embeddings, train_embeddings])
    pca.fit(combined_embeddings)
    
    # Transformer les embeddings
    gen_embeddings_pca = pca.transform(gen_embeddings)
    train_embeddings_pca = pca.transform(train_embeddings)
    
    print(f"Reduced dimensionality to {n_components} components")
    
    # Créer les modèles KDE
    print("Fitting KDE models...")
    kde_gen = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde_train = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    
    kde_gen.fit(gen_embeddings_pca)
    kde_train.fit(train_embeddings_pca)
    
    # Générer des échantillons pour estimer la divergence KL
    print("Sampling for KL divergence estimation...")
    np.random.seed(random_seed)
    
    # Échantillonner des points depuis les deux distributions
    gen_samples = kde_gen.sample(n_samples)
    train_samples = kde_train.sample(n_samples)
    
    # Calculer les log-densités
    log_p_gen_on_gen = kde_gen.score_samples(gen_samples)
    log_p_train_on_gen = kde_train.score_samples(gen_samples)
    
    log_p_train_on_train = kde_train.score_samples(train_samples)
    log_p_gen_on_train = kde_gen.score_samples(train_samples)
    
    # Calculer les divergences KL
    # KL(P_gen || P_train) = E_{x~P_gen}[log P_gen(x) - log P_train(x)]
    kl_gen_train = np.mean(log_p_gen_on_gen - log_p_train_on_gen)
    
    # KL(P_train || P_gen) = E_{x~P_train}[log P_train(x) - log P_gen(x)]
    kl_train_gen = np.mean(log_p_train_on_train - log_p_gen_on_train)
    
    print(f"KL(P_gen || P_train): {kl_gen_train:.6f}")
    print(f"KL(P_train || P_gen): {kl_train_gen:.6f}")
    
    return float(kl_gen_train), float(kl_train_gen)