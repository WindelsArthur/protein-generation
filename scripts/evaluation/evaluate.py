import yaml
from typing import List, Dict, Any, Union
import torch
from tqdm import tqdm
from datetime import datetime

import sys, pathlib, os
project_root = pathlib.Path.home() / "projets" / "protein-generation"
sys.path.append(str(project_root))

from transformers import EsmTokenizer, EsmForMaskedLM, EsmForProteinFolding  # type: ignore
from scripts.evaluation.evaluation_metrics import *

def evaluate_quality(
    sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    fold_model: EsmForProteinFolding,
    fold_tokenizer: EsmTokenizer,
    output_file: str,
    device: Union[str, torch.device] = "cuda"
) -> List[Dict[str, Any]]:
    """
    Evaluate individual sequence quality (pLDDT and perplexity) and save to YAML.
    """
    results: List[Dict[str, Any]] = []
    for seq in tqdm(sequences, desc="Quality evaluation"):
        # Compute pLDDT
        plddt_mean = calculate_plddt(
            seq, fold_model, fold_tokenizer, device
        )
        # Compute perplexity
        perplexity = calculate_perplexity(
            seq, ppl_model, ppl_tokenizer, device
        )
        results.append({
            "sequence": seq,
            "plddt_mean": float(plddt_mean),
            "perplexity": float(perplexity)
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    yaml_path = f"{output_file}_quality_{timestamp}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(results, f)
    return results


def evaluate_distributions(
    generated_sequences: List[str],
    training_sequences: List[str],
    ppl_model: EsmForMaskedLM,
    ppl_tokenizer: EsmTokenizer,
    output_file: str,
    device: Union[str, torch.device] = "cuda",
    k_neighbors: int = 5,
    soft_align_k: int = 5,
    n_projections: int = 100,
    kde_bandwidth: float = 0.1,
    kde_n_samples: int = 1000,
    toppr_alpha: float = 0.1,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate distribution-level metrics between generated and training sequences and save to YAML.
    """
    results: Dict[str, Any] = {}

    
    # 1. Hamming distance
    h_mean, h_dists = calculate_min_hamming_distance(
        generated_sequences, training_sequences
    )
    results["hamming_distance"] = {
        "mean": float(h_mean),
        "individual_distances": h_dists
    }

    # 2. Fréchet ESM distance
    results["frechet_esm_distance"] = float(
        calculate_frechet_esm_distance(
            generated_sequences, training_sequences,
            ppl_model, ppl_tokenizer, device
        )
    )

    # 3. Sliced Wasserstein distance
    sw_dist, sw_ind = calculate_sliced_wasserstein(
        generated_sequences, training_sequences,
        ppl_model, ppl_tokenizer, device,
        n_projections, random_seed
    )
    results["sliced_wasserstein"] = {
        "distance": float(sw_dist),
        "individual_projections": [float(x) for x in sw_ind]
    }

    # 4. TopPR score
    fidelity, diversity, f1 = 0, 0, 0
    #fidelity, diversity, f1 = calculate_toppr_score(
    #    generated_sequences, training_sequences,
    #    ppl_model, ppl_tokenizer, device, toppr_alpha
    #)
    results["toppr"] = {
        "fidelity": float(fidelity),
        "diversity": float(diversity),
        "f1_score": float(f1)
    }

    # 5. KDE KL divergence
    kl_gt, kl_tg = calculate_kde_kl_divergence(
        generated_sequences, training_sequences,
        ppl_model, ppl_tokenizer, device,
        kde_bandwidth, kde_n_samples, random_seed
    )
    results["kde_kl_divergence"] = {
        "kl_gen_to_train": float(kl_gt),
        "kl_train_to_gen": float(kl_tg)
    }

    # 6 & 7. KNN embedding distances
    knn_f_mean, knn_f_dists = calculate_knn_embedding_distance(
        generated_sequences, training_sequences,
        ppl_model, ppl_tokenizer, device, k_neighbors
    )
    knn_d_mean, knn_d_dists = calculate_knn_embedding_distance(
        training_sequences, generated_sequences,
        ppl_model, ppl_tokenizer, device, k_neighbors
    )
    results["knn_fidelity"] = {
        "mean_distance": float(knn_f_mean),
        "individual_distances": knn_f_dists
    }
    results["knn_diversity"] = {
        "mean_distance": float(knn_d_mean),
        "individual_distances": knn_d_dists
    }

    # 8 & 9. Soft alignment distances
    s_f_mean, s_f_paths, s_d_mean, s_d_paths = 0, [0 for k in range(100)], 0, [0 for k in range(100)]
    #s_f_mean, s_f_paths = calculate_min_soft_alignment_distance(
    #    generated_sequences, training_sequences,
    #    ppl_model, ppl_tokenizer, device, soft_align_k
    #)
    #s_d_mean, s_d_paths = calculate_min_soft_alignment_distance(
    #    training_sequences, generated_sequences,
    #    ppl_model, ppl_tokenizer, device, soft_align_k
    #)
    results["soft_alignment_fidelity"] = {
        "mean_distance": float(s_f_mean),
        "individual_paths": s_f_paths
    }
    results["soft_alignment_diversity"] = {
        "mean_distance": float(s_d_mean),
        "individual_paths": s_d_paths
    }

    # Metadata
    results["metadata"] = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_generated": len(generated_sequences),
        "n_training": len(training_sequences),
        "parameters": {
            "k_neighbors": k_neighbors,
            "soft_align_k": soft_align_k,
            "n_projections": n_projections,
            "kde_bandwidth": kde_bandwidth,
            "kde_n_samples": kde_n_samples,
            "toppr_alpha": toppr_alpha,
            "random_seed": random_seed
        }
    }

    yaml_path = f"{output_file}_distribution_{results['metadata']['timestamp']}.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(results, f)
    return results


def load_quality_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load quality results from a YAML file.
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_distribution_results(file_path: str) -> Dict[str, Any]:
    """
    Load distribution results from a YAML file.
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
