import torch
import pandas as pd
from pathlib import Path
import yaml
import random
import warnings
import os
import sys
from multiprocessing import Process
from datetime import datetime
import shutil

warnings.filterwarnings('ignore')

project_root = Path.home() / "projets" / "protein-generation"
sys.path.append(str(project_root))

from scripts.evaluation.evaluate import *

def load_config(config_path):
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file '{config_path}' not found!")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_on_gpu(gpu_id, total_gpus, config_path):
    """Fonction qui tourne sur un GPU spécifique"""
    # Définir le GPU pour ce process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')  # Toujours cuda:0 car on a masqué les autres
    
    print(f"[GPU {gpu_id}] Starting...")
    
    # Charger config
    config = load_config(config_path)
    
    # Charger les modèles
    print(f"[GPU {gpu_id}] Loading models...")
    ppl_model, ppl_tokenizer = load_perplexity_model(
        ppl_model_name=config['models']['perplexity'], 
        device=device
    )
    fold_model, fold_tokenizer = load_folding_model(
        fold_model_name=config['models']['folding'], 
        device=device
    )
    
    # Output directory avec suffixe GPU
    out_dir = Path(config['output_dir']) / f"gpu_{gpu_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger tous les datasets
    sequences = {}
    seq_column = config.get('sequence_column', 'sequence')
    
    for name, path in config['datasets'].items():
        df = pd.read_csv(path)
        all_seqs = df[seq_column].tolist()
        
        # Prendre seulement la partie pour ce GPU
        chunk_size = len(all_seqs) // total_gpus
        start_idx = gpu_id * chunk_size
        if gpu_id == total_gpus - 1:  # Dernier GPU prend le reste
            sequences[name] = all_seqs[start_idx:]
        else:
            sequences[name] = all_seqs[start_idx:start_idx + chunk_size]
        
        print(f"[GPU {gpu_id}] Loaded {name}: {len(sequences[name])} sequences")
    
    # Générer random sequences si configuré
    if config.get('random_baseline', {}).get('generate', False):
        rb = config['random_baseline']
        AA_SET = rb.get('amino_acids', 'ACDEFGHIKLMNPQRSTVWY')
        
        # Diviser le nombre de random sequences aussi
        n_samples_per_gpu = rb['n_samples'] // total_gpus
        if gpu_id == total_gpus - 1:
            n_samples_per_gpu += rb['n_samples'] % total_gpus
            
        sequences['random'] = [
            ''.join(random.choices(AA_SET, k=rb['sequence_length'])) 
            for _ in range(n_samples_per_gpu)
        ]
        print(f"[GPU {gpu_id}] Generated random: {len(sequences['random'])} sequences")
    
    # Training sequences
    training_name = config.get('training_dataset_name', 'training')
    training_sequences = sequences[training_name]
    
    # === QUALITY EVALUATION ===
    print(f"\n[GPU {gpu_id}] === Running Quality Evaluation ===")
    for name, seqs in sequences.items():
        print(f"[GPU {gpu_id}] Evaluating quality for '{name}'...")
        evaluate_quality(
            sequences=seqs,
            ppl_model=ppl_model,
            ppl_tokenizer=ppl_tokenizer,
            fold_model=fold_model,
            fold_tokenizer=fold_tokenizer,
            output_file=str(out_dir / f"{name}_sequences"),
            device=device
        )
    
    # === DISTRIBUTION EVALUATION ===
    print(f"\n[GPU {gpu_id}] === Running Distribution Evaluation ===")
    dist_params = config.get('distribution_parameters', {})
    
    for name, seqs in sequences.items():
        if name == training_name:
            continue
            
        print(f"[GPU {gpu_id}] Calculating distribution metrics for '{name}' vs training...")
        evaluate_distributions(
            generated_sequences=seqs,
            training_sequences=training_sequences,
            ppl_model=ppl_model,
            ppl_tokenizer=ppl_tokenizer,
            output_file=str(out_dir / f"{name}_vs_training"),
            device=device,
            **dist_params
        )
    
    print(f"\n[GPU {gpu_id}] Evaluation complete!")

def merge_results(base_dir, n_gpus):
    """Fusionner les résultats de tous les GPUs dans le format original"""
    base_path = Path(base_dir)
    
    # Dictionnaire pour stocker tous les résultats par type
    merged_files = {}
    
    # Parcourir tous les dossiers GPU
    for gpu_id in range(n_gpus):
        gpu_dir = base_path / f"gpu_{gpu_id}"
        
        if not gpu_dir.exists():
            continue
        
        # Parcourir tous les fichiers YAML dans le dossier GPU
        for yaml_file in gpu_dir.glob("*.yaml"):
            # Identifier le type de fichier (quality ou distribution)
            file_stem = yaml_file.stem
            
            # Extraire le nom de base du fichier (sans timestamp)
            if "_quality_" in file_stem:
                base_name = file_stem.split("_quality_")[0] + "_quality"
            elif "_distribution_" in file_stem:
                base_name = file_stem.split("_distribution_")[0] + "_distribution"
            else:
                continue
            
            # Charger le contenu
            with open(yaml_file, 'r') as f:
                content = yaml.safe_load(f)
            
            # Initialiser si nécessaire
            if base_name not in merged_files:
                if isinstance(content, list):
                    merged_files[base_name] = []
                else:
                    merged_files[base_name] = {"metadata": None, "results": []}
            
            # Fusionner selon le type
            if isinstance(content, list):
                # Fichiers quality (liste de résultats)
                merged_files[base_name].extend(content)
            else:
                # Fichiers distribution (dictionnaire avec métadonnées)
                if merged_files[base_name]["metadata"] is None:
                    merged_files[base_name]["metadata"] = content.get("metadata", {})
                
                # Fusionner les résultats individuels
                for key, value in content.items():
                    if key == "metadata":
                        continue
                    if key not in merged_files[base_name]:
                        merged_files[base_name][key] = value
                    elif isinstance(value, dict) and "individual_distances" in value:
                        # Fusionner les listes de distances individuelles
                        if "individual_distances" not in merged_files[base_name][key]:
                            merged_files[base_name][key] = value
                        else:
                            merged_files[base_name][key]["individual_distances"].extend(
                                value["individual_distances"]
                            )
                            # Recalculer la moyenne
                            all_dists = merged_files[base_name][key]["individual_distances"]
                            merged_files[base_name][key]["mean"] = sum(all_dists) / len(all_dists)
    
    # Sauver les fichiers fusionnés avec un nouveau timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for base_name, content in merged_files.items():
        output_file = base_path / f"{base_name}_{timestamp}.yaml"
        
        # Pour les fichiers distribution, restructurer si nécessaire
        if isinstance(content, dict) and "metadata" in content:
            # Mettre à jour les métadonnées
            if content["metadata"]:
                content["metadata"]["timestamp"] = timestamp
                content["metadata"]["n_gpus_used"] = n_gpus
            
            # Extraire juste le contenu sans le wrapper
            final_content = {k: v for k, v in content.items() if k != "results"}
        else:
            final_content = content
        
        with open(output_file, 'w') as f:
            yaml.safe_dump(final_content, f)
        
        print(f"Merged: {output_file.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_evaluation_parallel.py config.yaml [n_gpus]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Nombre de GPUs à utiliser
    if len(sys.argv) >= 3:
        n_gpus = int(sys.argv[2])
    else:
        n_gpus = torch.cuda.device_count()
    
    print(f"Running parallel evaluation on {n_gpus} GPUs")
    
    # Créer un process pour chaque GPU
    processes = []
    for gpu_id in range(n_gpus):
        p = Process(target=run_on_gpu, args=(gpu_id, n_gpus, config_path))
        p.start()
        processes.append(p)
    
    # Attendre que tous finissent
    for p in processes:
        p.join()
    
    print("\nAll GPUs finished!")
    
    # Merger automatiquement les résultats
    config = load_config(config_path)
    print("\nMerging results...")
    merge_results(config['output_dir'], n_gpus)
    
    # Nettoyer les dossiers gpu_* si souhaité
    print(f"\nMerged results saved in: {config['output_dir']}/")
    print("(Individual GPU results are in gpu_*/ subdirectories)")