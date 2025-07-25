# run_evaluation.py - Version simple avec config YAML
import torch
import pandas as pd
from pathlib import Path
import yaml
import random
import warnings
warnings.filterwarnings('ignore')

import sys
project_root = Path.home() / "projets" / "protein-generation"
sys.path.append(str(project_root))

from scripts.evaluation.evaluate import *

# Charger la config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Usage: python run_evaluation.py config.yaml
if __name__ == "__main__":
    # Charger config depuis le fichier passé en argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation_config.yaml"
    config = load_config(config_path)
    
    # Device
    device = torch.device(config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Charger les modèles
    print("Loading models...")
    ppl_model, ppl_tokenizer = load_perplexity_model(
        ppl_model_name=config['models']['perplexity'], 
        device=device
    )
    fold_model, fold_tokenizer = load_folding_model(
        fold_model_name=config['models']['folding'], 
        device=device
    )
    print("Models loaded successfully!")
    
    # Output directory
    out_dir = Path(config['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Charger tous les datasets
    sequences = {}
    seq_column = config.get('sequence_column', 'sequence')
    
    for name, path in config['datasets'].items():
        df = pd.read_csv(path)
        sequences[name] = df[seq_column].tolist()
        print(f"Loaded {name}: {len(sequences[name])} sequences")
    
    # Générer random sequences si configuré
    if config.get('random_baseline', {}).get('generate', False):
        rb = config['random_baseline']
        AA_SET = rb.get('amino_acids', 'ACDEFGHIKLMNPQRSTVWY')
        sequences['random'] = [
            ''.join(random.choices(AA_SET, k=rb['sequence_length'])) 
            for _ in range(rb['n_samples'])
        ]
        print(f"Generated random: {len(sequences['random'])} sequences")
    
    # Training sequences
    training_name = config.get('training_dataset_name', 'training')
    training_sequences = sequences[training_name]
    
    # === QUALITY EVALUATION ===
    print("\n=== Running Quality Evaluation ===")
    for name, seqs in sequences.items():
        print(f"\nEvaluating quality for '{name}'...")
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
    print("\n=== Running Distribution Evaluation ===")
    dist_params = config.get('distribution_parameters', {})
    
    for name, seqs in sequences.items():
        if name == training_name:
            continue  # Skip training vs training
            
        print(f"\nCalculating distribution metrics for '{name}' vs training...")
        evaluate_distributions(
            generated_sequences=seqs,
            training_sequences=training_sequences,
            ppl_model=ppl_model,
            ppl_tokenizer=ppl_tokenizer,
            output_file=str(out_dir / f"{name}_vs_training"),
            device=device,
            **dist_params  # Unpacks all distribution parameters
        )
    
    print("\nEvaluation complete!")