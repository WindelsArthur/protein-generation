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

project_root = Path.home() / "protein-generation"
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
    """Function that runs on a specific GPU"""
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')
    
    print(f"[GPU {gpu_id}] Starting...")
    
    config = load_config(config_path)
    
    # Load models
    print(f"[GPU {gpu_id}] Loading models...")
    ppl_model, ppl_tokenizer = load_perplexity_model(
        ppl_model_name=config['models']['perplexity'], 
        device=device
    )
    fold_model, fold_tokenizer = load_folding_model(
        fold_model_name=config['models']['folding'], 
        device=device
    )
    
    out_dir = Path(config['output_dir']) / f"gpu_{gpu_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets and split for this GPU
    sequences = {}
    seq_column = config.get('sequence_column', 'sequence')
    
    for name, path in config['datasets'].items():
        df = pd.read_csv(path)
        all_seqs = df[seq_column].tolist()
        
        chunk_size = len(all_seqs) // total_gpus
        start_idx = gpu_id * chunk_size
        if gpu_id == total_gpus - 1:  # Last GPU takes remainder
            sequences[name] = all_seqs[start_idx:]
        else:
            sequences[name] = all_seqs[start_idx:start_idx + chunk_size]
        
        print(f"[GPU {gpu_id}] Loaded {name}: {len(sequences[name])} sequences")
    
    # Generate random sequences if configured
    if config.get('random_baseline', {}).get('generate', False):
        rb = config['random_baseline']
        AA_SET = rb.get('amino_acids', 'ACDEFGHIKLMNPQRSTVWY')
        
        n_samples_per_gpu = rb['n_samples'] // total_gpus
        if gpu_id == total_gpus - 1:
            n_samples_per_gpu += rb['n_samples'] % total_gpus
            
        sequences['random'] = [
            ''.join(random.choices(AA_SET, k=rb['sequence_length'])) 
            for _ in range(n_samples_per_gpu)
        ]
        print(f"[GPU {gpu_id}] Generated random: {len(sequences['random'])} sequences")
    
    training_name = config.get('training_dataset_name', 'training')
    training_sequences = sequences[training_name]
    
    # Quality evaluation
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
    
    # Distribution evaluation
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
    """Merge results from all GPUs"""
    base_path = Path(base_dir)
    merged_files = {}
    
    # Process all GPU directories
    for gpu_id in range(n_gpus):
        gpu_dir = base_path / f"gpu_{gpu_id}"
        
        if not gpu_dir.exists():
            continue
        
        for yaml_file in gpu_dir.glob("*.yaml"):
            file_stem = yaml_file.stem
            
            # Extract base filename
            if "_quality_" in file_stem:
                base_name = file_stem.split("_quality_")[0] + "_quality"
            elif "_distribution_" in file_stem:
                base_name = file_stem.split("_distribution_")[0] + "_distribution"
            else:
                continue
            
            with open(yaml_file, 'r') as f:
                content = yaml.safe_load(f)
            
            if base_name not in merged_files:
                if isinstance(content, list):
                    merged_files[base_name] = []
                else:
                    merged_files[base_name] = {"metadata": None, "results": []}
            
            # Merge based on content type
            if isinstance(content, list):
                merged_files[base_name].extend(content)
            else:
                if merged_files[base_name]["metadata"] is None:
                    merged_files[base_name]["metadata"] = content.get("metadata", {})
                
                for key, value in content.items():
                    if key == "metadata":
                        continue
                    if key not in merged_files[base_name]:
                        merged_files[base_name][key] = value
                    elif isinstance(value, dict) and "individual_distances" in value:
                        if "individual_distances" not in merged_files[base_name][key]:
                            merged_files[base_name][key] = value
                        else:
                            merged_files[base_name][key]["individual_distances"].extend(
                                value["individual_distances"]
                            )
                            # Recalculate mean
                            all_dists = merged_files[base_name][key]["individual_distances"]
                            merged_files[base_name][key]["mean"] = sum(all_dists) / len(all_dists)
    
    # Save merged files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for base_name, content in merged_files.items():
        output_file = base_path / f"{base_name}_{timestamp}.yaml"
        
        if isinstance(content, dict) and "metadata" in content:
            if content["metadata"]:
                content["metadata"]["timestamp"] = timestamp
                content["metadata"]["n_gpus_used"] = n_gpus
            
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
    
    if len(sys.argv) >= 3:
        n_gpus = int(sys.argv[2])
    else:
        n_gpus = torch.cuda.device_count()
    
    print(f"Running parallel evaluation on {n_gpus} GPUs")
    
    # Create process for each GPU
    processes = []
    for gpu_id in range(n_gpus):
        p = Process(target=run_on_gpu, args=(gpu_id, n_gpus, config_path))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print("\nAll GPUs finished!")
    
    # Merge results
    config = load_config(config_path)
    print("\nMerging results...")
    merge_results(config['output_dir'], n_gpus)
    
    print(f"\nMerged results saved in: {config['output_dir']}/")
    print("(Individual GPU results are in gpu_*/ subdirectories)")