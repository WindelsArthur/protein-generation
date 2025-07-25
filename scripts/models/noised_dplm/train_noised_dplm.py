import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Union
import warnings
warnings.filterwarnings('ignore')

# Hugging Face transformers
from transformers import EsmTokenizer, EsmForMaskedLM # type: ignore

import sys, pathlib, os
project_root = pathlib.Path.home() / "protein-generation"
sys.path.append(str(project_root))
from scripts.utils import *
from scripts.models.noised_dplm.training.training_classic import *
from scripts.models.noised_dplm.classes.model import DenoisingTransformer
from scripts.models.noised_dplm.classes.noise_schedule import NoiseSchedule
from scripts.models.noised_dplm.classes.vocabulary import ProteinVocabulary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

def load_perplexity_model(
    ppl_model_name: str,
    device: str = "cuda"
) -> Tuple[EsmForMaskedLM, EsmTokenizer]:
    """Load ESM model and tokenizer for perplexity-based noising."""
    print(f"[INFO] Loading perplexity model: {ppl_model_name}")
    
    # Load tokenizer and model from Hugging Face
    ppl_tokenizer = EsmTokenizer.from_pretrained(ppl_model_name)
    ppl_model = EsmForMaskedLM.from_pretrained(ppl_model_name)
    
    # Set model to evaluation mode and move to specified device
    ppl_model.eval()
    ppl_model.to(device)
    
    print(f"[SUCCESS] Perplexity model loaded successfully")
    return ppl_model, ppl_tokenizer

# Model configuration
PPL_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# Load perplexity model
print("=" * 60)
print("LOADING MODELS")
print("=" * 60)
ppl_model, ppl_tokenizer = load_perplexity_model(ppl_model_name=PPL_MODEL_NAME, device=device)
print(f"[SUCCESS] All models loaded successfully\n")

# Main experiment workflow
print("=" * 60)
print("STARTING EXPERIMENT")
print("=" * 60)

# Load configuration
config_path = "/home/arthur/projets/protein-generation/configs/base_config.yaml"
print(f"[INFO] Loading configuration from: {config_path}")
config = load_experiment_config(config_path)
config = setup_experiment_directory(config)

print(f"[INFO] Experiment name: {config['experiment']['name']}")
print(f"[INFO] Experiment directory: {config['exp_dir']}")
print(f"[INFO] Configuration saved to experiment directory\n")

# Save configuration
save_experiment_config(config, config['exp_dir'])

# Set random seeds for reproducibility
torch.manual_seed(config['training']['seed'])
np.random.seed(config['training']['seed'])
print(f"[INFO] Random seeds set to: {config['training']['seed']}")

# Load and process protein sequences
print("\n" + "=" * 60)
print("LOADING AND PROCESSING DATA")
print("=" * 60)

print(f"[INFO] Loading protein sequences from: {config['data']['input_file']}")
protein_data = pd.read_csv(config['data']['input_file'])
sequences = protein_data['sequence'].tolist()
print(f"[INFO] Raw sequences loaded: {len(sequences)}")

# Filter sequences by maximum length
sequences = [seq for seq in sequences if len(seq) <= config['model']['max_seq_length']]
print(f"[INFO] Sequences after length filtering (≤{config['model']['max_seq_length']}): {len(sequences)}")

# Limit to training sample size
sequences = sequences[:config['training']['n_samples']]
print(f"[INFO] Final training sequences: {len(sequences)}")

# Calculate and display sequence statistics
seq_lengths = [len(seq) for seq in sequences]
print(f"[STATS] Sequence length - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {np.mean(seq_lengths):.1f}")

# Save training sequences for reference
df = pd.DataFrame({'sequence': sequences})
sequences_path = Path(config['exp_dir']) / 'training_sequences.csv'
df.to_csv(sequences_path, index=False)
print(f"[INFO] Training sequences saved to: {sequences_path}")

# Create vocabulary and encode sequences
print("\n" + "=" * 60)
print("ENCODING SEQUENCES")
print("=" * 60)

print(f"[INFO] Creating protein vocabulary")
vocabulary = ProteinVocabulary()
print(f"[INFO] Vocabulary size: {vocabulary.VOCAB_SIZE} (20 amino acids + PAD token)")

print(f"[INFO] Encoding sequences with max length: {config['model']['max_seq_length']}")
encoded = vocabulary.encode_batch(sequences, config['model']['max_seq_length'])
print(f"[SUCCESS] Encoded tensor shape: {encoded.shape}")

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(encoded)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True
)
print(f"[INFO] DataLoader created - Batch size: {config['training']['batch_size']}, Batches: {len(dataloader)}")

# Create denoising transformer model
print("\n" + "=" * 60)
print("CREATING MODEL")
print("=" * 60)

model = DenoisingTransformer(
    vocab=vocabulary,
    max_seq_length=config['model']['max_seq_length'],
    d_model=config['model']['d_model'],
    n_heads=config['model']['n_heads'],
    n_layers=config['model']['n_layers'],
    dropout=config['model']['dropout']
).to(device)

# Calculate model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"[INFO] Model architecture:")
print(f"  - Vocabulary size: {vocabulary.VOCAB_SIZE}")
print(f"  - Max sequence length: {config['model']['max_seq_length']}")
print(f"  - Model dimension: {config['model']['d_model']}")
print(f"  - Number of heads: {config['model']['n_heads']}")
print(f"  - Number of layers: {config['model']['n_layers']}")
print(f"  - Dropout rate: {config['model']['dropout']}")
print(f"[INFO] Total parameters: {total_params:,}")
print(f"[INFO] Trainable parameters: {trainable_params:,}")

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
print(f"[INFO] Optimizer: Adam with learning rate {config['training']['learning_rate']}")

# Setup noise schedule
noise_schedule = NoiseSchedule(config['diffusion']['noise_schedule'])
print(f"[INFO] Noise schedule: {config['diffusion']['noise_schedule']}")

# Training phase
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

print(f"[INFO] Starting training for {config['training']['n_epochs']} epochs")
print(f"[INFO] Training progress will be displayed every 10 epochs")

# Execute training loop
losses = train_model(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    noise_schedule=noise_schedule,
    n_epochs=config['training']['n_epochs'],
    vocab=vocabulary
    #ppl_model=ppl_model,
    #ppl_tokenizer=ppl_tokenizer
)

# Plot and save training losses
plot_and_save_losses(config, losses)
print(f"[SUCCESS] Training loss plot saved")
print(f"[SUCCESS] Training completed successfully!")
print(f"[RESULTS] Final loss: {losses[-1]:.6f}")
print(f"[RESULTS] Initial loss: {losses[0]:.6f}")
print(f"[RESULTS] Loss improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
print(f"[RESULTS] Total epochs trained: {len(losses)}")
print(f"[RESULTS] Results saved in: {config['exp_dir']}")


# Save final model checkpoint
print("\n" + "=" * 60)
print("SAVING MODEL CHECKPOINT")
print("=" * 60)
ckpt = {
    "model_state": model.state_dict(),
    "config": config,              
    "vocab_state": vocabulary.__dict__  
}
ckpt_path = Path(config['exp_dir']) / "checkpoint_latest.pt"
torch.save(ckpt, ckpt_path)
print(f"[INFO] Checkpoint saved to {ckpt_path}")
