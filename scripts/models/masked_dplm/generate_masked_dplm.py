import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys, pathlib
project_root = pathlib.Path.home() / "protein-generation"
sys.path.append(str(project_root))
from scripts.utils import *
from scripts.models.masked_dplm.classes.model import DenoisingTransformer
from scripts.models.masked_dplm.classes.noise_schedule import NoiseSchedule
from scripts.models.masked_dplm.classes.vocabulary import ProteinVocabulary
from scripts.models.masked_dplm.generation.generation_classic import generate_sequences

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# Load checkpoint
checkpoint_path = "/home/arthur/projets/protein-generation/experiments/models/masked_dplm_simple/checkpoint_latest.pt"
print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Extract from checkpoint
config = checkpoint['config']
model_state = checkpoint['model_state']

# Create vocabulary
vocabulary = ProteinVocabulary()

# Create model
model = DenoisingTransformer(
    vocab=vocabulary,
    max_seq_length=config['model']['max_seq_length'],
    d_model=config['model']['d_model'],
    n_heads=config['model']['n_heads'],
    n_layers=config['model']['n_layers'],
    dropout=config['model']['dropout']
).to(device)

# Load model weights
model.load_state_dict(model_state)
model.eval()
print(f"[SUCCESS] Model loaded")

# Create noise schedule
noise_schedule = NoiseSchedule(config['diffusion']['noise_schedule'])

# Generation parameters
n_samples = config['generation']['n_samples']
seq_length = config['generation']['seq_length']
dt = config['generation']['dt']

print(f"[INFO] Generating {n_samples} sequences...")

# Generate sequences
encoded_sequences = generate_sequences(
    model=model,
    vocab=vocabulary,
    n_samples=n_samples,
    seq_length=seq_length,
    noise_schedule=noise_schedule,
    dt=dt
)

# Decode sequences
cleaned_sequences = vocabulary.decode_batch(encoded_sequences)

print(f"[SUCCESS] Generated {len(cleaned_sequences)} sequences")

# Save to CSV
output_path = Path(config['exp_dir']) / 'generated_sequences.csv'
df = pd.DataFrame({'sequence': cleaned_sequences})
df.to_csv(output_path, index=False)
print(f"[SUCCESS] Sequences saved to: {output_path}")