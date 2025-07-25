import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import yaml
import os
import sys
import pathlib
from datetime import datetime
import pickle
import shutil

project_root = pathlib.Path.home() / "protein-generation"
sys.path.append(str(project_root))


def load_experiment_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment_directory(config):
    """Create the experiment directory and generate a name if needed."""
    if config['experiment']['name'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['experiment']['name'] = f"exp_{config['diffusion']['forward_process']}_{config['diffusion']['inference_process']}_{timestamp}"
    
    exp_dir = os.path.join(config['save']['base_dir'], config['experiment']['name'])
    os.makedirs(exp_dir, exist_ok=True)
    config['exp_dir'] = exp_dir
    return config


def save_experiment_config(config, exp_dir):
    """Save the configuration file in the experiment directory."""
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    return config_path


def save_results(config, model, losses, generated_sequences):
    """Save all experiment results."""
    exp_dir = config['exp_dir']
    
    # Save model weights
    if config['save']['save_model']:
        model_path = os.path.join(exp_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        if config['display']['verbose']:
            print(f"Model saved: {model_path}")
    
    # Save losses
    if config['save']['save_losses']:
        losses_path = os.path.join(exp_dir, 'losses.pkl')
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)
        
        # Also save CSV for easier analysis
        losses_csv_path = os.path.join(exp_dir, 'losses.csv')
        pd.DataFrame({'epoch': range(len(losses)), 'loss': losses}).to_csv(losses_csv_path, index=False)
        
        if config['display']['verbose']:
            print(f"Losses saved: {losses_path}")
    
    # Save generated sequences
    if config['save']['save_generated']:
        generated_path = os.path.join(exp_dir, 'generated_sequences.csv')
        df_generated = pd.DataFrame({config['data']['sequence_column']: generated_sequences})
        df_generated.to_csv(generated_path, index=False)
        
        if config['display']['verbose']:
            print(f"Generated sequences saved: {generated_path}")


def plot_and_save_losses(config, losses):
    """Plot and save training loss curve."""
    if config['display']['plot_losses']:
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f"Training Loss - {config['experiment']['name']}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        
        # Save figure
        plot_path = os.path.join(config['exp_dir'], 'training_loss.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        if config['display']['verbose']:
            print(f"Plot saved: {plot_path}")


def display_sample_sequences(config, generated_sequences):
    """Print a few example generated sequences."""
    n_show = min(config['display']['show_sample_sequences'], len(generated_sequences))
    if n_show > 0:
        print(f"\nDisplaying {n_show} generated sequences:")
        for i in range(n_show):
            seq = generated_sequences[i]
            print(f"Sequence {i+1}: {seq[:50]}{'...' if len(seq) > 50 else ''}")