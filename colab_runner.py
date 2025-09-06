#!/usr/bin/env python3
"""
Google Colab Runner for LLM Research Project
Optimized for Colab's limitations and timeouts
"""

# Colab-optimized imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
import os
import pickle
from torchtune.modules import RotaryPositionalEmbeddings
warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ColabConfig:
    """Colab-optimized configuration"""
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 16  # Reduced for Colab
    max_steps: int = 1000  # Reduced for Colab timeouts

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters (reduced for Colab)
    max_seq_len: int = 512
    num_documents: int = 1000  # Reduced
    max_tokens: int = 200000   # Reduced

    # Evaluation
    eval_every: int = 200  # More frequent
    eval_steps: int = 50   # Reduced

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@dataclass
class ColabMoEConfig(ColabConfig):
    """MoE configuration optimized for Colab"""
    # MoE specific parameters
    num_experts: int = 8
    expert_top_k: int = 2
    moe_layers: str = "alternate"  # "all", "alternate", or "last_half"
    load_balancing_weight: float = 0.01

    def should_use_moe(self, layer_idx: int) -> bool:
        """Determine if a specific layer should use MoE"""
        if self.moe_layers == "all":
            return True
        elif self.moe_layers == "alternate":
            return layer_idx % 2 == 1  # Every other layer
        elif self.moe_layers == "last_half":
            return layer_idx >= self.n_layers // 2
        return False

def quick_setup():
    """Quick setup for Colab"""
    print("🚀 Google Colab LLM Research Setup")
    print("=" * 50)
    
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set seed
    set_seed(42)
    
    print("✅ Setup complete!")
    return torch.cuda.is_available()

def load_data_quick(config: ColabConfig):
    """Quick data loading for Colab"""
    print("📦 Loading data (Colab optimized)...")
    
    # Load tokenizer
    tokenizer_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset (streaming for memory efficiency)
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:2000])  # Shorter texts for Colab
    
    print(f"Loaded {len(texts)} documents")
    
    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts[:500], desc="Tokenizing"):  # Limit for speed
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
    
    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size
    
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

def run_quick_experiment():
    """Run a quick experiment optimized for Colab"""
    print("🧪 Starting Quick LLM Comparison (Colab Optimized)")
    print("=" * 60)
    
    # Setup
    has_gpu = quick_setup()
    
    # Load data
    config = ColabConfig(use_amp=has_gpu)
    texts, tokenizer, tokens = load_data_quick(config)
    
    # Create datasets
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    val_size = min(len(dataset) // 10, 100)  # Limit validation size
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Test configurations
    models_to_test = [
        ("Regular Transformer", ColabConfig(vocab_size=config.vocab_size, use_amp=has_gpu)),
        ("Mixture of Experts", ColabMoEConfig(
            vocab_size=config.vocab_size,
            num_experts=8,
            expert_top_k=2,
            moe_layers="alternate",
            use_amp=has_gpu
        ))
    ]
    
    results = []
    
    for model_name, model_config in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {model_name}")
        print(f"{'='*60}")
        
        # For this demo, we'll simulate training results
        # In practice, you'd run your actual training code here
        start_time = time.time()
        
        # Simulate training time
        time.sleep(2)  # Simulate training
        
        # Mock results (replace with actual training)
        if "MoE" in model_name:
            final_metrics = {
                'val_loss': 0.0758,
                'val_accuracy': 0.9857,
                'val_perplexity': 1.08
            }
        else:
            final_metrics = {
                'val_loss': 0.1365,
                'val_accuracy': 0.9766,
                'val_perplexity': 1.15
            }
        
        total_time = time.time() - start_time
        
        results.append({
            'model': model_name,
            'time': total_time,
            'metrics': final_metrics
        })
        
        print(f"\n🎯 {model_name} Results:")
        print(f"⏱️ Training time: {total_time/60:.1f} minutes")
        print(f"🏆 Final Results:")
        print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    
    # Display comparison
    print(f"\n{'='*60}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*60}")
    
    for result in results:
        print(f"{result['model']:20} | Loss: {result['metrics']['val_loss']:.4f} | "
              f"Acc: {result['metrics']['val_accuracy']:.4f} | "
              f"PPL: {result['metrics']['val_perplexity']:.2f}")
    
    # Determine winner
    if len(results) >= 2:
        reg_loss = results[0]['metrics']['val_loss']
        moe_loss = results[1]['metrics']['val_loss']
        
        if moe_loss < reg_loss:
            improvement = ((reg_loss - moe_loss) / reg_loss) * 100
            print(f"\n🏆 MoE model wins! {improvement:.1f}% better validation loss")
        else:
            print(f"\n🏆 Regular Transformer wins!")
    
    print(f"\n✅ Experiment complete!")
    print("💡 To run the full training, replace the simulation with your actual training code")
    
    return results

if __name__ == "__main__":
    # Run the quick experiment
    results = run_quick_experiment()
    
    print("\n🚀 Next Steps:")
    print("1. Copy your full llm.py code into a Colab cell")
    print("2. Replace the simulation with actual training")
    print("3. Adjust config parameters as needed")
    print("4. Monitor GPU usage with !nvidia-smi")