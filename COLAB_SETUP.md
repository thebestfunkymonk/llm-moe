# 🚀 Google Colab Setup Guide

## Quick Start (3 Steps)

### 1. **Upload Your Code**
- Open [Google Colab](https://colab.research.google.com/)
- Create a new notebook
- Upload `llm.py` to the session storage (📁 icon → Upload to session storage)

### 2. **Install Dependencies**
```python
# Run this cell first
!pip install datasets transformers torchtune torchao torch numpy tqdm

# Check GPU
import torch
print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 3. **Run Your Code**
```python
# Run your main script
%run llm.py
```

## 🎯 Colab-Optimized Configuration

For better Colab performance, modify your config:

```python
# Add this to your llm.py or run in a separate cell
@dataclass
class ColabConfig(ModelConfig):
    # Reduced for Colab timeouts
    max_steps: int = 1000      # Reduced from 3000
    batch_size: int = 16       # Reduced from 24  
    max_tokens: int = 200000   # Reduced from 500000
    num_documents: int = 1000  # Reduced from 2000
    eval_every: int = 200      # More frequent evaluation

@dataclass  
class ColabMoEConfig(MoEModelConfig):
    # MoE with Colab optimizations
    max_steps: int = 1000
    batch_size: int = 16
    max_tokens: int = 200000
    num_documents: int = 1000
    eval_every: int = 200
```

## 🔧 Alternative: Direct Code Execution

If you prefer not to upload files, copy your entire `llm.py` content into a Colab cell:

```python
# Paste your complete llm.py code here
import torch
import torch.nn as nn
# ... (rest of your code)

# Then run the main execution
if __name__ == "__main__":
    # Your main code here
```

## 📊 Expected Results

| Model | Parameters | Active Params | Val Loss | Val Acc | Val PPL | Training Time |
|-------|------------|---------------|----------|---------|---------|---------------|
| Regular Transformer | ~29M | 29M | 0.1365 | 0.9766 | 1.15 | ~2 min |
| MoE (8 experts) | ~54M | ~25M | 0.0758 | 0.9857 | 1.08 | ~4 min |

## ⚡ Performance Tips

### **Memory Management:**
```python
# Add this if you run out of memory
import gc
torch.cuda.empty_cache()
gc.collect()
```

### **Checkpoint Saving:**
```python
# Add to your training loop
if step % 500 == 0:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, f'checkpoint_step_{step}.pth')
```

### **Progress Monitoring:**
```python
# Monitor GPU usage
!nvidia-smi
```

## 🚨 Troubleshooting

### **Common Issues:**

1. **"CUDA out of memory"**
   - Reduce `batch_size` to 8 or 12
   - Reduce `max_tokens` to 100000
   - Use `torch.cuda.empty_cache()`

2. **"Session timeout"**
   - Use Colab Pro for longer sessions
   - Save checkpoints frequently
   - Reduce `max_steps` for quick tests

3. **"Package not found"**
   - Run `!pip install package_name` in a new cell
   - Restart runtime if needed

4. **"Slow training"**
   - Ensure GPU is enabled: Runtime → Change runtime type → GPU
   - Use T4 (free) or V100/A100 (Pro)

## 🎮 Interactive Features

### **Real-time Monitoring:**
```python
# Add this to monitor training
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_training_progress(losses, accuracies):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    
    ax2.plot(accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
```

### **Model Comparison:**
```python
# Compare models side by side
import pandas as pd

results_df = pd.DataFrame([
    {'Model': 'Regular Transformer', 'Val Loss': 0.1365, 'Val Acc': 0.9766},
    {'Model': 'MoE (8 experts)', 'Val Loss': 0.0758, 'Val Acc': 0.9857}
])

print(results_df.to_string(index=False))
```

## 🔗 GitHub Integration

If your code is on GitHub:

```python
# Clone your repository
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo

# Install dependencies
!pip install -r requirements.txt

# Run your code
!python llm.py
```

## 📱 Mobile-Friendly Version

For quick tests on mobile Colab:

```python
# Ultra-lightweight config
@dataclass
class MobileConfig(ModelConfig):
    max_steps: int = 100        # Very quick test
    batch_size: int = 8         # Small batch
    max_tokens: int = 50000     # Minimal data
    num_documents: int = 100    # Few documents
    eval_every: int = 50        # Frequent eval
```

## 🎯 Next Steps

1. **Run the basic version first** to ensure everything works
2. **Experiment with different configurations** (more experts, different MoE layers)
3. **Add visualization** to track training progress
4. **Try different datasets** by modifying the data loading
5. **Implement model saving/loading** for longer experiments

## 📞 Support

If you encounter issues:
- Check the [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- Ensure all dependencies are installed
- Try restarting the runtime
- Use the troubleshooting section above

Happy training! 🚀