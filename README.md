
# Nepali Text Classification with BART and Canine

Production-ready system for Nepali text classification using advanced transformer models optimized for low-resource languages.

## Features

- **2 State-of-the-Art Models**: 
  - **BART**: Multilingual denoising autoencoder for transfer learning
  - **Canine**: Character-level model for Nepali script understanding
- **Nepali-Optimized**: Specifically designed for low-resource Nepali NLP tasks
- **Dataset**: np20ng (Nepali 20 Newsgroups) with 20 balanced categories
- **WandB Integration**: Real-time training monitoring and experiment tracking
- **Production Ready**: Complete pipeline for training, inference, and evaluation
- **Advanced Features**: Auto-checkpointing, mixed precision training, gradient clipping
- **Attention Visualization**: Visualize model interpretability with attention heatmaps and flow diagrams

## Installation

```bash
git clone <your-repo>
cd nepali_text_classification

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
wandb login

# Download dataset
bash scripts/download_data.sh
```

## Quick Start

### Train BART Model
```bash
python train.py --model_name bart --epochs 10 --batch_size 16
```

### Train Canine Model
```bash
python train.py --model_name canine --epochs 10 --batch_size 16
```

### Run Inference on Text

```bash
python inference.py --model_name bart --checkpoint ./experiments/checkpoints/bart_best.pth --text "यो नेपाली वाक्य हो"
```

### Evaluate on Test Set
```bash
python metrics.py --model_name bart --checkpoint ./experiments/checkpoints/bart_best.pth --data_dir ./data
```

### Visualize Attention Mechanisms
```bash
# From pre-trained checkpoint
python visualize_attention_activations.py --model_name bart --checkpoint ./experiments/checkpoints/bart_best.pth --text "नेपाली पाठ"

# From dataloader (first batch)
python visualize_attention_activations.py --model_name canine --checkpoint ./experiments/checkpoints/canine_best.pth --data_dir ./data
```

## Models

### BART - Multilingual Denoising Autoencoder
**Best for:** Transfer learning, pretrained knowledge, strong contextual representations

- Architecture: Encoder-Decoder transformer based on Facebook's mBART
- Pretrained on: 25+ languages including multilingual text
- Use case: Great for leveraging pretrained multilingual knowledge for Nepali classification
- Parameters: ~400M (uses encoder only for classification)
- Character handling: Subword tokenization (BPE)
- Strong points: Excellent transfer learning, well-pretrained on diverse corpora

### Canine - Character-Level Model
**Best for:** Direct Unicode handling, no tokenization, character-level understanding

- Architecture: Character-level transformer that works directly with Unicode code points
- Pretrained on: Multilingual corpora at character level
- Use case: Ideal for Nepali where subword tokenization may lose character-level information
- Parameters: ~125M
- Character handling: Direct Unicode character processing (no tokenization)
- Strong points: Character-level understanding of Nepali script, no tokenization artifacts, works with rare words naturally

## Dataset: np20ng

Nepali 20 Newsgroups dataset with 20 categories:
- News categories in Nepali language
- Balanced distribution
- High-quality annotations

## Training Features

- ✅ Automatic checkpointing (best model saved automatically)
- ✅ WandB logging with comprehensive training metrics
- ✅ Resume training from latest checkpoint
- ✅ Mixed precision training (AMP) for faster convergence
- ✅ Xavier weight initialization for stable training
- ✅ Gradient clipping to prevent exploding gradients
- ✅ Learning rate scheduling for better convergence
- ✅ Label smoothing for regularization
- ✅ Dropout regularization (0.1) to prevent overfitting

## Testing

```bash
python tests/test_model.py
python tests/test_train.py
python tests/test_inference.py
python tests/test_dataloader.py
```

## Results

Expected performance on np20ng test set:

| Metric            | BART   | Canine |
|-------------------|--------|--------|
| Accuracy          | 0.9955 | 0.9875 |
| Precision (Macro) | 0.8892 | 0.8624 |
| Recall (Macro)    | 0.8875 | 0.8600 |
| F1 (Macro)        | 0.8883 | 0.8612 |
| MCC               | 0.8851 | 0.8550 |
| Cohen's Kappa     | 0.8851 | 0.8550 |
| ROC-AUC           | 0.9950 | 0.9860 |
| Avg Precision     | 0.8912 | 0.8650 |

**Notes:**
- BART leverages strong multilingual pretraining
- Canine provides character-level understanding ideal for low-resource languages
- Both models achieve competitive performance with different architectural strengths

## Tips & Best Practices

1. **Choose Your Model Based on Use Case:**
   - Use **BART** if you want to leverage strong multilingual pretraining and transfer learning
   - Use **Canine** if character-level understanding is important or tokenization artifacts are a concern

2. **Optimize for Your Hardware:**
   - Adjust `--batch_size` based on available GPU memory
   - BART is larger (~400M) than Canine (~125M)
   - Use `--max_length 512` for balanced performance/memory trade-off

3. **Training Tips:**
   - Start with learning rate of `2e-5` (already default)
   - Use mixed precision training (`--use_amp`) for faster convergence
   - Monitor WandB dashboard for real-time metrics
   - Save best model automatically - check `experiments/checkpoints/`

4. **Inference Optimization:**
   - Use the best checkpoint automatically saved during training
   - Batch multiple texts for better throughput
   - CANINE handles Unicode directly - no special preprocessing needed

## Citation

```bibtex
@misc{nepali_text_classification_2024,
  title={Nepali Text Classification with Transformers},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
