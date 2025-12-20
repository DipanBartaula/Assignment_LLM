"""
model.py - Transformer models for Nepali text classification

This module provides BART and Canine models for classifying Nepali text.
Both models are powerful transformer architectures designed for handling
sequence classification tasks in low-resource languages.

Supported Models:
- BART: A denoising autoencoder with a transformer-based encoder-decoder
- Canine: A character-level model that works directly with Unicode characters
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    BertForSequenceClassification,
    BartForSequenceClassification,
    ElectraForSequenceClassification,
    ReformerForSequenceClassification,
    MBartForSequenceClassification,
    CanineForSequenceClassification,
    T5ForConditionalGeneration,
    T5EncoderModel,
    AutoModel
)
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')



class BARTClassifier(nn.Module):
    """
    BART-based sequence classifier for Nepali text.
    
    BART is a denoising autoencoder with a transformer encoder-decoder architecture.
    For classification, we use the encoder's output and apply a classification head.
    The model is pretrained on large multilingual corpora, making it suitable for
    low-resource language tasks like Nepali text classification.
    """
    def __init__(self, num_classes: int, model_name: str = "facebook/mbart-large-cc25", dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        
        # Load the encoder for classification task
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.d_model, num_classes)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Get encoder outputs
        outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Extract and pool representations
        hidden_state = outputs.last_hidden_state
        # Use first token (like CLS token in BERT) for sequence representation
        pooled = hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled
        }





class CanineClassifier(nn.Module):
    """
    CANINE-based sequence classifier for Nepali text.
    
    CANINE (Character Architecture with No In-word N-gram Embeddings) is a novel
    character-level model that works directly on Unicode code points instead of
    subword tokens. This makes it particularly well-suited for low-resource 
    languages like Nepali, where proper tokenization is challenging and character-level
    understanding can be more effective.
    """
    def __init__(self, num_classes: int, model_name: str = "google/canine-s", dropout: float = 0.1):
        super().__init__()
        self.canine = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.canine.config.hidden_size, num_classes)
        
        # Initialize classifier weights for better training stability
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Get character-level representations from CANINE
        outputs = self.canine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Use the pooled output (similar to BERT's CLS token) for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, {
            'attentions': outputs.attentions,
            'hidden_states': outputs.hidden_states,
            'pooled_output': pooled_output
        }





def get_model(model_name: str, num_classes: int, dropout: float = 0.1) -> nn.Module:
    """
    Load a transformer model for Nepali text classification.
    
    This factory function instantiates either BART or Canine based on your specification.
    Both models are excellent choices for Nepali NLP tasks with different strengths:
    - BART: Better for pretraining and transfer learning tasks
    - Canine: Better for character-level understanding of Nepali script
    
    Args:
        model_name: Model to load - either 'bart' or 'canine'
        num_classes: Number of output classification categories
        dropout: Dropout rate for regularization (default: 0.1)
        
    Returns:
        Initialized model ready for training or inference
        
    Raises:
        ValueError: If model_name is not one of the supported options
    """
    
    model_map = {
        'bart': lambda: BARTClassifier(num_classes, dropout=dropout),
        'canine': lambda: CanineClassifier(num_classes, dropout=dropout),
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"Model '{model_name}' is not supported. Please choose from: {list(model_map.keys())}")
    
    model = model_map[model_name.lower()]()
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Successfully loaded {model_name.upper()} model")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Quick test of both models
    batch_size = 2
    seq_len = 128
    num_classes = 20
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    for model_name in ['bart', 'canine']:
        print(f"\nTesting {model_name.upper()}...")
        try:
            model = get_model(model_name, num_classes)
            model.eval()
            
            with torch.no_grad():
                logits, aux = model(input_ids, attention_mask)
                print(f"  Input shape: {input_ids.shape}, Output shape: {logits.shape}")
                print(f"  Auxiliary outputs available: {list(aux.keys())}")
        except Exception as e:
            print(f"  Error: {e}")